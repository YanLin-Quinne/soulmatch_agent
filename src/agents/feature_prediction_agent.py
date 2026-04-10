"""Feature Prediction Agent — infer user traits with Bayesian updates + conformal prediction + CoT reasoning."""

import json
from typing import Dict, List, Optional, Any
from loguru import logger

from src.agents.bayesian_updater import BayesianFeatureUpdater
from src.agents.llm_router import router, AgentRole
from src.agents.reasoning import ChainOfThought, ReasoningTrace
from src.agents.conformal_calibrator import (
    ConformalCalibrator, ConformalResult, discretize_value, get_options,
    CATEGORICAL_DIMS, CONTINUOUS_BINS, ALL_DIMS,
)
from src.config import settings


# Confidence threshold: stop LLM extraction when average confidence exceeds this
CONVERGENCE_THRESHOLD = 0.80
# Minimum information gain to justify an LLM call
MIN_INFO_GAIN = 0.02
# Default calibrator path (fitted offline) — use absolute path via settings
DEFAULT_CALIBRATOR_PATH = str(settings.data_dir / "calibration" / "conformal_calibrator.json")


class FeaturePredictionAgent:
    """Predicts user features from conversation, using confidence-based stopping + conformal prediction."""

    def __init__(self, user_id: str, use_cot: bool = True, calibrator_path: Optional[str] = None):
        self.user_id = user_id
        self.bayesian_updater = BayesianFeatureUpdater()
        self.predicted_features: Dict[str, any] = {}
        self.feature_confidences: Dict[str, float] = {}
        self.conversation_count = 0
        self.use_cot = use_cot
        self.last_reasoning_trace: Optional[ReasoningTrace] = None

        # Conformal prediction calibrator
        self.conformal: Optional[ConformalCalibrator] = None
        self.last_conformal_result: Optional[ConformalResult] = None
        self._load_calibrator(calibrator_path or DEFAULT_CALIBRATOR_PATH)

    def _load_calibrator(self, path: str):
        """Load pre-fitted conformal calibrator if available."""
        try:
            from pathlib import Path
            if Path(path).exists():
                self.conformal = ConformalCalibrator()
                self.conformal.load(path)
                logger.info(f"Conformal calibrator loaded from {path}")
            else:
                logger.info("No conformal calibrator found — using LLM confidence only")
        except Exception as e:
            logger.warning(f"Failed to load conformal calibrator: {e}")
            self.conformal = None

    def predict_from_conversation(self, conversation_history: List[dict], post_threshold: bool = False) -> Dict[str, any]:
        """Extract and update features from conversation history.

        Args:
            conversation_history: list of {speaker, message} dicts.
            post_threshold: if True, allow periodic LLM refinement even after
                convergence (every 3rd turn) to keep predictions fresh.
        """
        self.conversation_count += 1

        # Check convergence — skip LLM if all features are high-confidence
        avg_conf = self._compute_overall_confidence()
        if avg_conf >= CONVERGENCE_THRESHOLD and self.conversation_count > 5:
            low = self._low_confidence_features()
            if not low:
                if post_threshold and self.conversation_count % 3 == 0:
                    logger.info(f"Post-threshold refinement at turn {self.conversation_count} (avg conf={avg_conf:.2f})")
                else:
                    logger.info(f"Features converged (avg conf={avg_conf:.2f}), skipping LLM extraction")
                    return self._result()

        new_obs = self._extract_features_llm(conversation_history)
        if not new_obs:
            return self._result()

        updated_f, updated_c = self._update_features(new_obs)
        self.predicted_features = updated_f
        self.feature_confidences = updated_c

        # ── v2.0: Infer MBTI from Big Five ──────────
        self._infer_mbti_from_big_five()

        # ── v2.0: Infer attachment style from personality ──────────
        self._infer_attachment_style()

        # ── Conformal Prediction: calibrate confidence scores ──────────
        if self.conformal and self.conformal.is_fitted:
            self.last_conformal_result = self.conformal.calibrate(
                predictions=self.predicted_features,
                llm_confidences=self.feature_confidences,
                turn=self.conversation_count,
            )
            logger.info(
                f"Conformal: turn={self.conversation_count}, "
                f"avg_set_size={self.last_conformal_result.avg_set_size:.2f}, "
                f"singletons={self.last_conformal_result.n_singleton}/{self.last_conformal_result.n_dimensions}"
            )

        logger.info(f"Feature update turn={self.conversation_count}, avg_conf={self._compute_overall_confidence():.2f}")
        return self._result()

    def _extract_features_llm(self, conversation_history: List[dict]) -> Optional[Dict[str, any]]:
        conversation_text = "\n".join(
            f"{msg['speaker']}: {msg['message']}" for msg in conversation_history[-10:]
        )

        if self.conversation_count >= 5 and len(conversation_history) >= 5:
            result = self._extract_features_contrastive(conversation_history)
            if result is not None:
                return result
            logger.debug("Contrastive extraction returned None, falling back")

        if self.use_cot and self.conversation_count >= 3:
            result = self._extract_features_with_cot(conversation_text)
            if result is not None:
                return result
            logger.debug("CoT extraction returned None, falling back to direct extraction")

        return self._extract_features_direct(conversation_text)

    def _extract_features_with_cot(self, conversation_text: str) -> Optional[Dict[str, any]]:
        """Use Chain-of-Thought reasoning to analyze the conversation step by step before extracting features."""
        low_conf = self._low_confidence_features()
        focus_hint = ""
        if low_conf:
            focus_hint = f"\nFocus especially on: {', '.join(low_conf[:6])}\n"

        # Current predictions context for the reasoner
        current_state = ""
        if self.predicted_features:
            current_state = "\nCurrent predictions (update these):\n" + json.dumps(
                {k: f"{v} (conf={self.feature_confidences.get(k, 0):.0%})"
                 for k, v in list(self.predicted_features.items())[:10]},
                indent=2,
            )

        cot_prompt = (
            f"Analyze this dating app conversation to infer personality traits.\n\n"
            f"Conversation:\n{conversation_text}\n{focus_hint}{current_state}\n\n"
            f"Think step by step:\n"
            f"1. What topics did the user bring up or respond to enthusiastically?\n"
            f"2. What does their language style reveal about personality (formal/casual, direct/indirect)?\n"
            f"3. What values or priorities are evident?\n"
            f"4. What Big Five traits can be inferred from their behavior?\n"
            f"5. What interests are clearly demonstrated vs. speculative?\n\n"
            f"After reasoning, produce the final feature JSON (same schema as before)."
        )

        try:
            trace = ChainOfThought.reason(
                role=AgentRole.FEATURE,
                system="You are a psychology expert analyzing dating conversations. Reason carefully before concluding.",
                messages=[{"role": "user", "content": cot_prompt}],
                temperature=0.3,
                max_tokens=1200,
            )
            self.last_reasoning_trace = trace
            logger.debug(f"CoT reasoning: {len(trace.steps)} steps, conf={trace.confidence:.2f}")

            # Try to extract JSON from the conclusion
            answer = trace.final_answer
            # Look for JSON block in the answer
            json_match = None
            if "{" in answer:
                start = answer.index("{")
                depth = 0
                for i in range(start, len(answer)):
                    if answer[i] == "{":
                        depth += 1
                    elif answer[i] == "}":
                        depth -= 1
                    if depth == 0:
                        json_match = answer[start:i+1]
                        break
            if json_match:
                return json.loads(json_match)
            return None

        except Exception as e:
            logger.error(f"CoT feature extraction failed: {e}")
            return None

    def _extract_features_contrastive(self, conversation_history: List[dict]) -> Optional[Dict[str, any]]:
        """Contrastive decoding: compare full-context vs minimal-context extraction to detect hallucinations.

        Inspired by Clinical Contrastive Decoding (CCD). If a feature appears with high confidence
        in the full-context extraction but has NO evidence in the minimal-context (last 3 turns),
        it's likely hallucinated and its confidence should be reduced.
        """
        full_text = "\n".join(f"{msg['speaker']}: {msg['message']}" for msg in conversation_history[-10:])
        full_result = self._extract_features_direct(full_text)

        if not full_result:
            return None

        minimal_text = "\n".join(f"{msg['speaker']}: {msg['message']}" for msg in conversation_history[-3:])
        minimal_result = self._extract_features_direct(minimal_text)

        if not minimal_result:
            return full_result

        full_conf = full_result.get("_confidence", {})
        minimal_conf = minimal_result.get("_confidence", {})

        adjusted_conf = dict(full_conf)
        hallucination_flags = []

        for key in full_conf:
            full_c = full_conf.get(key, 0)
            minimal_c = minimal_conf.get(key, 0)

            if full_c > 0.5 and minimal_c < 0.2:
                adjusted_conf[key] = full_c * 0.5
                hallucination_flags.append(key)
                logger.warning(
                    f"[Contrastive] Potential hallucination on '{key}': "
                    f"full_conf={full_c:.2f}, minimal_conf={minimal_c:.2f}"
                )
            elif full_c > 0.3 and minimal_c < 0.1:
                adjusted_conf[key] = full_c * 0.7
                hallucination_flags.append(key)

        full_result["_confidence"] = adjusted_conf
        full_result["_hallucination_flags"] = hallucination_flags

        if hallucination_flags:
            logger.info(
                f"[Contrastive] Flagged {len(hallucination_flags)} potential hallucinations: "
                f"{hallucination_flags}"
            )

        return full_result

    def _extract_features_direct(self, conversation_text: str) -> Optional[Dict[str, any]]:
        """Direct LLM extraction without reasoning chain (fallback)."""
        low_conf = self._low_confidence_features()
        focus_hint = ""
        if low_conf:
            focus_hint = f"\nPay special attention to: {', '.join(low_conf[:8])}\n"

        prompt = f"""Analyze this dating app conversation and infer the user's features. Respond ONLY with valid JSON.

Conversation:
{conversation_text}
{focus_hint}
Infer the following features (use null if uncertain):

{{
  "age": <estimated age 18-80 or null>,
  "sex": "<m/f or null>",
  "orientation": "<straight/gay/bisexual or null>",
  "location": "<city or null>",
  "big_five": {{
    "openness": <0.0-1.0 or null>,
    "conscientiousness": <0.0-1.0 or null>,
    "extraversion": <0.0-1.0 or null>,
    "agreeableness": <0.0-1.0 or null>,
    "neuroticism": <0.0-1.0 or null>
  }},
  "lifestyle": {{
    "diet": "<omnivore/vegetarian/vegan/other or null>",
    "drinks": "<not_at_all/rarely/socially/often/very_often or null>",
    "smokes": "<no/sometimes/yes or null>",
    "drugs": "<never/sometimes or null>"
  }},
  "background": {{
    "education": "<high_school/some_college/bachelors/masters/phd or null>",
    "job": "<field or null>",
    "religion": "<atheist/agnostic/christian/jewish/muslim/hindu/buddhist/other or null>"
  }},
  "interests": {{
    "music": <0.0-1.0>, "sports": <0.0-1.0>, "travel": <0.0-1.0>, "food": <0.0-1.0>,
    "arts": <0.0-1.0>, "tech": <0.0-1.0>, "outdoors": <0.0-1.0>, "books": <0.0-1.0>
  }},
  "communication_style": "<direct/indirect/humorous/serious/casual/formal or null>",
  "relationship_goals": "<casual/serious/friendship/unsure or null>",
  "_confidence": {{
    "age": <0.0-1.0>, "sex": <0.0-1.0>, "orientation": <0.0-1.0>,
    "big_five_openness": <0.0-1.0>, "big_five_conscientiousness": <0.0-1.0>,
    "big_five_extraversion": <0.0-1.0>, "big_five_agreeableness": <0.0-1.0>,
    "big_five_neuroticism": <0.0-1.0>,
    "lifestyle_diet": <0.0-1.0>, "lifestyle_drinks": <0.0-1.0>,
    "background_education": <0.0-1.0>,
    "interests_music": <0.0-1.0>,
    "communication_style": <0.0-1.0>, "relationship_goals": <0.0-1.0>
  }}
}}

Guidelines:
- Only infer what can be reasonably deduced
- Confidence: 0.0=no evidence, 1.0=very strong evidence
- Big Five: 0=low, 0.5=average, 1=high
- Interests: 0=none, 1=very strong"""

        try:
            text = router.chat(
                role=AgentRole.FEATURE,
                system="You are a psychology expert analyzing dating conversations to infer personality traits.",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1024,
                json_mode=True,
            )
            text = text.strip()
            if text.startswith("```json"):
                text = text[7:]
            if text.startswith("```"):
                text = text[3:]
            if text.endswith("```"):
                text = text[:-3]
            return json.loads(text.strip())
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return None

    def _update_features(self, new_obs: Dict[str, any]) -> tuple[Dict, Dict]:
        confidences = new_obs.pop("_confidence", {})
        hallucination_flags = {
            f"interest_{key[len('interests_'):]}" if key.startswith("interests_") else key
            for key in new_obs.pop("_hallucination_flags", [])
        }

        flat_obs: Dict[str, any] = {}
        flat_conf: Dict[str, float] = {}

        for key in ("age", "sex", "orientation", "location", "communication_style", "relationship_goals"):
            if key in new_obs and new_obs[key] is not None:
                flat_obs[key] = new_obs[key]
                flat_conf[key] = confidences.get(key, 0.5)

        for group in ("big_five", "lifestyle", "background", "interests"):
            if group in new_obs:
                for k, v in new_obs[group].items():
                    if v is not None:
                        fk = f"{group}_{k}" if group != "interests" else f"interest_{k}"
                        flat_obs[fk] = v
                        flat_conf[fk] = confidences.get(f"{group}_{k}" if group != "interests" else f"interests_{k}", 0.5)

        numeric_keys = [k for k in flat_obs if k.startswith("big_five_") or k.startswith("interest_")]
        categorical_keys = [k for k in flat_obs if k not in numeric_keys]

        updated_f: Dict[str, any] = {}
        updated_c: Dict[str, float] = {}

        for key in numeric_keys:
            pv = self.predicted_features.get(key, 0.5)
            pc = self.feature_confidences.get(key, 0.3)
            obs_conf = flat_conf[key]
            if key in hallucination_flags:
                obs_conf *= 0.5
                logger.debug(f"[Contrastive] Lowering Bayesian observation precision for flagged feature '{key}'")
            nv, nc = self.bayesian_updater.update_feature(pv, pc, flat_obs[key], obs_conf)
            updated_f[key] = nv
            updated_c[key] = nc

        for key in categorical_keys:
            nv, nc = flat_obs[key], flat_conf[key]
            if key in self.predicted_features:
                pc = self.feature_confidences.get(key, 0.3)
                if nc > pc:
                    updated_f[key] = nv
                    updated_c[key] = nc
                else:
                    updated_f[key] = self.predicted_features[key]
                    updated_c[key] = pc
            else:
                updated_f[key] = nv
                updated_c[key] = nc

        return updated_f, updated_c

    def _result(self) -> Dict[str, any]:
        result = {
            "features": self.predicted_features,
            "confidences": self.feature_confidences,
            "turn": self.conversation_count,
            "low_confidence": self._low_confidence_features(),
            "average_confidence": self._compute_overall_confidence(),
        }
        # Attach conformal prediction results if available
        if self.last_conformal_result:
            cr = self.last_conformal_result
            result["conformal"] = {
                "coverage_guarantee": cr.coverage_level,
                "avg_set_size": round(cr.avg_set_size, 2),
                "singletons": cr.n_singleton,
                "total_dims": cr.n_dimensions,
                "prediction_sets": {
                    dim: {
                        "set": ps.prediction_set,
                        "point": ps.point_prediction,
                        "size": ps.set_size,
                        "llm_conf": round(ps.llm_confidence, 3),
                        "calibrated_conf": round(ps.calibrated_confidence, 3),
                    }
                    for dim, ps in cr.prediction_sets.items()
                },
            }
        return result

    def _low_confidence_features(self, threshold: float = 0.5) -> List[str]:
        """Return feature keys with confidence below threshold."""
        all_keys = [
            "age", "sex", "orientation", "communication_style", "relationship_goals",
            "big_five_openness", "big_five_conscientiousness", "big_five_extraversion",
            "big_five_agreeableness", "big_five_neuroticism",
            "interest_music", "interest_sports", "interest_travel", "interest_food",
            "interest_arts", "interest_tech", "interest_outdoors", "interest_books",
        ]
        return [k for k in all_keys if self.feature_confidences.get(k, 0.0) < threshold]

    def _compute_overall_confidence(self) -> float:
        if not self.feature_confidences:
            return 0.0
        return sum(self.feature_confidences.values()) / len(self.feature_confidences)

    def compute_perception_accuracy(self, ground_truth: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Compute perception accuracy metrics.

        If *ground_truth* is provided (e.g. from a user profile), accuracy is
        calculated as the proportion of features whose predicted value matches
        or is within a tolerance of the true value.  Otherwise a proxy accuracy
        is derived from model confidence and prediction stability.

        Returns:
            A dict with overall_accuracy (0-1), per_feature scores, and
            metadata useful for the frontend AccuracyGauge.
        """
        if not self.predicted_features:
            return {"overall_accuracy": 0.0, "n_features": 0, "per_feature": {}, "method": "none"}

        per_feature: Dict[str, float] = {}

        if ground_truth:
            # --- Ground-truth comparison ---
            total, correct = 0, 0.0
            for key, true_val in ground_truth.items():
                pred_val = self.predicted_features.get(key)
                if pred_val is None:
                    continue
                total += 1
                if isinstance(true_val, (int, float)) and isinstance(pred_val, (int, float)):
                    # Continuous: accuracy = 1 - |error| clamped to [0, 1]
                    error = abs(pred_val - true_val)
                    score = max(0.0, 1.0 - error)
                    per_feature[key] = round(score, 3)
                    correct += score
                else:
                    # Categorical: exact match
                    match = 1.0 if str(pred_val).lower() == str(true_val).lower() else 0.0
                    per_feature[key] = match
                    correct += match
            overall = round(correct / total, 3) if total > 0 else 0.0
            return {
                "overall_accuracy": overall,
                "n_features": total,
                "per_feature": per_feature,
                "method": "ground_truth",
            }

        # --- Proxy accuracy: weighted confidence + stability heuristic ---
        high_conf_count = 0
        conf_sum = 0.0
        for key, val in self.predicted_features.items():
            conf = self.feature_confidences.get(key, 0.0)
            per_feature[key] = round(conf, 3)
            conf_sum += conf
            if conf >= 0.7:
                high_conf_count += 1

        n = len(self.predicted_features)
        avg_conf = conf_sum / n if n else 0.0

        # Stability bonus: if conformal calibrator narrows prediction sets
        stability_bonus = 0.0
        if self.last_conformal_result and self.last_conformal_result.n_dimensions > 0:
            singleton_ratio = (
                self.last_conformal_result.n_singleton / self.last_conformal_result.n_dimensions
            )
            stability_bonus = singleton_ratio * 0.1  # up to 10% bonus

        overall = round(min(1.0, avg_conf + stability_bonus), 3)
        return {
            "overall_accuracy": overall,
            "n_features": n,
            "high_confidence_count": high_conf_count,
            "per_feature": per_feature,
            "stability_bonus": round(stability_bonus, 3),
            "method": "proxy",
        }

    def get_feature_summary(self) -> Dict[str, any]:
        summary = {"demographics": {}, "personality": {}, "lifestyle": {}, "interests": {},
                   "overall_confidence": self._compute_overall_confidence()}
        for key in ("age", "sex", "orientation", "location"):
            if key in self.predicted_features:
                summary["demographics"][key] = {"value": self.predicted_features[key], "confidence": self.feature_confidences.get(key, 0.0)}
        for trait in ("openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"):
            k = f"big_five_{trait}"
            if k in self.predicted_features:
                summary["personality"][trait] = {"value": self.predicted_features[k], "confidence": self.feature_confidences.get(k, 0.0)}
        for interest in ("music", "sports", "travel", "food", "arts", "tech", "outdoors", "books"):
            k = f"interest_{interest}"
            if k in self.predicted_features:
                summary["interests"][interest] = {"value": self.predicted_features[k], "confidence": self.feature_confidences.get(k, 0.0)}

        # Conformal prediction summary
        if self.last_conformal_result:
            cr = self.last_conformal_result
            summary["conformal"] = {
                "coverage_guarantee": cr.coverage_level,
                "avg_set_size": round(cr.avg_set_size, 2),
                "singletons": cr.n_singleton,
                "total_dims": cr.n_dimensions,
                "determined_features": [
                    dim for dim, ps in cr.prediction_sets.items() if ps.set_size == 1
                ],
                "uncertain_features": [
                    dim for dim, ps in cr.prediction_sets.items() if ps.set_size > 2
                ],
            }

        return summary

    def _infer_mbti_from_big_five(self):
        """v2.0: 从Big Five映射推断MBTI类型"""
        # 基于心理测量学相关性映射
        # I/E ← extraversion (r=0.74)
        # S/N ← openness (r=0.72)
        # T/F ← agreeableness (r=-0.44)
        # J/P ← conscientiousness (r=0.49)

        e = self.predicted_features.get("big_five_extraversion")
        o = self.predicted_features.get("big_five_openness")
        a = self.predicted_features.get("big_five_agreeableness")
        c = self.predicted_features.get("big_five_conscientiousness")

        if None in [e, o, a, c]:
            return  # 需要完整Big Five才能推断

        # 计算MBTI轴
        mbti_ei = e  # 外向性直接映射
        mbti_sn = o  # 开放性映射到直觉
        mbti_tf = 1.0 - a  # 宜人性低→思考型
        mbti_jp = c  # 尽责性映射到判断型

        # 确定类型
        ei = "E" if mbti_ei > 0.5 else "I"
        sn = "N" if mbti_sn > 0.5 else "S"
        tf = "T" if mbti_tf > 0.5 else "F"
        jp = "J" if mbti_jp > 0.5 else "P"
        mbti_type = f"{ei}{sn}{tf}{jp}"

        # 置信度 = Big Five平均置信度
        avg_conf = (
            self.feature_confidences.get("big_five_extraversion", 0.5) +
            self.feature_confidences.get("big_five_openness", 0.5) +
            self.feature_confidences.get("big_five_agreeableness", 0.5) +
            self.feature_confidences.get("big_five_conscientiousness", 0.5)
        ) / 4.0

        self.predicted_features["mbti_type"] = mbti_type
        self.predicted_features["mbti_ei"] = mbti_ei
        self.predicted_features["mbti_sn"] = mbti_sn
        self.predicted_features["mbti_tf"] = mbti_tf
        self.predicted_features["mbti_jp"] = mbti_jp
        self.feature_confidences["mbti_type"] = avg_conf * 0.8  # 降低20%因为是推断
        self.feature_confidences["mbti_confidence"] = avg_conf * 0.8

        logger.debug(f"Inferred MBTI: {mbti_type} (conf={avg_conf*0.8:.2f})")

    def _infer_attachment_style(self):
        """v2.0: 从人格特征推断依恋风格"""
        # 简化映射规则:
        # 高神经质 + 低宜人性 → anxious
        # 低外向性 + 高神经质 → avoidant
        # 低神经质 + 高宜人性 → secure

        n = self.predicted_features.get("big_five_neuroticism")
        a = self.predicted_features.get("big_five_agreeableness")
        e = self.predicted_features.get("big_five_extraversion")

        if None in [n, a, e]:
            return

        # 计算依恋维度
        attachment_anxiety = n  # 神经质→焦虑
        attachment_avoidance = 1.0 - e  # 内向→回避

        # 确定风格
        if n < 0.4 and a > 0.6:
            style = "secure"
        elif n > 0.6 and a < 0.4:
            style = "anxious"
        elif e < 0.4 and n > 0.5:
            style = "avoidant"
        else:
            style = "disorganized"

        avg_conf = (
            self.feature_confidences.get("big_five_neuroticism", 0.5) +
            self.feature_confidences.get("big_five_agreeableness", 0.5) +
            self.feature_confidences.get("big_five_extraversion", 0.5)
        ) / 3.0

        self.predicted_features["attachment_style"] = style
        self.predicted_features["attachment_anxiety"] = attachment_anxiety
        self.predicted_features["attachment_avoidance"] = attachment_avoidance
        self.feature_confidences["attachment_style"] = avg_conf * 0.7

        logger.debug(f"Inferred attachment: {style} (conf={avg_conf*0.7:.2f})")
