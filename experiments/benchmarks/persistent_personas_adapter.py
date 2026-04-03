"""Persistent Personas benchmark adapter."""

from __future__ import annotations

from typing import Dict, List, Optional

from loguru import logger

from experiments.metrics import BIG_FIVE_TRAITS, compute_personality_metrics
from experiments.benchmarks.memory_metrics import temporal_consistency
from src.agents.llm_router import router, AgentRole, MODEL_ROUTING, MODELS


DEFAULT_CHECKPOINTS = [10, 30, 50, 80, 100]


class PersistentPersonasAdapter:
    """Adapter for Persistent Personas-style evaluation."""

    def __init__(self, llm_router=None, use_llm_assistance: bool = False):
        self.router = llm_router or router
        self.use_llm_assistance = use_llm_assistance

    def generate_extended_dialogue(self, persona: Dict, n_turns: int = 100) -> List[Dict]:
        """Generate extended dialogue with consistent persona."""
        dialogue: List[Dict] = []
        topic_cycle = [
            "intro",
            "weekend",
            "work",
            "food",
            "travel",
            "friends",
            "stress",
            "values",
            "hobbies",
            "future",
        ]

        for turn_idx in range(max(2, n_turns)):
            topic = topic_cycle[(turn_idx // 2) % len(topic_cycle)]
            if turn_idx % 2 == 0:
                dialogue.append(
                    {
                        "speaker": "match",
                        "message": self._interviewer_prompt(topic, turn_idx, persona),
                    }
                )
            else:
                dialogue.append(
                    {
                        "speaker": "user",
                        "message": self._persona_reply(topic, turn_idx, persona),
                    }
                )

        return self._maybe_rewrite_dialogue(dialogue[:n_turns], persona)

    def measure_fidelity_trajectory(
        self,
        predict_fn,
        dialogue: List[Dict],
        ground_truth: Dict,
        checkpoints: List[int] = DEFAULT_CHECKPOINTS,
    ) -> Dict:
        """Measure personality prediction accuracy at multiple checkpoints."""
        valid_checkpoints = sorted({cp for cp in checkpoints if 0 < cp <= len(dialogue)})
        checkpoint_results: Dict[int, Dict] = {}
        temporal_series: List[Dict] = []

        for checkpoint in valid_checkpoints:
            subset = dialogue[:checkpoint]
            raw_prediction = self._predict_persona(predict_fn, subset)
            normalized = self._normalize_prediction(raw_prediction)
            metrics = compute_personality_metrics([normalized], [ground_truth])

            checkpoint_results[checkpoint] = {
                "mae": metrics.get("big_five_mae", 0.0),
                "rmse": metrics.get("big_five_rmse", 0.0),
                "mbti_correct": normalized.get("mbti") == ground_truth.get("mbti"),
                "big_five": normalized.get("big_five", {}),
                "mbti": normalized.get("mbti"),
                "raw_prediction": raw_prediction,
            }
            temporal_series.append(
                {
                    "turn": checkpoint,
                    "big_five": normalized.get("big_five", {}),
                    "mbti": normalized.get("mbti"),
                }
            )

        consistency = temporal_consistency(temporal_series, ground_truth)
        final_checkpoint = valid_checkpoints[-1] if valid_checkpoints else 0
        final_prediction = checkpoint_results.get(final_checkpoint, {})
        final_accuracy = compute_personality_metrics(
            [self._normalize_prediction(final_prediction)],
            [ground_truth],
        ) if final_prediction else {}

        return {
            "checkpoints": checkpoint_results,
            "drift_score": consistency["drift_magnitude"],
            "convergence_turn": consistency["convergence_speed"],
            "final_accuracy": final_accuracy,
            "temporal_consistency": consistency,
        }

    def evaluate_systems(self, systems: Dict, personas: List[Dict], n_turns: int = 100) -> Dict:
        """Compare multiple systems' persona fidelity over extended dialogue."""
        results = {}
        checkpoint_template = sorted({cp for cp in DEFAULT_CHECKPOINTS if cp <= n_turns})

        for system_name, system in systems.items():
            per_persona = []
            aggregate_checkpoints = {
                checkpoint: {"mae": [], "mbti_correct": []}
                for checkpoint in checkpoint_template
            }
            final_predictions = []
            final_ground_truths = []
            drift_scores = []
            convergence_turns = []

            for persona in personas:
                dialogue = self.generate_extended_dialogue(persona, n_turns=n_turns)
                trajectory = self.measure_fidelity_trajectory(system, dialogue, persona, checkpoint_template)
                per_persona.append(
                    {
                        "persona_id": persona["id"],
                        "trajectory": trajectory,
                    }
                )

                for checkpoint, metrics in trajectory["checkpoints"].items():
                    aggregate_checkpoints[checkpoint]["mae"].append(metrics["mae"])
                    aggregate_checkpoints[checkpoint]["mbti_correct"].append(1.0 if metrics["mbti_correct"] else 0.0)

                final_prediction = trajectory["checkpoints"][checkpoint_template[-1]]
                final_predictions.append(self._normalize_prediction(final_prediction))
                final_ground_truths.append(persona)
                drift_scores.append(trajectory["drift_score"])
                convergence_turns.append(trajectory["convergence_turn"])

            results[system_name] = {
                "checkpoints": {
                    checkpoint: {
                        "avg_mae": self._mean(values["mae"]),
                        "mbti_accuracy": self._mean(values["mbti_correct"]),
                    }
                    for checkpoint, values in aggregate_checkpoints.items()
                },
                "avg_drift_score": self._mean(drift_scores),
                "avg_convergence_turn": self._mean(convergence_turns),
                "final_accuracy": compute_personality_metrics(final_predictions, final_ground_truths),
                "per_persona": per_persona,
            }

        return results

    def generate_personas(self, n_personas: int = 5) -> List[Dict]:
        base_personas = [
            {
                "id": "persona_1",
                "name": "Maya",
                "job": "UX designer",
                "city": "Bristol",
                "diet": "vegetarian",
                "interests": ["gallery hopping", "yoga", "recipe testing"],
                "communication_style": "warm and playful",
                "relationship_goals": "serious",
                "big_five": {
                    "openness": 0.84,
                    "conscientiousness": 0.68,
                    "extraversion": 0.74,
                    "agreeableness": 0.80,
                    "neuroticism": 0.34,
                },
                "mbti": "ENFP",
            },
            {
                "id": "persona_2",
                "name": "Jonah",
                "job": "civil engineer",
                "city": "Leeds",
                "diet": "omnivore",
                "interests": ["distance running", "board games", "meal prep"],
                "communication_style": "direct and grounded",
                "relationship_goals": "serious",
                "big_five": {
                    "openness": 0.46,
                    "conscientiousness": 0.88,
                    "extraversion": 0.42,
                    "agreeableness": 0.63,
                    "neuroticism": 0.29,
                },
                "mbti": "ISTJ",
            },
            {
                "id": "persona_3",
                "name": "Leila",
                "job": "radio producer",
                "city": "Glasgow",
                "diet": "vegan",
                "interests": ["live music", "night walks", "poetry"],
                "communication_style": "expressive and curious",
                "relationship_goals": "unsure",
                "big_five": {
                    "openness": 0.91,
                    "conscientiousness": 0.44,
                    "extraversion": 0.79,
                    "agreeableness": 0.69,
                    "neuroticism": 0.58,
                },
                "mbti": "ENTP",
            },
            {
                "id": "persona_4",
                "name": "Noah",
                "job": "physical therapist",
                "city": "Cardiff",
                "diet": "omnivore",
                "interests": ["climbing", "podcasts", "community coaching"],
                "communication_style": "calm and encouraging",
                "relationship_goals": "friendship",
                "big_five": {
                    "openness": 0.59,
                    "conscientiousness": 0.71,
                    "extraversion": 0.67,
                    "agreeableness": 0.83,
                    "neuroticism": 0.22,
                },
                "mbti": "ESFJ",
            },
            {
                "id": "persona_5",
                "name": "Aisha",
                "job": "research librarian",
                "city": "Oxford",
                "diet": "vegetarian",
                "interests": ["history books", "tea shops", "solo travel"],
                "communication_style": "thoughtful and precise",
                "relationship_goals": "serious",
                "big_five": {
                    "openness": 0.78,
                    "conscientiousness": 0.81,
                    "extraversion": 0.31,
                    "agreeableness": 0.76,
                    "neuroticism": 0.37,
                },
                "mbti": "INFJ",
            },
        ]
        return base_personas[: max(1, n_personas)]

    def _interviewer_prompt(self, topic: str, turn_idx: int, persona: Dict) -> str:
        prompts = {
            "intro": "What usually tells people who you are within the first few minutes?",
            "weekend": "What does an ideal weekend look like for you?",
            "work": "What part of your work actually energizes you?",
            "food": "What do you usually eat when you want comfort, not convenience?",
            "travel": "How do you approach trips: strict plan or loose sketch?",
            "friends": "What do your friends rely on you for most?",
            "stress": "What happens when your week gets messy?",
            "values": "What do you protect even when life gets busy?",
            "hobbies": "Which hobbies actually stay in your routine?",
            "future": "What kind of relationship are you hoping to build next?",
        }
        return prompts.get(topic, f"What has been on your mind around turn {turn_idx + 1}?")

    def _persona_reply(self, topic: str, turn_idx: int, persona: Dict) -> str:
        trait = persona["big_five"]
        interest_a = persona["interests"][turn_idx % len(persona["interests"])]
        interest_b = persona["interests"][(turn_idx + 1) % len(persona["interests"])]

        openness_line = (
            "I get restless if life feels too repetitive."
            if trait["openness"] >= 0.65
            else "I like familiar routines more than novelty for novelty's sake."
        )
        conscientiousness_line = (
            "I usually plan things properly and keep a running list."
            if trait["conscientiousness"] >= 0.65
            else "I improvise more than I probably should."
        )
        extraversion_line = (
            "I come alive around people and ideas bouncing around."
            if trait["extraversion"] >= 0.6
            else "I need quiet space before I feel like myself again."
        )
        agreeableness_line = (
            "People matter to me, so I try to keep the tone kind even when I'm direct."
            if trait["agreeableness"] >= 0.65
            else "I'm honest first and soft around the edges second."
        )
        neuroticism_line = (
            "I can overthink things a bit once the day finally slows down."
            if trait["neuroticism"] >= 0.5
            else "I stay fairly even unless something genuinely important is off."
        )

        replies = {
            "intro": (
                f"I'm {persona['name']}, I live in {persona['city']}, and I work as a {persona['job']}. "
                f"{extraversion_line}"
            ),
            "weekend": (
                f"My weekends usually include {interest_a} and then something slower with friends. "
                f"{openness_line}"
            ),
            "work": (
                f"I like that {persona['job']} work lets me be useful and thoughtful at the same time. "
                f"{conscientiousness_line}"
            ),
            "food": (
                f"I'm {persona['diet']}, so comfort food for me is usually something built around vegetables and good texture. "
                f"I'll happily talk about {interest_b} over dinner."
            ),
            "travel": (
                f"I travel with {'a rough map and a few anchor bookings' if trait['conscientiousness'] >= 0.65 else 'a list of possibilities and room to wander'}. "
                f"{openness_line}"
            ),
            "friends": (
                f"My friends come to me when they want {'steady advice' if trait['conscientiousness'] >= 0.65 else 'an honest read on things'}. "
                f"{agreeableness_line}"
            ),
            "stress": (
                f"When things get messy I {'tighten my routines' if trait['conscientiousness'] >= 0.65 else 'try to reset by changing scenery for a bit'}. "
                f"{neuroticism_line}"
            ),
            "values": (
                f"I protect curiosity, kindness, and making time for {interest_a}. "
                f"{agreeableness_line}"
            ),
            "hobbies": (
                f"The hobbies that actually stick are {interest_a} and {interest_b}. "
                f"{conscientiousness_line}"
            ),
            "future": (
                f"I'm looking for something {persona['relationship_goals']}, with communication that feels {persona['communication_style']}. "
                f"{extraversion_line}"
            ),
        }
        return replies.get(topic, f"I'd probably still bring it back to {interest_a} and how I like to live.")

    def _maybe_rewrite_dialogue(self, dialogue: List[Dict], persona: Dict) -> List[Dict]:
        if not self._llm_enabled(AgentRole.PERSONA):
            return dialogue
        try:
            prompt = f"""Make this dialogue slightly more natural while preserving persona facts exactly.

Persona:
{persona}

Dialogue:
{dialogue}

Return JSON only as a list of {{"speaker": "...", "message": "..."}} objects.
"""
            response = self.router.chat(
                role=AgentRole.PERSONA,
                system="You improve dialogue fluency while preserving semantics and personality cues exactly.",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4,
                max_tokens=2500,
                json_mode=True,
                timeout=20.0,
            )
            parsed = self._safe_parse_json(response)
            if isinstance(parsed, list) and parsed:
                return parsed[: len(dialogue)]
        except Exception as exc:
            logger.debug(f"[PersistentPersonasAdapter] Dialogue rewrite skipped: {exc}")
        return dialogue

    def _predict_persona(self, predict_fn, dialogue: List[Dict]) -> Dict:
        if hasattr(predict_fn, "predict_personality"):
            return predict_fn.predict_personality(dialogue) or {}
        if callable(predict_fn):
            return predict_fn(dialogue) or {}
        raise TypeError("predict_fn must be callable or expose predict_personality(dialogue)")

    def _normalize_prediction(self, prediction: Optional[Dict]) -> Dict:
        prediction = prediction or {}
        big_five = prediction.get("big_five", {})
        features = prediction.get("features", {})

        normalized_big_five = {}
        for trait in BIG_FIVE_TRAITS:
            if trait in big_five and big_five[trait] is not None:
                normalized_big_five[trait] = float(big_five[trait])
            else:
                flat_key = f"big_five_{trait}"
                if flat_key in prediction and prediction[flat_key] is not None:
                    normalized_big_five[trait] = float(prediction[flat_key])
                elif flat_key in features and features[flat_key] is not None:
                    normalized_big_five[trait] = float(features[flat_key])

        return {
            "big_five": normalized_big_five,
            "mbti": (
                prediction.get("mbti")
                or prediction.get("mbti_type")
                or features.get("mbti")
                or features.get("mbti_type")
            ),
        }

    def _safe_parse_json(self, text: str):
        cleaned = text.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        import json
        return json.loads(cleaned.strip())

    def _mean(self, values: List[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    def _llm_enabled(self, role: AgentRole) -> bool:
        if not self.use_llm_assistance:
            return False
        availability_check = getattr(self.router, "_is_available", None)
        if not callable(availability_check):
            return False

        for model_key in MODEL_ROUTING.get(role, []):
            spec = MODELS.get(model_key)
            if spec is None:
                continue
            try:
                if availability_check(spec.provider):
                    return True
            except Exception:
                continue
        return False


def run_persistent_personas_evaluation(systems: Dict, n_personas: int = 5) -> Dict:
    """Standalone evaluation runner."""
    adapter = PersistentPersonasAdapter()
    personas = adapter.generate_personas(n_personas=n_personas)
    return adapter.evaluate_systems(systems, personas, n_turns=100)
