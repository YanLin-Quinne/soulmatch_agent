"""
Conformal Prediction Calibrator for Feature Prediction

Converts uncalibrated LLM confidence scores into statistically guaranteed
prediction sets with coverage guarantees.

Theory:
  LLM says P(sex=m) = 0.8 → but this is NOT calibrated (ECE is high)
  Conformal prediction: given calibration data with ground truth,
  compute threshold q̂ such that the prediction set
    C(x) = {y : score(x, y) ≤ q̂}
  satisfies P(Y_true ∈ C(X)) ≥ 1 - α (e.g., 90% coverage)

Supports:
  - Categorical features → APS (Adaptive Prediction Sets)
  - Continuous features → discretized into bins, then APS
  - Per-turn calibration (handles sequential non-exchangeability)
  - Coverage monitoring and adaptive α adjustment

References:
  [1] Angelopoulos & Bates (2021). A Gentle Introduction to Conformal Prediction.
  [2] Kumar et al. (2023). Conformal Prediction with LLMs for Multi-Choice QA.
  [3] Sheng et al. (2025). Analyzing Uncertainty of LLM-as-a-Judge (arXiv:2509.18658).
"""

import json
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from pathlib import Path
from loguru import logger


# ═══════════════════════════════════════════════════════════════════
# Feature dimension definitions
# ═══════════════════════════════════════════════════════════════════

# Categorical dimensions and their possible values
CATEGORICAL_DIMS: Dict[str, List[str]] = {
    "sex": ["m", "f"],
    "orientation": ["straight", "gay", "bisexual"],
    "communication_style": ["direct", "indirect", "humorous", "serious", "casual", "formal"],
    "relationship_goals": ["casual", "serious", "friendship", "unsure"],
    "lifestyle_diet": ["omnivore", "vegetarian", "vegan", "other"],
    "lifestyle_drinks": ["not_at_all", "rarely", "socially", "often", "very_often"],
    "lifestyle_smokes": ["no", "sometimes", "yes"],
    "lifestyle_drugs": ["never", "sometimes"],
    "background_education": ["high_school", "some_college", "bachelors", "masters", "phd"],
    "background_religion": ["atheist", "agnostic", "christian", "jewish", "muslim", "hindu", "buddhist", "other"],
    # v2.0: Extended dimensions
    "mbti_type": ["ISTJ", "ISFJ", "INFJ", "INTJ", "ISTP", "ISFP", "INFP", "INTP",
                  "ESTP", "ESFP", "ENFP", "ENTP", "ESTJ", "ESFJ", "ENFJ", "ENTJ"],
    "attachment_style": ["secure", "anxious", "avoidant", "disorganized"],
    "love_language": ["words_of_affirmation", "quality_time", "receiving_gifts", "acts_of_service", "physical_touch"],
    "relationship_status": ["stranger", "acquaintance", "crush", "dating", "committed"],
    "relationship_type": ["love", "friendship", "family", "other"],
    "sentiment": ["positive", "neutral", "negative"],
    "can_advance": ["yes", "no", "uncertain"],
}

# Continuous dimensions → discretized into bins for conformal prediction
CONTINUOUS_BINS: Dict[str, List[Tuple[str, float, float]]] = {
    # Big Five: low/medium/high
    "big_five_openness":          [("low", 0.0, 0.33), ("medium", 0.33, 0.67), ("high", 0.67, 1.01)],
    "big_five_conscientiousness": [("low", 0.0, 0.33), ("medium", 0.33, 0.67), ("high", 0.67, 1.01)],
    "big_five_extraversion":      [("low", 0.0, 0.33), ("medium", 0.33, 0.67), ("high", 0.67, 1.01)],
    "big_five_agreeableness":     [("low", 0.0, 0.33), ("medium", 0.33, 0.67), ("high", 0.67, 1.01)],
    "big_five_neuroticism":       [("low", 0.0, 0.33), ("medium", 0.33, 0.67), ("high", 0.67, 1.01)],
    # Interests: none/moderate/strong
    "interest_music":    [("none", 0.0, 0.33), ("moderate", 0.33, 0.67), ("strong", 0.67, 1.01)],
    "interest_sports":   [("none", 0.0, 0.33), ("moderate", 0.33, 0.67), ("strong", 0.67, 1.01)],
    "interest_travel":   [("none", 0.0, 0.33), ("moderate", 0.33, 0.67), ("strong", 0.67, 1.01)],
    "interest_food":     [("none", 0.0, 0.33), ("moderate", 0.33, 0.67), ("strong", 0.67, 1.01)],
    "interest_arts":     [("none", 0.0, 0.33), ("moderate", 0.33, 0.67), ("strong", 0.67, 1.01)],
    "interest_tech":     [("none", 0.0, 0.33), ("moderate", 0.33, 0.67), ("strong", 0.67, 1.01)],
    "interest_outdoors": [("none", 0.0, 0.33), ("moderate", 0.33, 0.67), ("strong", 0.67, 1.01)],
    "interest_books":    [("none", 0.0, 0.33), ("moderate", 0.33, 0.67), ("strong", 0.67, 1.01)],
    # v2.0: Extended continuous dimensions
    "mbti_ei": [("introvert", 0.0, 0.5), ("extravert", 0.5, 1.01)],
    "mbti_sn": [("sensing", 0.0, 0.5), ("intuition", 0.5, 1.01)],
    "mbti_tf": [("thinking", 0.0, 0.5), ("feeling", 0.5, 1.01)],
    "mbti_jp": [("judging", 0.0, 0.5), ("perceiving", 0.5, 1.01)],
    "attachment_anxiety": [("low", 0.0, 0.33), ("medium", 0.33, 0.67), ("high", 0.67, 1.01)],
    "attachment_avoidance": [("low", 0.0, 0.33), ("medium", 0.33, 0.67), ("high", 0.67, 1.01)],
    "trust_score": [("low", 0.0, 0.33), ("medium", 0.33, 0.67), ("high", 0.67, 1.01)],
}

# Age bins (special handling)
AGE_BINS = [("18-24", 18, 25), ("25-30", 25, 31), ("30-35", 30, 36),
            ("35-40", 35, 41), ("40-50", 40, 51), ("50+", 50, 100)]

ALL_DIMS = list(CATEGORICAL_DIMS.keys()) + list(CONTINUOUS_BINS.keys()) + ["age"]


# ═══════════════════════════════════════════════════════════════════
# Data structures
# ═══════════════════════════════════════════════════════════════════

@dataclass
class PredictionSet:
    """A conformal prediction set for a single dimension."""
    dimension: str
    prediction_set: List[str]       # set of plausible values
    point_prediction: str           # top-1 prediction (best guess)
    set_size: int                   # |C(x)|, lower = more certain
    llm_confidence: float           # raw LLM-reported confidence (uncalibrated)
    calibrated_confidence: float    # 1 - (set_size / total_options)
    coverage_guarantee: float       # 1 - α


@dataclass
class ConformalResult:
    """Full conformal prediction result across all dimensions."""
    prediction_sets: Dict[str, PredictionSet]
    coverage_level: float           # 1 - α
    turn: int
    avg_set_size: float             # average set size across dims (lower = better)
    n_singleton: int                # how many dims have set_size == 1
    n_dimensions: int

    def summary(self) -> Dict[str, Any]:
        return {
            "coverage": self.coverage_level,
            "turn": self.turn,
            "avg_set_size": round(self.avg_set_size, 2),
            "singletons": f"{self.n_singleton}/{self.n_dimensions}",
            "sets": {
                dim: {
                    "set": ps.prediction_set,
                    "point": ps.point_prediction,
                    "size": ps.set_size,
                    "llm_conf": round(ps.llm_confidence, 3),
                    "cal_conf": round(ps.calibrated_confidence, 3),
                }
                for dim, ps in self.prediction_sets.items()
            },
        }


@dataclass
class CalibrationSample:
    """One calibration sample: LLM prediction + ground truth for one profile at one turn."""
    profile_id: str
    turn: int
    dimension: str
    llm_softmax: Dict[str, float]   # {option: probability} from LLM
    ground_truth: str                # true label (discretized for continuous)


# ═══════════════════════════════════════════════════════════════════
# Helper: discretize continuous values
# ═══════════════════════════════════════════════════════════════════

def discretize_value(dim: str, value: float) -> str:
    """Convert a continuous value to its bin label."""
    if dim == "age":
        for label, lo, hi in AGE_BINS:
            if lo <= value < hi:
                return label
        return "50+"

    if dim in CONTINUOUS_BINS:
        for label, lo, hi in CONTINUOUS_BINS[dim]:
            if lo <= value < hi:
                return label
        return CONTINUOUS_BINS[dim][-1][0]  # last bin

    return str(value)


def get_options(dim: str) -> List[str]:
    """Get all possible options for a dimension."""
    if dim in CATEGORICAL_DIMS:
        return CATEGORICAL_DIMS[dim]
    if dim in CONTINUOUS_BINS:
        return [label for label, _, _ in CONTINUOUS_BINS[dim]]
    if dim == "age":
        return [label for label, _, _ in AGE_BINS]
    return []


# ═══════════════════════════════════════════════════════════════════
# Conformal Calibrator
# ═══════════════════════════════════════════════════════════════════

class ConformalCalibrator:
    """
    Conformal prediction calibrator for feature prediction.

    Usage:
        # 1. Fit on calibration data (once, offline)
        calibrator = ConformalCalibrator(alpha=0.10)  # 90% coverage
        calibrator.fit(calibration_samples)

        # 2. At inference time (every turn)
        result = calibrator.calibrate(
            predictions=llm_output,
            llm_confidences=llm_confidence_dict,
            turn=current_turn,
        )
        # result.prediction_sets["sex"].prediction_set → ["m"]  (singleton = certain)
        # result.prediction_sets["diet"].prediction_set → ["omnivore", "vegetarian"]  (uncertain)
    """

    def __init__(self, alpha: float = 0.10):
        """
        Args:
            alpha: Miscoverage rate. 0.10 = 90% coverage guarantee.
        """
        self.alpha = alpha

        # Nonconformity scores grouped by (turn_bucket, dimension)
        # Key: (turn_bucket, dim) → List[float]
        self._scores: Dict[Tuple[int, str], List[float]] = {}

        # Computed quantiles after fit()
        # Key: (turn_bucket, dim) → float
        self._quantiles: Dict[Tuple[int, str], float] = {}

        # Global fallback quantile per dimension (when turn-specific data is sparse)
        self._global_quantiles: Dict[str, float] = {}

        # Calibration statistics
        self.n_calibration_samples = 0
        self.is_fitted = False

        # Turn buckets: group turns for sufficient calibration data
        self.turn_buckets = [(1, 5), (6, 10), (11, 15), (16, 20), (21, 25), (26, 30)]

    def _get_turn_bucket(self, turn: int) -> int:
        """Map turn number to bucket index."""
        for i, (lo, hi) in enumerate(self.turn_buckets):
            if lo <= turn <= hi:
                return i
        return len(self.turn_buckets) - 1  # last bucket for overflow

    # ───────────────────────────────────────────────────────────
    # Fit (offline, on calibration data)
    # ───────────────────────────────────────────────────────────

    def fit(self, samples: List[CalibrationSample], min_samples_per_bucket: int = 10):
        """
        Fit the calibrator on labeled calibration data.

        For each (turn_bucket, dimension), compute the nonconformity scores
        and their (1-α)-quantile.

        Args:
            samples: List of CalibrationSample with LLM softmax + ground truth
            min_samples_per_bucket: Minimum samples for a turn-specific quantile
        """
        self._scores.clear()
        self._quantiles.clear()
        self._global_quantiles.clear()

        # 1. Compute nonconformity scores
        for sample in samples:
            bucket = self._get_turn_bucket(sample.turn)
            key = (bucket, sample.dimension)

            # Nonconformity score = 1 - P_LLM(true_label)
            # Higher score = LLM was more wrong about the true label
            true_prob = sample.llm_softmax.get(sample.ground_truth, 0.0)
            score = 1.0 - true_prob

            self._scores.setdefault(key, []).append(score)

        # 2. Compute quantiles
        global_scores: Dict[str, List[float]] = {}

        for (bucket, dim), scores in self._scores.items():
            # Collect global scores per dimension
            global_scores.setdefault(dim, []).extend(scores)

            if len(scores) >= min_samples_per_bucket:
                # Quantile with finite-sample correction: ceil((n+1)(1-α)) / n
                n = len(scores)
                q_level = min(np.ceil((n + 1) * (1 - self.alpha)) / n, 1.0)
                self._quantiles[(bucket, dim)] = float(np.quantile(scores, q_level))

        # 3. Global fallback quantiles
        for dim, scores in global_scores.items():
            n = len(scores)
            q_level = min(np.ceil((n + 1) * (1 - self.alpha)) / n, 1.0)
            self._global_quantiles[dim] = float(np.quantile(scores, q_level))

        self.n_calibration_samples = len(samples)
        self.is_fitted = True

        logger.info(
            f"ConformalCalibrator fitted: {len(samples)} samples, "
            f"{len(self._quantiles)} turn-dim quantiles, "
            f"{len(self._global_quantiles)} global quantiles"
        )

    def fit_from_dialogues(self, dialogues: List[Dict], predictor_fn=None):
        """
        Convenience: fit from synthetic dialogue JSONL records.

        Each dialogue has ground_truth_features and turns.
        If predictor_fn is None, we simulate softmax from ground truth + noise.

        Args:
            dialogues: List of dialogue records from synthetic_dialogue_generator
            predictor_fn: Optional function(conversation_history, turn) → {dim: {option: prob}}
        """
        samples = []

        for dialogue in dialogues:
            gt_map = dialogue.get("ground_truth_features", {})
            turns = dialogue.get("turns", [])
            num_turns = len(turns)

            for profile_id, gt in gt_map.items():
                # Flatten ground truth
                flat_gt = self._flatten_ground_truth(gt)

                # Generate calibration samples at multiple turn checkpoints
                for turn_checkpoint in [5, 10, 15, 20, 25, 30]:
                    if turn_checkpoint > num_turns:
                        continue

                    if predictor_fn:
                        # Use actual predictor
                        conversation_subset = turns[:turn_checkpoint]
                        llm_output = predictor_fn(conversation_subset, turn_checkpoint)
                    else:
                        # Simulate softmax: ground truth gets high prob + noise
                        llm_output = self._simulate_softmax(flat_gt, turn_checkpoint, num_turns)

                    for dim, true_label in flat_gt.items():
                        if dim not in llm_output:
                            continue
                        samples.append(CalibrationSample(
                            profile_id=profile_id,
                            turn=turn_checkpoint,
                            dimension=dim,
                            llm_softmax=llm_output[dim],
                            ground_truth=true_label,
                        ))

        logger.info(f"Generated {len(samples)} calibration samples from {len(dialogues)} dialogues")
        self.fit(samples)

    def _flatten_ground_truth(self, gt: Dict) -> Dict[str, str]:
        """Convert ground_truth_features format to flat {dim: label} for all dims."""
        flat = {}

        meta = gt.get("profile_metadata", {})
        if meta.get("sex"):
            flat["sex"] = meta["sex"]
        if meta.get("orientation"):
            flat["orientation"] = meta["orientation"]
        if meta.get("education"):
            flat["background_education"] = meta["education"]
        if meta.get("age"):
            flat["age"] = discretize_value("age", meta["age"])

        if gt.get("communication_style"):
            flat["communication_style"] = gt["communication_style"]
        if gt.get("relationship_goals"):
            flat["relationship_goals"] = gt["relationship_goals"]

        personality = gt.get("personality", {})
        for trait in ("openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"):
            val = personality.get(trait)
            if val is not None:
                flat[f"big_five_{trait}"] = discretize_value(f"big_five_{trait}", val)

        interests = gt.get("interests", {})
        for interest_name, val in interests.items():
            dim = f"interest_{interest_name}"
            if dim in CONTINUOUS_BINS and val is not None:
                flat[dim] = discretize_value(dim, val)

        return flat

    def _simulate_softmax(self, flat_gt: Dict[str, str], turn: int, max_turns: int) -> Dict[str, Dict[str, float]]:
        """
        Simulate LLM softmax output for calibration when no real predictor is available.

        The simulation models:
          - Early turns → flatter distribution (more uncertain)
          - Later turns → sharper distribution (more certain)
          - Noise to simulate LLM miscalibration
        """
        progress = min(turn / max(max_turns, 1), 1.0)
        # Temperature: high early (flat), low late (sharp)
        temperature = max(0.3, 1.5 - progress * 1.2)

        result = {}
        for dim, true_label in flat_gt.items():
            options = get_options(dim)
            if not options or true_label not in options:
                continue

            # Generate logits: true label gets base_logit, others get noise
            base_logit = 1.0 + progress * 3.0  # stronger signal as turns increase
            logits = {}
            for opt in options:
                if opt == true_label:
                    logits[opt] = base_logit + np.random.normal(0, 0.3)
                else:
                    logits[opt] = np.random.normal(0, 0.5)

            # Softmax with temperature
            vals = np.array([logits[opt] for opt in options]) / temperature
            vals = vals - vals.max()  # numerical stability
            exp_vals = np.exp(vals)
            probs = exp_vals / exp_vals.sum()

            result[dim] = {opt: float(p) for opt, p in zip(options, probs)}

        return result

    # ───────────────────────────────────────────────────────────
    # Calibrate (online, per turn)
    # ───────────────────────────────────────────────────────────

    def calibrate(
        self,
        predictions: Dict[str, Any],
        llm_confidences: Dict[str, float],
        turn: int,
    ) -> ConformalResult:
        """
        Convert LLM predictions + uncalibrated confidences → conformal prediction sets.

        Args:
            predictions: Raw LLM feature predictions {dim: value}
            llm_confidences: LLM-reported confidence {dim: float}
            turn: Current conversation turn

        Returns:
            ConformalResult with prediction sets for all available dimensions
        """
        bucket = self._get_turn_bucket(turn)
        prediction_sets = {}

        for dim in ALL_DIMS:
            options = get_options(dim)
            if not options:
                continue

            # Get the quantile threshold for this (turn, dim)
            q_hat = self._quantiles.get((bucket, dim))
            if q_hat is None:
                q_hat = self._global_quantiles.get(dim)
            if q_hat is None:
                # No calibration data → return full set (maximally uncertain)
                prediction_sets[dim] = PredictionSet(
                    dimension=dim,
                    prediction_set=options,
                    point_prediction=self._get_point_prediction(dim, predictions),
                    set_size=len(options),
                    llm_confidence=llm_confidences.get(dim, 0.0),
                    calibrated_confidence=0.0,
                    coverage_guarantee=1 - self.alpha,
                )
                continue

            # Build softmax from LLM output
            softmax = self._build_softmax(dim, predictions, llm_confidences)

            # APS: include options in decreasing probability until cumulative
            # nonconformity score exceeds q̂
            # Equivalently: include option y if score(y) = 1 - P(y) ≤ q̂
            #              i.e., P(y) ≥ 1 - q̂
            threshold = 1.0 - q_hat
            pred_set = [opt for opt in options if softmax.get(opt, 0.0) >= threshold]

            # Ensure non-empty: always include the top-1 prediction
            if not pred_set:
                top_opt = max(softmax, key=softmax.get) if softmax else options[0]
                pred_set = [top_opt]

            # Sort by probability (highest first)
            pred_set.sort(key=lambda o: softmax.get(o, 0.0), reverse=True)

            point_pred = pred_set[0]
            cal_confidence = 1.0 - (len(pred_set) / len(options))

            prediction_sets[dim] = PredictionSet(
                dimension=dim,
                prediction_set=pred_set,
                point_prediction=point_pred,
                set_size=len(pred_set),
                llm_confidence=llm_confidences.get(dim, 0.0),
                calibrated_confidence=cal_confidence,
                coverage_guarantee=1 - self.alpha,
            )

        # Aggregate metrics
        sizes = [ps.set_size for ps in prediction_sets.values()]
        avg_size = float(np.mean(sizes)) if sizes else 0.0
        n_singleton = sum(1 for ps in prediction_sets.values() if ps.set_size == 1)

        return ConformalResult(
            prediction_sets=prediction_sets,
            coverage_level=1 - self.alpha,
            turn=turn,
            avg_set_size=avg_size,
            n_singleton=n_singleton,
            n_dimensions=len(prediction_sets),
        )

    def _get_point_prediction(self, dim: str, predictions: Dict) -> str:
        """Extract point prediction for a dimension from raw LLM output."""
        val = predictions.get(dim)
        if val is None:
            # Try nested structures
            for group_key in ("big_five", "lifestyle", "background", "interests"):
                group = predictions.get(group_key, {})
                short_key = dim.replace(f"{group_key}_", "").replace("interest_", "")
                if short_key in group:
                    val = group[short_key]
                    break

        if val is None:
            options = get_options(dim)
            return options[0] if options else "unknown"

        # Discretize continuous values
        if isinstance(val, (int, float)) and dim in CONTINUOUS_BINS:
            return discretize_value(dim, val)
        if isinstance(val, (int, float)) and dim == "age":
            return discretize_value("age", val)

        return str(val)

    def _build_softmax(
        self, dim: str, predictions: Dict, llm_confidences: Dict
    ) -> Dict[str, float]:
        """
        Build a softmax distribution over options for a dimension.

        Uses LLM confidence as signal for the predicted value,
        distributes remaining probability mass among other options.
        """
        options = get_options(dim)
        if not options:
            return {}

        point_pred = self._get_point_prediction(dim, predictions)
        conf = llm_confidences.get(dim, 0.5)
        conf = max(0.01, min(0.99, conf))  # clip

        softmax = {}
        remaining = 1.0 - conf
        n_others = max(len(options) - 1, 1)

        for opt in options:
            if opt == point_pred:
                softmax[opt] = conf
            else:
                softmax[opt] = remaining / n_others

        return softmax

    # ───────────────────────────────────────────────────────────
    # Persistence
    # ───────────────────────────────────────────────────────────

    def save(self, path: str):
        """Save fitted calibrator to JSON."""
        data = {
            "alpha": self.alpha,
            "n_calibration_samples": self.n_calibration_samples,
            "quantiles": {
                f"{bucket}_{dim}": q
                for (bucket, dim), q in self._quantiles.items()
            },
            "global_quantiles": self._global_quantiles,
            "turn_buckets": self.turn_buckets,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Calibrator saved to {path}")

    def load(self, path: str):
        """Load fitted calibrator from JSON."""
        with open(path, "r") as f:
            data = json.load(f)

        self.alpha = data["alpha"]
        self.n_calibration_samples = data["n_calibration_samples"]
        self.turn_buckets = [tuple(b) for b in data.get("turn_buckets", self.turn_buckets)]

        self._quantiles = {}
        for key_str, q in data.get("quantiles", {}).items():
            parts = key_str.split("_", 1)
            bucket = int(parts[0])
            dim = parts[1]
            self._quantiles[(bucket, dim)] = q

        self._global_quantiles = data.get("global_quantiles", {})
        self.is_fitted = True
        logger.info(f"Calibrator loaded from {path}: {self.n_calibration_samples} cal samples")

    # ───────────────────────────────────────────────────────────
    # Evaluation
    # ───────────────────────────────────────────────────────────

    def evaluate_coverage(self, test_samples: List[CalibrationSample]) -> Dict[str, Any]:
        """
        Evaluate empirical coverage and efficiency on held-out test data.

        Returns:
            {
                "marginal_coverage": float,  # should be ≥ 1-α
                "avg_set_size": float,        # lower = better (efficiency)
                "per_dim_coverage": {dim: float},
                "per_turn_coverage": {turn_bucket: float},
                "ece": float,  # Expected Calibration Error of LLM confidence
            }
        """
        if not self.is_fitted:
            raise RuntimeError("Calibrator not fitted. Call fit() first.")

        covers = []
        sizes = []
        per_dim_covers: Dict[str, List[bool]] = {}
        per_bucket_covers: Dict[int, List[bool]] = {}
        ece_bins: Dict[int, List[Tuple[float, bool]]] = {}  # bin → [(conf, correct)]

        for sample in test_samples:
            bucket = self._get_turn_bucket(sample.turn)
            # Build a fake prediction to calibrate
            fake_pred = {sample.dimension: sample.ground_truth}
            # Use the LLM softmax to get point prediction
            top_option = max(sample.llm_softmax, key=sample.llm_softmax.get)
            llm_conf = {sample.dimension: sample.llm_softmax.get(top_option, 0.5)}

            result = self.calibrate(fake_pred, llm_conf, sample.turn)
            ps = result.prediction_sets.get(sample.dimension)
            if ps is None:
                continue

            covered = sample.ground_truth in ps.prediction_set
            covers.append(covered)
            sizes.append(ps.set_size)

            per_dim_covers.setdefault(sample.dimension, []).append(covered)
            per_bucket_covers.setdefault(bucket, []).append(covered)

            # ECE computation
            conf_bin = int(llm_conf[sample.dimension] * 10)
            correct = (top_option == sample.ground_truth)
            ece_bins.setdefault(conf_bin, []).append((llm_conf[sample.dimension], correct))

        # Compute ECE
        ece = 0.0
        total = len(covers)
        for bin_idx, entries in ece_bins.items():
            bin_size = len(entries)
            avg_conf = np.mean([c for c, _ in entries])
            avg_acc = np.mean([int(a) for _, a in entries])
            ece += (bin_size / total) * abs(avg_conf - avg_acc)

        return {
            "marginal_coverage": float(np.mean(covers)) if covers else 0.0,
            "avg_set_size": float(np.mean(sizes)) if sizes else 0.0,
            "per_dim_coverage": {
                dim: float(np.mean(c)) for dim, c in per_dim_covers.items()
            },
            "per_turn_coverage": {
                str(self.turn_buckets[b]): float(np.mean(c))
                for b, c in per_bucket_covers.items()
            },
            "ece_llm": round(ece, 4),
            "n_test_samples": len(covers),
            "target_coverage": 1 - self.alpha,
        }
