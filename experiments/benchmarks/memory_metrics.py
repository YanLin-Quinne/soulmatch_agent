"""Persona-aware memory evaluation metrics."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


BIG_FIVE_TRAITS = [
    "openness",
    "conscientiousness",
    "extraversion",
    "agreeableness",
    "neuroticism",
]


def _normalize_text(value: object) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _fact_to_key(fact: object) -> str:
    if isinstance(fact, dict):
        if "key" in fact and "value" in fact:
            return f"{_normalize_text(fact['key'])}:{_normalize_text(fact['value'])}"
        if "fact" in fact:
            return _normalize_text(fact["fact"])
        ordered_items = sorted((str(k), _normalize_text(v)) for k, v in fact.items())
        return "|".join(f"{k}:{v}" for k, v in ordered_items)
    return _normalize_text(fact)


def _extract_numeric_profile(prediction: Dict) -> Dict[str, float]:
    profile: Dict[str, float] = {}
    big_five = prediction.get("big_five", {})
    features = prediction.get("features", {})

    for trait in BIG_FIVE_TRAITS:
        if trait in big_five and big_five[trait] is not None:
            profile[trait] = float(big_five[trait])
            continue

        flat_key = f"big_five_{trait}"
        if flat_key in prediction and prediction[flat_key] is not None:
            profile[trait] = float(prediction[flat_key])
        elif flat_key in features and features[flat_key] is not None:
            profile[trait] = float(features[flat_key])

    mbti = (
        prediction.get("mbti")
        or prediction.get("mbti_type")
        or features.get("mbti")
        or features.get("mbti_type")
    )
    if isinstance(mbti, str) and len(mbti) == 4:
        mbti = mbti.upper()
        profile["mbti_ei"] = 1.0 if mbti[0] == "E" else 0.0
        profile["mbti_sn"] = 1.0 if mbti[1] == "N" else 0.0
        profile["mbti_tf"] = 1.0 if mbti[2] == "T" else 0.0
        profile["mbti_jp"] = 1.0 if mbti[3] == "J" else 0.0

    for axis in ("mbti_ei", "mbti_sn", "mbti_tf", "mbti_jp"):
        if axis in prediction and prediction[axis] is not None:
            profile[axis] = float(prediction[axis])
        elif axis in features and features[axis] is not None:
            profile[axis] = float(features[axis])

    return profile


def _mean_or_zero(values: List[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def persona_fact_recall(predicted_facts: List, ground_truth_facts: List) -> Dict:
    """Measure recall of persona-relevant facts."""
    pred_set = {_fact_to_key(fact) for fact in predicted_facts if _fact_to_key(fact)}
    gt_set = {_fact_to_key(fact) for fact in ground_truth_facts if _fact_to_key(fact)}

    true_positives = pred_set & gt_set
    false_positives = pred_set - gt_set
    false_negatives = gt_set - pred_set

    precision = len(true_positives) / len(pred_set) if pred_set else 0.0
    recall = len(true_positives) / len(gt_set) if gt_set else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": len(true_positives),
        "false_positives": len(false_positives),
        "false_negatives": len(false_negatives),
        "support": len(gt_set),
    }


def temporal_consistency(predictions_over_time: List[Dict], ground_truth: Dict) -> Dict:
    """Measure how consistent personality predictions are over time."""
    if not predictions_over_time:
        return {
            "stability_score": 0.0,
            "drift_magnitude": 0.0,
            "convergence_speed": 0,
            "per_dimension": {},
        }

    turns = [int(pred.get("turn", idx + 1)) for idx, pred in enumerate(predictions_over_time)]
    numeric_predictions = [_extract_numeric_profile(pred) for pred in predictions_over_time]
    numeric_truth = _extract_numeric_profile(ground_truth)
    all_dims = sorted({dim for pred in numeric_predictions for dim in pred.keys()} | set(numeric_truth.keys()))

    per_dimension: Dict[str, Dict[str, float]] = {}
    stability_scores: List[float] = []
    drift_scores: List[float] = []

    for dim in all_dims:
        series = [pred[dim] for pred in numeric_predictions if dim in pred]
        if not series:
            continue

        diffs = [abs(curr - prev) for prev, curr in zip(series, series[1:])]
        drift = float(sum(diffs))
        stability = max(0.0, 1.0 - _mean_or_zero(diffs))
        final_error = abs(series[-1] - numeric_truth[dim]) if dim in numeric_truth else 0.0

        per_dimension[dim] = {
            "stability": stability,
            "drift": drift,
            "final_error": final_error,
        }
        stability_scores.append(stability)
        drift_scores.append(drift)

    convergence_turn = turns[-1]
    threshold = 0.05
    for idx in range(len(numeric_predictions)):
        suffix = numeric_predictions[idx:]
        comparable_dims = sorted({dim for pred in suffix for dim in pred.keys()})
        if not comparable_dims:
            continue

        max_future_change = 0.0
        for dim in comparable_dims:
            values = [pred[dim] for pred in suffix if dim in pred]
            if len(values) < 2:
                continue
            max_future_change = max(max_future_change, max(abs(val - values[-1]) for val in values[:-1]))
        if max_future_change <= threshold:
            convergence_turn = turns[idx]
            break

    return {
        "stability_score": _mean_or_zero(stability_scores),
        "drift_magnitude": _mean_or_zero(drift_scores),
        "convergence_speed": convergence_turn,
        "per_dimension": per_dimension,
    }


def cross_session_continuity(
    session_end_predictions: List[Dict],
    session_start_predictions: List[Dict],
) -> Dict:
    """Measure personality prediction continuity across session boundaries."""
    if not session_end_predictions or not session_start_predictions:
        return {
            "continuity_score": 0.0,
            "average_jump": 0.0,
            "n_boundaries": 0,
            "per_dimension": {},
        }

    boundary_count = min(len(session_end_predictions), len(session_start_predictions))
    paired_ends = [_extract_numeric_profile(pred) for pred in session_end_predictions[:boundary_count]]
    paired_starts = [_extract_numeric_profile(pred) for pred in session_start_predictions[:boundary_count]]

    dims = sorted({dim for pred in paired_ends + paired_starts for dim in pred.keys()})
    per_dimension: Dict[str, Dict[str, float]] = {}
    all_jumps: List[float] = []

    for dim in dims:
        jumps: List[float] = []
        for end_pred, start_pred in zip(paired_ends, paired_starts):
            if dim not in end_pred or dim not in start_pred:
                continue
            jumps.append(abs(start_pred[dim] - end_pred[dim]))

        if not jumps:
            continue

        avg_jump = _mean_or_zero(jumps)
        per_dimension[dim] = {
            "average_jump": avg_jump,
            "continuity": max(0.0, 1.0 - avg_jump),
        }
        all_jumps.extend(jumps)

    average_jump = _mean_or_zero(all_jumps)
    return {
        "continuity_score": max(0.0, 1.0 - average_jump),
        "average_jump": average_jump,
        "n_boundaries": boundary_count,
        "per_dimension": per_dimension,
    }
