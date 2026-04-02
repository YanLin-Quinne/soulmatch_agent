"""Evaluation metrics for SoulMatch baseline experiments.

Computes all metrics needed for the paper's experiment tables:
- Classification: accuracy, F1 (macro/micro/weighted)
- Regression: MAE, RMSE
- Calibration: ECE (Expected Calibration Error)
- Conformal: coverage rate, average prediction set size
- Aggregate: personality metrics, relationship metrics
- Reporting: LaTeX table generation, human-readable summaries
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple


# =====================================================================
# Primitive metrics
# =====================================================================

def accuracy(y_true: List, y_pred: List) -> float:
    """Classification accuracy.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.

    Returns:
        Fraction of correct predictions in [0, 1].
    """
    if not y_true or not y_pred:
        return 0.0
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Length mismatch: y_true has {len(y_true)}, y_pred has {len(y_pred)}"
        )
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    return correct / len(y_true)


def f1_score(y_true: List, y_pred: List, average: str = "macro") -> float:
    """F1 score with macro, micro, or weighted averaging.

    Attempts sklearn import for speed; falls back to manual implementation.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        average: One of 'macro', 'micro', 'weighted'.

    Returns:
        F1 score in [0, 1].
    """
    if not y_true or not y_pred:
        return 0.0
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Length mismatch: y_true has {len(y_true)}, y_pred has {len(y_pred)}"
        )
    if average not in ("macro", "micro", "weighted"):
        raise ValueError(f"average must be 'macro', 'micro', or 'weighted', got '{average}'")

    try:
        from sklearn.metrics import f1_score as sklearn_f1
        return float(sklearn_f1(y_true, y_pred, average=average, zero_division="warn"))
    except ImportError:
        pass

    return _f1_manual(y_true, y_pred, average)


def _f1_manual(y_true: List, y_pred: List, average: str) -> float:
    """Manual F1 implementation when sklearn is unavailable."""
    labels = sorted(set(y_true) | set(y_pred))

    if average == "micro":
        # Micro: aggregate TP/FP/FN across all classes, then compute
        tp_total = sum(1 for t, p in zip(y_true, y_pred) if t == p)
        # For micro, precision = recall = accuracy when every sample has exactly one label
        n = len(y_true)
        if n == 0:
            return 0.0
        precision = tp_total / n
        recall = tp_total / n
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    # Per-class precision, recall, F1
    per_class: List[Tuple[float, int]] = []  # (f1, support)
    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        support = tp + fn

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if precision + recall > 0:
            class_f1 = 2 * precision * recall / (precision + recall)
        else:
            class_f1 = 0.0
        per_class.append((class_f1, support))

    if not per_class:
        return 0.0

    if average == "macro":
        return sum(f for f, _ in per_class) / len(per_class)

    # weighted
    total_support = sum(s for _, s in per_class)
    if total_support == 0:
        return 0.0
    return sum(f * s for f, s in per_class) / total_support


def mae(y_true: List[float], y_pred: List[float]) -> float:
    """Mean Absolute Error for continuous values (e.g., Big Five scores).

    Args:
        y_true: Ground truth continuous values.
        y_pred: Predicted continuous values.

    Returns:
        Mean absolute error (non-negative).
    """
    if not y_true or not y_pred:
        return 0.0
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Length mismatch: y_true has {len(y_true)}, y_pred has {len(y_pred)}"
        )
    arr_true = np.asarray(y_true, dtype=np.float64)
    arr_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.mean(np.abs(arr_true - arr_pred)))


def rmse(y_true: List[float], y_pred: List[float]) -> float:
    """Root Mean Square Error.

    Args:
        y_true: Ground truth continuous values.
        y_pred: Predicted continuous values.

    Returns:
        RMSE (non-negative).
    """
    if not y_true or not y_pred:
        return 0.0
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Length mismatch: y_true has {len(y_true)}, y_pred has {len(y_pred)}"
        )
    arr_true = np.asarray(y_true, dtype=np.float64)
    arr_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.sqrt(np.mean((arr_true - arr_pred) ** 2)))


# =====================================================================
# Calibration metrics
# =====================================================================

def ece(confidences: List[float], accuracies: List[bool], n_bins: int = 10) -> float:
    """Expected Calibration Error.

    Bins predictions by confidence, computes weighted |accuracy - confidence|
    per bin. Lower is better.

    Args:
        confidences: Model confidence for each prediction, in [0, 1].
        accuracies: Whether each prediction was correct.
        n_bins: Number of equal-width bins.

    Returns:
        ECE value in [0, 1]. 0 means perfectly calibrated.
    """
    if not confidences or not accuracies:
        return 0.0
    if len(confidences) != len(accuracies):
        raise ValueError(
            f"Length mismatch: confidences has {len(confidences)}, "
            f"accuracies has {len(accuracies)}"
        )

    n = len(confidences)
    conf_arr = np.asarray(confidences, dtype=np.float64)
    acc_arr = np.asarray(accuracies, dtype=np.float64)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    total_ece = 0.0

    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        if i == n_bins - 1:
            # Last bin includes right boundary
            mask = (conf_arr >= lo) & (conf_arr <= hi)
        else:
            mask = (conf_arr >= lo) & (conf_arr < hi)

        bin_size = int(mask.sum())
        if bin_size == 0:
            continue

        bin_acc = float(acc_arr[mask].mean())
        bin_conf = float(conf_arr[mask].mean())
        total_ece += (bin_size / n) * abs(bin_acc - bin_conf)

    return total_ece


# =====================================================================
# Conformal prediction metrics
# =====================================================================

def conformal_coverage(
    prediction_sets: List[List[str]], ground_truths: List[str]
) -> float:
    """Conformal prediction coverage rate.

    Fraction of ground_truths that fall within their prediction_set.
    Should be >= 1-alpha (typically >= 0.90).

    Args:
        prediction_sets: List of prediction sets, each a list of candidate labels.
        ground_truths: Corresponding ground truth labels.

    Returns:
        Coverage rate in [0, 1].
    """
    if not prediction_sets or not ground_truths:
        return 0.0
    if len(prediction_sets) != len(ground_truths):
        raise ValueError(
            f"Length mismatch: prediction_sets has {len(prediction_sets)}, "
            f"ground_truths has {len(ground_truths)}"
        )

    covered = sum(
        1 for ps, gt in zip(prediction_sets, ground_truths)
        if gt in ps
    )
    return covered / len(ground_truths)


def avg_prediction_set_size(prediction_sets: List[List[str]]) -> float:
    """Average size of conformal prediction sets. Smaller is better.

    Args:
        prediction_sets: List of prediction sets.

    Returns:
        Mean set size (>= 0).
    """
    if not prediction_sets:
        return 0.0
    return float(np.mean([len(ps) for ps in prediction_sets]))


# =====================================================================
# Aggregate personality metrics
# =====================================================================

BIG_FIVE_TRAITS = [
    "openness", "conscientiousness", "extraversion",
    "agreeableness", "neuroticism",
]


def compute_personality_metrics(
    predictions: List[Dict], ground_truths: List[Dict]
) -> Dict[str, Any]:
    """Compute all personality inference metrics.

    Each prediction/ground_truth dict has:
        - big_five: {openness: float, conscientiousness: float, ...}
        - mbti: str (e.g., "ENFP")
        - confidences: {dim: float} (optional, for ECE)
        - conformal_sets: {dim: [str]} (optional, for coverage)

    Args:
        predictions: List of prediction dicts.
        ground_truths: Corresponding ground truth dicts.

    Returns:
        Dict with mbti_accuracy, big_five_mae, big_five_rmse, per-trait
        breakdown, and optional ECE / conformal metrics.
    """
    if not predictions or not ground_truths:
        return {}
    if len(predictions) != len(ground_truths):
        raise ValueError(
            f"Length mismatch: predictions has {len(predictions)}, "
            f"ground_truths has {len(ground_truths)}"
        )

    result: Dict[str, Any] = {}

    # -- MBTI accuracy --
    mbti_true = [
        gt.get("mbti") for gt in ground_truths if gt.get("mbti") is not None
    ]
    mbti_pred = [
        pred.get("mbti") for pred, gt in zip(predictions, ground_truths)
        if gt.get("mbti") is not None
    ]
    if mbti_true and mbti_pred and len(mbti_true) == len(mbti_pred):
        result["mbti_accuracy"] = accuracy(mbti_true, mbti_pred)
        result["mbti_f1"] = f1_score(mbti_true, mbti_pred, average="macro")
    else:
        result["mbti_accuracy"] = 0.0

    # -- Big Five regression metrics --
    all_true_vals: List[float] = []
    all_pred_vals: List[float] = []
    per_trait: Dict[str, Dict[str, float]] = {}

    for trait in BIG_FIVE_TRAITS:
        trait_true: List[float] = []
        trait_pred: List[float] = []

        for pred, gt in zip(predictions, ground_truths):
            gt_bf = gt.get("big_five", {})
            pred_bf = pred.get("big_five", {})
            gt_val = gt_bf.get(trait)
            pred_val = pred_bf.get(trait)
            if gt_val is not None and pred_val is not None:
                trait_true.append(float(gt_val))
                trait_pred.append(float(pred_val))

        if trait_true:
            per_trait[trait] = {
                "mae": mae(trait_true, trait_pred),
                "rmse": rmse(trait_true, trait_pred),
                "n": len(trait_true),
            }
            all_true_vals.extend(trait_true)
            all_pred_vals.extend(trait_pred)
        else:
            per_trait[trait] = {"mae": 0.0, "rmse": 0.0, "n": 0}

    result["big_five_per_trait"] = per_trait
    if all_true_vals:
        result["big_five_mae"] = mae(all_true_vals, all_pred_vals)
        result["big_five_rmse"] = rmse(all_true_vals, all_pred_vals)
    else:
        result["big_five_mae"] = 0.0
        result["big_five_rmse"] = 0.0

    # -- ECE (if confidences provided) --
    conf_list: List[float] = []
    acc_list: List[bool] = []

    for pred, gt in zip(predictions, ground_truths):
        confs = pred.get("confidences", {})
        if not confs:
            continue
        # MBTI confidence vs correctness
        mbti_conf = confs.get("mbti")
        if mbti_conf is not None and gt.get("mbti") is not None:
            conf_list.append(float(mbti_conf))
            acc_list.append(pred.get("mbti") == gt.get("mbti"))
        # Big Five trait confidences
        for trait in BIG_FIVE_TRAITS:
            trait_conf = confs.get(trait)
            if trait_conf is not None:
                gt_val = (gt.get("big_five") or {}).get(trait)
                pred_val = (pred.get("big_five") or {}).get(trait)
                if gt_val is not None and pred_val is not None:
                    conf_list.append(float(trait_conf))
                    # "Correct" for continuous: within 0.1 tolerance
                    acc_list.append(abs(float(gt_val) - float(pred_val)) <= 0.1)

    if conf_list:
        result["ece"] = ece(conf_list, acc_list)

    # -- Conformal coverage (if conformal_sets provided) --
    mbti_sets: List[List[str]] = []
    mbti_gts: List[str] = []

    for pred, gt in zip(predictions, ground_truths):
        cs = pred.get("conformal_sets", {})
        if not cs:
            continue
        mbti_set = cs.get("mbti")
        gt_mbti = gt.get("mbti")
        if mbti_set is not None and gt_mbti is not None:
            mbti_sets.append(mbti_set)
            mbti_gts.append(gt_mbti)

    if mbti_sets:
        result["coverage"] = conformal_coverage(mbti_sets, mbti_gts)
        result["avg_set_size"] = avg_prediction_set_size(mbti_sets)

    return result


# =====================================================================
# Aggregate relationship metrics
# =====================================================================

def compute_relationship_metrics(
    predictions: List[Dict], ground_truths: List[Dict]
) -> Dict[str, Any]:
    """Compute all relationship prediction metrics.

    Each prediction/ground_truth dict has:
        - rel_type: str
        - rel_status: str
        - confidences: {dim: float} (optional)

    Args:
        predictions: List of prediction dicts.
        ground_truths: Corresponding ground truth dicts.

    Returns:
        Dict with type/status accuracy and F1, plus optional ECE.
    """
    if not predictions or not ground_truths:
        return {}
    if len(predictions) != len(ground_truths):
        raise ValueError(
            f"Length mismatch: predictions has {len(predictions)}, "
            f"ground_truths has {len(ground_truths)}"
        )

    result: Dict[str, Any] = {}

    # -- Relationship type --
    type_true = [
        gt["rel_type"] for gt in ground_truths
        if gt.get("rel_type") is not None
    ]
    type_pred = [
        pred["rel_type"] for pred, gt in zip(predictions, ground_truths)
        if gt.get("rel_type") is not None
    ]
    if type_true and type_pred and len(type_true) == len(type_pred):
        result["type_accuracy"] = accuracy(type_true, type_pred)
        result["type_f1"] = f1_score(type_true, type_pred, average="macro")
    else:
        result["type_accuracy"] = 0.0
        result["type_f1"] = 0.0

    # -- Relationship status --
    status_true = [
        gt["rel_status"] for gt in ground_truths
        if gt.get("rel_status") is not None
    ]
    status_pred = [
        pred["rel_status"] for pred, gt in zip(predictions, ground_truths)
        if gt.get("rel_status") is not None
    ]
    if status_true and status_pred and len(status_true) == len(status_pred):
        result["status_accuracy"] = accuracy(status_true, status_pred)
        result["status_f1"] = f1_score(status_true, status_pred, average="macro")
    else:
        result["status_accuracy"] = 0.0
        result["status_f1"] = 0.0

    # -- ECE across both dimensions --
    conf_list: List[float] = []
    acc_list: List[bool] = []

    for pred, gt in zip(predictions, ground_truths):
        confs = pred.get("confidences", {})
        if not confs:
            continue
        # Type confidence
        type_conf = confs.get("rel_type")
        if type_conf is not None and gt.get("rel_type") is not None:
            conf_list.append(float(type_conf))
            acc_list.append(pred.get("rel_type") == gt.get("rel_type"))
        # Status confidence
        status_conf = confs.get("rel_status")
        if status_conf is not None and gt.get("rel_status") is not None:
            conf_list.append(float(status_conf))
            acc_list.append(pred.get("rel_status") == gt.get("rel_status"))

    if conf_list:
        result["ece"] = ece(conf_list, acc_list)

    return result


# =====================================================================
# LaTeX table generation
# =====================================================================

def _bold_best(values: List[Optional[float]], lower_is_better: bool = False) -> List[str]:
    """Format a column of values, bolding the best one.

    Args:
        values: Metric values (None means not available).
        lower_is_better: If True, the minimum value is best.

    Returns:
        List of formatted strings with \\textbf{} on the best.
    """
    formatted = []
    valid = [(i, v) for i, v in enumerate(values) if v is not None]

    if not valid:
        return ["--" for _ in values]

    if lower_is_better:
        best_idx = min(valid, key=lambda x: x[1])[0]
    else:
        best_idx = max(valid, key=lambda x: x[1])[0]

    for i, v in enumerate(values):
        if v is None:
            formatted.append("--")
        elif i == best_idx:
            formatted.append(f"\\textbf{{{v:.4f}}}")
        else:
            formatted.append(f"{v:.4f}")

    return formatted


def generate_latex_table(results: Dict[str, Dict], task: str = "personality") -> str:
    """Generate LaTeX table string for paper.

    Args:
        results: {method_name: metrics_dict} from compute_personality_metrics
                 or compute_relationship_metrics.
        task: "personality" or "relationship".

    Returns:
        LaTeX table string using booktabs style.
    """
    if not results:
        return "% No results to display."

    methods = list(results.keys())

    if task == "personality":
        return _latex_personality_table(methods, results)
    elif task == "relationship":
        return _latex_relationship_table(methods, results)
    else:
        raise ValueError(f"Unknown task '{task}', expected 'personality' or 'relationship'")


def _latex_personality_table(methods: List[str], results: Dict[str, Dict]) -> str:
    """Personality task table: Method | MBTI Acc | Big Five MAE | ECE | Coverage | Avg Set Size."""
    # Collect column values
    mbti_acc = [results[m].get("mbti_accuracy") for m in methods]
    bf_mae = [results[m].get("big_five_mae") for m in methods]
    ece_vals = [results[m].get("ece") for m in methods]
    cov_vals = [results[m].get("coverage") for m in methods]
    setsize_vals = [results[m].get("avg_set_size") for m in methods]

    # Format with best bolded
    col_mbti = _bold_best(mbti_acc, lower_is_better=False)
    col_mae = _bold_best(bf_mae, lower_is_better=True)
    col_ece = _bold_best(ece_vals, lower_is_better=True)
    col_cov = _bold_best(cov_vals, lower_is_better=False)
    col_set = _bold_best(setsize_vals, lower_is_better=True)

    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Personality Inference Results}",
        "\\label{tab:personality}",
        "\\begin{tabular}{lccccc}",
        "\\toprule",
        "Method & MBTI Acc $\\uparrow$ & Big Five MAE $\\downarrow$ "
        "& ECE $\\downarrow$ & Coverage $\\uparrow$ & Avg Set Size $\\downarrow$ \\\\",
        "\\midrule",
    ]

    for i, method in enumerate(methods):
        row = (
            f"{method} & {col_mbti[i]} & {col_mae[i]} "
            f"& {col_ece[i]} & {col_cov[i]} & {col_set[i]} \\\\"
        )
        lines.append(row)

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])
    return "\n".join(lines)


def _latex_relationship_table(methods: List[str], results: Dict[str, Dict]) -> str:
    """Relationship task table: Method | Accuracy | F1 | ECE | Coverage."""
    type_acc = [results[m].get("type_accuracy") for m in methods]
    type_f1 = [results[m].get("type_f1") for m in methods]
    status_acc = [results[m].get("status_accuracy") for m in methods]
    status_f1 = [results[m].get("status_f1") for m in methods]
    ece_vals = [results[m].get("ece") for m in methods]

    col_tacc = _bold_best(type_acc, lower_is_better=False)
    col_tf1 = _bold_best(type_f1, lower_is_better=False)
    col_sacc = _bold_best(status_acc, lower_is_better=False)
    col_sf1 = _bold_best(status_f1, lower_is_better=False)
    col_ece = _bold_best(ece_vals, lower_is_better=True)

    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Relationship Prediction Results}",
        "\\label{tab:relationship}",
        "\\begin{tabular}{lccccc}",
        "\\toprule",
        "Method & Type Acc $\\uparrow$ & Type F1 $\\uparrow$ "
        "& Status Acc $\\uparrow$ & Status F1 $\\uparrow$ & ECE $\\downarrow$ \\\\",
        "\\midrule",
    ]

    for i, method in enumerate(methods):
        row = (
            f"{method} & {col_tacc[i]} & {col_tf1[i]} "
            f"& {col_sacc[i]} & {col_sf1[i]} & {col_ece[i]} \\\\"
        )
        lines.append(row)

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])
    return "\n".join(lines)


# =====================================================================
# Human-readable summary
# =====================================================================

def generate_results_summary(all_results: Dict[str, Dict]) -> str:
    """Generate a human-readable summary of all results.

    Args:
        all_results: {method_name: metrics_dict} for any task.

    Returns:
        Multi-line summary string.
    """
    if not all_results:
        return "No results available."

    lines = ["=" * 60, "Results Summary", "=" * 60, ""]

    for method, metrics in all_results.items():
        lines.append(f"--- {method} ---")

        for key, value in sorted(metrics.items()):
            if isinstance(value, dict):
                lines.append(f"  {key}:")
                for sub_key, sub_val in sorted(value.items()):
                    if isinstance(sub_val, dict):
                        parts = ", ".join(
                            f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                            for k, v in sorted(sub_val.items())
                        )
                        lines.append(f"    {sub_key}: {parts}")
                    elif isinstance(sub_val, float):
                        lines.append(f"    {sub_key}: {sub_val:.4f}")
                    else:
                        lines.append(f"    {sub_key}: {sub_val}")
            elif isinstance(value, float):
                lines.append(f"  {key}: {value:.4f}")
            else:
                lines.append(f"  {key}: {value}")

        lines.append("")

    # Comparative ranking
    lines.append("=" * 60)
    lines.append("Ranking")
    lines.append("=" * 60)

    # Find common numeric metrics across methods for ranking
    all_keys: set = set()
    for metrics in all_results.values():
        for k, v in metrics.items():
            if isinstance(v, float):
                all_keys.add(k)

    lower_better = {"big_five_mae", "big_five_rmse", "ece", "avg_set_size"}

    for key in sorted(all_keys):
        pairs = [
            (m, metrics[key])
            for m, metrics in all_results.items()
            if key in metrics and isinstance(metrics[key], float)
        ]
        if len(pairs) < 2:
            continue

        reverse = key not in lower_better
        ranked = sorted(pairs, key=lambda x: x[1], reverse=reverse)
        direction = "(lower is better)" if key in lower_better else "(higher is better)"
        ranking_str = " > ".join(f"{m}({v:.4f})" for m, v in ranked)
        lines.append(f"  {key} {direction}: {ranking_str}")

    return "\n".join(lines)
