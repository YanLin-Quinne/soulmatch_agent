"""
Calibration Pipeline for Conformal Feature Prediction

This script:
  1. Loads synthetic dialogue data (from synthetic_dialogue_generator)
  2. Runs the feature predictor on each dialogue at turn checkpoints
  3. Collects (LLM_softmax, ground_truth) pairs
  4. Fits the ConformalCalibrator
  5. Evaluates coverage and efficiency on held-out test set
  6. Saves the fitted calibrator for inference

Usage:
    # Quick mode (simulated softmax, no LLM calls — for testing)
    python -m src.training.calibration_pipeline --mode simulate --dialogues data/training/synthetic_dialogues.jsonl

    # Full mode (actual LLM calls — expensive but accurate)
    python -m src.training.calibration_pipeline --mode live --dialogues data/training/synthetic_dialogues.jsonl

    # Evaluate only (after calibrator is fitted)
    python -m src.training.calibration_pipeline --mode evaluate --calibrator data/calibration/conformal_calibrator.json
"""

import json
import argparse
import random
from pathlib import Path
from typing import List, Dict, Optional
from loguru import logger

from src.agents.conformal_calibrator import (
    ConformalCalibrator,
    CalibrationSample,
    discretize_value,
    get_options,
    CATEGORICAL_DIMS,
    CONTINUOUS_BINS,
    ALL_DIMS,
)


def load_dialogues(path: str) -> List[Dict]:
    """Load JSONL dialogue file."""
    dialogues = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                dialogues.append(json.loads(line))
    logger.info(f"Loaded {len(dialogues)} dialogues from {path}")
    return dialogues


def flatten_ground_truth(gt: Dict) -> Dict[str, str]:
    """
    Convert ground_truth_features to flat {dim: discretized_label}.
    Matches the format used by ConformalCalibrator.
    """
    flat = {}

    meta = gt.get("profile_metadata", {})
    if meta.get("sex"):
        flat["sex"] = meta["sex"]
    if meta.get("orientation"):
        flat["orientation"] = meta["orientation"]
    if meta.get("education"):
        flat["background_education"] = str(meta["education"]).lower().replace(" ", "_")
    if meta.get("age") is not None:
        flat["age"] = discretize_value("age", meta["age"])

    if gt.get("communication_style"):
        flat["communication_style"] = gt["communication_style"]
    if gt.get("relationship_goals"):
        flat["relationship_goals"] = gt["relationship_goals"]

    # Big Five personality
    personality = gt.get("personality", {})
    for trait in ("openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"):
        val = personality.get(trait)
        if val is not None:
            flat[f"big_five_{trait}"] = discretize_value(f"big_five_{trait}", val)

    # Interests
    interests = gt.get("interests", {})
    for name, val in interests.items():
        dim = f"interest_{name}"
        if dim in CONTINUOUS_BINS and val is not None:
            flat[dim] = discretize_value(dim, val)

    return flat


def generate_calibration_samples_simulated(
    dialogues: List[Dict],
    turn_checkpoints: List[int] = None,
) -> List[CalibrationSample]:
    """
    Generate calibration samples using simulated softmax (no LLM calls).

    Simulates what the LLM *would* output by creating noisy softmax
    distributions centered on the ground truth, with noise level
    proportional to how early in the conversation we are.
    """
    import numpy as np

    if turn_checkpoints is None:
        turn_checkpoints = [3, 5, 8, 10, 15, 20, 25, 30]

    samples = []

    for dialogue in dialogues:
        gt_map = dialogue.get("ground_truth_features", {})
        num_turns = dialogue.get("num_turns", 10)

        for profile_id, gt in gt_map.items():
            flat_gt = flatten_ground_truth(gt)

            for turn in turn_checkpoints:
                if turn > num_turns:
                    continue

                progress = min(turn / max(num_turns, 1), 1.0)

                for dim, true_label in flat_gt.items():
                    options = get_options(dim)
                    if not options or true_label not in options:
                        continue

                    # Simulate LLM softmax with noise
                    # Early turns: flat (uncertain), late turns: peaked (confident)
                    temperature = max(0.3, 1.5 - progress * 1.2)
                    base_logit = 1.0 + progress * 3.0

                    logits = {}
                    for opt in options:
                        if opt == true_label:
                            logits[opt] = base_logit + np.random.normal(0, 0.3)
                        else:
                            logits[opt] = np.random.normal(0, 0.5)

                    # Softmax
                    vals = np.array([logits[opt] for opt in options]) / temperature
                    vals -= vals.max()
                    exp_vals = np.exp(vals)
                    probs = exp_vals / exp_vals.sum()

                    softmax = {opt: float(p) for opt, p in zip(options, probs)}

                    # Add miscalibration noise (simulate LLM being overconfident)
                    # ~20% of the time, flip the top prediction (LLM gets it wrong)
                    if np.random.random() < max(0.05, 0.35 - progress * 0.3):
                        wrong_opt = np.random.choice([o for o in options if o != true_label])
                        softmax[wrong_opt], softmax[true_label] = softmax[true_label], softmax[wrong_opt]

                    samples.append(CalibrationSample(
                        profile_id=profile_id,
                        turn=turn,
                        dimension=dim,
                        llm_softmax=softmax,
                        ground_truth=true_label,
                    ))

    logger.info(f"Generated {len(samples)} simulated calibration samples")
    return samples


def generate_calibration_samples_live(
    dialogues: List[Dict],
    turn_checkpoints: List[int] = None,
) -> List[CalibrationSample]:
    """
    Generate calibration samples using actual LLM calls.

    This is expensive but gives accurate calibration.
    Runs the FeaturePredictionAgent on each dialogue prefix.
    """
    from src.agents.feature_prediction_agent import FeaturePredictionAgent

    if turn_checkpoints is None:
        turn_checkpoints = [5, 10, 15, 20, 25, 30]

    samples = []

    for i, dialogue in enumerate(dialogues):
        gt_map = dialogue.get("ground_truth_features", {})
        turns = dialogue.get("turns", [])
        num_turns = len(turns)

        for profile_id, gt in gt_map.items():
            flat_gt = flatten_ground_truth(gt)

            # Create a fresh predictor for each profile
            predictor = FeaturePredictionAgent(
                user_id=f"cal_{profile_id}",
                use_cot=False,  # faster without CoT for calibration
                calibrator_path=None,  # don't load calibrator during calibration!
            )

            for turn_idx in turn_checkpoints:
                if turn_idx > num_turns:
                    break

                # Feed conversation up to this turn
                conversation_prefix = turns[:turn_idx]
                result = predictor.predict_from_conversation(conversation_prefix)

                features = result.get("features", {})
                confidences = result.get("confidences", {})

                # Convert to calibration samples
                for dim, true_label in flat_gt.items():
                    options = get_options(dim)
                    if not options or true_label not in options:
                        continue

                    # Build softmax from LLM output
                    point_pred = features.get(dim)
                    if point_pred is None:
                        continue

                    # Discretize if needed
                    if isinstance(point_pred, (int, float)):
                        point_pred = discretize_value(dim, point_pred)

                    conf = confidences.get(dim, 0.5)
                    conf = max(0.01, min(0.99, conf))

                    softmax = {}
                    remaining = 1.0 - conf
                    n_others = max(len(options) - 1, 1)
                    for opt in options:
                        if opt == str(point_pred):
                            softmax[opt] = conf
                        else:
                            softmax[opt] = remaining / n_others

                    samples.append(CalibrationSample(
                        profile_id=profile_id,
                        turn=turn_idx,
                        dimension=dim,
                        llm_softmax=softmax,
                        ground_truth=true_label,
                    ))

        if (i + 1) % 10 == 0:
            logger.info(f"Processed {i + 1}/{len(dialogues)} dialogues, {len(samples)} samples so far")

    logger.info(f"Generated {len(samples)} live calibration samples")
    return samples


def run_calibration(
    dialogues_path: str,
    output_dir: str = "data/calibration",
    mode: str = "simulate",
    alpha: float = 0.10,
    test_fraction: float = 0.2,
):
    """
    Full calibration pipeline.

    Args:
        dialogues_path: Path to synthetic dialogues JSONL
        output_dir: Where to save calibrator and evaluation results
        mode: "simulate" (fast, no LLM) or "live" (accurate, uses LLM)
        alpha: Miscoverage rate (0.10 = 90% coverage)
        test_fraction: Fraction of data held out for evaluation
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load dialogues
    dialogues = load_dialogues(dialogues_path)

    # 2. Split into calibration and test
    random.shuffle(dialogues)
    split_idx = int(len(dialogues) * (1 - test_fraction))
    cal_dialogues = dialogues[:split_idx]
    test_dialogues = dialogues[split_idx:]
    logger.info(f"Split: {len(cal_dialogues)} calibration, {len(test_dialogues)} test")

    # 3. Generate calibration samples
    if mode == "simulate":
        cal_samples = generate_calibration_samples_simulated(cal_dialogues)
        test_samples = generate_calibration_samples_simulated(test_dialogues)
    elif mode == "live":
        cal_samples = generate_calibration_samples_live(cal_dialogues)
        test_samples = generate_calibration_samples_live(test_dialogues)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # 4. Fit calibrator
    calibrator = ConformalCalibrator(alpha=alpha)
    calibrator.fit(cal_samples)

    # 5. Evaluate
    eval_results = calibrator.evaluate_coverage(test_samples)
    logger.info(f"Evaluation results:")
    logger.info(f"  Marginal coverage: {eval_results['marginal_coverage']:.3f} (target: {1 - alpha:.2f})")
    logger.info(f"  Avg set size: {eval_results['avg_set_size']:.2f}")
    logger.info(f"  LLM ECE: {eval_results['ece_llm']:.4f}")
    logger.info(f"  Per-turn coverage: {eval_results['per_turn_coverage']}")

    # 6. Save
    calibrator_path = output_dir / "conformal_calibrator.json"
    calibrator.save(str(calibrator_path))

    eval_path = output_dir / "evaluation_results.json"
    with open(eval_path, "w") as f:
        json.dump(eval_results, f, indent=2)
    logger.info(f"Evaluation saved to {eval_path}")

    return calibrator, eval_results


def main():
    parser = argparse.ArgumentParser(description="Calibrate conformal predictor for feature prediction")
    parser.add_argument("--dialogues", type=str, default="data/training/synthetic_dialogues.jsonl",
                        help="Path to synthetic dialogues JSONL")
    parser.add_argument("--output", type=str, default="data/calibration",
                        help="Output directory for calibrator and evaluation")
    parser.add_argument("--mode", type=str, choices=["simulate", "live", "evaluate"], default="simulate",
                        help="simulate=fast/no LLM, live=actual LLM calls, evaluate=eval existing calibrator")
    parser.add_argument("--alpha", type=float, default=0.10,
                        help="Miscoverage rate (0.10 = 90%% coverage)")
    parser.add_argument("--calibrator", type=str, default=None,
                        help="Path to existing calibrator (for evaluate mode)")

    args = parser.parse_args()

    if args.mode == "evaluate":
        calibrator = ConformalCalibrator()
        calibrator.load(args.calibrator or f"{args.output}/conformal_calibrator.json")
        dialogues = load_dialogues(args.dialogues)
        test_samples = generate_calibration_samples_simulated(dialogues)
        results = calibrator.evaluate_coverage(test_samples)
        print(json.dumps(results, indent=2))
    else:
        run_calibration(
            dialogues_path=args.dialogues,
            output_dir=args.output,
            mode=args.mode,
            alpha=args.alpha,
        )


if __name__ == "__main__":
    main()
