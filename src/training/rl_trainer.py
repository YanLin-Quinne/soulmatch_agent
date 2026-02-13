"""
RL (GRPO) Trainer — Improve pass@1 accuracy for feature prediction.

Uses Group Relative Policy Optimization to align the model's feature predictions
with ground truth, rewarding accurate JSON output and penalizing hallucination.

Usage:
    python -m src.training.rl_trainer --sft-model models/sft/final --data data/training/synthetic_dialogues.jsonl
"""

import json
import argparse
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

from loguru import logger

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from trl import GRPOTrainer, GRPOConfig
    TRL_AVAILABLE = True
except ImportError:
    TRL_AVAILABLE = False


# ---------------------------------------------------------------------------
# Reward functions
# ---------------------------------------------------------------------------

def feature_prediction_reward(completions: list[list[dict]], ground_truths: list[dict], **kwargs) -> list[float]:
    """
    Reward function for feature prediction quality.

    Scores each completion based on:
    1. Valid JSON output (+0.2)
    2. Feature accuracy vs ground truth (up to +0.5)
    3. Confidence calibration (+0.15)
    4. Completeness of predictions (+0.15)

    Args:
        completions: List of completion groups (GRPO generates multiple per prompt)
        ground_truths: Corresponding ground truth feature dicts

    Returns:
        Flat list of rewards, one per completion.
    """
    rewards = []

    for group, truth in zip(completions, ground_truths):
        for completion in group:
            content = completion.get("content", "")
            score = _score_prediction(content, truth)
            rewards.append(score)

    return rewards


def _score_prediction(text: str, truth: dict) -> float:
    """Score a single prediction against ground truth."""
    score = 0.0

    # 1. Valid JSON (+0.2)
    try:
        text = text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
        predicted = json.loads(text.strip())
        score += 0.2
    except (json.JSONDecodeError, ValueError):
        return 0.0  # Invalid JSON gets zero reward

    if not isinstance(predicted, dict):
        return 0.1  # Parsed but not a dict

    # 2. Feature accuracy (+0.5 max)
    accuracy_score = _feature_accuracy(predicted, truth)
    score += accuracy_score * 0.5

    # 3. Confidence calibration (+0.15 max)
    calibration_score = _calibration_score(predicted, truth)
    score += calibration_score * 0.15

    # 4. Completeness (+0.15 max)
    expected_keys = {"age", "sex", "communication_style", "relationship_goals"}
    for trait in ("openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"):
        expected_keys.add(f"big_five_{trait}")
    present = sum(1 for k in expected_keys if k in predicted)
    completeness = present / len(expected_keys)
    score += completeness * 0.15

    return round(min(score, 1.0), 4)


def _feature_accuracy(predicted: dict, truth: dict) -> float:
    """Compute accuracy of predicted features vs ground truth."""
    matches = 0
    total = 0

    # Numeric features (Big Five, interests)
    for prefix in ("big_five_", "interest_"):
        for key in predicted:
            if key.startswith(prefix):
                total += 1
                truth_key = key
                truth_val = truth.get("personality", {}).get(key.replace("big_five_", "")) if "big_five" in key else None
                if truth_val is None:
                    interests = truth.get("interests", {})
                    truth_val = interests.get(key.replace("interest_", ""))
                if truth_val is not None and isinstance(predicted[key], (int, float)):
                    diff = abs(float(predicted[key]) - float(truth_val))
                    if diff < 0.2:
                        matches += 1
                    elif diff < 0.4:
                        matches += 0.5

    # Categorical features
    for key in ("communication_style", "relationship_goals", "sex"):
        if key in predicted and key in truth.get("profile_metadata", {}):
            total += 1
            if str(predicted[key]).lower() == str(truth["profile_metadata"][key]).lower():
                matches += 1
        elif key in predicted and key in truth:
            total += 1
            if str(predicted[key]).lower() == str(truth[key]).lower():
                matches += 1

    return matches / max(total, 1)


def _calibration_score(predicted: dict, truth: dict) -> float:
    """Reward well-calibrated confidence scores."""
    confidences = predicted.get("_confidence", {})
    if not confidences:
        return 0.3  # No confidence = neutral

    scores = []
    for key, conf in confidences.items():
        if not isinstance(conf, (int, float)):
            continue
        # Check if the prediction exists and is correct
        pred_val = predicted.get(key)
        if pred_val is None:
            # Low confidence for missing = good
            if conf < 0.3:
                scores.append(1.0)
            else:
                scores.append(0.0)
        else:
            # High confidence for present predictions
            scores.append(min(conf, 1.0))

    return sum(scores) / max(len(scores), 1)


# ---------------------------------------------------------------------------
# RL Training
# ---------------------------------------------------------------------------

def build_rl_prompts(data_path: str, max_examples: int = 500) -> tuple[list[str], list[dict]]:
    """Build prompts and ground truths from synthetic dialogue data."""
    data_path = Path(data_path)
    with open(data_path, "r") as f:
        dialogues = [json.loads(line) for line in f if line.strip()]

    prompts = []
    truths = []

    for dialogue in dialogues[:max_examples]:
        turns = dialogue.get("turns", [])
        ground_truth = dialogue.get("ground_truth_features", {})

        if len(turns) < 4:
            continue

        context = "\n".join(f"{t['speaker']}: {t['message']}" for t in turns[-10:])

        for bot_id, truth in ground_truth.items():
            prompt = (
                f"<|im_start|>system\n"
                f"You are a psychology expert. Analyze this dating conversation and predict personality features as JSON.\n"
                f"<|im_end|>\n"
                f"<|im_start|>user\n"
                f"Conversation:\n{context}\n\n"
                f"Predict features for {bot_id}.\n"
                f"<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
            prompts.append(prompt)
            truths.append(truth)

    logger.info(f"Built {len(prompts)} RL training prompts from {len(dialogues)} dialogues")
    return prompts, truths


def train_rl(
    sft_model_path: str = "models/sft/final",
    data_path: str = "data/training/synthetic_dialogues.jsonl",
    output_dir: str = "models/rl",
    num_epochs: int = 1,
    batch_size: int = 2,
    num_generations: int = 4,
    learning_rate: float = 5e-6,
    max_examples: int = 500,
    device: Optional[str] = None,
):
    """
    Run GRPO training on top of SFT model.

    Args:
        sft_model_path: Path to SFT-trained model
        data_path: Path to synthetic dialogue data
        output_dir: Where to save RL-trained model
        num_epochs: Training epochs
        batch_size: Per-device batch size
        num_generations: GRPO group size (completions per prompt)
        learning_rate: Learning rate
        max_examples: Max training examples
        device: Device override
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch not installed")
    if not TRL_AVAILABLE:
        raise RuntimeError("TRL not installed. Run: pip install trl>=0.12")

    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    # Load SFT model
    logger.info(f"Loading SFT model from {sft_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(sft_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        sft_model_path,
        torch_dtype=torch.bfloat16 if device != "cpu" else torch.float32,
        trust_remote_code=True,
    )

    # Build prompts and ground truths
    prompts, truths = build_rl_prompts(data_path, max_examples)

    # Convert to dataset format expected by GRPOTrainer
    dataset = [{"prompt": p, "ground_truth": t} for p, t in zip(prompts, truths)]

    # GRPO config
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    grpo_config = GRPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        num_generations=num_generations,
        max_completion_length=512,
        logging_steps=5,
        save_strategy="epoch",
        save_total_limit=2,
        bf16=device != "cpu",
        gradient_accumulation_steps=2,
        report_to="none",
    )

    # Wrap reward function to pass ground truths
    ground_truth_map = {p: t for p, t in zip(prompts, truths)}

    def reward_fn(completions, prompts_batch, **kwargs):
        batch_truths = [ground_truth_map.get(p, {}) for p in prompts_batch]
        return feature_prediction_reward(completions, batch_truths)

    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        processing_class=tokenizer,
        train_dataset=dataset,
        reward_funcs=reward_fn,
    )

    logger.info("Starting GRPO training...")
    trainer.train()

    trainer.save_model(str(output_dir / "final"))
    tokenizer.save_pretrained(str(output_dir / "final"))
    logger.info(f"RL model saved to {output_dir / 'final'}")

    return str(output_dir / "final")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GRPO RL Training")
    parser.add_argument("--sft-model", default="models/sft/final")
    parser.add_argument("--data", default="data/training/synthetic_dialogues.jsonl")
    parser.add_argument("--output", default="models/rl")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--max-examples", type=int, default=500)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    train_rl(
        sft_model_path=args.sft_model,
        data_path=args.data,
        output_dir=args.output,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        num_generations=args.num_generations,
        learning_rate=args.lr,
        max_examples=args.max_examples,
        device=args.device,
    )
