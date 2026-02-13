"""
SFT (Supervised Fine-Tuning) Cold Start Trainer.

Takes synthetic dialogue data and fine-tunes a small model (Qwen3-0.6B by default)
on feature prediction and memory summarization tasks.

Training data format:
  - Input: conversation context (last N turns)
  - Output: JSON feature predictions with confidence scores

Usage:
    python -m src.training.sft_trainer --data data/training/synthetic_dialogues.jsonl --epochs 3
"""

import json
import argparse
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field

from loguru import logger

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
    from trl import SFTTrainer, SFTConfig
    TRL_AVAILABLE = True
except ImportError:
    TRL_AVAILABLE = False


@dataclass
class SFTDataConfig:
    """Configuration for SFT data preparation."""
    max_context_turns: int = 10
    max_input_length: int = 1024
    max_output_length: int = 512
    train_split: float = 0.9


class FeaturePredictionDataset(Dataset):
    """Dataset that converts synthetic dialogues into feature prediction training examples."""

    def __init__(self, dialogues: list[dict], config: SFTDataConfig):
        self.examples = []
        self.config = config
        self._prepare(dialogues)

    def _prepare(self, dialogues: list[dict]):
        for dialogue in dialogues:
            turns = dialogue.get("turns", [])
            ground_truth = dialogue.get("ground_truth_features", {})

            if len(turns) < 4:
                continue

            for bot_id, truth in ground_truth.items():
                # Build conversation context
                context_turns = turns[-self.config.max_context_turns:]
                conversation = "\n".join(
                    f"{t['speaker']}: {t['message']}" for t in context_turns
                )

                # Build target JSON
                target = self._format_target(truth)

                prompt = (
                    f"<|im_start|>system\n"
                    f"You are a psychology expert. Analyze this dating conversation and predict personality features as JSON.\n"
                    f"<|im_end|>\n"
                    f"<|im_start|>user\n"
                    f"Conversation:\n{conversation}\n\n"
                    f"Predict features for {bot_id}.\n"
                    f"<|im_end|>\n"
                    f"<|im_start|>assistant\n"
                    f"{target}\n"
                    f"<|im_end|>"
                )

                self.examples.append({"text": prompt})

        logger.info(f"Prepared {len(self.examples)} training examples from {len(dialogues)} dialogues")

    @staticmethod
    def _format_target(truth: dict) -> str:
        """Format ground truth features into the prediction target format."""
        target = {}

        # Personality
        personality = truth.get("personality", {})
        for trait in ("openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"):
            val = personality.get(trait)
            if val is not None:
                target[f"big_five_{trait}"] = round(val, 2)

        # Communication
        if truth.get("communication_style"):
            target["communication_style"] = truth["communication_style"]

        # Relationship goals
        if truth.get("relationship_goals"):
            target["relationship_goals"] = truth["relationship_goals"]

        # Interests
        interests = truth.get("interests", {})
        for k, v in interests.items():
            if isinstance(v, (int, float)):
                target[f"interest_{k}"] = round(v, 2)

        # Demographics
        meta = truth.get("profile_metadata", {})
        if meta.get("age"):
            target["age"] = meta["age"]
        if meta.get("sex"):
            target["sex"] = meta["sex"]

        return json.dumps(target, indent=2)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


class MemorySummarizationDataset(Dataset):
    """Dataset for training memory summarization from conversation context."""

    def __init__(self, dialogues: list[dict], config: SFTDataConfig):
        self.examples = []
        self.config = config
        self._prepare(dialogues)

    def _prepare(self, dialogues: list[dict]):
        for dialogue in dialogues:
            turns = dialogue.get("turns", [])
            if len(turns) < 6:
                continue

            # Build conversation text
            conversation = "\n".join(f"{t['speaker']}: {t['message']}" for t in turns)

            # Build memory summary from ground truth
            truth = dialogue.get("ground_truth_features", {})
            memory_items = []
            for bot_id, features in truth.items():
                meta = features.get("profile_metadata", {})
                personality = features.get("personality", {})
                summary = features.get("personality_summary", "")
                if summary:
                    memory_items.append(f"{bot_id}: {summary}")
                interests = features.get("interests", {})
                if interests:
                    top = sorted(interests.items(), key=lambda x: x[1] if isinstance(x[1], (int, float)) else 0, reverse=True)[:3]
                    memory_items.append(f"{bot_id} interests: {', '.join(k for k, v in top)}")

            if not memory_items:
                continue

            memory_text = "\n".join(f"- {item}" for item in memory_items)

            prompt = (
                f"<|im_start|>system\n"
                f"You are a memory manager. Extract key facts worth remembering from this conversation.\n"
                f"<|im_end|>\n"
                f"<|im_start|>user\n"
                f"Conversation:\n{conversation}\n\n"
                f"What should we remember about these users?\n"
                f"<|im_end|>\n"
                f"<|im_start|>assistant\n"
                f"{memory_text}\n"
                f"<|im_end|>"
            )

            self.examples.append({"text": prompt})

        logger.info(f"Prepared {len(self.examples)} memory examples from {len(dialogues)} dialogues")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def train_sft(
    data_path: str = "data/training/synthetic_dialogues.jsonl",
    model_name: str = "Qwen/Qwen3-0.6B",
    output_dir: str = "models/sft",
    num_epochs: int = 3,
    batch_size: int = 2,
    learning_rate: float = 2e-5,
    max_seq_length: int = 1536,
    task: str = "feature_prediction",  # "feature_prediction" or "memory"
    device: Optional[str] = None,
):
    """
    Run SFT training.

    Args:
        data_path: Path to synthetic dialogue JSONL
        model_name: HuggingFace model ID
        output_dir: Where to save the fine-tuned model
        num_epochs: Training epochs
        batch_size: Per-device batch size
        learning_rate: Learning rate
        max_seq_length: Max sequence length
        task: Which task to train ("feature_prediction" or "memory")
        device: Device override ("cuda", "mps", "cpu")
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch not installed. Run: pip install torch")
    if not TRL_AVAILABLE:
        raise RuntimeError("TRL not installed. Run: pip install trl transformers")

    # Auto-detect device
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    logger.info(f"Training on device: {device}")

    # Load data
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Training data not found: {data_path}")

    with open(data_path, "r") as f:
        dialogues = [json.loads(line) for line in f if line.strip()]
    logger.info(f"Loaded {len(dialogues)} dialogues from {data_path}")

    # Build dataset
    config = SFTDataConfig()
    if task == "feature_prediction":
        dataset = FeaturePredictionDataset(dialogues, config)
    elif task == "memory":
        dataset = MemorySummarizationDataset(dialogues, config)
    else:
        raise ValueError(f"Unknown task: {task}")

    if len(dataset) == 0:
        raise ValueError("No training examples generated from data")

    # Split train/eval
    train_size = int(len(dataset) * config.train_split)
    eval_size = len(dataset) - train_size
    train_dataset, eval_dataset = torch.utils.data.random_split(dataset, [train_size, eval_size])

    logger.info(f"Train: {train_size}, Eval: {eval_size}")

    # Load model and tokenizer
    logger.info(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if device != "cpu" else torch.float32,
        trust_remote_code=True,
    )

    # Training config
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sft_config = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        max_seq_length=max_seq_length,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        bf16=device != "cpu",
        gradient_accumulation_steps=4,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    logger.info("Starting SFT training...")
    trainer.train()

    # Save
    trainer.save_model(str(output_dir / "final"))
    tokenizer.save_pretrained(str(output_dir / "final"))
    logger.info(f"Model saved to {output_dir / 'final'}")

    return str(output_dir / "final")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SFT Cold Start Training")
    parser.add_argument("--data", default="data/training/synthetic_dialogues.jsonl")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--output", default="models/sft")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--task", choices=["feature_prediction", "memory"], default="feature_prediction")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    train_sft(
        data_path=args.data,
        model_name=args.model,
        output_dir=args.output,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        task=args.task,
        device=args.device,
    )
