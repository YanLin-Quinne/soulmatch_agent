"""Unified baseline evaluation runner for SoulMatch experiments.

Usage:
    python experiments/run_baselines.py --dataset data/evaluation/eval_dataset.json --output experiments/results/
    python experiments/run_baselines.py --dataset data/evaluation/eval_dataset.json --methods direct cot sc --output experiments/results/
"""
import json
import time
import sys
import os
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.baselines import DirectPromptingBaseline, CoTBaseline, SelfConsistencyBaseline
from experiments.metrics import (  # noqa: E402
    compute_personality_metrics,
    compute_relationship_metrics,
    generate_latex_table,
    generate_results_summary,
)
from src.agents.llm_router import router, AgentRole
from src.agents.feature_prediction_agent import FeaturePredictionAgent


# =====================================================================
# SoulMatch system wrappers (Full + Ablation variants)
# =====================================================================

class SoulMatchWrapper:
    """Wrapper around full SoulMatch system for evaluation.

    Supports ablation configs via constructor flags:
    - use_cot: enable/disable CoT reasoning in feature prediction
    - use_bayesian: enable/disable Bayesian updates
    - use_conformal: enable/disable conformal prediction calibration
    - use_discussion_room: enable/disable multi-agent discussion (social agents)
    - use_three_layer_memory: enable/disable three-layer memory (not used in per-sample eval)
    """

    def __init__(
        self,
        name: str = "SoulMatch (Full)",
        use_cot: bool = True,
        use_bayesian: bool = True,
        use_conformal: bool = True,
        use_discussion_room: bool = True,
    ):
        self.name = name
        self.use_cot = use_cot
        self.use_bayesian = use_bayesian
        self.use_conformal = use_conformal
        self.use_discussion_room = use_discussion_room

    def predict_personality(self, dialogue: List[Dict]) -> Dict:
        """Run SoulMatch personality inference pipeline."""
        t0 = time.time()

        agent = FeaturePredictionAgent(
            user_id="eval_user",
            use_cot=self.use_cot,
            calibrator_path=None if not self.use_conformal else None,  # uses default path
        )

        # If not using conformal, disable it
        if not self.use_conformal:
            agent.conformal = None

        # Feed conversation turns incrementally (simulating real usage)
        conversation_so_far = []
        result = None
        for msg in dialogue:
            conversation_so_far.append(msg)
            if len(conversation_so_far) >= 4:  # Start predicting after 2 exchanges
                if self.use_bayesian:
                    result = agent.predict_from_conversation(conversation_so_far)
                else:
                    # Without Bayesian: only use the last observation, no accumulation
                    # Reset prior each time
                    agent.predicted_features = {}
                    agent.feature_confidences = {}
                    result = agent.predict_from_conversation(conversation_so_far)

        if result is None:
            result = agent.predict_from_conversation(dialogue)

        elapsed = time.time() - t0

        # Transform to standard evaluation format
        features = result.get("features", {})
        confidences = result.get("confidences", {})

        return {
            "big_five": {
                "openness": features.get("big_five_openness"),
                "conscientiousness": features.get("big_five_conscientiousness"),
                "extraversion": features.get("big_five_extraversion"),
                "agreeableness": features.get("big_five_agreeableness"),
                "neuroticism": features.get("big_five_neuroticism"),
            },
            "mbti": features.get("mbti_type"),
            "confidences": {
                "big_five_openness": confidences.get("big_five_openness", 0.5),
                "big_five_conscientiousness": confidences.get("big_five_conscientiousness", 0.5),
                "big_five_extraversion": confidences.get("big_five_extraversion", 0.5),
                "big_five_agreeableness": confidences.get("big_five_agreeableness", 0.5),
                "big_five_neuroticism": confidences.get("big_five_neuroticism", 0.5),
                "mbti": confidences.get("mbti_type", 0.5),
            },
            "conformal_sets": result.get("conformal", {}).get("prediction_sets", {}),
            "elapsed_seconds": elapsed,
            "method": self.name,
        }

    def predict_relationship(self, dialogue: List[Dict], _context: Optional[Dict] = None) -> Dict:
        """Simplified relationship prediction (sync wrapper).
        Uses direct LLM call matching the single_llm_assessment path."""
        t0 = time.time()

        from experiments.baselines.utils import format_dialogue, parse_json_response

        dialogue_text = format_dialogue(dialogue)

        prompt = f"""You are a relationship assessment panel with 3 experts:
1. Emotional analyst (weight: 0.4)
2. Values compatibility analyst (weight: 0.3)
3. Behavioral pattern analyst (weight: 0.3)

{"Multi-agent discussion enabled. Consider diverse perspectives from different demographic backgrounds." if self.use_discussion_room else "Single analyst assessment."}

Conversation:
{dialogue_text}

Assess the relationship and output JSON:
{{
  "rel_type": "love|friendship|family|other",
  "rel_type_probs": {{"love": 0.x, "friendship": 0.y, ...}},
  "rel_status": "stranger|acquaintance|crush|dating|committed",
  "rel_status_probs": {{"stranger": 0.x, "acquaintance": 0.y, ...}},
  "confidences": {{"rel_type": 0.x, "rel_status": 0.y}},
  "reasoning": "brief explanation"
}}"""

        try:
            response = router.chat(
                role=AgentRole.PERSONA if self.use_discussion_room else AgentRole.GENERAL,
                system="You are a relationship assessment panel. Output valid JSON only.",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.3,
                json_mode=True,
            )
            result = parse_json_response(response)
            if result is None:
                raise ValueError("Failed to parse response")
        except Exception:
            result = {
                "rel_type": "other",
                "rel_status": "stranger",
                "rel_type_probs": {"other": 1.0},
                "rel_status_probs": {"stranger": 1.0},
            }

        elapsed = time.time() - t0
        result["elapsed_seconds"] = elapsed
        result["method"] = self.name
        result.setdefault("confidences", {"rel_type": 0.5, "rel_status": 0.5})
        return result


# =====================================================================
# Method registry
# =====================================================================

ABLATION_CONFIGS = {
    "SoulMatch (Full)": {
        "use_cot": True,
        "use_bayesian": True,
        "use_conformal": True,
        "use_discussion_room": True,
    },
    "w/o Multi-Agent": {
        "use_cot": True,
        "use_bayesian": True,
        "use_conformal": True,
        "use_discussion_room": False,
    },
    "w/o Bayesian": {
        "use_cot": True,
        "use_bayesian": False,
        "use_conformal": True,
        "use_discussion_room": True,
    },
    "w/o Conformal": {
        "use_cot": True,
        "use_bayesian": True,
        "use_conformal": False,
        "use_discussion_room": True,
    },
    "w/o CoT": {
        "use_cot": False,
        "use_bayesian": True,
        "use_conformal": True,
        "use_discussion_room": True,
    },
}


def build_methods(method_names: Optional[List[str]] = None) -> Dict[str, Any]:
    """Build method instances. If method_names specified, only build those."""
    all_methods = {}

    # Baselines
    all_methods["Direct Prompting"] = DirectPromptingBaseline()
    all_methods["CoT"] = CoTBaseline()
    all_methods["Self-Consistency"] = SelfConsistencyBaseline(n_samples=5)

    # SoulMatch variants
    for name, config in ABLATION_CONFIGS.items():
        all_methods[name] = SoulMatchWrapper(name=name, **config)

    if method_names:
        return {k: v for k, v in all_methods.items() if k in method_names}
    return all_methods


# =====================================================================
# Evaluation loop
# =====================================================================

def load_dataset(path: str) -> List[Dict]:
    """Load evaluation dataset."""
    with open(path) as f:
        return json.load(f)


def evaluate_method(method, dataset: List[Dict], task: str = "personality") -> Dict:
    """Run a single method on the dataset and return predictions."""
    predictions = []
    errors = 0

    for sample in dataset:
        dialogue = sample["dialogue"]

        try:
            if task == "personality":
                pred = method.predict_personality(dialogue)
            else:
                pred = method.predict_relationship(dialogue, sample.get("ground_truth", {}))
            predictions.append(pred)
        except Exception as e:
            print(f"  Error on {sample['conversation_id']}: {e}")
            errors += 1
            predictions.append(None)

    return {
        "predictions": predictions,
        "errors": errors,
        "total": len(dataset),
    }


def run_evaluation(
    dataset_path: str,
    output_dir: str,
    method_names: Optional[List[str]] = None,
    seed: int = 42,
):
    """Run full evaluation pipeline."""
    # Load dataset
    dataset = load_dataset(dataset_path)
    print(f"Loaded {len(dataset)} samples from {dataset_path}")

    # Build methods
    methods = build_methods(method_names)
    print(f"Running {len(methods)} methods: {list(methods.keys())}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # -- Task 1: Personality Inference --
    print("\n" + "=" * 60)
    print("Task 1: Personality Inference")
    print("=" * 60)

    personality_results = {}
    for name, method in methods.items():
        print(f"\n  Running: {name}...")
        eval_result = evaluate_method(method, dataset, task="personality")

        # Extract ground truths and valid predictions
        gt_list = []
        pred_list = []
        for sample, pred in zip(dataset, eval_result["predictions"]):
            if pred is None:
                continue
            gt_list.append(sample["ground_truth"]["personality"])
            pred_list.append(pred)

        # Compute metrics
        if pred_list:
            metrics = compute_personality_metrics(pred_list, gt_list)
        else:
            metrics = {}

        # Add timing info
        valid_preds = [p for p in eval_result["predictions"] if p is not None]
        if valid_preds:
            metrics["avg_latency"] = sum(p.get("elapsed_seconds", 0) for p in valid_preds) / len(valid_preds)

        metrics["errors"] = eval_result["errors"]
        personality_results[name] = metrics
        print(f"    MBTI Acc={metrics.get('mbti_accuracy', 'N/A')}, Big Five MAE={metrics.get('big_five_mae', 'N/A')}")

    # -- Task 2: Relationship Prediction --
    print("\n" + "=" * 60)
    print("Task 2: Relationship Prediction")
    print("=" * 60)

    relationship_results = {}
    for name, method in methods.items():
        print(f"\n  Running: {name}...")
        eval_result = evaluate_method(method, dataset, task="relationship")

        gt_list = []
        pred_list = []
        for sample, pred in zip(dataset, eval_result["predictions"]):
            if pred is None:
                continue
            gt_list.append(sample["ground_truth"].get("relationship", {}))
            pred_list.append(pred)

        if pred_list:
            metrics = compute_relationship_metrics(pred_list, gt_list)
        else:
            metrics = {}

        valid_preds = [p for p in eval_result["predictions"] if p is not None]
        if valid_preds:
            metrics["avg_latency"] = sum(p.get("elapsed_seconds", 0) for p in valid_preds) / len(valid_preds)

        metrics["errors"] = eval_result["errors"]
        relationship_results[name] = metrics
        print(f"    Acc={metrics.get('type_accuracy', 'N/A')}, F1={metrics.get('type_f1', 'N/A')}")

    # -- Save results --
    all_results = {
        "personality": personality_results,
        "relationship": relationship_results,
        "dataset_path": dataset_path,
        "n_samples": len(dataset),
        "methods": list(methods.keys()),
        "seed": seed,
    }

    # Save JSON
    json_path = os.path.join(output_dir, "metrics.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {json_path}")

    # Save LaTeX tables
    if personality_results:
        latex_p = generate_latex_table(personality_results, task="personality")
        with open(os.path.join(output_dir, "table_personality.tex"), "w") as f:
            f.write(latex_p)
        print(f"Personality table: {output_dir}/table_personality.tex")

    if relationship_results:
        latex_r = generate_latex_table(relationship_results, task="relationship")
        with open(os.path.join(output_dir, "table_relationship.tex"), "w") as f:
            f.write(latex_r)
        print(f"Relationship table: {output_dir}/table_relationship.tex")

    # Print summary
    print("\n" + generate_results_summary(all_results))

    # Save usage report
    usage = router.get_usage_report()
    with open(os.path.join(output_dir, "llm_usage.json"), "w") as f:
        json.dump(usage, f, indent=2)
    print(f"LLM usage report: {output_dir}/llm_usage.json")

    return all_results


# =====================================================================
# CLI
# =====================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SoulMatch baseline evaluation")
    parser.add_argument("--dataset", type=str, required=True, help="Path to eval dataset JSON")
    parser.add_argument("--output", type=str, default="experiments/results/", help="Output directory")
    parser.add_argument("--methods", nargs="*", default=None, help="Method names to run (default: all)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    run_evaluation(
        dataset_path=args.dataset,
        output_dir=args.output,
        method_names=args.methods,
        seed=args.seed,
    )
