#!/usr/bin/env python3
"""Master experiment runner for AI YOU paper.

Usage:
    # Generate data + run all experiments
    python experiments/run_full_experiment.py --all

    # Only generate eval dataset
    python experiments/run_full_experiment.py --generate-data

    # Only run baselines (assumes data exists)
    python experiments/run_full_experiment.py --run-baselines

    # Only run MPI benchmark
    python experiments/run_full_experiment.py --run-mpi
"""

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.generate_eval_dataset import (
    DEFAULT_OUTPUT_PATH as DEFAULT_DATASET_PATH,
    USER_PERSONAS,
    generate_eval_dataset,
    generate_eval_dataset_for_personas,
    load_bot_personas,
    parse_turn_counts,
)
from experiments.metrics import generate_results_summary
from experiments.run_baselines import AIYouWrapper, build_methods, run_evaluation
from src.agents.llm_router import AgentRole, MODELS, MODEL_ROUTING, UsageRecord, router
from src.config import settings


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = PROJECT_ROOT / "experiments" / "results"
MPI_BENCHMARK_PATH = PROJECT_ROOT / "experiments" / "benchmarks" / "mpi_benchmark.py"

ESTIMATED_ROLE_TOKENS = {
    AgentRole.GENERAL: {"input_tokens": 220, "output_tokens": 45},
    AgentRole.PERSONA: {"input_tokens": 240, "output_tokens": 70},
    AgentRole.FEATURE: {"input_tokens": 700, "output_tokens": 260},
}


@dataclass
class StepRecord:
    name: str
    status: str
    started_at: str
    finished_at: str
    duration_seconds: float
    details: Dict[str, Any] = field(default_factory=dict)
    artifacts: Dict[str, str] = field(default_factory=dict)
    error: Optional[str] = None


def configured_providers() -> List[str]:
    """Return routed providers with usable credentials."""
    providers = []
    if settings.anthropic_api_key:
        providers.append("anthropic")
    if settings.openai_api_key:
        providers.append("openai")
    if settings.gemini_api_key:
        providers.append("gemini")
    if settings.deepseek_api_key:
        providers.append("deepseek")
    if settings.qwen_api_key:
        providers.append("qwen")
    return providers


def ensure_provider_configured() -> List[str]:
    """Fail fast if no routed provider is configured."""
    providers = configured_providers()
    if not providers:
        raise RuntimeError(
            "No routed LLM provider is configured. Set at least one of: "
            "ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY, DEEPSEEK_API_KEY, QWEN_API_KEY."
        )
    return providers


def timestamp_now() -> str:
    """Return a filesystem-safe timestamp."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def usage_snapshot() -> Dict[str, Any]:
    """Take a snapshot of current router usage counters."""
    usage = router.usage
    return {
        "total_calls": usage.call_count,
        "total_errors": usage.errors,
        "total_input_tokens": usage.total_input_tokens,
        "total_output_tokens": usage.total_output_tokens,
        "total_cost_usd": usage.total_cost_usd,
    }


def usage_delta(before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, Any]:
    """Compute a router usage delta between two snapshots."""
    return {
        "total_calls": after["total_calls"] - before["total_calls"],
        "total_errors": after["total_errors"] - before["total_errors"],
        "total_input_tokens": after["total_input_tokens"] - before["total_input_tokens"],
        "total_output_tokens": after["total_output_tokens"] - before["total_output_tokens"],
        "total_cost_usd": round(after["total_cost_usd"] - before["total_cost_usd"], 6),
    }


def reset_router_usage() -> None:
    """Reset router usage so downstream reports stay step-local."""
    router.usage = UsageRecord()


def first_available_model(role: AgentRole) -> Optional[str]:
    """Pick the first model in the routing chain backed by a configured provider."""
    provider_names = set(configured_providers())
    for model_key in MODEL_ROUTING.get(role, []):
        spec = MODELS[model_key]
        if spec.provider.value in provider_names:
            return model_key
    return None


def estimate_role_cost(role: AgentRole, n_calls: int) -> Dict[str, Any]:
    """Estimate cost for a given role using the first configured routed model."""
    model_key = first_available_model(role)
    token_guess = ESTIMATED_ROLE_TOKENS[role]
    input_tokens = token_guess["input_tokens"] * n_calls
    output_tokens = token_guess["output_tokens"] * n_calls

    if model_key is None:
        return {
            "model": None,
            "calls": n_calls,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "estimated_cost_usd": None,
        }

    spec = MODELS[model_key]
    estimated_cost = (
        (input_tokens / 1000) * spec.input_cost_per_1k
        + (output_tokens / 1000) * spec.output_cost_per_1k
    )
    return {
        "model": model_key,
        "calls": n_calls,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "estimated_cost_usd": round(estimated_cost, 6),
    }


def estimate_dataset_generation_cost(
    bot_count: int,
    n_conversations_per_bot: int,
    turn_counts: List[int],
) -> Dict[str, Any]:
    """Estimate dataset generation token usage and cost."""
    user_calls = 0
    bot_calls = 0

    for target_turns in turn_counts:
        conversations = bot_count * len(USER_PERSONAS) * n_conversations_per_bot
        user_calls += conversations * ((target_turns + 1) // 2)
        bot_calls += conversations * (target_turns // 2)

    general_estimate = estimate_role_cost(AgentRole.GENERAL, user_calls)
    persona_estimate = estimate_role_cost(AgentRole.PERSONA, bot_calls)

    total_cost = 0.0
    known_cost = True
    for estimate in (general_estimate, persona_estimate):
        if estimate["estimated_cost_usd"] is None:
            known_cost = False
            continue
        total_cost += estimate["estimated_cost_usd"]

    total_conversations = bot_count * len(USER_PERSONAS) * len(turn_counts) * n_conversations_per_bot
    return {
        "bots": bot_count,
        "user_personas": len(USER_PERSONAS),
        "turn_counts": turn_counts,
        "replicates_per_cell": n_conversations_per_bot,
        "total_conversations": total_conversations,
        "role_breakdown": {
            "general": general_estimate,
            "persona": persona_estimate,
        },
        "estimated_cost_usd": round(total_cost, 6) if known_cost else None,
    }


def estimate_baseline_cost(dataset: List[Dict], method_names: Optional[List[str]] = None) -> Dict[str, Any]:
    """Estimate baseline token usage and cost from dataset size and dialogue length."""
    methods = build_methods(method_names)
    role_calls = {
        AgentRole.FEATURE: 0,
        AgentRole.GENERAL: 0,
        AgentRole.PERSONA: 0,
    }

    for sample in dataset:
        turns = sample.get("n_turns", len(sample.get("dialogue", [])))

        if "Direct Prompting" in methods:
            role_calls[AgentRole.FEATURE] += 1
            role_calls[AgentRole.GENERAL] += 1

        if "CoT" in methods:
            role_calls[AgentRole.FEATURE] += 1
            role_calls[AgentRole.GENERAL] += 1

        if "Self-Consistency" in methods:
            n_samples = getattr(methods["Self-Consistency"], "n_samples", 5)
            role_calls[AgentRole.FEATURE] += n_samples
            role_calls[AgentRole.GENERAL] += n_samples

        for name in methods:
            if name not in {
                "AI YOU (Full)",
                "w/o Multi-Agent",
                "w/o Bayesian",
                "w/o Conformal",
                "w/o CoT",
            }:
                continue

            role_calls[AgentRole.FEATURE] += max(1, turns - 3)
            if name == "w/o Multi-Agent":
                role_calls[AgentRole.GENERAL] += 1
            else:
                role_calls[AgentRole.PERSONA] += 1

    estimates = {
        "feature": estimate_role_cost(AgentRole.FEATURE, role_calls[AgentRole.FEATURE]),
        "general": estimate_role_cost(AgentRole.GENERAL, role_calls[AgentRole.GENERAL]),
        "persona": estimate_role_cost(AgentRole.PERSONA, role_calls[AgentRole.PERSONA]),
    }

    total_cost = 0.0
    known_cost = True
    for estimate in estimates.values():
        if estimate["estimated_cost_usd"] is None:
            known_cost = False
            continue
        total_cost += estimate["estimated_cost_usd"]

    return {
        "samples": len(dataset),
        "methods": list(methods.keys()),
        "role_breakdown": estimates,
        "estimated_cost_usd": round(total_cost, 6) if known_cost else None,
    }


def estimate_mpi_cost(n_personas: int = 20, n_turns: int = 10) -> Dict[str, Any]:
    """Estimate cost for the bundled MPI benchmark path."""
    dialogue_entries = n_turns * 2
    role_breakdown = {
        "persona": estimate_role_cost(AgentRole.PERSONA, n_personas),
        "feature": estimate_role_cost(AgentRole.FEATURE, n_personas * max(1, dialogue_entries - 3)),
    }

    total_cost = 0.0
    known_cost = True
    for estimate in role_breakdown.values():
        if estimate["estimated_cost_usd"] is None:
            known_cost = False
            continue
        total_cost += estimate["estimated_cost_usd"]

    return {
        "n_personas": n_personas,
        "n_turns": n_turns,
        "role_breakdown": role_breakdown,
        "estimated_cost_usd": round(total_cost, 6) if known_cost else None,
    }


def load_dataset(dataset_path: Path) -> List[Dict]:
    """Load an evaluation dataset file."""
    with open(dataset_path) as f:
        return json.load(f)


def write_text(path: Path, content: str) -> None:
    """Write text content to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def save_json(path: Path, data: Dict[str, Any]) -> None:
    """Write JSON content to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def run_dataset_generation_step(
    dataset_path: Path,
    n_conversations_per_bot: int,
    turn_counts: List[int],
    force_regenerate: bool,
    max_bots: Optional[int],
) -> StepRecord:
    """Generate the evaluation dataset if needed."""
    started = datetime.now().isoformat()
    t0 = time.time()

    personas = load_bot_personas(max_bots=max_bots)
    estimate = estimate_dataset_generation_cost(
        bot_count=len(personas),
        n_conversations_per_bot=n_conversations_per_bot,
        turn_counts=turn_counts,
    )

    if dataset_path.exists() and not force_regenerate:
        existing_dataset = load_dataset(dataset_path)
        return StepRecord(
            name="generate_data",
            status="skipped",
            started_at=started,
            finished_at=datetime.now().isoformat(),
            duration_seconds=round(time.time() - t0, 3),
            details={
                "reason": "dataset already exists",
                "dataset_path": str(dataset_path),
                "samples": len(existing_dataset),
                "estimate": estimate,
            },
            artifacts={"dataset": str(dataset_path)},
        )

    usage_before = usage_snapshot()
    if max_bots is None:
        dataset = generate_eval_dataset(
            n_conversations_per_bot=n_conversations_per_bot,
            turn_counts=turn_counts,
            output_path=str(dataset_path),
        )
    else:
        dataset = generate_eval_dataset_for_personas(
            personas=personas,
            n_conversations_per_bot=n_conversations_per_bot,
            turn_counts=turn_counts,
            output_path=str(dataset_path),
        )
    usage_after = usage_snapshot()

    return StepRecord(
        name="generate_data",
        status="completed",
        started_at=started,
        finished_at=datetime.now().isoformat(),
        duration_seconds=round(time.time() - t0, 3),
        details={
            "dataset_path": str(dataset_path),
            "samples": len(dataset),
            "bots": len(personas),
            "turn_counts": turn_counts,
            "max_bots": max_bots,
            "estimate": estimate,
            "actual_usage": usage_delta(usage_before, usage_after),
        },
        artifacts={"dataset": str(dataset_path)},
    )


def run_baselines_step(
    dataset_path: Path,
    run_dir: Path,
    method_names: Optional[List[str]] = None,
) -> StepRecord:
    """Run baseline experiments and store outputs in the run directory."""
    started = datetime.now().isoformat()
    t0 = time.time()
    dataset = load_dataset(dataset_path)
    estimate = estimate_baseline_cost(dataset, method_names=method_names)
    baseline_dir = run_dir / "baselines"
    reset_router_usage()

    results = run_evaluation(
        dataset_path=str(dataset_path),
        output_dir=str(baseline_dir),
        method_names=method_names,
    )

    return StepRecord(
        name="run_baselines",
        status="completed",
        started_at=started,
        finished_at=datetime.now().isoformat(),
        duration_seconds=round(time.time() - t0, 3),
        details={
            "estimate": estimate,
            "actual_usage": router.get_usage_report(),
            "n_samples": len(dataset),
            "methods": results.get("methods", []),
        },
        artifacts={
            "metrics": str(baseline_dir / "metrics.json"),
            "personality_table": str(baseline_dir / "table_personality.tex"),
            "relationship_table": str(baseline_dir / "table_relationship.tex"),
            "usage": str(baseline_dir / "llm_usage.json"),
        },
    )


def run_mpi_step(run_dir: Path, allow_missing: bool) -> StepRecord:
    """Run the optional MPI benchmark script if present."""
    started = datetime.now().isoformat()
    t0 = time.time()
    mpi_dir = run_dir / "mpi"
    mpi_dir.mkdir(parents=True, exist_ok=True)
    estimate = estimate_mpi_cost()

    if not MPI_BENCHMARK_PATH.exists():
        status = "skipped" if allow_missing else "failed"
        return StepRecord(
            name="run_mpi",
            status=status,
            started_at=started,
            finished_at=datetime.now().isoformat(),
            duration_seconds=round(time.time() - t0, 3),
            details={
                "benchmark_path": str(MPI_BENCHMARK_PATH),
                "estimate": estimate,
            },
            error=None if allow_missing else f"MPI benchmark not found at {MPI_BENCHMARK_PATH}",
        )

    from experiments.benchmarks.mpi_benchmark import run_mpi_benchmark

    reset_router_usage()
    predictor = AIYouWrapper(name="AI YOU (Full)")
    results = run_mpi_benchmark(predictor)
    results_path = mpi_dir / "mpi_results.json"
    usage_path = mpi_dir / "llm_usage.json"
    save_json(results_path, results)
    save_json(usage_path, router.get_usage_report())

    return StepRecord(
        name="run_mpi",
        status="completed",
        started_at=started,
        finished_at=datetime.now().isoformat(),
        duration_seconds=round(time.time() - t0, 3),
        details={
            "benchmark_path": str(MPI_BENCHMARK_PATH),
            "estimate": estimate,
            "benchmark": results.get("benchmark"),
            "n_personas": results.get("n_personas"),
            "n_turns": results.get("n_turns"),
            "mean_mae": results.get("mean_mae"),
            "mean_rmse": results.get("mean_rmse"),
            "mean_correlation": results.get("mean_correlation"),
            "actual_usage": router.get_usage_report(),
        },
        artifacts={
            "results": str(results_path),
            "usage": str(usage_path),
        },
    )


def generate_summary_report(run_dir: Path, step_records: List[StepRecord]) -> Path:
    """Generate a consolidated text summary for the run."""
    lines = [
        "AI YOU Full Experiment Summary",
        "=" * 60,
        f"Run directory: {run_dir}",
        "",
        "Steps",
        "-" * 60,
    ]

    for record in step_records:
        lines.append(
            f"{record.name}: {record.status} "
            f"({record.duration_seconds:.2f}s)"
        )
        if record.error:
            lines.append(f"  error: {record.error}")

    baseline_metrics_path = run_dir / "baselines" / "metrics.json"
    if baseline_metrics_path.exists():
        with open(baseline_metrics_path) as f:
            baseline_metrics = json.load(f)

        lines.extend(["", "Personality Results", "-" * 60])
        lines.append(generate_results_summary(baseline_metrics.get("personality", {})))
        lines.extend(["", "Relationship Results", "-" * 60])
        lines.append(generate_results_summary(baseline_metrics.get("relationship", {})))

    mpi_results_path = run_dir / "mpi" / "mpi_results.json"
    if mpi_results_path.exists():
        with open(mpi_results_path) as f:
            mpi_results = json.load(f)
        lines.extend(
            [
                "",
                "MPI Results",
                "-" * 60,
                f"mean_mae: {mpi_results.get('mean_mae')}",
                f"mean_rmse: {mpi_results.get('mean_rmse')}",
                f"mean_correlation: {mpi_results.get('mean_correlation')}",
                f"failed_predictions: {mpi_results.get('failed_predictions')}",
            ]
        )

    report_path = run_dir / "summary_report.txt"
    write_text(report_path, "\n".join(lines) + "\n")
    return report_path


def run_full_experiment(
    *,
    do_generate_data: bool = False,
    do_run_baselines: bool = False,
    do_run_mpi: bool = False,
    force_regenerate: bool = False,
    n_conversations_per_bot: int = 3,
    turn_counts: Optional[List[int]] = None,
    dataset_path: Optional[str] = None,
    output_root: Optional[str] = None,
    max_bots: Optional[int] = None,
    method_names: Optional[List[str]] = None,
    allow_missing_mpi: bool = False,
) -> Dict[str, Any]:
    """Run the requested experiment stages and return a manifest."""
    turn_counts = turn_counts or [10, 20, 30]
    dataset_path_obj = Path(dataset_path) if dataset_path else DEFAULT_DATASET_PATH
    output_root_path = Path(output_root) if output_root else RESULTS_ROOT
    run_dir = output_root_path / timestamp_now()
    run_dir.mkdir(parents=True, exist_ok=True)

    providers = ensure_provider_configured()
    manifest: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "run_dir": str(run_dir),
        "providers": providers,
        "dataset_path": str(dataset_path_obj),
        "turn_counts": turn_counts,
        "n_conversations_per_bot": n_conversations_per_bot,
        "max_bots": max_bots,
        "steps": [],
    }

    step_records: List[StepRecord] = []

    def record_failed_step(name: str, started_at: str, exc: Exception) -> StepRecord:
        return StepRecord(
            name=name,
            status="failed",
            started_at=started_at,
            finished_at=datetime.now().isoformat(),
            duration_seconds=0.0,
            error=str(exc),
        )

    if do_generate_data:
        started_at = datetime.now().isoformat()
        try:
            step_records.append(
                run_dataset_generation_step(
                    dataset_path=dataset_path_obj,
                    n_conversations_per_bot=n_conversations_per_bot,
                    turn_counts=turn_counts,
                    force_regenerate=force_regenerate,
                    max_bots=max_bots,
                )
            )
        except Exception as exc:
            step_records.append(record_failed_step("generate_data", started_at, exc))

    if do_run_baselines:
        started_at = datetime.now().isoformat()
        if not dataset_path_obj.exists():
            step_records.append(
                StepRecord(
                    name="run_baselines",
                    status="failed",
                    started_at=started_at,
                    finished_at=datetime.now().isoformat(),
                    duration_seconds=0.0,
                    error=(
                        f"Dataset not found at {dataset_path_obj}. "
                        "Generate it first or pass --dataset-path."
                    ),
                )
            )
        else:
            try:
                step_records.append(
                    run_baselines_step(
                        dataset_path=dataset_path_obj,
                        run_dir=run_dir,
                        method_names=method_names,
                    )
                )
            except Exception as exc:
                step_records.append(record_failed_step("run_baselines", started_at, exc))

    if do_run_mpi:
        started_at = datetime.now().isoformat()
        try:
            step_records.append(run_mpi_step(run_dir=run_dir, allow_missing=allow_missing_mpi))
        except Exception as exc:
            step_records.append(record_failed_step("run_mpi", started_at, exc))

    summary_path = generate_summary_report(run_dir, step_records)
    manifest["steps"] = [asdict(record) for record in step_records]
    manifest["summary_report"] = str(summary_path)
    manifest["status"] = (
        "partial_failure"
        if any(record.status == "failed" for record in step_records)
        else "completed"
    )

    save_json(run_dir / "run_manifest.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Run AI YOU experiment pipeline")
    parser.add_argument("--all", action="store_true", help="Generate data, run baselines, and run MPI")
    parser.add_argument("--generate-data", action="store_true", help="Only generate the evaluation dataset")
    parser.add_argument("--run-baselines", action="store_true", help="Run baseline experiments")
    parser.add_argument("--run-mpi", action="store_true", help="Run MPI benchmark")
    parser.add_argument("--force-regenerate", action="store_true", help="Regenerate dataset even if it exists")
    parser.add_argument("--n-per-bot", type=int, default=3, help="Replicates per bot/persona/turn-count cell")
    parser.add_argument("--turns", type=str, default="10,20,30", help="Comma-separated total dialogue lengths")
    parser.add_argument("--dataset-path", type=str, default=str(DEFAULT_DATASET_PATH), help="Evaluation dataset path")
    parser.add_argument("--output-root", type=str, default=str(RESULTS_ROOT), help="Root output directory")
    parser.add_argument("--max-bots", type=int, default=None, help="Limit personas during dataset generation")
    parser.add_argument("--methods", nargs="*", default=None, help="Subset of baseline methods to run")
    return parser.parse_args()


def main() -> int:
    """CLI entrypoint."""
    args = parse_args()
    turn_counts = parse_turn_counts(args.turns)

    do_generate_data = args.all or args.generate_data
    do_run_baselines = args.all or args.run_baselines
    do_run_mpi = args.all or args.run_mpi

    if not any([do_generate_data, do_run_baselines, do_run_mpi]):
        raise SystemExit("Specify one of --all, --generate-data, --run-baselines, or --run-mpi.")

    manifest = run_full_experiment(
        do_generate_data=do_generate_data,
        do_run_baselines=do_run_baselines,
        do_run_mpi=do_run_mpi,
        force_regenerate=args.force_regenerate,
        n_conversations_per_bot=args.n_per_bot,
        turn_counts=turn_counts,
        dataset_path=args.dataset_path,
        output_root=args.output_root,
        max_bots=args.max_bots,
        method_names=args.methods,
        allow_missing_mpi=args.all,
    )

    print(json.dumps(manifest, indent=2))
    return 1 if manifest["status"] == "partial_failure" else 0


if __name__ == "__main__":
    raise SystemExit(main())
