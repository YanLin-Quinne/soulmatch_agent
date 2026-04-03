"""Benchmark adapters for AI YOU experiments."""

try:
    from .mpi_benchmark import MPIBenchmark, run_mpi_benchmark
except ImportError:  # pragma: no cover - optional until MPI benchmark lands
    MPIBenchmark = None

    def run_mpi_benchmark(*args, **kwargs):
        raise ImportError("experiments.benchmarks.mpi_benchmark is not available in this checkout")

from .locomo_adapter import LoCoMoAdapter, run_locomo_evaluation
from .persistent_personas_adapter import (
    PersistentPersonasAdapter,
    run_persistent_personas_evaluation,
)

__all__ = [
    "MPIBenchmark",
    "run_mpi_benchmark",
    "LoCoMoAdapter",
    "run_locomo_evaluation",
    "PersistentPersonasAdapter",
    "run_persistent_personas_evaluation",
]
