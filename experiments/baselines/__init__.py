"""Baseline methods for SoulMatch evaluation."""

from .direct_prompting import DirectPromptingBaseline
from .cot_prompting import CoTBaseline
from .self_consistency import SelfConsistencyBaseline

__all__ = [
    "DirectPromptingBaseline",
    "CoTBaseline",
    "SelfConsistencyBaseline",
]
