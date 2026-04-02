"""Abstract base for LLM provider clients."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncIterator


@dataclass
class StreamEvent:
    """A single event from a streaming LLM response."""
    event_type: str  # "text_delta" | "usage" | "done"
    text: str = ""
    input_tokens: int = 0
    output_tokens: int = 0


class ProviderClient(ABC):
    """Unified interface for LLM provider backends."""

    @abstractmethod
    def chat_sync(
        self,
        model_id: str,
        system: str,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 300,
    ) -> tuple[str, int, int]:
        """Synchronous chat. Returns (text, input_tokens, output_tokens)."""
        ...

    @abstractmethod
    async def chat_stream(
        self,
        model_id: str,
        system: str,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 300,
    ) -> AsyncIterator[StreamEvent]:
        """Async streaming chat. Yields StreamEvent objects."""
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this provider is configured and reachable."""
        ...
