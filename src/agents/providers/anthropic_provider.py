"""Anthropic (Claude) provider implementation."""

from __future__ import annotations

from typing import AsyncIterator

from loguru import logger

from src.agents.providers.base import ProviderClient, StreamEvent


class AnthropicProvider(ProviderClient):
    """Provider for Anthropic Claude models."""

    def __init__(self):
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import anthropic
                from src.config import settings
                if settings.anthropic_api_key:
                    self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
            except Exception as e:
                logger.warning(f"Anthropic client init failed: {e}")
        return self._client

    def is_available(self) -> bool:
        return self._get_client() is not None

    def chat_sync(self, model_id, system, messages, temperature=0.7, max_tokens=300):
        client = self._get_client()
        if not client:
            raise RuntimeError("Anthropic client not available")
        resp = client.messages.create(
            model=model_id,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=messages,
        )
        text = " ".join(b.text for b in resp.content if hasattr(b, "text"))
        return text, resp.usage.input_tokens, resp.usage.output_tokens

    async def chat_stream(self, model_id, system, messages, temperature=0.7, max_tokens=300):
        client = self._get_client()
        if not client:
            raise RuntimeError("Anthropic client not available")
        with client.messages.stream(
            model=model_id,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=messages,
        ) as stream:
            input_tokens = 0
            output_tokens = 0
            for event in stream:
                if hasattr(event, "type"):
                    if event.type == "content_block_delta" and hasattr(event.delta, "text"):
                        yield StreamEvent(event_type="text_delta", text=event.delta.text)
                    elif event.type == "message_delta" and hasattr(event.usage, "output_tokens"):
                        output_tokens = event.usage.output_tokens
                    elif event.type == "message_start" and hasattr(event.message, "usage"):
                        input_tokens = event.message.usage.input_tokens
            yield StreamEvent(event_type="usage", input_tokens=input_tokens, output_tokens=output_tokens)
            yield StreamEvent(event_type="done")
