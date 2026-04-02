"""OpenAI-compatible provider (OpenAI, DeepSeek, Qwen, Local)."""

from __future__ import annotations

from typing import AsyncIterator, Optional

from loguru import logger

from src.agents.providers.base import ProviderClient, StreamEvent


class OpenAICompatProvider(ProviderClient):
    """Provider for OpenAI-compatible APIs (OpenAI, DeepSeek, Qwen, Local LLM)."""

    def __init__(self, api_key: str, base_url: Optional[str] = None, provider_name: str = "openai"):
        self._api_key = api_key
        self._base_url = base_url
        self._provider_name = provider_name
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import openai
                kwargs = {"api_key": self._api_key}
                if self._base_url:
                    kwargs["base_url"] = self._base_url
                self._client = openai.OpenAI(**kwargs)
            except Exception as e:
                logger.warning(f"{self._provider_name} client init failed: {e}")
        return self._client

    def is_available(self) -> bool:
        return bool(self._api_key) and self._get_client() is not None

    def chat_sync(self, model_id, system, messages, temperature=0.7, max_tokens=300):
        client = self._get_client()
        if not client:
            raise RuntimeError(f"{self._provider_name} client not available")
        api_messages = [{"role": "system", "content": system}] + messages
        kwargs = {
            "model": model_id,
            "messages": api_messages,
            "temperature": temperature,
        }
        # GPT-5+ uses max_completion_tokens
        if "gpt-5" in model_id or "o1" in model_id or "o3" in model_id:
            kwargs["max_completion_tokens"] = max_tokens
        else:
            kwargs["max_tokens"] = max_tokens
        resp = client.chat.completions.create(**kwargs)
        text = resp.choices[0].message.content or ""
        usage = resp.usage
        return text, usage.prompt_tokens if usage else 0, usage.completion_tokens if usage else 0

    async def chat_stream(self, model_id, system, messages, temperature=0.7, max_tokens=300):
        client = self._get_client()
        if not client:
            raise RuntimeError(f"{self._provider_name} client not available")
        api_messages = [{"role": "system", "content": system}] + messages
        kwargs = {
            "model": model_id,
            "messages": api_messages,
            "temperature": temperature,
            "stream": True,
        }
        if "gpt-5" in model_id or "o1" in model_id or "o3" in model_id:
            kwargs["max_completion_tokens"] = max_tokens
        else:
            kwargs["max_tokens"] = max_tokens
        stream = client.chat.completions.create(**kwargs)
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                yield StreamEvent(event_type="text_delta", text=chunk.choices[0].delta.content)
        yield StreamEvent(event_type="done")
