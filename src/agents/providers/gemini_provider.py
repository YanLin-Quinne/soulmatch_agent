"""Google Gemini provider implementation."""

from __future__ import annotations

from typing import AsyncIterator

from loguru import logger

from src.agents.providers.base import ProviderClient, StreamEvent


class GeminiProvider(ProviderClient):
    """Provider for Google Gemini models."""

    def __init__(self):
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import google.genai as genai
                from src.config import settings
                if settings.gemini_api_key:
                    self._client = genai.Client(api_key=settings.gemini_api_key)
            except Exception as e:
                logger.warning(f"Gemini client init failed: {e}")
        return self._client

    def is_available(self) -> bool:
        return self._get_client() is not None

    def chat_sync(self, model_id, system, messages, temperature=0.7, max_tokens=300):
        client = self._get_client()
        if not client:
            raise RuntimeError("Gemini client not available")
        import google.genai as genai
        contents = []
        for msg in messages:
            role = "user" if msg["role"] == "user" else "model"
            contents.append(genai.types.Content(role=role, parts=[genai.types.Part(text=msg["content"])]))
        config = genai.types.GenerateContentConfig(
            system_instruction=system,
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        resp = client.models.generate_content(model=model_id, contents=contents, config=config)
        text = resp.text or ""
        input_tokens = getattr(resp.usage_metadata, "prompt_token_count", 0) if resp.usage_metadata else 0
        output_tokens = getattr(resp.usage_metadata, "candidates_token_count", 0) if resp.usage_metadata else 0
        return text, input_tokens, output_tokens

    async def chat_stream(self, model_id, system, messages, temperature=0.7, max_tokens=300):
        client = self._get_client()
        if not client:
            raise RuntimeError("Gemini client not available")
        import google.genai as genai
        contents = []
        for msg in messages:
            role = "user" if msg["role"] == "user" else "model"
            contents.append(genai.types.Content(role=role, parts=[genai.types.Part(text=msg["content"])]))
        config = genai.types.GenerateContentConfig(
            system_instruction=system,
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        for chunk in client.models.generate_content_stream(model=model_id, contents=contents, config=config):
            if chunk.text:
                yield StreamEvent(event_type="text_delta", text=chunk.text)
        yield StreamEvent(event_type="done")
