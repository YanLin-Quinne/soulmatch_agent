"""
LLM Router — unified multi-provider client with fallback, model routing, and cost tracking.

Supports: Claude (Anthropic), GPT (OpenAI), Gemini (Google), DeepSeek, Qwen (Alibaba).
Each agent type maps to a preferred model; if that provider fails, the router tries
the next one in the fallback chain.
"""

import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from loguru import logger

from src.config import settings


# ---------------------------------------------------------------------------
# Provider & role enums
# ---------------------------------------------------------------------------

class Provider(str, Enum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GEMINI = "gemini"
    DEEPSEEK = "deepseek"
    QWEN = "qwen"
    LOCAL = "local"   # vLLM / Ollama / llama.cpp (OpenAI-compatible)
    HF = "hf"         # HuggingFace transformers (in-process)


class AgentRole(str, Enum):
    """Logical roles — each maps to a preferred model via MODEL_ROUTING."""
    PERSONA = "persona"           # bot role-play (quality matters)
    EMOTION = "emotion"           # emotion classification (speed matters)
    FEATURE = "feature"           # feature extraction (reasoning matters)
    SCAM = "scam"                 # scam semantic analysis
    MEMORY = "memory"             # memory decision
    QUESTION = "question"         # question strategy
    GENERAL = "general"           # fallback / misc


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

@dataclass
class ModelSpec:
    provider: Provider
    model_id: str
    input_cost_per_1k: float = 0.0   # USD per 1K input tokens
    output_cost_per_1k: float = 0.0  # USD per 1K output tokens


# Available models — ordered by quality within each provider (2026-02-22 Latest)
MODELS: dict[str, ModelSpec] = {
    # Anthropic Claude Opus 4.6 (2026)
    "claude-opus-4": ModelSpec(Provider.ANTHROPIC, "claude-opus-4-6", 0.015, 0.075),
    "claude-sonnet": ModelSpec(Provider.ANTHROPIC, "claude-sonnet-4-20250514", 0.003, 0.015),
    "claude-haiku": ModelSpec(Provider.ANTHROPIC, "claude-haiku-4-20250414", 0.00025, 0.00125),
    # OpenAI GPT-5.2 (2026)
    "gpt-5": ModelSpec(Provider.OPENAI, "gpt-5.2", 0.010, 0.030),
    "gpt-4o": ModelSpec(Provider.OPENAI, "gpt-4o", 0.005, 0.015),
    "gpt-4o-mini": ModelSpec(Provider.OPENAI, "gpt-4o-mini", 0.00015, 0.0006),
    # Google Gemini 3.1 Pro Preview / 2.5 Flash (2026)
    "gemini-3-pro": ModelSpec(Provider.GEMINI, "gemini-3.1-pro-preview", 0.002, 0.008),
    "gemini-flash": ModelSpec(Provider.GEMINI, "gemini-2.5-flash", 0.0001, 0.0004),
    "gemini-pro": ModelSpec(Provider.GEMINI, "gemini-2.5-pro-preview-06-05", 0.00125, 0.01),
    # DeepSeek V3.2 Reasoner (2026)
    "deepseek-reasoner": ModelSpec(Provider.DEEPSEEK, "deepseek-reasoner", 0.00055, 0.0022),
    "deepseek-chat": ModelSpec(Provider.DEEPSEEK, "deepseek-chat", 0.00014, 0.00028),
    # Qwen 3 Max (2026)
    "qwen-max": ModelSpec(Provider.QWEN, "qwen3-max", 0.0008, 0.0024),
    "qwen-plus": ModelSpec(Provider.QWEN, "qwen-plus", 0.0004, 0.0012),
    "qwen-turbo": ModelSpec(Provider.QWEN, "qwen-turbo", 0.0001, 0.0002),
    # Local (costs are zero — your hardware)
    "local": ModelSpec(Provider.LOCAL, "local", 0.0, 0.0),
    # HuggingFace in-process (costs are zero — your hardware)
    "hf": ModelSpec(Provider.HF, "hf", 0.0, 0.0),
}

# Default routing: role → ordered list of model keys to try (最适配策略)
MODEL_ROUTING: dict[AgentRole, list[str]] = {
    AgentRole.PERSONA:  ["gpt-5", "deepseek-reasoner", "gemini-flash", "qwen-max"],  # 角色扮演需要高质量
    AgentRole.EMOTION:  ["gemini-flash", "gpt-4o-mini", "deepseek-chat"],  # 情绪分类速度优先
    AgentRole.FEATURE:  ["gpt-5", "deepseek-reasoner", "gemini-flash", "qwen-max"],  # 特征推理需要推理能力
    AgentRole.SCAM:     ["gemini-flash", "gpt-4o-mini", "deepseek-chat"],  # 诈骗检测速度优先
    AgentRole.MEMORY:   ["gemini-flash", "gpt-4o-mini", "deepseek-chat"],  # 记忆决策速度优先
    AgentRole.QUESTION: ["gemini-flash", "deepseek-chat", "gpt-4o-mini"],  # 问题策略速度优先
    AgentRole.GENERAL:  ["gemini-flash", "gpt-4o-mini", "deepseek-chat"],  # 通用任务速度优先
}



# ---------------------------------------------------------------------------
# Usage tracking
# ---------------------------------------------------------------------------

@dataclass
class UsageRecord:
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0
    call_count: int = 0
    errors: int = 0
    per_provider: dict[str, dict[str, float]] = field(default_factory=dict)

    def record(self, provider: str, model_key: str, input_tok: int, output_tok: int, cost: float):
        self.total_input_tokens += input_tok
        self.total_output_tokens += output_tok
        self.total_cost_usd += cost
        self.call_count += 1
        if provider not in self.per_provider:
            self.per_provider[provider] = {"input_tokens": 0, "output_tokens": 0, "cost": 0.0, "calls": 0}
        self.per_provider[provider]["input_tokens"] += input_tok
        self.per_provider[provider]["output_tokens"] += output_tok
        self.per_provider[provider]["cost"] += cost
        self.per_provider[provider]["calls"] += 1


# ---------------------------------------------------------------------------
# Provider clients (lazy-initialized singletons)
# ---------------------------------------------------------------------------

class _Clients:
    """Lazy-initialized provider clients."""

    _anthropic = None
    _openai = None
    _gemini = None
    _deepseek = None
    _qwen = None

    @classmethod
    def anthropic(cls):
        if cls._anthropic is None:
            import anthropic
            cls._anthropic = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        return cls._anthropic

    @classmethod
    def openai(cls):
        if cls._openai is None:
            import openai
            cls._openai = openai.OpenAI(api_key=settings.openai_api_key)
        return cls._openai

    @classmethod
    def gemini(cls):
        if cls._gemini is None:
            from google import genai
            cls._gemini = genai.Client(api_key=settings.gemini_api_key)
        return cls._gemini

    @classmethod
    def deepseek(cls):
        if cls._deepseek is None:
            import openai as _openai
            cls._deepseek = _openai.OpenAI(
                api_key=settings.deepseek_api_key,
                base_url="https://api.deepseek.com"
            )
        return cls._deepseek

    @classmethod
    def qwen(cls):
        if cls._qwen is None:
            import openai as _openai
            cls._qwen = _openai.OpenAI(
                api_key=settings.qwen_api_key,
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
            )
        return cls._qwen

    _local = None

    @classmethod
    def local(cls):
        if cls._local is None:
            import openai as _openai
            cls._local = _openai.OpenAI(
                api_key=settings.local_llm_api_key,
                base_url=settings.local_llm_base_url,
            )
        return cls._local

    _hf_pipeline = None

    @classmethod
    def hf_pipeline(cls):
        if cls._hf_pipeline is None:
            try:
                import torch
                from transformers import pipeline as hf_pipeline_fn
                device = settings.hf_device or ("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
                logger.info(f"Loading HF model {settings.hf_model_name} on {device}...")
                cls._hf_pipeline = hf_pipeline_fn(
                    "text-generation",
                    model=settings.hf_model_name,
                    device=device,
                    torch_dtype=torch.float16 if device != "cpu" else torch.float32,
                )
                logger.info(f"HF model loaded: {settings.hf_model_name}")
            except Exception as e:
                raise RuntimeError(f"Failed to load HF model: {e}") from e
        return cls._hf_pipeline


def _provider_available(provider: Provider) -> bool:
    """Check if provider has a valid API key / config configured."""
    if provider == Provider.LOCAL:
        return bool(settings.local_llm_base_url and settings.local_llm_model)
    if provider == Provider.HF:
        return bool(settings.hf_model_name)
    key_map = {
        Provider.ANTHROPIC: settings.anthropic_api_key,
        Provider.OPENAI: settings.openai_api_key,
        Provider.GEMINI: settings.gemini_api_key,
        Provider.DEEPSEEK: settings.deepseek_api_key,
        Provider.QWEN: settings.qwen_api_key,
    }
    return bool(key_map.get(provider, ""))


# ---------------------------------------------------------------------------
# Core call helpers (one per provider)
# ---------------------------------------------------------------------------

def _call_anthropic(
    model_id: str, system: str, messages: list[dict], temperature: float, max_tokens: int
) -> tuple[str, int, int]:
    """Returns (text, input_tokens, output_tokens)."""
    resp = _Clients.anthropic().messages.create(
        model=model_id,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system,
        messages=messages,
    )
    if not resp.content:
        raise ValueError(f"Anthropic returned empty content (stop_reason={resp.stop_reason})")
    text = resp.content[0].text if hasattr(resp.content[0], 'text') else str(resp.content[0])
    return text, resp.usage.input_tokens, resp.usage.output_tokens


def _call_openai_compat(
    client, model_id: str, system: str, messages: list[dict], temperature: float, max_tokens: int
) -> tuple[str, int, int]:
    """OpenAI-compatible call (works for OpenAI, DeepSeek, Qwen)."""
    api_messages = [{"role": "system", "content": system}] + messages

    # GPT-5+ uses max_completion_tokens instead of max_tokens
    kwargs = {
        "model": model_id,
        "messages": api_messages,
        "temperature": temperature,
    }
    if "gpt-5" in model_id or "gpt-6" in model_id:
        kwargs["max_completion_tokens"] = max_tokens
    else:
        kwargs["max_tokens"] = max_tokens

    resp = client.chat.completions.create(**kwargs)
    if not resp.choices:
        raise ValueError(f"OpenAI-compat returned no choices (model={model_id})")
    content = resp.choices[0].message.content or ""
    usage = resp.usage
    return content, usage.prompt_tokens, usage.completion_tokens


def _call_gemini(
    model_id: str, system: str, messages: list[dict], temperature: float, max_tokens: int
) -> tuple[str, int, int]:
    """Google Gemini call via google-genai SDK."""
    from google.genai import types

    # Convert messages to Gemini content format
    contents = []
    for msg in messages:
        role = "user" if msg["role"] == "user" else "model"
        contents.append(types.Content(role=role, parts=[types.Part(text=msg["content"])]))

    resp = _Clients.gemini().models.generate_content(
        model=model_id,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=system,
            temperature=temperature,
            max_output_tokens=max_tokens,
        ),
    )
    input_tok = resp.usage_metadata.prompt_token_count or 0
    output_tok = resp.usage_metadata.candidates_token_count or 0
    text = resp.text or ""
    if not text:
        raise ValueError("Gemini returned empty text")
    return text, input_tok, output_tok


def _call_hf(
    system: str, messages: list[dict], temperature: float, max_tokens: int
) -> tuple[str, int, int]:
    """HuggingFace transformers in-process inference."""
    pipe = _Clients.hf_pipeline()

    # Build a ChatML-style prompt
    chat = [{"role": "system", "content": system}]
    for msg in messages:
        chat.append({"role": msg["role"], "content": msg["content"]})

    result = pipe(
        chat,
        max_new_tokens=max_tokens,
        temperature=max(temperature, 0.01),
        do_sample=temperature > 0,
        return_full_text=False,
    )
    text = result[0]["generated_text"].strip() if result else ""
    if not text:
        raise ValueError("HF model returned empty text")
    # Rough token estimate (no exact count from pipeline)
    input_tok = sum(len(m["content"].split()) for m in chat)
    output_tok = len(text.split())
    return text, input_tok, output_tok


# ---------------------------------------------------------------------------
# LLMRouter — the public interface
# ---------------------------------------------------------------------------

class LLMRouter:
    """
    Unified LLM interface.  Usage:

        router = LLMRouter()
        text = router.chat(
            role=AgentRole.EMOTION,
            system="You are an emotion detector...",
            messages=[{"role": "user", "content": "I feel great!"}],
        )
    """

    def __init__(self):
        self.usage = UsageRecord()
        self._available_cache: dict[Provider, bool] = {}

    # ------------------------------------------------------------------
    def _is_available(self, provider: Provider) -> bool:
        if provider not in self._available_cache:
            self._available_cache[provider] = _provider_available(provider)
        return self._available_cache[provider]

    # ------------------------------------------------------------------
    def chat(
        self,
        role: AgentRole,
        system: str,
        messages: list[dict],
        *,
        temperature: float = 0.7,
        max_tokens: int = 300,
        preferred_model: Optional[str] = None,
        json_mode: bool = False,
    ) -> str:
        """
        Send a chat request with automatic fallback.

        Args:
            role: Agent role (determines model routing).
            system: System prompt.
            messages: List of {"role": "user"|"assistant", "content": "..."}.
            temperature: Sampling temperature.
            max_tokens: Max output tokens.
            preferred_model: Override model key (e.g. "claude-sonnet").
            json_mode: If True, append JSON instruction to system prompt.

        Returns:
            Generated text.

        Raises:
            RuntimeError: If all providers in the fallback chain fail.
        """
        if json_mode:
            system = system + "\n\nIMPORTANT: Respond with valid JSON only, no markdown fences."

        # Build ordered model list
        if preferred_model and preferred_model in MODELS:
            chain = [preferred_model] + [m for m in MODEL_ROUTING.get(role, []) if m != preferred_model]
        else:
            chain = MODEL_ROUTING.get(role, MODEL_ROUTING[AgentRole.GENERAL])

        errors: list[str] = []
        for model_key in chain:
            spec = MODELS[model_key]
            if not self._is_available(spec.provider):
                continue

            try:
                t0 = time.time()
                text, in_tok, out_tok = self._dispatch(spec, system, messages, temperature, max_tokens)
                elapsed = time.time() - t0

                cost = (in_tok / 1000) * spec.input_cost_per_1k + (out_tok / 1000) * spec.output_cost_per_1k
                self.usage.record(spec.provider.value, model_key, in_tok, out_tok, cost)

                logger.debug(
                    f"[LLMRouter] {model_key} ok | {in_tok}+{out_tok} tok | "
                    f"${cost:.5f} | {elapsed:.2f}s"
                )
                return text

            except Exception as e:
                self.usage.errors += 1
                errors.append(f"{model_key}: {e}")
                logger.warning(f"[LLMRouter] {model_key} failed: {e}")
                continue

        raise RuntimeError(f"All providers failed for role={role.value}: {errors}")

    # ------------------------------------------------------------------
    def _dispatch(
        self, spec: ModelSpec, system: str, messages: list[dict],
        temperature: float, max_tokens: int,
    ) -> tuple[str, int, int]:
        """Route to the correct provider call."""
        if spec.provider == Provider.ANTHROPIC:
            return _call_anthropic(spec.model_id, system, messages, temperature, max_tokens)
        elif spec.provider == Provider.OPENAI:
            return _call_openai_compat(_Clients.openai(), spec.model_id, system, messages, temperature, max_tokens)
        elif spec.provider == Provider.GEMINI:
            return _call_gemini(spec.model_id, system, messages, temperature, max_tokens)
        elif spec.provider == Provider.DEEPSEEK:
            return _call_openai_compat(_Clients.deepseek(), spec.model_id, system, messages, temperature, max_tokens)
        elif spec.provider == Provider.QWEN:
            return _call_openai_compat(_Clients.qwen(), spec.model_id, system, messages, temperature, max_tokens)
        elif spec.provider == Provider.LOCAL:
            return _call_openai_compat(_Clients.local(), settings.local_llm_model, system, messages, temperature, max_tokens)
        elif spec.provider == Provider.HF:
            return _call_hf(system, messages, temperature, max_tokens)
        else:
            raise ValueError(f"Unknown provider: {spec.provider}")

    # ------------------------------------------------------------------
    def get_usage_report(self) -> dict[str, Any]:
        """Return usage statistics."""
        return {
            "total_calls": self.usage.call_count,
            "total_errors": self.usage.errors,
            "total_input_tokens": self.usage.total_input_tokens,
            "total_output_tokens": self.usage.total_output_tokens,
            "total_cost_usd": round(self.usage.total_cost_usd, 6),
            "per_provider": self.usage.per_provider,
        }


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
router = LLMRouter()
