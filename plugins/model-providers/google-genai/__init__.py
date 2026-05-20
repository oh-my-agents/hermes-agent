"""Google GenAI SDK provider profile.

google-genai: Google AI Studio via the official google-genai Python SDK.
Uses GoogleGenAIClient (agent/google_genai_adapter.py) which subclasses
GeminiNativeClient, so all existing async wrappers and credential-pool
checks work without modification.
"""

from typing import Any

from providers import register_provider
from providers.base import ProviderProfile


class GoogleGenAIProfile(ProviderProfile):
    """google-genai SDK provider — passes thinking_config via extra_body."""

    def build_extra_body(
        self, *, session_id: str | None = None, **context: Any
    ) -> dict[str, Any]:
        from agent.transports.chat_completions import (
            _build_gemini_thinking_config,
            _snake_case_gemini_thinking_config,
        )

        model = context.get("model") or ""
        reasoning_config = context.get("reasoning_config")

        raw_thinking_config = _build_gemini_thinking_config(model, reasoning_config)
        if not raw_thinking_config:
            return {}

        thinking_config = _snake_case_gemini_thinking_config(raw_thinking_config)
        if thinking_config:
            return {"thinking_config": thinking_config}
        return {}


google_genai = GoogleGenAIProfile(
    name="google-genai",
    aliases=("google-generative-ai", "genai"),
    api_mode="chat_completions",
    env_vars=("GOOGLE_API_KEY", "GEMINI_API_KEY", "GOOGLE_GENAI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta",
    auth_type="api_key",
    default_aux_model="gemini-2.5-flash",
)

register_provider(google_genai)
