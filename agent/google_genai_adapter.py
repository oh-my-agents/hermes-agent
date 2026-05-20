"""google-genai SDK adapter for Hermes.

Wraps the official ``google-genai`` Python SDK (google.genai) behind the same
OpenAI-compatible interface used by GeminiNativeClient, so existing
isinstance checks and async wrappers in auxiliary_client / agent_runtime_helpers
work without modification.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from types import SimpleNamespace
from typing import Any, Dict, Iterator, List, Optional

from agent.gemini_native_adapter import (
    GeminiNativeClient,
    _GeminiStreamChunk,
    _make_stream_chunk,
    build_gemini_request,
)

logger = logging.getLogger(__name__)

_FINISH_REASON_MAP = {
    "STOP": "stop",
    "MAX_TOKENS": "length",
    "SAFETY": "content_filter",
    "RECITATION": "content_filter",
    "OTHER": "stop",
    "FINISH_REASON_UNSPECIFIED": "stop",
}


def _get_sdk():
    """Lazily import google-genai, installing via lazy_deps if needed."""
    try:
        from tools.lazy_deps import ensure
        ensure("provider.google_genai")
    except Exception:
        pass
    import google.genai as genai  # type: ignore[import-untyped]
    return genai


def _sdk_parts_to_openai(parts: Any, model: str) -> tuple[
    list[str], list[str], list[SimpleNamespace]
]:
    """Convert SDK Part list → (text_pieces, reasoning_pieces, tool_calls)."""
    text_pieces: list[str] = []
    reasoning_pieces: list[str] = []
    tool_calls: list[SimpleNamespace] = []

    for index, part in enumerate(parts or []):
        # Thought part
        if getattr(part, "thought", False) and part.text:
            reasoning_pieces.append(part.text)
            continue
        # Text part
        if part.text:
            text_pieces.append(part.text)
            continue
        # Function call part
        fc = getattr(part, "function_call", None)
        if fc is not None and getattr(fc, "name", None):
            try:
                args_str = json.dumps(dict(fc.args), ensure_ascii=False)
            except (TypeError, ValueError):
                args_str = "{}"
            tool_call = SimpleNamespace(
                id=f"call_{uuid.uuid4().hex[:12]}",
                type="function",
                index=index,
                function=SimpleNamespace(name=str(fc.name), arguments=args_str),
            )
            # Preserve thought signature if present
            ts = getattr(fc, "thought_signature", None) or getattr(fc, "thoughtSignature", None)
            if isinstance(ts, str) and ts:
                tool_call.extra_content = {"google": {"thought_signature": ts}}
            tool_calls.append(tool_call)

    return text_pieces, reasoning_pieces, tool_calls


def _translate_sdk_response(response: Any, model: str) -> SimpleNamespace:
    """Convert a google-genai GenerateContentResponse → OpenAI-compatible SimpleNamespace."""
    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        message = SimpleNamespace(
            role="assistant", content="", tool_calls=None,
            reasoning=None, reasoning_content=None, reasoning_details=None,
        )
        choice = SimpleNamespace(index=0, message=message, finish_reason="stop")
        usage = SimpleNamespace(
            prompt_tokens=0, completion_tokens=0, total_tokens=0,
            prompt_tokens_details=SimpleNamespace(cached_tokens=0),
        )
        return SimpleNamespace(
            id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
            object="chat.completion",
            created=int(time.time()),
            model=model,
            choices=[choice],
            usage=usage,
        )

    cand = candidates[0]
    content_obj = getattr(cand, "content", None)
    parts = getattr(content_obj, "parts", None) or []

    text_pieces, reasoning_pieces, tool_calls = _sdk_parts_to_openai(parts, model)

    finish_reason_raw = str(getattr(getattr(cand, "finish_reason", None), "name", "STOP") or "STOP")
    finish_reason = "tool_calls" if tool_calls else _FINISH_REASON_MAP.get(finish_reason_raw, "stop")

    usage_meta = getattr(response, "usage_metadata", None)
    usage = SimpleNamespace(
        prompt_tokens=int(getattr(usage_meta, "prompt_token_count", 0) or 0),
        completion_tokens=int(getattr(usage_meta, "candidates_token_count", 0) or 0),
        total_tokens=int(getattr(usage_meta, "total_token_count", 0) or 0),
        prompt_tokens_details=SimpleNamespace(
            cached_tokens=int(getattr(usage_meta, "cached_content_token_count", 0) or 0),
        ),
    )

    reasoning = "".join(reasoning_pieces) or None
    message = SimpleNamespace(
        role="assistant",
        content="".join(text_pieces) if text_pieces else None,
        tool_calls=tool_calls or None,
        reasoning=reasoning,
        reasoning_content=reasoning,
        reasoning_details=None,
    )
    choice = SimpleNamespace(index=0, message=message, finish_reason=finish_reason)
    return SimpleNamespace(
        id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
        object="chat.completion",
        created=int(time.time()),
        model=model,
        choices=[choice],
        usage=usage,
    )


def _iter_sdk_stream(sdk_stream: Any, model: str) -> Iterator[_GeminiStreamChunk]:
    """Yield _GeminiStreamChunk objects from a google-genai streaming response."""
    tool_call_index: Dict[str, int] = {}
    next_tc_index = 0

    for chunk in sdk_stream:
        candidates = getattr(chunk, "candidates", None) or []
        if not candidates:
            continue
        cand = candidates[0]
        content_obj = getattr(cand, "content", None)
        parts = getattr(content_obj, "parts", None) or []

        finish_reason_raw = str(
            getattr(getattr(cand, "finish_reason", None), "name", None) or ""
        )
        finish_reason = _FINISH_REASON_MAP.get(finish_reason_raw) if finish_reason_raw else None

        for part in parts:
            if getattr(part, "thought", False) and part.text:
                yield _make_stream_chunk(model=model, reasoning=part.text)
                continue
            if part.text:
                yield _make_stream_chunk(model=model, content=part.text)
                continue
            fc = getattr(part, "function_call", None)
            if fc is not None and getattr(fc, "name", None):
                name = str(fc.name)
                if name not in tool_call_index:
                    tool_call_index[name] = next_tc_index
                    next_tc_index += 1
                try:
                    args_str = json.dumps(dict(fc.args), ensure_ascii=False)
                except (TypeError, ValueError):
                    args_str = "{}"
                tc_delta: Dict[str, Any] = {
                    "index": tool_call_index[name],
                    "id": f"call_{uuid.uuid4().hex[:12]}",
                    "name": name,
                    "arguments": args_str,
                }
                ts = getattr(fc, "thought_signature", None) or getattr(fc, "thoughtSignature", None)
                if isinstance(ts, str) and ts:
                    tc_delta["extra_content"] = {"google": {"thought_signature": ts}}
                yield _make_stream_chunk(model=model, tool_call_delta=tc_delta)

        if finish_reason:
            yield _make_stream_chunk(model=model, finish_reason=finish_reason)


_KNOWN_API_VERSIONS = ("v1beta", "v1alpha", "v1")


def _split_api_version(base_url: str) -> tuple[str, str]:
    """Split a base_url that may already contain an API version suffix.

    google-genai SDK constructs: {base_url}/{api_version}/models/...
    If base_url already ends with e.g. '/v1beta', the SDK would produce
    '.../v1beta/v1beta/models/...' — a doubled path that 404s.

    Returns (stripped_base_url, api_version) so the SDK receives them
    separately and builds the correct single-segment path.
    """
    url = base_url.rstrip("/")
    for ver in _KNOWN_API_VERSIONS:
        if url.endswith("/" + ver):
            return url[: -len("/" + ver)], ver
    return url, "v1beta"


class GoogleGenAIClient(GeminiNativeClient):
    """Client backed by the official google-genai SDK for direct Google endpoints,
    with automatic fallback to OpenAI-compatible format for third-party proxies
    (sub2api, etc.) that don't speak Gemini native protocol.

    Subclasses GeminiNativeClient so existing isinstance checks in
    auxiliary_client.py and AsyncGeminiNativeClient wrapping continue
    to work without modification.
    """

    def __init__(
        self,
        *,
        api_key: str,
        base_url: Optional[str] = None,
        default_headers: Optional[Dict[str, str]] = None,
        timeout: Any = None,
        **_: Any,
    ) -> None:
        # Initialise parent to pass isinstance checks; parent's httpx.Client
        # is reused for the OpenAI-compat proxy fallback path.
        super().__init__(api_key=api_key, base_url=base_url, default_headers=default_headers, timeout=timeout)
        genai = _get_sdk()
        # google-genai SDK constructs: {base_url}/{api_version}/models/{model}:{method}
        # Strip a trailing api_version segment from base_url so it isn't doubled.
        sdk_base, api_version = _split_api_version(self.base_url)
        self._sdk_client = genai.Client(
            api_key=api_key,
            http_options={"base_url": sdk_base, "api_version": api_version},
        )
        self._sdk_timeout = timeout

    def _create_chat_completion(
        self,
        *,
        model: str = "gemini-2.5-flash",
        messages: Optional[List[Dict[str, Any]]] = None,
        stream: bool = False,
        tools: Any = None,
        tool_choice: Any = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        stop: Any = None,
        extra_body: Optional[Dict[str, Any]] = None,
        timeout: Any = None,
        **_: Any,
    ) -> Any:
        genai = _get_sdk()

        thinking_config = None
        if isinstance(extra_body, dict):
            thinking_config = extra_body.get("thinking_config") or extra_body.get("thinkingConfig")

        request = build_gemini_request(
            messages=messages or [],
            tools=tools,
            tool_choice=tool_choice,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stop=stop,
            thinking_config=thinking_config,
        )

        contents = request["contents"]
        gen_cfg = request.get("generationConfig") or {}

        config_kwargs: Dict[str, Any] = {}
        if request.get("systemInstruction"):
            parts = request["systemInstruction"].get("parts") or []
            text = " ".join(p.get("text", "") for p in parts if isinstance(p, dict)).strip()
            if text:
                config_kwargs["system_instruction"] = text
        if request.get("tools"):
            config_kwargs["tools"] = request["tools"]
        if request.get("toolConfig"):
            config_kwargs["tool_config"] = request["toolConfig"]
        if "maxOutputTokens" in gen_cfg:
            config_kwargs["max_output_tokens"] = gen_cfg["maxOutputTokens"]
        if "temperature" in gen_cfg:
            config_kwargs["temperature"] = gen_cfg["temperature"]
        if "topP" in gen_cfg:
            config_kwargs["top_p"] = gen_cfg["topP"]
        if "stopSequences" in gen_cfg:
            config_kwargs["stop_sequences"] = gen_cfg["stopSequences"]
        if "thinkingConfig" in gen_cfg:
            config_kwargs["thinking_config"] = gen_cfg["thinkingConfig"]

        sdk_config = genai.types.GenerateContentConfig(**config_kwargs) if config_kwargs else None

        if stream:
            return self._stream_sdk(model=model, contents=contents, config=sdk_config)

        response = self._sdk_client.models.generate_content(
            model=model,
            contents=contents,
            config=sdk_config,
        )
        return _translate_sdk_response(response, model=model)

    def _stream_sdk(
        self, *, model: str, contents: Any, config: Any
    ) -> Iterator[_GeminiStreamChunk]:
        sdk_stream = self._sdk_client.models.generate_content_stream(
            model=model,
            contents=contents,
            config=config,
        )
        return _iter_sdk_stream(sdk_stream, model=model)
