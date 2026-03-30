"""Gemini provider adapter."""

from __future__ import annotations

from typing import Any

import httpx

from ..models import ChatRequest, ChatResponse, Message
from ..settings import Settings
from .base import BaseProvider, ProviderError, ProviderNotConfiguredError

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"
DEFAULT_GEMINI_MODEL = "gemini-2.5-pro-preview-03-25"
JSON_MIME_TYPE = "application/json"
GEMINI_25_EFFORT_TO_BUDGET = {
    "minimal": 1024,
    "low": 1024,
    "medium": 8192,
    "high": 24576,
    "none": 0,
}
GEMINI_3_FLASH_EFFORT_TO_LEVEL = {
    "minimal": "low",
    "low": "low",
    "medium": "medium",
    "high": "high",
}
GEMINI_3_PRO_EFFORT_TO_LEVEL = {
    "minimal": "low",
    "low": "low",
    "high": "high",
}


class GeminiProvider(BaseProvider):
    name = "gemini"

    def __init__(self, client: httpx.AsyncClient, settings: Settings) -> None:
        super().__init__()
        self._client = client
        self._settings = settings

    async def chat(self, request: ChatRequest, trace_id: str) -> ChatResponse:
        api_key = self._settings.gemini_api_key
        if not api_key:
            raise ProviderNotConfiguredError("GEMINI_KEY is not configured")

        model_name = _normalize_model(request.model or DEFAULT_GEMINI_MODEL)
        url = f"{GEMINI_BASE_URL}/{model_name}:generateContent?key={api_key}"

        contents = [_message_to_gemini(message) for message in request.messages]
        payload: dict[str, Any] = {"contents": contents}
        generation_config = _build_generation_config(request, model_name=model_name)
        if generation_config:
            payload["generationConfig"] = generation_config

        try:
            response = await self._client.post(
                url,
                json=payload,
                timeout=self._settings.gateway_timeout_seconds,
            )
        except httpx.TimeoutException as exc:
            raise ProviderError(f"Gemini request timed out: {exc}", status_code=504) from exc
        except httpx.RequestError as exc:
            raise ProviderError(f"Gemini request failed: {exc}", status_code=502) from exc
        if response.status_code >= 400:
            body = response.text or ""
            raise ProviderError(
                f"Gemini error {response.status_code}: {_truncate(body)}",
                status_code=response.status_code,
                provider_request_id=_provider_request_id(response),
            )

        data = response.json()
        output_text = _extract_output_text(data)
        usage = data.get("usageMetadata", {})

        return ChatResponse(
            provider=self.name,
            model=model_name,
            output_text=output_text,
            usage=usage,
            trace_id=trace_id,
            conversation_id=request.conversation_id,
            agent_id=request.agent_id,
            provider_request_id=_provider_request_id(response),
        )


def _message_to_gemini(message: Message) -> dict[str, Any]:
    role = "user" if message.role.value == "user" else "model"

    def _chunk_to_part(chunk: Any) -> dict[str, Any]:
        # Text chunk
        if isinstance(chunk, str):
            return {"text": chunk}

        if isinstance(chunk, dict):
            # Explicit input_text chunk
            if chunk.get("type") == "input_text" and "text" in chunk:
                return {"text": str(chunk.get("text", ""))}

            # Input image provided as data URL
            if chunk.get("type") == "input_image":
                data_url = chunk.get("image_url") or ""
                if data_url.startswith("data:") and ";base64," in data_url:
                    mime, b64 = data_url.split(";base64,", 1)
                    mime = mime.split(":", 1)[1] if ":" in mime else "image/png"
                    return {"inline_data": {"mime_type": mime, "data": b64}}

            # Raw inline data already base64 encoded
            if "image_base64" in chunk:
                return {"inline_data": {"mime_type": "image/png", "data": chunk["image_base64"]}}

            if "text" in chunk:
                return {"text": str(chunk["text"])}

        # Fallback to text-only representation
        return {"text": message.as_text()}

    content = message.content
    parts: list[dict[str, Any]] = []

    if isinstance(content, list):
        parts = [_chunk_to_part(c) for c in content]
    elif isinstance(content, dict):
        parts = [_chunk_to_part(content)]
    else:
        parts = [{"text": message.as_text()}]

    return {"role": role, "parts": parts}


def _normalize_model(model_name: str) -> str:
    return model_name.removeprefix("models/")


def _build_generation_config(request: ChatRequest, *, model_name: str) -> dict[str, Any]:
    metadata = request.metadata or {}
    cfg: dict[str, Any] = {}
    if request.temperature is not None:
        cfg["temperature"] = request.temperature
    if request.max_tokens is not None:
        cfg["maxOutputTokens"] = request.max_tokens

    response_cfg = _extract_response_format(metadata)
    if response_cfg:
        cfg.update(response_cfg)

    thinking_cfg = _extract_thinking_config(metadata, model_name=model_name)
    if thinking_cfg:
        cfg["thinkingConfig"] = thinking_cfg

    return cfg


def _extract_response_format(metadata: dict[str, Any]) -> dict[str, Any]:
    text_cfg = metadata.get("text")
    response_format = metadata.get("response_format")

    response_mime_type = _first_non_empty_str(
        [
            text_cfg.get("responseMimeType") if isinstance(text_cfg, dict) else None,
            text_cfg.get("response_mime_type") if isinstance(text_cfg, dict) else None,
            response_format.get("responseMimeType") if isinstance(response_format, dict) else None,
            response_format.get("response_mime_type") if isinstance(response_format, dict) else None,
        ]
    )
    response_schema: dict[str, Any] | None = None

    for candidate in (_extract_openai_text_format(text_cfg), response_format if isinstance(response_format, dict) else None):
        if not isinstance(candidate, dict):
            continue
        fmt_type = str(candidate.get("type") or "").strip().lower()
        if fmt_type in {"json_object", "json_schema"} and not response_mime_type:
            response_mime_type = JSON_MIME_TYPE
        schema = _extract_schema_from_format(candidate)
        if schema is not None:
            response_schema = schema
            response_mime_type = response_mime_type or JSON_MIME_TYPE

    out: dict[str, Any] = {}
    if response_mime_type:
        out["responseMimeType"] = response_mime_type
    if response_schema is not None:
        out["responseSchema"] = response_schema
    return out


def _extract_openai_text_format(text_cfg: Any) -> dict[str, Any] | None:
    if not isinstance(text_cfg, dict):
        return None
    fmt = text_cfg.get("format")
    return fmt if isinstance(fmt, dict) else None


def _extract_schema_from_format(fmt: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(fmt, dict):
        return None
    json_schema = fmt.get("json_schema")
    if isinstance(json_schema, dict):
        schema = json_schema.get("schema")
        if isinstance(schema, dict):
            return schema
    schema = fmt.get("schema")
    if isinstance(schema, dict):
        return schema
    return None


def _extract_thinking_config(metadata: dict[str, Any], *, model_name: str) -> dict[str, Any]:
    reasoning = metadata.get("reasoning")
    if not isinstance(reasoning, dict) or not reasoning:
        return {}

    if _is_gemini_3_model(model_name):
        return _gemini_3_thinking_config(reasoning, model_name=model_name)
    return _gemini_25_thinking_config(reasoning)


def _gemini_25_thinking_config(reasoning: dict[str, Any]) -> dict[str, Any]:
    explicit_budget = _extract_int(reasoning, keys=("thinkingBudget", "thinking_budget"))
    if explicit_budget is not None:
        return {"thinkingBudget": explicit_budget}

    effort = str(reasoning.get("effort") or "").strip().lower()
    if not effort:
        return {}
    if effort not in GEMINI_25_EFFORT_TO_BUDGET:
        raise ProviderError(
            f"Unsupported Gemini 2.5 reasoning effort '{effort}'. Expected one of {sorted(GEMINI_25_EFFORT_TO_BUDGET)}.",
            status_code=422,
        )
    return {"thinkingBudget": GEMINI_25_EFFORT_TO_BUDGET[effort]}


def _gemini_3_thinking_config(reasoning: dict[str, Any], *, model_name: str) -> dict[str, Any]:
    explicit_level = _first_non_empty_str([reasoning.get("thinkingLevel"), reasoning.get("thinking_level")])
    if explicit_level:
        return {"thinkingLevel": explicit_level}

    effort = str(reasoning.get("effort") or "").strip().lower()
    if not effort:
        return {}
    if effort == "none":
        raise ProviderError(
            f"Gemini 3 model '{model_name}' does not support reasoning.effort='none'.",
            status_code=422,
        )

    mapping = GEMINI_3_FLASH_EFFORT_TO_LEVEL if "flash" in model_name.lower() else GEMINI_3_PRO_EFFORT_TO_LEVEL
    if effort not in mapping:
        allowed = sorted(mapping)
        raise ProviderError(
            f"Gemini 3 model '{model_name}' does not support reasoning.effort='{effort}'. Expected one of {allowed}.",
            status_code=422,
        )
    return {"thinkingLevel": mapping[effort]}


def _extract_output_text(payload: dict[str, Any]) -> str:
    candidates = payload.get("candidates")
    if not isinstance(candidates, list):
        return ""
    collected: list[str] = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        content = candidate.get("content")
        if not isinstance(content, dict):
            continue
        parts = content.get("parts")
        if not isinstance(parts, list):
            continue
        for part in parts:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                collected.append(str(part.get("text") or ""))
    return "".join(collected)


def _extract_int(obj: dict[str, Any], *, keys: tuple[str, ...]) -> int | None:
    for key in keys:
        value = obj.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            return value
    return None


def _first_non_empty_str(values: list[Any]) -> str:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _is_gemini_3_model(model_name: str) -> bool:
    normalized = model_name.lower()
    return normalized.startswith("gemini-3") or normalized.startswith("models/gemini-3")


def _provider_request_id(response: httpx.Response) -> str | None:
    return (
        response.headers.get("x-request-id")
        or response.headers.get("x-goog-request-id")
        or response.headers.get("x-cloud-trace-context")
    )


def _truncate(text: str, limit: int = 2000) -> str:
    if len(text) <= limit:
        return text
    return f"{text[:limit]}...[truncated {len(text) - limit} chars]"
