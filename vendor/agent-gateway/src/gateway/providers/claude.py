"""Claude (Anthropic) provider adapter."""

from __future__ import annotations

from typing import Any

import httpx

from ..models import ChatRequest, ChatResponse
from ..settings import Settings
from .base import BaseProvider, ProviderError, ProviderNotConfiguredError

CLAUDE_BASE_URL = "https://api.anthropic.com/v1/messages"


class ClaudeProvider(BaseProvider):
    name = "claude"

    def __init__(self, client: httpx.AsyncClient, settings: Settings) -> None:
        super().__init__()
        self._client = client
        self._settings = settings

    async def chat(self, request: ChatRequest, trace_id: str) -> ChatResponse:
        api_key = self._settings.claude_api_key
        if not api_key:
            raise ProviderNotConfiguredError("CLAUDE_KEY is not configured")

        headers = {
            "Authorization": f"Bearer {api_key}",
            "x-api-key": api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        }

        payload: dict[str, Any] = {
            "model": request.model,
            "max_tokens": request.max_tokens or 1024,
            "temperature": request.temperature,
            "messages": [message.model_dump(exclude_none=True) for message in request.messages],
        }

        try:
            response = await self._client.post(
                CLAUDE_BASE_URL,
                json=payload,
                headers=headers,
                timeout=self._settings.gateway_timeout_seconds,
            )
        except httpx.TimeoutException as exc:
            raise ProviderError(f"Claude request timed out: {exc}", status_code=504) from exc
        except httpx.RequestError as exc:
            raise ProviderError(f"Claude request failed: {exc}", status_code=502) from exc
        if response.status_code >= 400:
            body = response.text or ""
            raise ProviderError(
                f"Claude error {response.status_code}: {_truncate(body)}",
                status_code=response.status_code,
                provider_request_id=_provider_request_id(response),
            )

        data = response.json()
        content = data.get("content", [])
        output_text = "".join(block.get("text", "") for block in content if isinstance(block, dict))
        usage = data.get("usage", {})

        return ChatResponse(
            provider=self.name,
            model=request.model,
            output_text=output_text,
            usage=usage,
            trace_id=trace_id,
            conversation_id=request.conversation_id,
            agent_id=request.agent_id,
        )


def _provider_request_id(response: httpx.Response) -> str | None:
    return (
        response.headers.get("x-request-id")
        or response.headers.get("anthropic-request-id")
        or response.headers.get("x-cloud-trace-context")
    )


def _truncate(text: str, limit: int = 2000) -> str:
    if len(text) <= limit:
        return text
    return f"{text[:limit]}...[truncated {len(text) - limit} chars]"
