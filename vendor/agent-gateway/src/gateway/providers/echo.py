"""Deterministic provider useful for tests and local dev."""

from __future__ import annotations

from ..models import ChatRequest, ChatResponse
from .base import BaseProvider


class EchoProvider(BaseProvider):
    """Simple provider that echoes the latest user message."""

    name = "echo"

    async def chat(self, request: ChatRequest, trace_id: str) -> ChatResponse:
        latest = request.messages[-1]
        latest_text = latest.as_text()
        output = f"[echo::{request.model}] {latest_text}"
        return ChatResponse(
            provider=self.name,
            model=request.model,
            output_text=output,
            usage={
                "prompt_tokens": len(latest_text.split()) if latest_text else 0,
                "completion_tokens": len(output),
            },
            trace_id=trace_id,
            conversation_id=request.conversation_id,
            agent_id=request.agent_id,
        )
