"""Shared request/response models."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class Role(str, Enum):
    """Chat roles supported by the gateway."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    DEVELOPER = "developer"


class Message(BaseModel):
    """Generic message representation for multi-agent orchestration."""

    role: Role
    content: Any
    name: str | None = None
    metadata: dict[str, Any] | None = None

    def as_text(self) -> str:
        """Best-effort textual representation for providers that only accept text."""

        return _flatten_content_to_text(self.content)


class ChatRequest(BaseModel):
    """Incoming gateway request for an LLM call."""

    provider: str = Field(..., description="Identifier of the provider adapter to use.")
    model: str = Field(..., description="Model name or version to target.")
    messages: list[Message] = Field(..., min_length=1)
    temperature: float | None = Field(default=0.3, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, gt=0)
    metadata: dict[str, Any] | None = None
    conversation_id: str | None = Field(
        default=None, description="Traceable conversation identifier for agent hand-offs."
    )
    agent_id: str | None = Field(default=None, description="Originating agent identifier.")


class ChatResponse(BaseModel):
    """Normalized response coming back from providers."""

    provider: str
    model: str
    output_text: str
    usage: dict[str, Any] = Field(default_factory=dict)
    trace_id: str
    conversation_id: str | None = None
    agent_id: str | None = None
    provider_request_id: str | None = None


class AgentEnvelope(BaseModel):
    """Payload for inter-agent messaging through the gateway."""

    conversation_id: str
    sender_agent_id: str
    recipient_agent_id: str
    payload: dict[str, Any]


def _flatten_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = [item for item in (_flatten_content_to_text(entry) for entry in content) if item]
        return "\n".join(parts)

    if isinstance(content, dict):
        text_value = content.get("text")
        if isinstance(text_value, str):
            return text_value
        if isinstance(text_value, list):
            return "\n".join(str(item) for item in text_value if item)

        image_url = content.get("image_url")
        if isinstance(image_url, dict) and image_url.get("url"):
            return image_url["url"]
        if "image_base64" in content or "image" in content:
            return "<image>"

        audio_field = content.get("audio") or content.get("audio_base64")
        if audio_field:
            return "<audio>"

        if "type" in content:
            return f"<{content['type']}>"

    return str(content)
