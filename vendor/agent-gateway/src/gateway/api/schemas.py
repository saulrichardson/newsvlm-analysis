"""API request schemas for the Responses contract."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class ResponseInputMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool", "developer"]
    content: Any


class ResponseRequest(BaseModel):
    model: str
    input: list[ResponseInputMessage]
    temperature: float | None = Field(default=None, ge=0, le=2)
    max_output_tokens: int | None = Field(default=None, gt=0)
    stream: bool | None = Field(default=True)
    text: dict[str, Any] | None = None
    response_format: dict[str, Any] | None = None
    reasoning: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None

    @field_validator("input")
    @classmethod
    def _input_not_empty(cls, value: list[ResponseInputMessage]) -> list[ResponseInputMessage]:
        if not value:
            raise ValueError("input cannot be empty")
        return value

    @field_validator("stream")
    @classmethod
    def _require_streaming(cls, value: bool | None) -> bool:
        if value is False:
            raise ValueError("Only streaming responses are supported (stream=true).")
        return True
