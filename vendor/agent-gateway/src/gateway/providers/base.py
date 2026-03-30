"""Provider abstraction for the gateway."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol

from ..models import ChatRequest, ChatResponse


class ProviderNotConfiguredError(RuntimeError):
    """Raised when a provider lacks the credentials to execute requests."""


class ProviderError(RuntimeError):
    """Raised on provider-specific failures."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        provider_request_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.provider_request_id = provider_request_id


class ModelProvider(Protocol):
    """Interface for provider adapters."""

    name: str

    async def chat(self, request: ChatRequest, trace_id: str) -> ChatResponse:
        """Execute the request and return a normalized response."""


class BaseProvider(ABC):
    """Helper base class with shared utilities."""

    name: str

    def __init__(self, metadata: dict[str, str] | None = None) -> None:
        self.metadata = metadata or {}

    @abstractmethod
    async def chat(self, request: ChatRequest, trace_id: str) -> ChatResponse:
        """Execute the request and return a normalized response."""
