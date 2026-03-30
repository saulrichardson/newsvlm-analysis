"""Gateway service orchestrating providers and agent bus."""

from __future__ import annotations

import uuid

import httpx

from ..logging import bind_trace
from ..models import AgentEnvelope, ChatRequest, ChatResponse
from ..providers import (
    ClaudeProvider,
    EchoProvider,
    GeminiProvider,
    OpenAIProvider,
    ProviderError,
    ProviderRegistry,
)
from ..settings import Settings
from .agent_bus import AgentBus


class GatewayService:
    """Main entry point for executing chat calls and relaying agent messages."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = httpx.AsyncClient(timeout=settings.gateway_timeout_seconds)
        self.providers = self._build_registry(settings)
        self.agent_bus = AgentBus()

    def _build_registry(self, settings: Settings) -> ProviderRegistry:
        registry = ProviderRegistry()
        registry.register(EchoProvider())
        registry.register(OpenAIProvider(client=self._client, settings=settings))
        registry.register(GeminiProvider(client=self._client, settings=settings))
        registry.register(ClaudeProvider(client=self._client, settings=settings))
        return registry

    async def shutdown(self) -> None:
        await self._client.aclose()

    async def chat(self, request: ChatRequest, trace_id: str | None = None) -> ChatResponse:
        provider_name = request.provider or self._settings.default_provider
        provider = self.providers.get(provider_name)
        if provider is None:
            raise ProviderError(f"Provider '{provider_name}' is not registered")

        trace_id = trace_id or uuid.uuid4().hex
        bind_trace(trace_id=trace_id, provider=provider.name)
        result = await provider.chat(request, trace_id=trace_id)
        return result

    def publish_agent_message(self, envelope: AgentEnvelope) -> None:
        self.agent_bus.publish(envelope)

    def drain_agent_messages(self, agent_id: str, conversation_id: str) -> list[AgentEnvelope]:
        return self.agent_bus.consume(agent_id, conversation_id)
