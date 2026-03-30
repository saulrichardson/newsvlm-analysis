"""Provider registry for dependency injection."""

from __future__ import annotations

from .base import ModelProvider


class ProviderRegistry:
    """Simple in-memory registry mapping provider names to adapters."""

    def __init__(self) -> None:
        self._providers: dict[str, ModelProvider] = {}

    def register(self, provider: ModelProvider) -> None:
        key = provider.name.lower()
        self._providers[key] = provider

    def get(self, name: str) -> ModelProvider | None:
        return self._providers.get(name.lower())

    def available_providers(self) -> list[str]:
        return sorted(self._providers.keys())
