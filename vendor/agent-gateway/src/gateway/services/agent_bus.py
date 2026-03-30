"""In-memory agent messaging bus for orchestration."""

from __future__ import annotations

from collections import defaultdict, deque

from ..models import AgentEnvelope


class AgentBus:
    """Keeps short-lived message queues per agent and conversation."""

    def __init__(self, max_messages: int = 100) -> None:
        self._queues: dict[str, deque[AgentEnvelope]] = defaultdict(deque)
        self._max_messages = max_messages

    def publish(self, envelope: AgentEnvelope) -> None:
        key = self._queue_key(envelope.recipient_agent_id, envelope.conversation_id)
        queue = self._queues[key]
        queue.append(envelope)
        while len(queue) > self._max_messages:
            queue.popleft()

    def consume(self, agent_id: str, conversation_id: str) -> list[AgentEnvelope]:
        key = self._queue_key(agent_id, conversation_id)
        queue = self._queues.get(key)
        if not queue:
            return []
        items = list(queue)
        queue.clear()
        return items

    @staticmethod
    def _queue_key(agent_id: str, conversation_id: str) -> str:
        return f"{agent_id}:{conversation_id}"
