"""Lightweight ad-hoc checks for the gateway.

Run with:
    PYTHONPATH=src python scripts/ad_hoc_checks.py

These are intentionally minimal, fail fast, and cover the high-level
behaviors the gateway promises. No external keys or network calls are
required; everything is exercised against the in-process FastAPI app
and pure helper functions.
"""

from __future__ import annotations

import json
from typing import Any

from fastapi.testclient import TestClient

from gateway.api.routes import _parse_model_identifier, _to_chat_request
from gateway.api.schemas import ResponseInputMessage, ResponseRequest
from gateway.app import create_app
from gateway.models import AgentEnvelope
from gateway.services.agent_bus import AgentBus
from gateway.settings import Settings


def assert_stream_events() -> None:
    settings = Settings(default_provider="echo")
    app = create_app(settings=settings)
    client = TestClient(app)

    payload = {
        "model": "echo:test-model",
        "input": [
            {"role": "user", "content": [{"type": "input_text", "text": "ping"}]},
        ],
    }

    with client.stream("POST", "/v1/responses", json=payload) as response:
        body = "".join(list(response.iter_text()))

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    events = [chunk.strip() for chunk in body.split("\n\n") if chunk.strip()]
    names = [chunk.split("\n", 1)[0].replace("event: ", "") for chunk in events]
    assert "response.created" in names
    assert "response.output_text.delta" in names
    assert "response.completed" in names

    created_idx = names.index("response.created")
    delta_idx = names.index("response.output_text.delta")
    completed_idx = names.index("response.completed")
    assert created_idx < delta_idx < completed_idx


def assert_missing_provider_rejected() -> None:
    import os

    os.environ["DEFAULT_PROVIDER"] = ""
    settings = Settings()
    app = create_app(settings=settings)
    client = TestClient(app)

    payload = {
        "model": "gpt-5-nano",
        "input": [{"role": "user", "content": "hi"}],
    }

    response = client.post("/v1/responses", json=payload)
    assert response.status_code == 422
    detail = response.json()["detail"]["error"]
    assert detail["code"] == "provider_required"


def assert_unknown_provider_rejected() -> None:
    settings = Settings(default_provider="echo")
    app = create_app(settings=settings)
    client = TestClient(app)

    payload = {
        "model": "bogus:model",
        "input": [{"role": "user", "content": "hi"}],
    }

    response = client.post("/v1/responses", json=payload)
    assert response.status_code == 502
    detail = response.json()["detail"]["error"]
    assert detail["provider"] == "bogus"


def assert_structured_content_preserved() -> None:
    payload = ResponseRequest(
        model="echo:test-model",
        input=[
            ResponseInputMessage(
                role="user",
                content=[
                    {"type": "input_text", "text": "Describe this image."},
                    {"type": "input_image", "image_url": {"url": "https://example.com/img.png"}},
                ],
            )
        ],
    )

    provider, upstream = _parse_model_identifier(payload.model, default_provider=None)
    chat_request = _to_chat_request(payload, provider, upstream)

    message_content = chat_request.messages[0].content
    assert isinstance(message_content, list)
    assert message_content[1]["type"] == "input_image"


def assert_agent_bus_round_trip() -> None:
    bus = AgentBus(max_messages=2)

    envelope = AgentEnvelope(
        conversation_id="conv1",
        sender_agent_id="agent_a",
        recipient_agent_id="agent_b",
        payload={"msg": "hello"},
    )

    bus.publish(envelope)
    drained = bus.consume("agent_b", "conv1")
    assert len(drained) == 1
    assert drained[0].payload["msg"] == "hello"
    drained_again = bus.consume("agent_b", "conv1")
    assert drained_again == []


def run_checks() -> None:
    checks: dict[str, Any] = {
        "streaming": assert_stream_events,
        "missing_provider": assert_missing_provider_rejected,
        "unknown_provider": assert_unknown_provider_rejected,
        "structured_content": assert_structured_content_preserved,
        "agent_bus": assert_agent_bus_round_trip,
    }

    for name, fn in checks.items():
        fn()
        print(json.dumps({"check": name, "status": "pass"}))


if __name__ == "__main__":
    run_checks()
