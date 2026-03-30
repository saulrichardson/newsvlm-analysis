"""Helper utilities for agents that need to call the gateway frequently."""

from __future__ import annotations

import asyncio
import base64
import json
import mimetypes
import os
from pathlib import Path
from typing import Any, AsyncIterator, Sequence

import httpx

DEFAULT_BASE_URL = os.getenv("GATEWAY_URL", "http://127.0.0.1:8000")


class GatewayAgentClient:
    """Thin async wrapper around the /v1/responses streaming endpoint."""

    def __init__(self, base_url: str = DEFAULT_BASE_URL, timeout: float = 30.0) -> None:
        self._client = httpx.AsyncClient(base_url=base_url, timeout=timeout)

    async def __aenter__(self) -> "GatewayAgentClient":
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._client.aclose()

    async def stream_response(
        self,
        *,
        model: str,
        input_messages: list[dict[str, Any]],
        text: dict[str, Any] | None = None,
        response_format: dict[str, Any] | None = None,
        reasoning: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Yield SSE events mirroring the OpenAI Responses stream."""

        payload: dict[str, Any] = {
            "model": model,
            "input": input_messages,
            "stream": True,
        }
        if text:
            payload["text"] = text
        if response_format:
            payload["response_format"] = response_format
        if reasoning:
            payload["reasoning"] = reasoning
        if temperature is not None:
            payload["temperature"] = temperature
        if max_output_tokens is not None:
            payload["max_output_tokens"] = max_output_tokens
        if metadata:
            payload["metadata"] = metadata

        async with self._client.stream("POST", "/v1/responses", json=payload) as response:
            response.raise_for_status()
            current_event: str | None = None
            async for raw_line in response.aiter_lines():
                if not raw_line:
                    continue
                line = raw_line.strip()
                if not line:
                    continue
                if line.startswith("event:"):
                    current_event = line.replace("event:", "", 1).strip()
                    continue
                if line.startswith("data:") and current_event:
                    data = json.loads(line.replace("data:", "", 1).strip())
                    yield {"event": current_event, "data": data}
                    current_event = None

    async def complete_response(
        self,
        *,
        model: str,
        input_messages: list[dict[str, Any]],
        text: dict[str, Any] | None = None,
        response_format: dict[str, Any] | None = None,
        reasoning: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Collect the streaming response and return the final text + metadata."""

        deltas: list[str] = []
        completed_payload: dict[str, Any] | None = None
        failed_payload: dict[str, Any] | None = None

        async for event in self.stream_response(
            model=model,
            input_messages=input_messages,
            text=text,
            response_format=response_format,
            reasoning=reasoning,
            metadata=metadata,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        ):
            if event["event"] == "response.output_text.delta":
                data = event["data"] or {}
                if "delta" in data:
                    delta_text = data.get("delta") or ""
                else:
                    output_text = data.get("output_text") or []
                    delta_text = "".join(str(chunk) for chunk in output_text)
                deltas.append(str(delta_text))
            elif event["event"] == "response.failed":
                failed_payload = event["data"] or {}
            elif event["event"] == "response.completed":
                completed_payload = event["data"]

        if failed_payload is not None:
            raise RuntimeError(_format_failed_response_error(failed_payload))

        text = "".join(deltas)
        if not text and completed_payload is not None:
            text = _extract_completed_output_text(completed_payload)

        return {
            "text": text,
            "meta": completed_payload or {},
        }


def _extract_completed_output_text(payload: dict[str, Any]) -> str:
    response = payload.get("response")
    if not isinstance(response, dict):
        return ""

    output_text = response.get("output_text")
    if isinstance(output_text, list):
        return "".join(str(chunk) for chunk in output_text if chunk).strip()
    if isinstance(output_text, str):
        return output_text.strip()

    output = response.get("output")
    if not isinstance(output, list):
        return ""

    parts: list[str] = []
    for item in output:
        if not isinstance(item, dict):
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for chunk in content:
            if not isinstance(chunk, dict):
                continue
            if chunk.get("type") not in {"output_text", "text"}:
                continue
            text = chunk.get("text")
            if isinstance(text, str) and text.strip():
                parts.append(text)
    return "\n".join(parts).strip()


def _format_failed_response_error(payload: dict[str, Any]) -> str:
    response = payload.get("response")
    if isinstance(response, dict):
        err = response.get("error")
        if isinstance(err, dict):
            message = err.get("message")
            if isinstance(message, str) and message.strip():
                return f"Gateway response.failed: {message.strip()}"
        message = response.get("message")
        if isinstance(message, str) and message.strip():
            return f"Gateway response.failed: {message.strip()}"

    error = payload.get("error")
    if isinstance(error, dict):
        message = error.get("message")
        if isinstance(message, str) and message.strip():
            return f"Gateway response.failed: {message.strip()}"
    if isinstance(error, str) and error.strip():
        return f"Gateway response.failed: {error.strip()}"

    return f"Gateway response.failed: {json.dumps(payload, ensure_ascii=False)}"


def build_user_message(
    prompt: str,
    *,
    image_paths: Sequence[str] | None = None,
    image_bytes: Sequence[bytes] | None = None,
    image_mime_type: str | None = None,
) -> dict[str, Any]:
    """Create a user message that optionally bundles images as input_image parts.

    Supports passing paths on disk or already-loaded image bytes (PNG/JPEG, etc.).
    """

    chunks: list[dict[str, Any]] = []
    if prompt:
        chunks.append({"type": "input_text", "text": prompt})

    for path in image_paths or []:
        chunks.append(_image_chunk_from_path(path))

    for blob in image_bytes or []:
        chunks.append(_image_chunk_from_bytes(blob, mime_type=image_mime_type))

    if not chunks:
        raise ValueError("A prompt and/or at least one image must be supplied.")

    if len(chunks) == 1 and chunks[0]["type"] == "input_text":
        return {"role": "user", "content": prompt}

    return {"role": "user", "content": chunks}


def _image_chunk_from_path(path: str) -> dict[str, Any]:
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")

    mime_type, _ = mimetypes.guess_type(file_path.name)
    encoded = base64.b64encode(file_path.read_bytes()).decode("utf-8")
    data_url = f"data:{mime_type or 'image/png'};base64,{encoded}"
    chunk: dict[str, Any] = {
        "type": "input_image",
        "image_url": data_url,
    }
    return chunk


def _image_chunk_from_bytes(data: bytes, *, mime_type: str | None = None) -> dict[str, Any]:
    if not isinstance(data, (bytes, bytearray)):
        raise TypeError("image_bytes entries must be bytes or bytearray")

    encoded = base64.b64encode(bytes(data)).decode("utf-8")
    data_url = f"data:{mime_type or 'image/png'};base64,{encoded}"
    return {
        "type": "input_image",
        "image_url": data_url,
    }


# --- Synchronous convenience wrappers -------------------------------------------------


def complete_response_sync(
    *,
    model: str,
    prompt: str,
    base_url: str = DEFAULT_BASE_URL,
    text: dict[str, Any] | None = None,
    response_format: dict[str, Any] | None = None,
    reasoning: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    temperature: float | None = None,
    max_output_tokens: int | None = None,
    timeout: float | None = None,
) -> str:
    """Blocking helper that wraps GatewayAgentClient.complete_response.

    Note: uses asyncio.run under the hood, so prefer the async API when already
    inside an event loop.
    """

    async def _run() -> str:
        message = build_user_message(prompt)
        async with GatewayAgentClient(base_url=base_url, timeout=timeout or 30.0) as client:
            result = await client.complete_response(
                model=model,
                input_messages=[message],
                text=text,
                response_format=response_format,
                reasoning=reasoning,
                metadata=metadata,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            )
            return result["text"]

    return asyncio.run(_run())
