from __future__ import annotations

import random
import time
from typing import Any, Dict, Iterable, Optional

import requests

API_URL = "https://api.openai.com/v1/responses"


class OpenAIResponsesClient:
    def __init__(
        self,
        *,
        api_key: str,
        timeout: float = 120.0,
        max_retries: int = 5,
        initial_backoff: float = 2.0,
    ) -> None:
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.initial_backoff = initial_backoff

    def create_response(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        attempt = 0
        while True:
            attempt += 1
            try:
                response = requests.post(
                    API_URL,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                )
            except requests.RequestException as exc:  # noqa: PERF203
                error = f"network error: {exc}"
            else:
                if response.status_code == 200:
                    return response.json()
                error = f"HTTP {response.status_code}: {response.text}"

            if attempt >= self.max_retries:
                raise RuntimeError(f"OpenAI request failed after {attempt} attempts: {error}")

            delay = self.initial_backoff * (2 ** (attempt - 1))
            delay += random.random()
            time.sleep(delay)

    @staticmethod
    def extract_text(output: Dict[str, Any]) -> Optional[str]:
        if not output:
            return None
        if output.get("output_text"):
            return "\n".join(output["output_text"])
        candidates: Iterable[Dict[str, Any]] = output.get("output") or output.get("data") or []
        for message in candidates:
            content = message.get("content", [])
            for part in content:
                if part.get("type") in {"text", "output_text"}:
                    text = part.get("text") or part.get("output_text")
                    if text:
                        return text
            if message.get("text"):
                return message["text"]
        return None
