from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class GatewaySettings:
    tar_root: Path
    manifest_path: Path
    openai_api_key: str
    default_model: str = "gpt-5-mini"
    openai_timeout: float = 120.0
    openai_max_retries: int = 5
    openai_initial_backoff: float = 2.0
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False
