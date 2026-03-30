"""Structlog-based logging helpers."""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog


def configure_logging() -> None:
    """Configure structlog for JSON-friendly, trace-aware logs."""

    timestamper = structlog.processors.TimeStamper(fmt="iso", utc=True)

    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        timestamper,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.EventRenamer("message"),
    ]

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=logging.INFO,
    )

    # Avoid leaking provider secrets (e.g., Gemini `key=` query param) via httpx
    # request logs. We still keep our own structured logs at INFO.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    structlog.configure(
        processors=shared_processors
        + [
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def bind_trace(**kwargs: Any) -> None:
    """Attach contextual trace metadata to the current context."""

    structlog.contextvars.bind_contextvars(**kwargs)
