from __future__ import annotations

import argparse
import os
from pathlib import Path

import uvicorn

from .api import create_app
from .settings import GatewaySettings


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the EDGAR pipeline gateway server.")
    parser.add_argument("--tar-root", required=True, type=Path, help="Path to the EDGAR tarball root directory.")
    parser.add_argument("--manifest", required=True, type=Path, help="Parquet/CSV manifest file covering available segments.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--openai-key", default=os.getenv("OPENAI_API_KEY"), help="OpenAI API key (defaults to OPENAI_API_KEY env var)")
    parser.add_argument("--openai-timeout", type=float, default=120.0)
    parser.add_argument("--openai-max-retries", type=int, default=5)
    parser.add_argument("--openai-initial-backoff", type=float, default=2.0)
    parser.add_argument("--reload", action="store_true", help="Enable autoreload (development only)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if not args.openai_key:
        raise SystemExit("Error: provide an OpenAI API key via --openai-key or OPENAI_API_KEY env var.")
    settings = GatewaySettings(
        tar_root=args.tar_root,
        manifest_path=args.manifest,
        openai_api_key=args.openai_key,
        openai_timeout=args.openai_timeout,
        openai_max_retries=args.openai_max_retries,
        openai_initial_backoff=args.openai_initial_backoff,
        host=args.host,
        port=args.port,
        reload=args.reload,
    )
    app = create_app(settings)
    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":  # pragma: no cover
    main()
