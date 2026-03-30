"""Gateway package CLI entry point."""

from __future__ import annotations

import argparse

import uvicorn


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the LLM Gateway API.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload (development only).",
    )
    return parser.parse_args(args=argv)


def main(argv: list[str] | None = None) -> int:
    """Run the FastAPI gateway."""

    args = parse_args(argv)
    uvicorn.run(
        "gateway.app:create_app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        factory=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
