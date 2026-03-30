#!/usr/bin/env python3
"""
Set explicit control fields on staged OpenAI-Batch-style request shards.

This is intended for live gateway reruns where we want to keep the original
prompt payloads but override request controls such as:
  - body.reasoning = {"effort": "..."}
  - body.text = {"format": {"type": "json_object"}}

The script rewrites `openai_requests_shard*.jsonl` in place under --request-dir.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Set explicit controls on openai_requests_shard*.jsonl.")
    ap.add_argument("--request-dir", required=True, help="Directory containing openai_requests_shard*.jsonl.")
    ap.add_argument(
        "--reasoning-effort",
        default="",
        choices=["", "none", "minimal", "low", "medium", "high"],
        help="If set, replace body.reasoning with {'effort': <value>}. Empty leaves reasoning unchanged.",
    )
    ap.add_argument(
        "--json-object",
        action="store_true",
        help="Set body.text.format.type=json_object on every request shard line.",
    )
    ap.add_argument(
        "--clear-text",
        action="store_true",
        help="Remove body.text from every request shard line.",
    )
    return ap.parse_args()


def _iter_request_paths(request_dir: Path) -> list[Path]:
    paths = sorted(request_dir.glob("openai_requests_shard*.jsonl"))
    if not paths:
        raise SystemExit(f"No openai_requests_shard*.jsonl files found in {request_dir}")
    return paths


def _rewrite_path(path: Path, *, reasoning_effort: str, json_object: bool, clear_text: bool) -> int:
    fd, tmp_name = tempfile.mkstemp(prefix=f"{path.stem}_", suffix=".tmp", dir=str(path.parent))
    tmp_path = Path(tmp_name)
    os.close(fd)
    rows = 0
    try:
        with path.open("r", encoding="utf-8") as src, tmp_path.open("w", encoding="utf-8") as dst:
            for line_no, raw in enumerate(src, start=1):
                line = raw.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception as exc:
                    raise SystemExit(f"{path}:{line_no}: invalid JSON: {exc}") from exc
                body = obj.get("body")
                if not isinstance(body, dict):
                    raise SystemExit(f"{path}:{line_no}: missing body object")
                if reasoning_effort:
                    body["reasoning"] = {"effort": reasoning_effort}
                if clear_text:
                    body.pop("text", None)
                if json_object:
                    body["text"] = {"format": {"type": "json_object"}}
                obj["body"] = body
                dst.write(json.dumps(obj, ensure_ascii=False) + "\n")
                rows += 1
        tmp_path.replace(path)
        return rows
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def main() -> None:
    args = _parse_args()
    request_dir = Path(args.request_dir).expanduser().resolve()
    if not request_dir.is_dir():
        raise SystemExit(f"--request-dir is not a directory: {request_dir}")
    if args.json_object and args.clear_text:
        raise SystemExit("--json-object and --clear-text are mutually exclusive")

    total_rows = 0
    paths = _iter_request_paths(request_dir)
    for path in paths:
        total_rows += _rewrite_path(
            path,
            reasoning_effort=str(args.reasoning_effort).strip().lower(),
            json_object=bool(args.json_object),
            clear_text=bool(args.clear_text),
        )
    print(
        json.dumps(
            {
                "request_dir": str(request_dir),
                "files_rewritten": len(paths),
                "rows_rewritten": total_rows,
                "reasoning_effort": str(args.reasoning_effort).strip().lower(),
                "json_object": bool(args.json_object),
                "clear_text": bool(args.clear_text),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
