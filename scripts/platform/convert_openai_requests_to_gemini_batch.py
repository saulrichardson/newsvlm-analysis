#!/usr/bin/env python3
"""
Convert OpenAI-Batch-shaped request shards into Gemini Batch JSONL shards.

This is intended for cases where we already exported issue-level prompts for OpenAI and
want to resubmit the exact same prompts to Gemini Batch without rebuilding the upstream
transcript aggregation run.

Input:
  - openai_requests_shardNNN.jsonl
  - mapping_shardNNN.jsonl (used only to verify shard parity when present)

Output:
  - gemini_requests_shardNNN.jsonl
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Convert openai_requests_shard*.jsonl to gemini_requests_shard*.jsonl.")
    ap.add_argument("--request-dir", required=True, help="Directory containing openai_requests_shard*.jsonl files.")
    ap.add_argument(
        "--response-mime-type",
        default="application/json",
        help="Gemini request generation_config.response_mime_type (empty disables).",
    )
    ap.add_argument(
        "--thinking-budget",
        type=int,
        default=32768,
        help="Gemini 2.5 thinking_budget to include in request generation_config.thinking_config (omit with --thinking-budget 0).",
    )
    ap.add_argument(
        "--thinking-level",
        default="",
        help="Gemini 3 thinking_level to include in request generation_config.thinking_config (e.g. high).",
    )
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing gemini_requests_shard*.jsonl files.")
    return ap.parse_args()


def _line_count(path: Path) -> int:
    out = subprocess.check_output(["wc", "-l", str(path)], text=True).strip()
    return int(out.split()[0])


def _parse_prompt_text(obj: dict[str, Any]) -> str:
    body = obj.get("body") or {}
    input_items = body.get("input") or []
    text_parts: list[str] = []
    for msg in input_items:
        for part in msg.get("content") or []:
            if part.get("type") == "input_text":
                text = part.get("text")
                if isinstance(text, str) and text:
                    text_parts.append(text)
    return "\n\n".join(text_parts).strip()


def _build_gemini_request(
    key: str,
    prompt: str,
    *,
    response_mime_type: str,
    thinking_budget: int,
    thinking_level: str,
) -> dict[str, Any]:
    request: dict[str, Any] = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
    }
    generation_config: dict[str, Any] = {}
    if response_mime_type.strip():
        generation_config["response_mime_type"] = response_mime_type.strip()
    thinking_config: dict[str, Any] = {}
    if thinking_budget > 0:
        thinking_config["thinking_budget"] = int(thinking_budget)
    if thinking_level.strip():
        thinking_config["thinking_level"] = thinking_level.strip().lower()
    if thinking_config:
        generation_config["thinking_config"] = thinking_config
    if generation_config:
        request["generation_config"] = generation_config
    return {"key": key, "request": request}


def _parse_shard_index(filename: str, *, prefix: str, suffix: str) -> int:
    if not (filename.startswith(prefix) and filename.endswith(suffix)):
        raise ValueError(f"Unexpected filename: {filename}")
    token = filename[len(prefix) : -len(suffix)]
    if not token.isdigit():
        raise ValueError(f"Unexpected shard token in filename: {filename}")
    return int(token)


def main() -> None:
    args = _parse_args()
    request_dir = Path(args.request_dir).expanduser().resolve()
    if not request_dir.is_dir():
        raise SystemExit(f"--request-dir is not a directory: {request_dir}")

    openai_paths: dict[int, Path] = {}
    mapping_paths: dict[int, Path] = {}
    for path in request_dir.glob("openai_requests_shard*.jsonl"):
        idx = _parse_shard_index(path.name, prefix="openai_requests_shard", suffix=".jsonl")
        openai_paths[idx] = path
    for path in request_dir.glob("mapping_shard*.jsonl"):
        idx = _parse_shard_index(path.name, prefix="mapping_shard", suffix=".jsonl")
        mapping_paths[idx] = path

    if not openai_paths:
        raise SystemExit(f"No openai_requests_shard*.jsonl files found in {request_dir}")

    converted = 0
    skipped = 0
    for idx in sorted(openai_paths):
        openai_path = openai_paths[idx]
        gemini_path = request_dir / f"gemini_requests_shard{idx:03d}.jsonl"
        if gemini_path.exists() and not args.overwrite:
            skipped += 1
            print(f"skip_exists shard={idx:03d} path={gemini_path}")
            continue

        rows_written = 0
        with openai_path.open("r", encoding="utf-8") as src, gemini_path.open("w", encoding="utf-8") as dst:
            for line_no, raw in enumerate(src, start=1):
                line = raw.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception as exc:
                    raise SystemExit(f"{openai_path}:{line_no}: invalid JSON: {exc}") from exc
                key = str(obj.get("custom_id") or "").strip()
                if not key:
                    raise SystemExit(f"{openai_path}:{line_no}: missing custom_id")
                prompt = _parse_prompt_text(obj)
                if not prompt:
                    raise SystemExit(f"{openai_path}:{line_no}: could not extract prompt text")
                out_obj = _build_gemini_request(
                    key,
                    prompt,
                    response_mime_type=str(args.response_mime_type),
                    thinking_budget=int(args.thinking_budget),
                    thinking_level=str(args.thinking_level),
                )
                dst.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
                rows_written += 1

        mapping_path = mapping_paths.get(idx)
        if mapping_path is not None:
            mapping_lines = _line_count(mapping_path)
            if rows_written != mapping_lines:
                raise SystemExit(
                    f"Shard {idx:03d} line mismatch after conversion: mapping={mapping_lines} gemini={rows_written}"
                )
        converted += 1
        print(f"converted shard={idx:03d} rows={rows_written} path={gemini_path}")

    print(
        json.dumps(
            {
                "request_dir": str(request_dir),
                "total_openai_shards": len(openai_paths),
                "converted": converted,
                "skipped": skipped,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
