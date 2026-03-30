#!/usr/bin/env python3
"""
Replay OpenAI Batch-style /v1/responses request shards directly against OpenAI.

The request shard format is the same one already exported for OpenAI Batch:
each line contains `custom_id`, `url=/v1/responses`, and a JSON `body`.
This runner sends those bodies directly to the API and writes batch-shaped
result and error shards so the existing rehydration code can consume them.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Replay /v1/responses request shards directly against OpenAI.")
    ap.add_argument("--request-dir", required=True, help="Directory containing openai_requests_shard*.jsonl")
    ap.add_argument("--output-dir", required=True, help="Directory to write openai_results_shard*.jsonl")
    ap.add_argument("--env-file", default=".env", help="Path to .env with OPENAI_API_KEY / OPENAI_KEY / CODEX_API_KEY")
    ap.add_argument("--api-base", default="https://api.openai.com", help="Base URL for the OpenAI API")
    ap.add_argument("--timeout", type=float, default=180.0, help="Per-request timeout in seconds")
    ap.add_argument("--max-concurrency", type=int, default=8, help="Thread count for concurrent calls")
    ap.add_argument("--max-retries", type=int, default=4, help="Retry count for transient failures")
    ap.add_argument("--resume", action="store_true", help="Skip custom_ids already present in result/error shards")
    ap.add_argument(
        "--retry-errors",
        action="store_true",
        help="With --resume, skip completed result rows but retry any custom_ids that only exist in error shards.",
    )
    ap.add_argument(
        "--shard-indices",
        default="",
        help=(
            "Optional shard subset to process, e.g. '0,1,4-7'. "
            "When omitted, all request shards in --request-dir are processed."
        ),
    )
    return ap.parse_args()


def _parse_env_file(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.is_file():
        return out
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        out[key.strip()] = value.strip().strip('"').strip("'")
    return out


def _resolve_openai_key(env_file: Path) -> str:
    file_env = _parse_env_file(env_file)
    for name in ("OPENAI_API_KEY", "OPENAI_KEY", "CODEX_API_KEY"):
        val = os.environ.get(name) or file_env.get(name)
        if val:
            return val
    for idx in range(1, 6):
        for name in (f"OPENAI_KEY_BACKUP_{idx}", f"OPENAI_API_KEY_BACKUP_{idx}"):
            val = os.environ.get(name) or file_env.get(name)
            if val:
                return val
    raise SystemExit(f"Missing OpenAI key in env or env-file: {env_file}")


def _parse_shard_index(name: str, *, prefix: str, suffix: str) -> int:
    if not name.startswith(prefix) or not name.endswith(suffix):
        raise ValueError(f"Not a shard filename: {name}")
    middle = name[len(prefix) : -len(suffix)]
    if not middle.isdigit():
        raise ValueError(f"Shard index is not digits: {name}")
    return int(middle)


def _parse_shard_indices(raw: str) -> set[int]:
    text = str(raw or "").strip()
    if not text:
        return set()
    out: set[int] = set()
    for part in text.split(","):
        token = part.strip()
        if not token:
            continue
        if "-" in token:
            start_text, end_text = token.split("-", 1)
            start = int(start_text.strip())
            end = int(end_text.strip())
            if end < start:
                raise SystemExit(f"Invalid shard range '{token}': end < start")
            out.update(range(start, end + 1))
        else:
            out.add(int(token))
    return out


def _stable_fake_id(custom_id: str, *, model: str) -> str:
    digest = hashlib.sha256(f"{model}::{custom_id}".encode("utf-8")).hexdigest()[:24]
    return f"resp_req_{digest}"


def _read_existing_custom_ids(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    out: set[str] = set()
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        cid = obj.get("custom_id")
        if isinstance(cid, str) and cid:
            out.add(cid)
    return out


@dataclass(frozen=True)
class WorkItem:
    shard: int
    custom_id: str
    body: dict[str, Any]
    model: str


def _collect_work(request_dir: Path, *, shard_filter: set[int] | None = None) -> list[WorkItem]:
    req_paths = sorted(request_dir.glob("openai_requests_shard*.jsonl"))
    if not req_paths:
        raise SystemExit(f"No openai_requests_shard*.jsonl found in {request_dir}")
    out: list[WorkItem] = []
    for path in req_paths:
        shard = _parse_shard_index(path.name, prefix="openai_requests_shard", suffix=".jsonl")
        if shard_filter and shard not in shard_filter:
            continue
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line:
                continue
            obj = json.loads(line)
            cid = obj.get("custom_id")
            body = obj.get("body")
            if obj.get("url") != "/v1/responses":
                raise SystemExit(f"Request line in {path} is not a /v1/responses request")
            if not isinstance(cid, str) or not cid:
                raise SystemExit(f"Request line in {path} missing custom_id")
            if not isinstance(body, dict):
                raise SystemExit(f"Request line in {path} missing body")
            model = body.get("model")
            if not isinstance(model, str) or not model:
                raise SystemExit(f"Responses request in {path} missing body.model")
            out.append(WorkItem(shard=shard, custom_id=cid, body=body, model=model))
    return out


def _result_line(*, custom_id: str, model: str, body: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": _stable_fake_id(custom_id, model=model),
        "custom_id": custom_id,
        "response": {
            "status_code": 200,
            "request_id": "",
            "body": body,
        },
        "error": None,
    }


def _error_line(*, custom_id: str, model: str, status_code: int, message: str) -> dict[str, Any]:
    return {
        "id": _stable_fake_id(custom_id, model=model),
        "custom_id": custom_id,
        "response": {
            "status_code": int(status_code),
            "request_id": "",
            "body": {"error": {"message": str(message), "type": "responses_error"}},
        },
        "error": None,
    }


def _should_retry(status_code: int) -> bool:
    return int(status_code) in {408, 409, 429, 500, 502, 503, 504}


def _post_responses(*, api_base: str, api_key: str, item: WorkItem, timeout: float, max_retries: int) -> tuple[bool, dict[str, Any]]:
    url = api_base.rstrip("/") + "/v1/responses"
    payload = json.dumps(item.body).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    backoff = 1.0
    for attempt in range(max(1, int(max_retries))):
        req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
                return True, _result_line(custom_id=item.custom_id, model=item.model, body=body)
        except urllib.error.HTTPError as exc:
            raw = exc.read().decode("utf-8", errors="replace")
            try:
                parsed = json.loads(raw)
                message = (
                    parsed.get("error", {}).get("message")
                    if isinstance(parsed, dict)
                    else raw[:500]
                ) or raw[:500]
            except Exception:
                message = raw[:500]
            if attempt + 1 < max_retries and _should_retry(exc.code):
                time.sleep(backoff)
                backoff *= 2.0
                continue
            return False, _error_line(custom_id=item.custom_id, model=item.model, status_code=exc.code, message=message)
        except Exception as exc:  # noqa: BLE001
            if attempt + 1 < max_retries:
                time.sleep(backoff)
                backoff *= 2.0
                continue
            return False, _error_line(custom_id=item.custom_id, model=item.model, status_code=599, message=str(exc))
    return False, _error_line(custom_id=item.custom_id, model=item.model, status_code=599, message="responses request exhausted retries")


def main() -> None:
    args = _parse_args()
    request_dir = Path(args.request_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    env_file = Path(args.env_file).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    api_key = _resolve_openai_key(env_file)
    shard_filter = _parse_shard_indices(args.shard_indices)
    items = _collect_work(request_dir, shard_filter=shard_filter)

    existing_by_shard: dict[int, set[str]] = {}
    if args.resume:
        for item in items:
            existing_by_shard.setdefault(item.shard, set())
        for shard in sorted(existing_by_shard):
            existing_by_shard[shard] |= _read_existing_custom_ids(output_dir / f"openai_results_shard{shard:03d}.jsonl")
            if not args.retry_errors:
                existing_by_shard[shard] |= _read_existing_custom_ids(output_dir / f"openai_errors_shard{shard:03d}.jsonl")
        items = [item for item in items if item.custom_id not in existing_by_shard.get(item.shard, set())]

    result_handles: dict[int, Any] = {}
    error_handles: dict[int, Any] = {}
    try:
        for shard in sorted({item.shard for item in items}):
            result_handles[shard] = (output_dir / f"openai_results_shard{shard:03d}.jsonl").open("a", encoding="utf-8")
            error_handles[shard] = (output_dir / f"openai_errors_shard{shard:03d}.jsonl").open("a", encoding="utf-8")

        ok_count = 0
        error_count = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, int(args.max_concurrency))) as pool:
            future_map = {
                pool.submit(
                    _post_responses,
                    api_base=str(args.api_base),
                    api_key=api_key,
                    item=item,
                    timeout=float(args.timeout),
                    max_retries=int(args.max_retries),
                ): item
                for item in items
            }
            for future in concurrent.futures.as_completed(future_map):
                item = future_map[future]
                ok, line = future.result()
                handle = result_handles[item.shard] if ok else error_handles[item.shard]
                handle.write(json.dumps(line, ensure_ascii=False) + "\n")
                handle.flush()
                if ok:
                    ok_count += 1
                else:
                    error_count += 1
        summary = {
            "request_dir": str(request_dir),
            "output_dir": str(output_dir),
            "submitted_count": int(len(items)),
            "ok_count": int(ok_count),
            "error_count": int(error_count),
            "model_names": sorted({item.model for item in items}),
        }
        (output_dir / "sync_responses_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(summary, indent=2))
    finally:
        for handle in result_handles.values():
            handle.close()
        for handle in error_handles.values():
            handle.close()


if __name__ == "__main__":
    main()
