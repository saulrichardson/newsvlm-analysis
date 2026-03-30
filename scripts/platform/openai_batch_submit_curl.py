#!/usr/bin/env python3
"""
Submit OpenAI Batch jobs for shard JSONL files using curl (no OpenAI SDK dependency).

Expected request files:
  <request-dir>/openai_requests_shardNNN.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Submit OpenAI batch shards via curl.")
    ap.add_argument("--request-dir", required=True, help="Directory containing openai_requests_shard*.jsonl")
    ap.add_argument(
        "--record-path",
        default="",
        help="JSONL submission record path (default: <request-dir>/../openai_batch_submission/batch_jobs_<timestamp>.jsonl)",
    )
    ap.add_argument(
        "--env-file",
        default=".env",
        help="Path to .env file with OPENAI_API_KEY / OPENAI_KEY / CODEX_API_KEY",
    )
    ap.add_argument("--endpoint", default="/v1/responses", help="OpenAI batch endpoint")
    ap.add_argument("--completion-window", default="24h", help="Batch completion window")
    ap.add_argument("--run-label", default="", help="Optional run label stored in batch metadata")
    ap.add_argument("--model-label", default="", help="Optional model label stored in batch metadata")
    ap.add_argument("--reasoning-effort", default="", help="Optional reasoning-effort label stored in batch metadata")
    ap.add_argument("--sleep-seconds", type=float, default=1.0, help="Pause between submissions")
    ap.add_argument("--dry-run", action="store_true", help="Print planned submissions without API calls")
    return ap.parse_args()


def _parse_env_file(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.is_file():
        return out
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def _resolve_openai_key(env_file: Path) -> str:
    file_env = _parse_env_file(env_file)
    for name in ("OPENAI_API_KEY", "OPENAI_KEY", "CODEX_API_KEY"):
        val = os.environ.get(name) or file_env.get(name)
        if val:
            return val
    raise SystemExit(f"Missing OpenAI key in env or env-file: {env_file}")


def _write_auth_header_file(*, key: str, env_file: Path, record_path: Path) -> Path:
    """
    Write an Authorization header to a temp file so the key does NOT appear in `ps` output.

    We avoid passing `Authorization: Bearer <key>` on the curl command line, because that
    exposes the key to other users on multi-tenant machines via process listings.

    curl supports: `-H @/path/to/headers.txt` (see `curl --help all`).
    """
    # Prefer a secure location adjacent to the env file when possible (Torch: /scratch/$USER/.secrets).
    base_dir = env_file.parent
    try:
        base_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        base_dir = record_path.parent

    try:
        fd, tmp_path = tempfile.mkstemp(prefix="openai_headers_", suffix=".txt", dir=str(base_dir))
        path = Path(tmp_path)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(f"Authorization: Bearer {key}\n")
        os.chmod(path, 0o600)
        return path
    except Exception:
        # Last-resort: write to record dir (file perms still 600).
        fallback = record_path.parent / "openai_headers.txt"
        fallback.write_text(f"Authorization: Bearer {key}\n", encoding="utf-8")
        os.chmod(fallback, 0o600)
        return fallback


def _run_json(cmd: list[str]) -> dict[str, Any]:
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        safe = []
        for a in cmd:
            if isinstance(a, str) and "Authorization: Bearer " in a:
                safe.append("Authorization: Bearer [REDACTED]")
            else:
                safe.append(a)
        raise RuntimeError(f"Command failed ({p.returncode}): {' '.join(safe)}\n{p.stderr[:600]}")
    try:
        return json.loads(p.stdout)
    except Exception as exc:
        raise RuntimeError(f"Expected JSON response, got parse error: {exc}\n{p.stdout[:600]}") from exc


def _line_count(path: Path) -> int:
    out = subprocess.check_output(["wc", "-l", str(path)], text=True).strip()
    return int(out.split()[0])


def _load_existing_submitted(record_path: Path) -> set[str]:
    out: set[str] = set()
    if not record_path.is_file():
        return out
    for raw in record_path.read_text(encoding="utf-8").splitlines():
        s = raw.strip()
        if not s:
            continue
        try:
            obj = json.loads(s)
        except Exception:
            continue
        shard = str(obj.get("shard") or "").strip()
        batch_id = str(obj.get("batch_id") or "").strip()
        if shard and batch_id:
            out.add(shard)
    return out


def main() -> None:
    args = _parse_args()
    request_dir = Path(args.request_dir).expanduser().resolve()
    if not request_dir.is_dir():
        raise SystemExit(f"--request-dir is not a directory: {request_dir}")

    env_file = Path(args.env_file).expanduser().resolve()
    key = _resolve_openai_key(env_file)

    shards = sorted(request_dir.glob("openai_requests_shard*.jsonl"))
    if not shards:
        raise SystemExit(f"No openai request shards found in {request_dir}")

    if str(args.record_path).strip():
        record_path = Path(args.record_path).expanduser().resolve()
    else:
        sub_dir = request_dir.parent / "openai_batch_submission"
        sub_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        record_path = sub_dir / f"batch_jobs_{ts}.jsonl"
    record_path.parent.mkdir(parents=True, exist_ok=True)

    header_path = _write_auth_header_file(key=key, env_file=env_file, record_path=record_path)

    existing = _load_existing_submitted(record_path)
    submitted = 0
    skipped = 0

    for shard_path in shards:
        shard = shard_path.name
        if shard in existing:
            skipped += 1
            print(f"skip_already_submitted shard={shard}")
            continue

        bytes_size = shard_path.stat().st_size
        lines = _line_count(shard_path)
        print(f"submit_start shard={shard} lines={lines} bytes={bytes_size}")

        metadata = {"shard": shard}
        if str(args.run_label).strip():
            metadata["run"] = str(args.run_label).strip()
        if str(args.model_label).strip():
            metadata["model"] = str(args.model_label).strip()
        if str(args.reasoning_effort).strip():
            metadata["reasoning_effort"] = str(args.reasoning_effort).strip()

        if args.dry_run:
            print(f"dry_run shard={shard}")
            continue

        upload = _run_json(
            [
                "curl",
                "-sS",
                "https://api.openai.com/v1/files",
                "-H",
                f"@{str(header_path)}",
                "-F",
                "purpose=batch",
                "-F",
                f"file=@{str(shard_path)}",
            ]
        )
        file_id = str(upload.get("id") or "").strip()
        if not file_id:
            raise RuntimeError(f"Upload did not return file id for shard={shard}: {json.dumps(upload)[:600]}")

        batch_payload = {
            "input_file_id": file_id,
            "endpoint": str(args.endpoint),
            "completion_window": str(args.completion_window),
            "metadata": metadata,
        }
        batch = _run_json(
            [
                "curl",
                "-sS",
                "https://api.openai.com/v1/batches",
                "-H",
                f"@{str(header_path)}",
                "-H",
                "Content-Type: application/json",
                "-d",
                json.dumps(batch_payload, ensure_ascii=False),
            ]
        )
        batch_id = str(batch.get("id") or "").strip()
        status = str(batch.get("status") or "").strip()
        if not batch_id:
            raise RuntimeError(f"Batch create did not return id for shard={shard}: {json.dumps(batch)[:600]}")

        rec = {
            "shard": shard,
            "submitted": True,
            "file_id": file_id,
            "batch_id": batch_id,
            "status": status,
            "created_at": int(time.time()),
            "request_counts": {"total": 0, "completed": 0, "failed": 0},
            "requests_path": str(shard_path),
            "request_lines": int(lines),
            "request_bytes": int(bytes_size),
            "endpoint": str(args.endpoint),
            "completion_window": str(args.completion_window),
            "metadata": metadata,
        }
        with record_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        submitted += 1
        print(f"submit_ok shard={shard} batch_id={batch_id} status={status}")
        time.sleep(max(0.0, float(args.sleep_seconds)))

    print(
        json.dumps(
            {
                "record_path": str(record_path),
                "total_shards": len(shards),
                "submitted": submitted,
                "skipped": skipped,
            },
            ensure_ascii=False,
        )
    )

    # Best-effort cleanup (header file contains a live secret).
    try:
        header_path.unlink(missing_ok=True)  # py3.8+: ok
    except Exception:
        pass


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f"fatal: {exc}", file=sys.stderr)
        sys.exit(1)
