#!/usr/bin/env python3
"""
Monitor Gemini Batch jobs via REST and download completed output files.

Expected input:
  - record JSONL from scripts/platform/gemini_batch_submit.py

Outputs:
  - latest_status_snapshot.json under <run_root>/gemini_batch_submission/
  - completed_outputs/*.jsonl for succeeded batches with downloadable response files
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


FINAL_SUCCESS = {"BATCH_STATE_SUCCEEDED", "JOB_STATE_SUCCEEDED", "SUCCEEDED"}
FINAL_FAILURE = {
    "BATCH_STATE_FAILED",
    "FAILED",
    "JOB_STATE_FAILED",
    "BATCH_STATE_CANCELLED",
    "JOB_STATE_CANCELLED",
    "CANCELLED",
    "BATCH_STATE_EXPIRED",
    "JOB_STATE_EXPIRED",
    "EXPIRED",
}


def _eprint(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Monitor Gemini batch jobs and download completed outputs.")
    ap.add_argument("--record-path", required=True, help="JSONL submission record from gemini_batch_submit.py")
    ap.add_argument("--env-file", default="", help="Optional .env file with GEMINI_KEY / GEMINI_API_KEY")
    ap.add_argument("--poll-seconds", type=float, default=600.0, help="Seconds between polls")
    ap.add_argument("--stop-when-final", action="store_true", help="Exit when all jobs are final")
    ap.add_argument("--max-consecutive-errors", type=int, default=25, help="Abort after this many consecutive poll errors")
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


def _resolve_gemini_key(env_file: Path | None) -> str:
    for name in ("GEMINI_KEY", "GEMINI_API_KEY"):
        val = os.environ.get(name)
        if val:
            return str(val)
    if env_file is not None:
        file_env = _parse_env_file(env_file)
        for name in ("GEMINI_KEY", "GEMINI_API_KEY"):
            val = file_env.get(name)
            if val:
                return val
    raise SystemExit("Missing Gemini key. Set GEMINI_KEY in env or pass --env-file with GEMINI_KEY=...")


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True)


def _run_json(cmd: list[str]) -> dict[str, Any]:
    p = _run(cmd)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed ({p.returncode}): {' '.join(cmd)}\n{p.stderr[:800]}")
    try:
        return json.loads(p.stdout)
    except Exception as exc:
        raise RuntimeError(f"Expected JSON response, got parse error: {exc}\n{p.stdout[:1000]}") from exc


def _write_key_header_file(*, key: str, env_file: Path | None, record_path: Path) -> Path:
    base_dir = env_file.parent if env_file is not None else record_path.parent
    try:
        base_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        base_dir = record_path.parent
    fd, tmp_path = tempfile.mkstemp(prefix="gemini_headers_", suffix=".txt", dir=str(base_dir))
    path = Path(tmp_path)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(f"x-goog-api-key: {key}\n")
    os.chmod(path, 0o600)
    return path


def _load_records(record_path: Path) -> list[dict[str, Any]]:
    if not record_path.is_file():
        raise SystemExit(f"Record file not found: {record_path}")
    out: list[dict[str, Any]] = []
    for raw in record_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        batch_name = str(obj.get("batch_name") or "").strip()
        if not batch_name:
            continue
        out.append(obj)
    if not out:
        raise SystemExit(f"No batch records found in {record_path}")
    return out


def _extract_state(batch_obj: dict[str, Any]) -> str:
    value = batch_obj.get("state")
    if isinstance(value, str) and value.strip():
        return value.strip()
    metadata = batch_obj.get("metadata")
    if isinstance(metadata, dict):
        value = metadata.get("state")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _extract_response_file_name(batch_obj: dict[str, Any]) -> str:
    candidates: list[Any] = [
        batch_obj.get("dest"),
        batch_obj.get("destination"),
        batch_obj.get("response"),
        batch_obj.get("output"),
        batch_obj.get("result"),
    ]
    for candidate in candidates:
        if isinstance(candidate, dict):
            for key in ("fileName", "file_name", "responsesFile", "responses_file", "responseFile", "response_file"):
                value = candidate.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
                if isinstance(value, dict):
                    nested = value.get("fileName") or value.get("file_name")
                    if isinstance(nested, str) and nested.strip():
                        return nested.strip()
    for key in ("responsesFile", "responses_file", "responseFile", "response_file"):
        value = batch_obj.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, dict):
            nested = value.get("fileName") or value.get("file_name")
            if isinstance(nested, str) and nested.strip():
                return nested.strip()
    return ""


def _download_file(*, file_name: str, out_path: Path, header_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "curl",
        "-sS",
        f"https://generativelanguage.googleapis.com/download/v1beta/{file_name}:download?alt=media",
        "-H",
        f"@{str(header_path)}",
        "-o",
        str(out_path),
    ]
    p = _run(cmd)
    if p.returncode != 0:
        raise RuntimeError(f"Failed to download Gemini batch output {file_name}\n{p.stderr[:800]}")


def _fetch_batch(*, batch_name: str, header_path: Path) -> dict[str, Any]:
    return _run_json(
        [
            "curl",
            "-sS",
            f"https://generativelanguage.googleapis.com/v1beta/{batch_name}",
            "-H",
            f"@{str(header_path)}",
        ]
    )


def _status_bucket(state: str) -> str:
    if state in FINAL_SUCCESS:
        return "completed"
    if state in FINAL_FAILURE:
        return "failed"
    if not state:
        return "unknown"
    return "in_progress"


def main() -> None:
    args = _parse_args()
    record_path = Path(args.record_path).expanduser().resolve()
    env_file = Path(args.env_file).expanduser().resolve() if str(args.env_file).strip() else None
    key = _resolve_gemini_key(env_file)
    header_path = _write_key_header_file(key=key, env_file=env_file, record_path=record_path)
    records = _load_records(record_path)

    run_root = record_path.parent.parent if record_path.parent.name == "requests" else record_path.parent
    sub_dir = run_root / "gemini_batch_submission"
    completed_dir = sub_dir / "completed_outputs"
    sub_dir.mkdir(parents=True, exist_ok=True)
    completed_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = sub_dir / "latest_status_snapshot.json"

    poll_index = 0
    consecutive_errors = 0
    try:
        while True:
            poll_index += 1
            try:
                status_counts: dict[str, int] = {}
                batches_out: list[dict[str, Any]] = []
                downloaded_outputs = 0
                new_downloads = 0

                for rec in records:
                    batch_name = str(rec.get("batch_name") or "").strip()
                    batch_obj = _fetch_batch(batch_name=batch_name, header_path=header_path)
                    state = _extract_state(batch_obj)
                    bucket = _status_bucket(state)
                    status_counts[bucket] = status_counts.get(bucket, 0) + 1

                    response_file_name = _extract_response_file_name(batch_obj)
                    local_output_path = completed_dir / (batch_name.replace("/", "__") + "_output.jsonl")
                    downloaded = local_output_path.is_file()
                    if bucket == "completed" and response_file_name and not downloaded:
                        _download_file(file_name=response_file_name, out_path=local_output_path, header_path=header_path)
                        downloaded = True
                        new_downloads += 1
                    if downloaded:
                        downloaded_outputs += 1

                    batches_out.append(
                        {
                            "batch_name": batch_name,
                            "status": state,
                            "response_file_name": response_file_name,
                            "downloaded_output": downloaded,
                        }
                    )

                snapshot = {
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "poll_index": poll_index,
                    "status_counts": status_counts,
                    "downloaded_output_files": downloaded_outputs,
                    "new_downloads": new_downloads,
                    "batches": batches_out,
                }
                snapshot_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                print(json.dumps({"poll_index": poll_index, "status_counts": status_counts, "new_downloads": new_downloads}), flush=True)

                consecutive_errors = 0
                if args.stop_when_final and status_counts.get("in_progress", 0) == 0 and status_counts.get("unknown", 0) == 0:
                    break
                time.sleep(max(1.0, float(args.poll_seconds)))
            except Exception as exc:
                consecutive_errors += 1
                _eprint(f"poll_error consecutive={consecutive_errors}: {exc}")
                if consecutive_errors >= int(args.max_consecutive_errors):
                    raise
                time.sleep(min(60.0, max(5.0, float(args.poll_seconds) / 2.0)))
    finally:
        try:
            header_path.unlink()
        except Exception:
            pass


if __name__ == "__main__":
    main()
