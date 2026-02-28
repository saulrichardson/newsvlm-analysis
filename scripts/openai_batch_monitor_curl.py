#!/usr/bin/env python3
"""
Poll OpenAI Batch jobs and download completed output/error files using curl.

Input:
  - JSONL submission record from openai_batch_submit_curl.py
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


FINAL_STATES = {"completed", "failed", "cancelled", "expired"}


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Monitor OpenAI batch jobs and download outputs.")
    ap.add_argument("--record-path", required=True, help="Submission record JSONL")
    ap.add_argument("--env-file", default=".env", help="Path to .env with OpenAI key")
    ap.add_argument(
        "--output-dir",
        default="",
        help="Directory for downloaded outputs (default: <record-dir>/completed_outputs)",
    )
    ap.add_argument("--poll-seconds", type=float, default=120.0, help="Polling interval")
    ap.add_argument("--max-polls", type=int, default=0, help="Stop after N polls (0 = no limit)")
    ap.add_argument("--snapshot-path", default="", help="Status snapshot path")
    ap.add_argument("--poll-dir", default="", help="Directory to archive each poll response")
    ap.add_argument("--stop-when-final", action="store_true", help="Exit once all batches are in final states")
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


def _run_json(cmd: list[str]) -> dict[str, Any]:
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed ({p.returncode}): {' '.join(cmd)}\n{p.stderr[:600]}")
    try:
        return json.loads(p.stdout)
    except Exception as exc:
        raise RuntimeError(f"Expected JSON response, got parse error: {exc}\n{p.stdout[:600]}") from exc


def _download_file_content(key: str, file_id: str, out_path: Path) -> int:
    p = subprocess.run(
        [
            "curl",
            "-sS",
            f"https://api.openai.com/v1/files/{file_id}/content",
            "-H",
            f"Authorization: Bearer {key}",
        ],
        capture_output=True,
    )
    if p.returncode != 0:
        raise RuntimeError(f"Failed downloading file_id={file_id}: {p.stderr.decode('utf-8', errors='ignore')[:500]}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(p.stdout)
    return len(p.stdout)


def _load_batch_ids(record_path: Path) -> list[str]:
    if not record_path.is_file():
        raise SystemExit(f"Record path not found: {record_path}")
    ids: list[str] = []
    for raw in record_path.read_text(encoding="utf-8").splitlines():
        s = raw.strip()
        if not s:
            continue
        try:
            obj = json.loads(s)
        except Exception:
            continue
        bid = str(obj.get("batch_id") or "").strip()
        if bid:
            ids.append(bid)
    if not ids:
        raise SystemExit(f"No batch_id rows found in {record_path}")
    return ids


def main() -> None:
    args = _parse_args()
    record_path = Path(args.record_path).expanduser().resolve()
    env_file = Path(args.env_file).expanduser().resolve()
    key = _resolve_openai_key(env_file)
    batch_ids = _load_batch_ids(record_path)

    out_dir = (
        Path(args.output_dir).expanduser().resolve()
        if str(args.output_dir).strip()
        else (record_path.parent / "completed_outputs")
    )
    poll_dir = (
        Path(args.poll_dir).expanduser().resolve()
        if str(args.poll_dir).strip()
        else (record_path.parent / "polls")
    )
    snapshot_path = (
        Path(args.snapshot_path).expanduser().resolve()
        if str(args.snapshot_path).strip()
        else (record_path.parent / "latest_status_snapshot.json")
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    poll_dir.mkdir(parents=True, exist_ok=True)

    poll_idx = 0
    while True:
        poll_idx += 1
        rows: list[dict[str, Any]] = []
        for bid in batch_ids:
            obj = _run_json(
                [
                    "curl",
                    "-sS",
                    f"https://api.openai.com/v1/batches/{bid}",
                    "-H",
                    f"Authorization: Bearer {key}",
                ]
            )
            rows.append(obj)

        downloaded = 0
        for obj in rows:
            bid = str(obj.get("id") or "").strip()
            if not bid:
                continue
            if str(obj.get("status") or "") != "completed":
                continue
            out_fid = str(obj.get("output_file_id") or "").strip()
            err_fid = str(obj.get("error_file_id") or "").strip()

            if out_fid:
                out_path = out_dir / f"{bid}_output.jsonl"
                if not out_path.exists() or out_path.stat().st_size == 0:
                    _download_file_content(key, out_fid, out_path)
                    downloaded += 1
            if err_fid:
                err_path = out_dir / f"{bid}_error.jsonl"
                if not err_path.exists() or err_path.stat().st_size == 0:
                    _download_file_content(key, err_fid, err_path)
                    downloaded += 1

        status_counts: dict[str, int] = {}
        completed_requests = 0
        failed_requests = 0
        total_requests = 0
        for obj in rows:
            st = str(obj.get("status") or "unknown")
            status_counts[st] = status_counts.get(st, 0) + 1
            rc = obj.get("request_counts") or {}
            try:
                completed_requests += int(rc.get("completed") or 0)
                failed_requests += int(rc.get("failed") or 0)
                total_requests += int(rc.get("total") or 0)
            except Exception:
                pass

        snapshot = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "poll_index": poll_idx,
            "status_counts": status_counts,
            "completed_requests": completed_requests,
            "failed_requests": failed_requests,
            "total_requests": total_requests,
            "completion_pct": (100.0 * completed_requests / total_requests) if total_requests else 0.0,
            "downloaded_output_files": len(list(out_dir.glob("*_output.jsonl"))),
            "downloaded_error_files": len(list(out_dir.glob("*_error.jsonl"))),
            "new_downloads": downloaded,
            "batches": [
                {
                    "batch_id": str(o.get("id") or ""),
                    "status": str(o.get("status") or ""),
                    "request_counts": o.get("request_counts"),
                    "output_file_id": str(o.get("output_file_id") or ""),
                    "error_file_id": str(o.get("error_file_id") or ""),
                }
                for o in rows
            ],
        }
        snapshot_path.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        poll_stamp = time.strftime("%Y%m%d_%H%M%S")
        (poll_dir / f"poll_{poll_stamp}.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps({"poll_index": poll_idx, "status_counts": status_counts, "completion_pct": snapshot["completion_pct"]}, ensure_ascii=False))

        if args.stop_when_final and rows and all((str(o.get("status") or "") in FINAL_STATES) for o in rows):
            break
        if int(args.max_polls) > 0 and poll_idx >= int(args.max_polls):
            break
        time.sleep(max(1.0, float(args.poll_seconds)))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f"fatal: {exc}", file=sys.stderr)
        sys.exit(1)
