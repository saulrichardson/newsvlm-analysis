#!/usr/bin/env python3
"""
Stage unresolved OpenAI Batch requests for the live gateway runner.

This script is stricter than `prepare_issue_classifier_gateway_live_run.py`:

  - completed batches are excluded
  - expired/cancelled/failed batches contribute only the custom_ids present in
    their error files
  - in-progress batches can either be skipped, cancelled and finalized so we can
    extract the exact unresolved custom_ids, or included whole (overlap risk)

The output layout is designed for:

  slurm/pipelines/run_issue_classifier_gateway_live_array.sbatch

and produces:

  <out>/full/requests/openai_requests_shardNNN.jsonl
  <out>/full/requests/mapping_shardNNN.jsonl
  <out>/full/workers/workerNNN/requests/{openai_requests,mapping}_shardNNN.jsonl
  <out>/full/results/
  <out>/manifest.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


FINAL_STATES = {"completed", "failed", "cancelled", "expired"}
FINAL_INCOMPLETE_STATES = {"failed", "cancelled", "expired"}
ACTIVE_STATES = {"validating", "in_progress", "finalizing", "cancelling"}


def _eprint(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


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
        value = os.environ.get(name) or file_env.get(name)
        if value:
            return value
    raise SystemExit(f"Missing OpenAI key in env or env-file: {env_file}")


def _request_json(
    method: str,
    url: str,
    *,
    api_key: str,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=90) as resp:
            return json.load(resp)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"{method} {url} failed: HTTP {exc.code}: {body[:800]}") from exc


def _fetch_batch(batch_id: str, *, api_key: str) -> dict[str, Any]:
    return _request_json("GET", f"https://api.openai.com/v1/batches/{batch_id}", api_key=api_key)


def _cancel_batch(batch_id: str, *, api_key: str) -> dict[str, Any]:
    return _request_json("POST", f"https://api.openai.com/v1/batches/{batch_id}/cancel", api_key=api_key, payload={})


def _download_file(*, file_id: str, dest_path: Path, api_key: str) -> None:
    req = urllib.request.Request(
        f"https://api.openai.com/v1/files/{file_id}/content",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        data = resp.read()
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    dest_path.write_bytes(data)


def _iter_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def _line_count(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle if _.strip())


def _batch_output_dir(run_root: Path) -> Path:
    return run_root / "openai_batch_submission" / "completed_outputs"


def _download_final_files_if_needed(
    *,
    run_root: Path,
    batch_id: str,
    output_file_id: str,
    error_file_id: str,
    api_key: str,
) -> tuple[Path | None, Path | None]:
    out_dir = _batch_output_dir(run_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{batch_id}_output.jsonl" if output_file_id else None
    err_path = out_dir / f"{batch_id}_error.jsonl" if error_file_id else None

    if out_path is not None and (not out_path.exists() or out_path.stat().st_size == 0):
        _download_file(file_id=output_file_id, dest_path=out_path, api_key=api_key)
    if err_path is not None and (not err_path.exists() or err_path.stat().st_size == 0):
        _download_file(file_id=error_file_id, dest_path=err_path, api_key=api_key)
    return out_path, err_path


def _read_error_custom_ids(path: Path | None) -> set[str]:
    out: set[str] = set()
    if path is None or not path.is_file():
        return out
    for obj in _iter_jsonl(path):
        cid = obj.get("custom_id")
        if isinstance(cid, str) and cid:
            out.add(cid)
    return out


def _read_output_custom_ids(path: Path | None) -> set[str]:
    out: set[str] = set()
    if path is None or not path.is_file():
        return out
    for obj in _iter_jsonl(path):
        cid = obj.get("custom_id")
        if isinstance(cid, str) and cid:
            out.add(cid)
    return out


@dataclass(frozen=True)
class _SelectedRequest:
    source_run_root: str
    source_run_label: str
    batch_id: str
    batch_status: str
    source_shard_name: str
    source_line_index: int
    custom_id: str
    request_line: str
    mapping_line: str
    selection_mode: str


def _select_whole_shard(
    *,
    requests_path: Path,
    mapping_path: Path,
    source_run_root: Path,
    source_run_label: str,
    batch_id: str,
    batch_status: str,
    source_shard_name: str,
    selection_mode: str,
) -> list[_SelectedRequest]:
    out: list[_SelectedRequest] = []
    with requests_path.open("r", encoding="utf-8") as req_f, mapping_path.open("r", encoding="utf-8") as map_f:
        for idx, (req_raw, map_raw) in enumerate(zip(req_f, map_f)):
            req_line = req_raw.strip()
            map_line = map_raw.strip()
            if not req_line:
                continue
            if not map_line:
                raise SystemExit(f"Blank mapping line at {mapping_path}:{idx + 1}")
            req_obj = json.loads(req_line)
            custom_id = str(req_obj.get("custom_id") or "").strip()
            if not custom_id:
                raise SystemExit(f"Missing custom_id at {requests_path}:{idx + 1}")
            out.append(
                _SelectedRequest(
                    source_run_root=str(source_run_root),
                    source_run_label=source_run_label,
                    batch_id=batch_id,
                    batch_status=batch_status,
                    source_shard_name=source_shard_name,
                    source_line_index=idx,
                    custom_id=custom_id,
                    request_line=req_line,
                    mapping_line=map_line,
                    selection_mode=selection_mode,
                )
            )
    return out


def _select_custom_ids(
    *,
    requests_path: Path,
    mapping_path: Path,
    source_run_root: Path,
    source_run_label: str,
    batch_id: str,
    batch_status: str,
    source_shard_name: str,
    target_custom_ids: set[str],
    selection_mode: str,
) -> list[_SelectedRequest]:
    out: list[_SelectedRequest] = []
    remaining = set(target_custom_ids)
    with requests_path.open("r", encoding="utf-8") as req_f, mapping_path.open("r", encoding="utf-8") as map_f:
        for idx, (req_raw, map_raw) in enumerate(zip(req_f, map_f)):
            req_line = req_raw.strip()
            map_line = map_raw.strip()
            if not req_line:
                continue
            if not map_line:
                raise SystemExit(f"Blank mapping line at {mapping_path}:{idx + 1}")
            req_obj = json.loads(req_line)
            custom_id = str(req_obj.get("custom_id") or "").strip()
            if not custom_id:
                raise SystemExit(f"Missing custom_id at {requests_path}:{idx + 1}")
            if custom_id not in remaining:
                continue
            out.append(
                _SelectedRequest(
                    source_run_root=str(source_run_root),
                    source_run_label=source_run_label,
                    batch_id=batch_id,
                    batch_status=batch_status,
                    source_shard_name=source_shard_name,
                    source_line_index=idx,
                    custom_id=custom_id,
                    request_line=req_line,
                    mapping_line=map_line,
                    selection_mode=selection_mode,
                )
            )
            remaining.remove(custom_id)
            if not remaining:
                break
    if remaining:
        preview = ", ".join(sorted(remaining)[:5])
        raise SystemExit(
            f"Did not find {len(remaining)} target custom_ids in {requests_path}; first few: {preview}"
        )
    return out


def _poll_to_final(batch_id: str, *, api_key: str, poll_seconds: float, max_wait_seconds: float) -> dict[str, Any]:
    deadline = time.time() + max_wait_seconds
    while True:
        batch = _fetch_batch(batch_id, api_key=api_key)
        status = str(batch.get("status") or "").strip()
        if status in FINAL_STATES:
            return batch
        if time.time() >= deadline:
            raise SystemExit(f"Timed out waiting for batch {batch_id} to reach a final state; last status={status}")
        time.sleep(poll_seconds)


def _stage_requests(
    *,
    output_dir: Path,
    items: list[_SelectedRequest],
    worker_count: int,
) -> dict[str, Any]:
    layout_root = output_dir / "full"
    requests_dir = layout_root / "requests"
    results_dir = layout_root / "results"
    workers_root = layout_root / "workers"
    requests_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    workers_root.mkdir(parents=True, exist_ok=True)

    if not items:
        raise SystemExit("No unresolved requests selected for staging.")

    actual_worker_count = min(max(1, int(worker_count)), len(items))
    base = len(items) // actual_worker_count
    remainder = len(items) % actual_worker_count
    target_counts = [base + (1 if idx < remainder else 0) for idx in range(actual_worker_count)]

    worker_summaries: list[dict[str, Any]] = []
    cursor = 0
    for worker_idx, target in enumerate(target_counts):
        request_path = requests_dir / f"openai_requests_shard{worker_idx:03d}.jsonl"
        mapping_path = requests_dir / f"mapping_shard{worker_idx:03d}.jsonl"
        shard_items = items[cursor : cursor + target]
        cursor += target
        with request_path.open("w", encoding="utf-8") as req_f, mapping_path.open("w", encoding="utf-8") as map_f:
            for item in shard_items:
                req_f.write(item.request_line + "\n")
                map_f.write(item.mapping_line + "\n")

        worker_dir = workers_root / f"worker{worker_idx:03d}"
        request_dir = worker_dir / "requests"
        request_dir.mkdir(parents=True, exist_ok=True)
        for src_path in (request_path, mapping_path):
            dst_path = request_dir / src_path.name
            if dst_path.exists() or dst_path.is_symlink():
                dst_path.unlink()
            os.symlink(src_path, dst_path)

        counts_by_mode: dict[str, int] = {}
        for item in shard_items:
            counts_by_mode[item.selection_mode] = counts_by_mode.get(item.selection_mode, 0) + 1

        worker_summaries.append(
            {
                "worker_index": worker_idx,
                "request_count": len(shard_items),
                "worker_dir": str(worker_dir),
                "request_dir": str(request_dir),
                "selection_mode_counts": counts_by_mode,
                "sample_custom_ids": [item.custom_id for item in shard_items[:3]],
            }
        )

    if cursor != len(items):
        raise SystemExit(f"Internal error: staged {cursor} requests, expected {len(items)}")

    return {
        "layout_root": str(layout_root),
        "requests_dir": str(requests_dir),
        "results_dir": str(results_dir),
        "worker_count": actual_worker_count,
        "workers": worker_summaries,
    }


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Stage unresolved OpenAI Batch requests for the live gateway runner.")
    ap.add_argument(
        "--run-root",
        action="append",
        required=True,
        help="Run root containing openai_batch_submission/batch_jobs.jsonl. Repeat for prod/inc/retry.",
    )
    ap.add_argument("--output-dir", required=True, help="Prepared output root.")
    ap.add_argument("--env-file", default=".env", help="Path to .env with OPENAI_API_KEY / OPENAI_KEY / CODEX_API_KEY.")
    ap.add_argument("--worker-count", type=int, default=100, help="How many worker shards to write.")
    ap.add_argument(
        "--in-progress-mode",
        choices=["skip", "cancel", "whole"],
        default="skip",
        help=(
            "How to handle in-progress batches: skip them, cancel and finalize to extract exact unresolved "
            "custom_ids, or include the whole shard with overlap risk."
        ),
    )
    ap.add_argument(
        "--cancel-poll-seconds",
        type=float,
        default=5.0,
        help="Polling interval after issuing an OpenAI Batch cancel.",
    )
    ap.add_argument(
        "--cancel-max-wait-seconds",
        type=float,
        default=900.0,
        help="Max seconds to wait for a cancelled batch to reach a final state.",
    )
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    env_file = Path(args.env_file).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists():
        raise SystemExit(f"--output-dir already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)

    api_key = _resolve_openai_key(env_file)
    run_roots = [Path(raw).expanduser().resolve() for raw in args.run_root]
    selected_items: list[_SelectedRequest] = []
    batch_summaries: list[dict[str, Any]] = []

    for run_root in run_roots:
        record_path = run_root / "openai_batch_submission" / "batch_jobs.jsonl"
        if not record_path.is_file():
            raise SystemExit(f"Missing batch_jobs.jsonl: {record_path}")
        for row in _iter_jsonl(record_path):
            batch_id = str(row.get("batch_id") or "").strip()
            if not batch_id:
                raise SystemExit(f"Missing batch_id in {record_path}")
            requests_path = Path(str(row.get("requests_path") or "")).expanduser().resolve()
            if not requests_path.is_file():
                raise SystemExit(f"Missing requests_path for batch {batch_id}: {requests_path}")
            mapping_path = requests_path.with_name(requests_path.name.replace("openai_requests_", "mapping_"))
            if not mapping_path.is_file():
                raise SystemExit(f"Missing mapping shard for {requests_path}: {mapping_path}")

            batch = _fetch_batch(batch_id, api_key=api_key)
            status = str(batch.get("status") or "").strip()
            request_counts = dict(batch.get("request_counts") or {})
            total = int(request_counts.get("total") or 0)
            completed = int(request_counts.get("completed") or 0)
            failed = int(request_counts.get("failed") or 0)
            expected_missing = total - completed - failed
            selection_mode = ""
            selected_count = 0

            if status == "completed":
                batch_summaries.append(
                    {
                        "run_root": str(run_root),
                        "shard": str(row.get("shard") or requests_path.name),
                        "batch_id": batch_id,
                        "status": status,
                        "selected_requests": 0,
                        "selection_mode": "completed_skip",
                        "request_counts": request_counts,
                    }
                )
                continue

            if status in ACTIVE_STATES:
                if args.in_progress_mode == "skip":
                    batch_summaries.append(
                        {
                            "run_root": str(run_root),
                            "shard": str(row.get("shard") or requests_path.name),
                            "batch_id": batch_id,
                            "status": status,
                            "selected_requests": 0,
                            "selection_mode": "in_progress_skip",
                            "request_counts": request_counts,
                        }
                    )
                    continue
                if args.in_progress_mode == "whole":
                    selection_mode = f"{status}_whole_shard"
                    shard_items = _select_whole_shard(
                        requests_path=requests_path,
                        mapping_path=mapping_path,
                        source_run_root=run_root,
                        source_run_label=run_root.name,
                        batch_id=batch_id,
                        batch_status=status,
                        source_shard_name=str(row.get("shard") or requests_path.name),
                        selection_mode=selection_mode,
                    )
                    selected_items.extend(shard_items)
                    selected_count = len(shard_items)
                    batch_summaries.append(
                        {
                            "run_root": str(run_root),
                            "shard": str(row.get("shard") or requests_path.name),
                            "batch_id": batch_id,
                            "status": status,
                            "selected_requests": selected_count,
                            "selection_mode": selection_mode,
                            "request_counts": request_counts,
                        }
                    )
                    continue
                else:
                    _eprint(f"Cancelling in-progress batch {batch_id} ({row.get('shard')})")
                    _cancel_batch(batch_id, api_key=api_key)
                    batch = _poll_to_final(
                        batch_id,
                        api_key=api_key,
                        poll_seconds=float(args.cancel_poll_seconds),
                        max_wait_seconds=float(args.cancel_max_wait_seconds),
                    )
                    status = str(batch.get("status") or "").strip()
                    if status not in FINAL_STATES:
                        raise SystemExit(f"Batch {batch_id} did not reach final state after cancel: {status}")

            if status in FINAL_INCOMPLETE_STATES:
                output_file_id = str(batch.get("output_file_id") or "").strip()
                error_file_id = str(batch.get("error_file_id") or "").strip()
                out_path, err_path = _download_final_files_if_needed(
                    run_root=run_root,
                    batch_id=batch_id,
                    output_file_id=output_file_id,
                    error_file_id=error_file_id,
                    api_key=api_key,
                )
                output_ids = _read_output_custom_ids(out_path)
                error_ids = _read_error_custom_ids(err_path)
                if expected_missing != len(error_ids):
                    raise SystemExit(
                        f"Batch {batch_id} expected {expected_missing} unresolved requests from API counts, "
                        f"but error file has {len(error_ids)} rows."
                    )
                if total != len(output_ids) + len(error_ids):
                    raise SystemExit(
                        f"Batch {batch_id} accounting mismatch: total={total}, "
                        f"output_rows={len(output_ids)}, error_rows={len(error_ids)}"
                    )
                if not error_ids:
                    batch_summaries.append(
                        {
                            "run_root": str(run_root),
                            "shard": str(row.get("shard") or requests_path.name),
                            "batch_id": batch_id,
                            "status": status,
                            "selected_requests": 0,
                            "selection_mode": "final_incomplete_zero_error_rows",
                            "request_counts": dict(batch.get("request_counts") or {}),
                        }
                    )
                    continue
                selection_mode = f"{status}_error_file"
                shard_items = _select_custom_ids(
                    requests_path=requests_path,
                    mapping_path=mapping_path,
                    source_run_root=run_root,
                    source_run_label=run_root.name,
                    batch_id=batch_id,
                    batch_status=status,
                    source_shard_name=str(row.get("shard") or requests_path.name),
                    target_custom_ids=error_ids,
                    selection_mode=selection_mode,
                )
                selected_items.extend(shard_items)
                selected_count = len(shard_items)
            elif status != "completed":
                raise SystemExit(f"Unhandled batch status for {batch_id}: {status}")

            batch_summaries.append(
                {
                    "run_root": str(run_root),
                    "shard": str(row.get("shard") or requests_path.name),
                    "batch_id": batch_id,
                    "status": status,
                    "selected_requests": selected_count,
                    "selection_mode": selection_mode,
                    "request_counts": dict(batch.get("request_counts") or {}),
                }
            )

    selected_items.sort(key=lambda item: (item.source_run_label, item.source_shard_name, item.source_line_index))
    staged = _stage_requests(output_dir=output_dir, items=selected_items, worker_count=int(args.worker_count))

    by_mode: dict[str, int] = {}
    by_run: dict[str, int] = {}
    for item in selected_items:
        by_mode[item.selection_mode] = by_mode.get(item.selection_mode, 0) + 1
        by_run[item.source_run_label] = by_run.get(item.source_run_label, 0) + 1

    manifest = {
        "prepared_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "output_dir": str(output_dir),
        "run_roots": [str(path) for path in run_roots],
        "in_progress_mode": str(args.in_progress_mode),
        "worker_count_requested": int(args.worker_count),
        "selected_request_count": len(selected_items),
        "selected_request_count_by_run": by_run,
        "selected_request_count_by_mode": by_mode,
        "batches": batch_summaries,
        "full": staged,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
