#!/usr/bin/env python3
"""
Prepare staged live gateway reruns for unresolved OpenAI issue-classifier shards.

This script queries the OpenAI Batch API directly, selects all shards that are
not yet completed, and stages:

  - a full live-run layout with normalized shard IDs and worker partitions
  - a smaller smoke-test layout with a bounded number of requests

The staged directories are designed for `scripts/platform/run_openai_requests_via_gateway.py`.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


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


def _fetch_batch(batch_id: str, *, api_key: str) -> dict[str, Any]:
    req = urllib.request.Request(
        f"https://api.openai.com/v1/batches/{batch_id}",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.load(resp)


def _iter_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def _copy_first_n_jsonl_lines(src_path: Path, dst_path: Path, *, n_lines: int) -> int:
    if n_lines <= 0:
        raise SystemExit(f"n_lines must be > 0 for {src_path}")
    wrote = 0
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with src_path.open("r", encoding="utf-8") as src, dst_path.open("w", encoding="utf-8") as dst:
        for raw in src:
            if not raw.strip():
                continue
            dst.write(raw.rstrip("\n") + "\n")
            wrote += 1
            if wrote >= n_lines:
                break
    if wrote != n_lines:
        raise SystemExit(f"Requested {n_lines} lines from {src_path}, but only found {wrote}")
    return wrote


def _copy_jsonl_slice(src_path: Path, dst_path: Path, *, start_line: int, n_lines: int) -> int:
    if start_line < 0:
        raise SystemExit(f"start_line must be >= 0 for {src_path}")
    if n_lines <= 0:
        raise SystemExit(f"n_lines must be > 0 for {src_path}")
    wrote = 0
    seen = 0
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with src_path.open("r", encoding="utf-8") as src, dst_path.open("w", encoding="utf-8") as dst:
        for raw in src:
            if not raw.strip():
                continue
            if seen < start_line:
                seen += 1
                continue
            dst.write(raw.rstrip("\n") + "\n")
            wrote += 1
            seen += 1
            if wrote >= n_lines:
                break
    if wrote != n_lines:
        raise SystemExit(
            f"Requested slice start={start_line} n_lines={n_lines} from {src_path}, but only found {wrote}"
        )
    return wrote


def _link_or_copy(src: Path, dst: Path, *, mode: str) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copy2(src, dst)
        return
    os.symlink(src, dst)


def _worker_name(idx: int) -> str:
    return f"worker{idx:03d}"


@dataclass(frozen=True)
class _SelectedShard:
    source_run_root: Path
    source_run_label: str
    source_shard_name: str
    source_requests_path: Path
    source_mapping_path: Path
    request_lines: int
    batch_id: str
    batch_status: str
    batch_request_counts: dict[str, Any]


@dataclass(frozen=True)
class _StagedShard:
    staged_shard_index: int
    source_run_root: Path
    source_run_label: str
    source_shard_name: str
    source_requests_path: Path
    source_mapping_path: Path
    requests_path: Path
    mapping_path: Path
    request_lines: int
    batch_id: str
    batch_status: str
    batch_request_counts: dict[str, Any]


def _load_selected_shards(*, run_root: Path, api_key: str) -> list[_SelectedShard]:
    record_path = run_root / "openai_batch_submission" / "batch_jobs.jsonl"
    if not record_path.is_file():
        raise SystemExit(f"Missing batch record: {record_path}")
    rows = _iter_jsonl(record_path)
    if not rows:
        raise SystemExit(f"No rows in batch record: {record_path}")

    selected: list[_SelectedShard] = []
    for row in rows:
        batch_id = str(row.get("batch_id") or "").strip()
        requests_path = Path(str(row.get("requests_path") or "")).expanduser().resolve()
        if not batch_id:
            raise SystemExit(f"Missing batch_id in {record_path}")
        if not requests_path.is_file():
            raise SystemExit(f"Missing requests_path for batch {batch_id}: {requests_path}")
        mapping_name = requests_path.name.replace("openai_requests_", "mapping_")
        mapping_path = requests_path.with_name(mapping_name)
        if not mapping_path.is_file():
            raise SystemExit(f"Missing mapping shard for {requests_path}: {mapping_path}")

        batch_obj = _fetch_batch(batch_id, api_key=api_key)
        status = str(batch_obj.get("status") or "").strip()
        if status == "completed":
            continue

        request_lines = int(row.get("request_lines") or 0)
        if request_lines <= 0:
            raise SystemExit(f"Invalid request_lines for {requests_path}")

        selected.append(
            _SelectedShard(
                source_run_root=run_root,
                source_run_label=run_root.name,
                source_shard_name=str(row.get("shard") or requests_path.name),
                source_requests_path=requests_path,
                source_mapping_path=mapping_path,
                request_lines=request_lines,
                batch_id=batch_id,
                batch_status=status,
                batch_request_counts=dict(batch_obj.get("request_counts") or {}),
            )
        )

    return sorted(selected, key=lambda x: (x.source_run_label, x.source_shard_name))


def _assign_workers(staged_shards: list[_StagedShard], *, worker_count: int) -> dict[int, list[_StagedShard]]:
    if worker_count <= 0:
        raise SystemExit("worker_count must be > 0")
    buckets: dict[int, list[_StagedShard]] = {idx: [] for idx in range(worker_count)}
    loads: dict[int, int] = {idx: 0 for idx in range(worker_count)}
    for shard in sorted(staged_shards, key=lambda x: (-x.request_lines, x.staged_shard_index)):
        worker_idx = min(loads, key=lambda idx: (loads[idx], idx))
        buckets[worker_idx].append(shard)
        loads[worker_idx] += shard.request_lines
    return buckets


def _stage_worker_dirs(*, layout_root: Path, worker_assignments: dict[int, list[_StagedShard]]) -> list[dict[str, Any]]:
    workers_root = layout_root / "workers"
    workers_root.mkdir(parents=True, exist_ok=True)
    out: list[dict[str, Any]] = []
    for worker_idx, shards in sorted(worker_assignments.items()):
        worker_dir = workers_root / _worker_name(worker_idx)
        request_dir = worker_dir / "requests"
        request_dir.mkdir(parents=True, exist_ok=True)
        for shard in sorted(shards, key=lambda x: x.staged_shard_index):
            for src_path in (shard.requests_path, shard.mapping_path):
                dst_path = request_dir / src_path.name
                if dst_path.exists() or dst_path.is_symlink():
                    dst_path.unlink()
                os.symlink(src_path, dst_path)
        out.append(
            {
                "worker_index": worker_idx,
                "worker_dir": str(worker_dir),
                "request_dir": str(request_dir),
                "request_count": sum(s.request_lines for s in shards),
                "staged_shards": [s.staged_shard_index for s in sorted(shards, key=lambda x: x.staged_shard_index)],
            }
        )
    return out


def _stage_full_layout(
    *,
    output_dir: Path,
    selected: list[_SelectedShard],
    worker_count: int,
    stage_mode: str,
) -> dict[str, Any]:
    layout_root = output_dir / "full"
    requests_dir = layout_root / "requests"
    results_dir = layout_root / "results"
    requests_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    staged: list[_StagedShard] = []
    for staged_idx, sel in enumerate(selected):
        request_dst = requests_dir / f"openai_requests_shard{staged_idx:03d}.jsonl"
        mapping_dst = requests_dir / f"mapping_shard{staged_idx:03d}.jsonl"
        _link_or_copy(sel.source_requests_path, request_dst, mode=stage_mode)
        _link_or_copy(sel.source_mapping_path, mapping_dst, mode=stage_mode)
        staged.append(
            _StagedShard(
                staged_shard_index=staged_idx,
                source_run_root=sel.source_run_root,
                source_run_label=sel.source_run_label,
                source_shard_name=sel.source_shard_name,
                source_requests_path=sel.source_requests_path,
                source_mapping_path=sel.source_mapping_path,
                requests_path=request_dst,
                mapping_path=mapping_dst,
                request_lines=sel.request_lines,
                batch_id=sel.batch_id,
                batch_status=sel.batch_status,
                batch_request_counts=sel.batch_request_counts,
            )
        )

    assignments = _assign_workers(staged, worker_count=worker_count)
    workers = _stage_worker_dirs(layout_root=layout_root, worker_assignments=assignments)
    return {
        "layout_root": str(layout_root),
        "requests_dir": str(requests_dir),
        "results_dir": str(results_dir),
        "worker_count": worker_count,
        "workers": workers,
        "staged_shards": [
            {
                "staged_shard_index": s.staged_shard_index,
                "request_lines": s.request_lines,
                "source_run_root": str(s.source_run_root),
                "source_run_label": s.source_run_label,
                "source_shard_name": s.source_shard_name,
                "source_requests_path": str(s.source_requests_path),
                "source_mapping_path": str(s.source_mapping_path),
                "batch_id": s.batch_id,
                "batch_status": s.batch_status,
                "batch_request_counts": s.batch_request_counts,
            }
            for s in staged
        ],
    }


def _stage_partitioned_layout(
    *,
    output_dir: Path,
    layout_name: str,
    selected: list[_SelectedShard],
    request_budget_by_run: dict[str, int],
    worker_count: int,
) -> dict[str, Any]:
    layout_root = output_dir / layout_name
    requests_dir = layout_root / "requests"
    results_dir = layout_root / "results"
    workers_root = layout_root / "workers"
    requests_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    workers_root.mkdir(parents=True, exist_ok=True)

    total_requests = sum(max(0, int(v)) for v in request_budget_by_run.values())
    if total_requests <= 0:
        raise SystemExit(f"{layout_name} layout would be empty")

    actual_worker_count = min(max(1, int(worker_count)), total_requests)
    base = total_requests // actual_worker_count
    remainder = total_requests % actual_worker_count
    target_counts = [base + (1 if idx < remainder else 0) for idx in range(actual_worker_count)]

    request_paths = [requests_dir / f"openai_requests_shard{idx:03d}.jsonl" for idx in range(actual_worker_count)]
    mapping_paths = [requests_dir / f"mapping_shard{idx:03d}.jsonl" for idx in range(actual_worker_count)]
    request_files = [path.open("w", encoding="utf-8") for path in request_paths]
    mapping_files = [path.open("w", encoding="utf-8") for path in mapping_paths]
    worker_summaries: list[dict[str, Any]] = [
        {
            "worker_index": idx,
            "worker_dir": str(workers_root / _worker_name(idx)),
            "request_dir": str((workers_root / _worker_name(idx) / "requests")),
            "request_count": 0,
            "staged_shards": [idx],
            "source_fragments": [],
        }
        for idx in range(actual_worker_count)
    ]
    staged_summaries: list[dict[str, Any]] = [
        {
            "staged_shard_index": idx,
            "request_lines": 0,
            "source_fragments": [],
        }
        for idx in range(actual_worker_count)
    ]

    worker_idx = 0
    worker_written = 0
    line_limits = {key: max(0, int(val)) for key, val in request_budget_by_run.items()}
    total_written = 0

    def _record_fragment(*, summary: dict[str, Any], fragment: dict[str, Any]) -> None:
        fragments = summary["source_fragments"]
        if fragments:
            last = fragments[-1]
            contiguous = int(last.get("source_line_start", 0)) + int(last.get("line_count", 0)) == int(
                fragment.get("source_line_start", 0)
            )
            if (
                last.get("source_run_label") == fragment.get("source_run_label")
                and last.get("source_shard_name") == fragment.get("source_shard_name")
                and contiguous
            ):
                last["line_count"] += int(fragment["line_count"])
                return
        fragments.append(fragment)

    try:
        for sel in selected:
            remaining_for_run = line_limits.get(sel.source_run_label, 0)
            if remaining_for_run <= 0:
                continue
            take = min(sel.request_lines, remaining_for_run)
            if take <= 0:
                continue

            with sel.source_requests_path.open("r", encoding="utf-8") as req_f, sel.source_mapping_path.open(
                "r", encoding="utf-8"
            ) as map_f:
                for source_line_idx, (req_raw, map_raw) in enumerate(zip(req_f, map_f)):
                    if source_line_idx >= take:
                        break
                    if not req_raw.strip() or not map_raw.strip():
                        raise SystemExit(
                            f"Encountered blank JSONL line while partitioning {sel.source_requests_path} / {sel.source_mapping_path}"
                        )
                    while worker_idx < actual_worker_count and worker_written >= target_counts[worker_idx]:
                        worker_idx += 1
                        worker_written = 0
                    if worker_idx >= actual_worker_count:
                        raise SystemExit(f"Internal error: exhausted worker slots while staging {layout_name}")

                    request_files[worker_idx].write(req_raw.rstrip("\n") + "\n")
                    mapping_files[worker_idx].write(map_raw.rstrip("\n") + "\n")
                    worker_summaries[worker_idx]["request_count"] += 1
                    staged_summaries[worker_idx]["request_lines"] += 1
                    fragment = {
                        "source_run_label": sel.source_run_label,
                        "source_shard_name": sel.source_shard_name,
                        "source_line_start": source_line_idx,
                        "line_count": 1,
                    }
                    _record_fragment(summary=worker_summaries[worker_idx], fragment=dict(fragment))
                    _record_fragment(summary=staged_summaries[worker_idx], fragment=dict(fragment))
                    worker_written += 1
                    total_written += 1

            line_limits[sel.source_run_label] = remaining_for_run - take
    finally:
        for f in request_files:
            f.close()
        for f in mapping_files:
            f.close()

    if total_written != total_requests:
        raise SystemExit(f"Partitioned {total_written} requests for {layout_name}, expected {total_requests}")

    for idx, target in enumerate(target_counts):
        if worker_summaries[idx]["request_count"] != target:
            raise SystemExit(
                f"Worker {idx} in {layout_name} wrote {worker_summaries[idx]['request_count']} requests, expected {target}"
            )

        worker_dir = workers_root / _worker_name(idx)
        request_dir = worker_dir / "requests"
        request_dir.mkdir(parents=True, exist_ok=True)
        for src_path in (request_paths[idx], mapping_paths[idx]):
            dst_path = request_dir / src_path.name
            if dst_path.exists() or dst_path.is_symlink():
                dst_path.unlink()
            os.symlink(src_path, dst_path)

    return {
        "layout_root": str(layout_root),
        "requests_dir": str(requests_dir),
        "results_dir": str(results_dir),
        "worker_count": actual_worker_count,
        "workers": worker_summaries,
        "staged_shards": staged_summaries,
        "actual_requests": total_written,
        "request_budget_by_run": request_budget_by_run,
    }


def _smoke_run_quotas(selected: list[_SelectedShard], *, total_requests: int) -> dict[str, int]:
    if total_requests <= 0:
        return {}
    grouped: dict[str, int] = {}
    for item in selected:
        grouped[item.source_run_label] = grouped.get(item.source_run_label, 0) + item.request_lines
    run_labels = sorted(grouped)
    if not run_labels:
        return {}

    quotas: dict[str, int] = {label: 0 for label in run_labels}
    remaining = total_requests
    base = total_requests // len(run_labels)
    remainder = total_requests % len(run_labels)

    for idx, label in enumerate(run_labels):
        target = base + (1 if idx < remainder else 0)
        target = min(target, grouped[label])
        quotas[label] = target
        remaining -= target

    while remaining > 0:
        made_progress = False
        for label in run_labels:
            available = grouped[label] - quotas[label]
            if available <= 0:
                continue
            quotas[label] += 1
            remaining -= 1
            made_progress = True
            if remaining <= 0:
                break
        if not made_progress:
            break

    return quotas


def _stage_smoke_layout(
    *,
    output_dir: Path,
    selected: list[_SelectedShard],
    smoke_requests: int,
    worker_count: int,
) -> dict[str, Any]:
    quotas = _smoke_run_quotas(selected, total_requests=smoke_requests)
    staged = _stage_partitioned_layout(
        output_dir=output_dir,
        layout_name="smoke",
        selected=selected,
        request_budget_by_run=quotas,
        worker_count=worker_count,
    )
    staged["requested_smoke_requests"] = smoke_requests
    staged["actual_smoke_requests"] = staged["actual_requests"]
    staged["quotas_by_run"] = quotas
    return staged


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Prepare staged live gateway reruns for unresolved OpenAI shards.")
    ap.add_argument(
        "--run-root",
        action="append",
        required=True,
        help="Issue-classifier run root containing openai_batch_submission/batch_jobs.jsonl. Repeat for prod/inc/retry.",
    )
    ap.add_argument("--output-dir", required=True, help="Prepared live-run output root.")
    ap.add_argument("--env-file", default=".env", help="Path to .env with OPENAI_API_KEY / OPENAI_KEY / CODEX_API_KEY.")
    ap.add_argument(
        "--full-worker-count",
        type=int,
        default=0,
        help="Worker count for the full rerun (0 = one worker per unresolved shard).",
    )
    ap.add_argument(
        "--full-layout-mode",
        choices=["shard", "repack"],
        default="shard",
        help="How to stage the full rerun layout.",
    )
    ap.add_argument(
        "--smoke-requests",
        type=int,
        default=200,
        help="How many requests to stage in the smoke subset (0 disables smoke staging).",
    )
    ap.add_argument(
        "--smoke-worker-count",
        type=int,
        default=4,
        help="Worker count for the smoke subset.",
    )
    ap.add_argument(
        "--stage-mode",
        choices=["symlink", "copy"],
        default="symlink",
        help="How to stage full-run request shards into the prepared requests dir.",
    )
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    env_file = Path(args.env_file).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    api_key = _resolve_openai_key(env_file)
    run_roots = [Path(x).expanduser().resolve() for x in args.run_root]
    selected: list[_SelectedShard] = []
    for run_root in run_roots:
        selected.extend(_load_selected_shards(run_root=run_root, api_key=api_key))

    if not selected:
        raise SystemExit("No unresolved OpenAI shards found.")

    full_worker_count = int(args.full_worker_count) if int(args.full_worker_count) > 0 else len(selected)
    if str(args.full_layout_mode) == "shard":
        full_worker_count = min(full_worker_count, len(selected))

    manifest: dict[str, Any] = {
        "prepared_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "output_dir": str(output_dir),
        "run_roots": [str(p) for p in run_roots],
        "total_selected_shards": len(selected),
        "total_selected_requests": sum(s.request_lines for s in selected),
        "source_runs": sorted({s.source_run_label for s in selected}),
        "full_layout_mode": str(args.full_layout_mode),
    }
    if str(args.full_layout_mode) == "repack":
        manifest["full"] = _stage_partitioned_layout(
            output_dir=output_dir,
            layout_name="full",
            selected=selected,
            request_budget_by_run={label: sum(s.request_lines for s in selected if s.source_run_label == label) for label in sorted({s.source_run_label for s in selected})},
            worker_count=max(1, int(full_worker_count)),
        )
    else:
        manifest["full"] = _stage_full_layout(
            output_dir=output_dir,
            selected=selected,
            worker_count=full_worker_count,
            stage_mode=str(args.stage_mode),
        )

    smoke_requests = max(0, int(args.smoke_requests))
    if smoke_requests > 0:
        manifest["smoke"] = _stage_smoke_layout(
            output_dir=output_dir,
            selected=selected,
            smoke_requests=smoke_requests,
            worker_count=max(1, int(args.smoke_worker_count)),
        )
    else:
        manifest["smoke"] = None

    manifest_path = output_dir / "gateway_live_prepare_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"manifest_path": str(manifest_path), "total_selected_shards": len(selected), "total_selected_requests": sum(s.request_lines for s in selected)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
