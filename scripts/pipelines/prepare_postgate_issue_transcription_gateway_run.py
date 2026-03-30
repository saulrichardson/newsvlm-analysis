#!/usr/bin/env python3
"""
Prepare a large post-gate issue transcription run for Slurm gateway workers.

This script:
  - reads a manifest with transcript_path and image_paths (or dataset-style pages[])
  - resolves and validates those paths
  - balances rows across worker manifests by estimated request weight
  - writes one JSONL manifest per worker for run_postgate_issue_transcription.py
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from run_postgate_issue_transcription import _load_manifest, _norm


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]], *, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})


@dataclass(frozen=True)
class _PreparedRow:
    worker_request_id: str
    request_id: str
    issue_id: str
    transcript_path: Path
    image_paths: tuple[Path, ...]
    page_ids: tuple[str, ...]
    transcript_bytes: int
    image_bytes_total: int
    weight_bytes: int


def _parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser(description="Prepare worker manifests for large post-gate issue transcription runs.")
    ap.add_argument(
        "--manifest-path",
        required=True,
        help="CSV or JSONL manifest with transcript_path and image_paths, or dataset-style issue rows with pages[].",
    )
    ap.add_argument(
        "--issue-txt-dir",
        default="",
        help="Optional fallback issue transcript dir for manifests that omit transcript_path.",
    )
    ap.add_argument("--output-dir", required=True, help="Prepared run root.")
    ap.add_argument("--worker-count", type=int, default=100, help="Target worker count.")
    ap.add_argument(
        "--prompt-path",
        default=str(repo_root / "prompts" / "transcription_v13_issue_selected_images_postgate.txt"),
        help="Prompt path to record in the prepared config.",
    )
    ap.add_argument(
        "--request-model",
        default="gemini-3.1-pro-preview",
        help="Model name embedded in each worker request body.",
    )
    ap.add_argument(
        "--gateway-model",
        default="gemini:gemini-3.1-pro-preview",
        help="Gateway provider:model to use from Slurm workers.",
    )
    ap.add_argument("--reasoning-effort", default="high")
    ap.add_argument("--max-concurrency", type=int, default=4)
    ap.add_argument("--timeout", type=float, default=21600.0)
    return ap.parse_args()


def _resolve_rows(*, manifest_path: Path, issue_txt_dir: Path | None) -> list[_PreparedRow]:
    raw_rows = _load_manifest(manifest_path, issue_txt_dir=issue_txt_dir)
    prepared: list[_PreparedRow] = []

    seen_request_ids: set[str] = set()
    for idx, row in enumerate(raw_rows):
        issue_id = _norm(row.get("issue_id"))
        request_id = _norm(row.get("request_id")) or issue_id
        if request_id in seen_request_ids:
            worker_request_id = f"{request_id}__dup{idx:06d}"
        else:
            worker_request_id = request_id
        seen_request_ids.add(worker_request_id)

        transcript_raw = _norm(row.get("transcript_path"))
        if not transcript_raw:
            raise SystemExit(
                f"Prepared-run manifests must resolve transcript_path explicitly. Missing transcript_path for issue_id={issue_id}"
            )
        transcript_path = Path(transcript_raw).expanduser().resolve()
        if not transcript_path.is_file():
            raise SystemExit(f"Missing transcript_path for issue_id={issue_id}: {transcript_path}")

        raw_image_paths = list(row.get("image_paths") or [])
        if not raw_image_paths:
            raise SystemExit(f"Missing image_paths for issue_id={issue_id}. This scale-out path expects direct image paths.")

        image_paths: list[Path] = []
        for raw in raw_image_paths:
            p = Path(str(raw)).expanduser().resolve()
            if not p.is_file():
                raise SystemExit(f"Missing image_path for issue_id={issue_id}: {p}")
            image_paths.append(p)

        page_ids = tuple(str(x) for x in (row.get("page_ids") or [])) or tuple(p.stem for p in image_paths)
        if len(page_ids) != len(image_paths):
            raise SystemExit(
                f"issue_id={issue_id} has {len(page_ids)} page_ids but {len(image_paths)} image_paths"
            )

        transcript_bytes = transcript_path.stat().st_size
        image_bytes_total = sum(p.stat().st_size for p in image_paths)
        weight_bytes = transcript_bytes + image_bytes_total

        prepared.append(
            _PreparedRow(
                worker_request_id=worker_request_id,
                request_id=request_id,
                issue_id=issue_id,
                transcript_path=transcript_path,
                image_paths=tuple(image_paths),
                page_ids=tuple(page_ids),
                transcript_bytes=transcript_bytes,
                image_bytes_total=image_bytes_total,
                weight_bytes=weight_bytes,
            )
        )

    if not prepared:
        raise SystemExit("No usable rows found for prepared run")
    return prepared


def _assign_workers(rows: list[_PreparedRow], *, worker_count: int) -> dict[int, list[_PreparedRow]]:
    if worker_count <= 0:
        raise SystemExit("worker_count must be > 0")
    actual_worker_count = min(worker_count, len(rows))
    buckets: dict[int, list[_PreparedRow]] = {i: [] for i in range(actual_worker_count)}
    loads: dict[int, int] = {i: 0 for i in range(actual_worker_count)}
    for row in sorted(rows, key=lambda r: (-r.weight_bytes, r.worker_request_id)):
        worker_idx = min(loads, key=lambda i: (loads[i], i))
        buckets[worker_idx].append(row)
        loads[worker_idx] += row.weight_bytes
    return buckets


def main() -> None:
    args = _parse_args()
    manifest_path = Path(args.manifest_path).expanduser().resolve()
    issue_txt_dir = Path(args.issue_txt_dir).expanduser().resolve() if _norm(args.issue_txt_dir) else None
    output_dir = Path(args.output_dir).expanduser().resolve()
    prompt_path = Path(args.prompt_path).expanduser().resolve()

    if not manifest_path.is_file():
        raise SystemExit(f"Missing manifest: {manifest_path}")
    if issue_txt_dir is not None and not issue_txt_dir.is_dir():
        raise SystemExit(f"Missing issue_txt_dir: {issue_txt_dir}")
    if not prompt_path.is_file():
        raise SystemExit(f"Missing prompt path: {prompt_path}")

    rows = _resolve_rows(manifest_path=manifest_path, issue_txt_dir=issue_txt_dir)
    assignments = _assign_workers(rows, worker_count=int(args.worker_count))

    workers_root = output_dir / "workers"
    workers_root.mkdir(parents=True, exist_ok=True)
    results_root = output_dir / "results"
    results_root.mkdir(parents=True, exist_ok=True)

    worker_rows: list[dict[str, Any]] = []
    for worker_idx, worker_items in sorted(assignments.items()):
        worker_dir = workers_root / f"worker{worker_idx:03d}"
        worker_manifest_path = worker_dir / "manifest.jsonl"
        manifest_rows: list[dict[str, Any]] = []
        for item in worker_items:
            manifest_rows.append(
                {
                    "request_id": item.worker_request_id,
                    "issue_id": item.issue_id,
                    "transcript_path": str(item.transcript_path),
                    "page_ids": list(item.page_ids),
                    "image_paths": [str(p) for p in item.image_paths],
                }
            )
        _write_jsonl(worker_manifest_path, manifest_rows)
        worker_rows.append(
            {
                "worker_index": worker_idx,
                "worker_dir": str(worker_dir),
                "manifest_path": str(worker_manifest_path),
                "issue_count": len(worker_items),
                "transcript_bytes": sum(item.transcript_bytes for item in worker_items),
                "image_bytes_total": sum(item.image_bytes_total for item in worker_items),
                "weight_bytes": sum(item.weight_bytes for item in worker_items),
                "request_ids": "|".join(item.worker_request_id for item in worker_items),
            }
        )

    item_rows: list[dict[str, Any]] = []
    worker_lookup = {row["request_id"]: idx for idx, items in assignments.items() for row in [
        {"request_id": item.worker_request_id} for item in items
    ]}
    for item in rows:
        item_rows.append(
            {
                "worker_index": worker_lookup[item.worker_request_id],
                "request_id": item.worker_request_id,
                "issue_id": item.issue_id,
                "transcript_path": str(item.transcript_path),
                "page_ids": "|".join(item.page_ids),
                "image_paths": "|".join(str(p) for p in item.image_paths),
                "transcript_bytes": item.transcript_bytes,
                "image_bytes_total": item.image_bytes_total,
                "weight_bytes": item.weight_bytes,
            }
        )

    _write_csv(
        output_dir / "worker_assignments.csv",
        worker_rows,
        fieldnames=[
            "worker_index",
            "worker_dir",
            "manifest_path",
            "issue_count",
            "transcript_bytes",
            "image_bytes_total",
            "weight_bytes",
            "request_ids",
        ],
    )
    _write_csv(
        output_dir / "prepared_items.csv",
        item_rows,
        fieldnames=[
            "worker_index",
            "request_id",
            "issue_id",
            "transcript_path",
            "page_ids",
            "image_paths",
            "transcript_bytes",
            "image_bytes_total",
            "weight_bytes",
        ],
    )

    summary = {
        "manifest_path": str(manifest_path),
        "issue_txt_dir": str(issue_txt_dir) if issue_txt_dir is not None else "",
        "output_dir": str(output_dir),
        "worker_count_requested": int(args.worker_count),
        "worker_count_actual": len(assignments),
        "issues": len(rows),
        "prompt_path": str(prompt_path),
        "request_model": str(args.request_model),
        "gateway_model": str(args.gateway_model),
        "reasoning_effort": str(args.reasoning_effort),
        "max_concurrency": int(args.max_concurrency),
        "timeout": float(args.timeout),
        "workers_root": str(workers_root),
        "results_root": str(results_root),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
