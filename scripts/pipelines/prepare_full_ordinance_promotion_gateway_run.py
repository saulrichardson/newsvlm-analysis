#!/usr/bin/env python3
"""
Prepare a shardable request run for second-stage full-ordinance promotion.

This script assumes you already have one issue transcript per issue in
`--issue-txt-dir`, usually created from parsed classifier outputs via
`build_issue_classifier_dataset_manifest.py`.

It writes a prepared run root:

  <output_dir>/
    summary.json
    prepared_items.csv
    worker_assignments.csv
    workers/
      worker000/
        requests/openai_requests_shard000.jsonl
      ...
    results/

Each shard request JSONL is ready to run through either:
  - `run_openai_requests_via_gateway.py`
  - `convert_openai_requests_to_gemini_batch.py` + `gemini_batch_submit.py`
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _norm(value: Any) -> str:
    return str(value or "").strip()


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]], *, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _make_openai_request(*, custom_id: str, prompt: str, model: str, reasoning_effort: str) -> dict[str, Any]:
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/responses",
        "body": {
            "model": model,
            "input": [
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": prompt}],
                }
            ],
            "reasoning": {"effort": reasoning_effort},
            "stream": False,
        },
    }


@dataclass(frozen=True)
class _PreparedIssue:
    issue_id: str
    transcript_path: Path
    transcript_chars: int
    transcript_bytes: int
    prompt_chars: int
    request_chars: int
    request_bytes: int


def _parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--issue-txt-dir", required=True, help="Directory containing <issue_id>.txt transcripts.")
    parser.add_argument("--output-dir", required=True, help="Prepared run root.")
    parser.add_argument(
        "--prompt-path",
        default=str(repo_root / "prompts" / "second_stage_full_ordinance_promotion_v2.txt"),
        help="Promotion prompt to prepend to each issue transcript.",
    )
    parser.add_argument("--worker-count", type=int, default=4, help="Target worker count.")
    parser.add_argument(
        "--request-model",
        default="gemini-3-flash-preview",
        help="Model written into each request JSON body.",
    )
    parser.add_argument(
        "--gateway-model",
        default="gemini:gemini-3-flash-preview",
        help="Gateway provider:model to run from Slurm workers.",
    )
    parser.add_argument("--reasoning-effort", default="high")
    parser.add_argument("--max-concurrency", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=21600.0)
    return parser.parse_args()


def _discover_issue_paths(issue_txt_dir: Path) -> list[Path]:
    paths = sorted(issue_txt_dir.glob("*.txt"))
    if not paths:
        raise SystemExit(f"No .txt issue transcripts found under {issue_txt_dir}")
    return paths


def _prepare_issues(*, issue_txt_dir: Path, prompt_text: str, request_model: str, reasoning_effort: str) -> list[_PreparedIssue]:
    prepared: list[_PreparedIssue] = []
    for path in _discover_issue_paths(issue_txt_dir):
        issue_id = path.stem
        transcript = path.read_text(encoding="utf-8", errors="replace").strip()
        prompt = prompt_text + "\n\n" + f"ISSUE_ID: {issue_id}\n\nISSUE_TRANSCRIPT:\n" + transcript
        request = _make_openai_request(
            custom_id=issue_id,
            prompt=prompt,
            model=request_model,
            reasoning_effort=reasoning_effort,
        )
        request_bytes = len(json.dumps(request, ensure_ascii=False).encode("utf-8"))
        prepared.append(
            _PreparedIssue(
                issue_id=issue_id,
                transcript_path=path.resolve(),
                transcript_chars=len(transcript),
                transcript_bytes=path.stat().st_size,
                prompt_chars=len(prompt_text),
                request_chars=len(prompt),
                request_bytes=request_bytes,
            )
        )
    return prepared


def _assign_workers(issues: list[_PreparedIssue], *, worker_count: int) -> dict[int, list[_PreparedIssue]]:
    if worker_count <= 0:
        raise SystemExit("worker_count must be > 0")
    actual_worker_count = min(worker_count, len(issues))
    buckets: dict[int, list[_PreparedIssue]] = {i: [] for i in range(actual_worker_count)}
    load_by_worker: dict[int, int] = {i: 0 for i in range(actual_worker_count)}
    for issue in sorted(issues, key=lambda row: (-row.request_bytes, row.issue_id)):
        worker_idx = min(load_by_worker, key=lambda idx: (load_by_worker[idx], idx))
        buckets[worker_idx].append(issue)
        load_by_worker[worker_idx] += issue.request_bytes
    return buckets


def main() -> None:
    args = _parse_args()
    issue_txt_dir = Path(args.issue_txt_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    prompt_path = Path(args.prompt_path).expanduser().resolve()

    if not issue_txt_dir.is_dir():
        raise SystemExit(f"Missing issue transcript dir: {issue_txt_dir}")
    if not prompt_path.is_file():
        raise SystemExit(f"Missing prompt: {prompt_path}")

    prompt_text = prompt_path.read_text(encoding="utf-8").strip()
    prepared = _prepare_issues(
        issue_txt_dir=issue_txt_dir,
        prompt_text=prompt_text,
        request_model=str(args.request_model),
        reasoning_effort=str(args.reasoning_effort),
    )
    assignments = _assign_workers(prepared, worker_count=int(args.worker_count))

    workers_root = output_dir / "workers"
    results_root = output_dir / "results"
    workers_root.mkdir(parents=True, exist_ok=True)
    results_root.mkdir(parents=True, exist_ok=True)

    worker_rows: list[dict[str, Any]] = []
    item_rows: list[dict[str, Any]] = []

    for worker_idx, issues in sorted(assignments.items()):
        worker_dir = workers_root / f"worker{worker_idx:03d}"
        request_dir = worker_dir / "requests"
        request_path = request_dir / "openai_requests_shard000.jsonl"
        mapping_path = request_dir / "mapping_shard000.jsonl"
        request_rows: list[dict[str, Any]] = []
        mapping_rows: list[dict[str, Any]] = []
        total_request_bytes = 0
        for issue in issues:
            transcript = issue.transcript_path.read_text(encoding="utf-8", errors="replace").strip()
            prompt = prompt_text + "\n\n" + f"ISSUE_ID: {issue.issue_id}\n\nISSUE_TRANSCRIPT:\n" + transcript
            request_obj = _make_openai_request(
                custom_id=issue.issue_id,
                prompt=prompt,
                model=str(args.request_model),
                reasoning_effort=str(args.reasoning_effort),
            )
            request_rows.append(request_obj)
            mapping_rows.append(
                {
                    "request_id": issue.issue_id,
                    "issue_id": issue.issue_id,
                    "transcript_path": str(issue.transcript_path),
                    "transcript_chars": issue.transcript_chars,
                    "transcript_bytes": issue.transcript_bytes,
                }
            )
            total_request_bytes += len(json.dumps(request_obj, ensure_ascii=False).encode("utf-8"))
            item_rows.append(
                {
                    "worker_index": worker_idx,
                    "issue_id": issue.issue_id,
                    "transcript_path": str(issue.transcript_path),
                    "transcript_chars": issue.transcript_chars,
                    "transcript_bytes": issue.transcript_bytes,
                    "request_chars": issue.request_chars,
                    "request_bytes": issue.request_bytes,
                }
            )
        _write_jsonl(request_path, request_rows)
        _write_jsonl(mapping_path, mapping_rows)
        worker_rows.append(
            {
                "worker_index": worker_idx,
                "worker_dir": str(worker_dir),
                "request_dir": str(request_dir),
                "request_path": str(request_path),
                "mapping_path": str(mapping_path),
                "issue_count": len(issues),
                "transcript_bytes": sum(issue.transcript_bytes for issue in issues),
                "request_bytes": total_request_bytes,
                "issue_ids": "|".join(issue.issue_id for issue in issues),
            }
        )

    _write_csv(
        output_dir / "worker_assignments.csv",
        worker_rows,
        fieldnames=[
            "worker_index",
            "worker_dir",
            "request_dir",
            "request_path",
            "mapping_path",
            "issue_count",
            "transcript_bytes",
            "request_bytes",
            "issue_ids",
        ],
    )
    _write_csv(
        output_dir / "prepared_items.csv",
        item_rows,
        fieldnames=[
            "worker_index",
            "issue_id",
            "transcript_path",
            "transcript_chars",
            "transcript_bytes",
            "request_chars",
            "request_bytes",
        ],
    )

    summary = {
        "issue_txt_dir": str(issue_txt_dir),
        "output_dir": str(output_dir),
        "prompt_path": str(prompt_path),
        "worker_count_requested": int(args.worker_count),
        "worker_count_actual": len(assignments),
        "issues": len(prepared),
        "request_model": str(args.request_model),
        "gateway_model": str(args.gateway_model),
        "reasoning_effort": str(args.reasoning_effort),
        "max_concurrency": int(args.max_concurrency),
        "timeout": float(args.timeout),
        "workers_root": str(workers_root),
        "results_root": str(results_root),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
