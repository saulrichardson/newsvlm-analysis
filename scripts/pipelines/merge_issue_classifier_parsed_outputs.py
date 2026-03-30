#!/usr/bin/env python3
"""
Merge parsed issue-classifier outputs into one canonical JSONL.

The intended OpenAI/GPT merge order for the current production workflow is:

1. prod batch parsed outputs
2. retry parsed outputs, but only when retry row status == ok
3. prod live rerun parsed outputs (overwrite exact unresolved prod misses)
4. incremental live rerun parsed outputs (append / overwrite by issue_id)

Rows are keyed by `issue_id`, falling back to `custom_id` if needed.
The merged row is annotated with:
  - merged_source
  - merged_source_rank
  - merged_at
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from collections import Counter
from pathlib import Path
from typing import Any


def _norm(value: Any) -> str:
    return str(value or "").strip()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except Exception as exc:
                raise SystemExit(f"Invalid JSON in {path}:{line_number}: {exc}") from exc
            if not isinstance(payload, dict):
                raise SystemExit(f"Expected object JSON in {path}:{line_number}")
            rows.append(payload)
    return rows


def _row_key(row: dict[str, Any]) -> str:
    key = _norm(row.get("issue_id") or row.get("custom_id") or row.get("id"))
    if not key:
        raise SystemExit(f"Row missing issue_id/custom_id/id: {row}")
    return key


def _parse_spec(raw: str, *, index: int) -> tuple[str, Path, bool]:
    parts = raw.split("::")
    if len(parts) == 2:
        name, path_raw = parts
        ok_only = False
    elif len(parts) == 3:
        name, path_raw, mode = parts
        ok_only = mode.strip().lower() == "ok_only"
        if mode.strip().lower() not in {"ok_only", "all"}:
            raise SystemExit(
                f"Invalid source spec mode in --source #{index}: {raw!r}; expected ok_only or all"
            )
    else:
        raise SystemExit(
            f"Invalid --source #{index}: {raw!r}. Expected NAME::/abs/path or NAME::/abs/path::ok_only"
        )
    path = Path(path_raw).expanduser().resolve()
    if not path.is_file():
        raise SystemExit(f"--source #{index} path not found: {path}")
    return name.strip(), path, ok_only


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        help=(
            "Merge source spec in precedence order: NAME::/abs/path or NAME::/abs/path::ok_only. "
            "Later sources overwrite earlier rows by issue_id."
        ),
    )
    parser.add_argument("--out-jsonl", type=Path, required=True, help="Merged output JSONL path.")
    parser.add_argument("--summary-json", type=Path, required=True, help="Merge summary JSON path.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    source_specs = [_parse_spec(raw, index=i + 1) for i, raw in enumerate(args.source)]

    merged: dict[str, dict[str, Any]] = {}
    kept_by_source: Counter[str] = Counter()
    skipped_by_source: Counter[str] = Counter()

    merged_at = dt.datetime.now(dt.timezone.utc).isoformat()
    for rank, (name, path, ok_only) in enumerate(source_specs, start=1):
        for row in _read_jsonl(path):
            if ok_only and _norm(row.get("status")).lower() != "ok":
                skipped_by_source[name] += 1
                continue
            key = _row_key(row)
            row_copy = dict(row)
            row_copy["merged_source"] = name
            row_copy["merged_source_rank"] = rank
            row_copy["merged_at"] = merged_at
            merged[key] = row_copy
            kept_by_source[name] += 1

    out_jsonl = args.out_jsonl.expanduser().resolve()
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w", encoding="utf-8") as handle:
        for key in sorted(merged):
            handle.write(json.dumps(merged[key], ensure_ascii=False) + "\n")

    status_counts = Counter(_norm(row.get("status")) for row in merged.values())
    merged_source_counts = Counter(_norm(row.get("merged_source")) for row in merged.values())
    summary = {
        "created_at": merged_at,
        "sources": [
            {
                "name": name,
                "path": str(path),
                "ok_only": ok_only,
            }
            for name, path, ok_only in source_specs
        ],
        "merged_issue_count": len(merged),
        "status_counts": dict(sorted(status_counts.items())),
        "merged_source_counts": dict(sorted(merged_source_counts.items())),
        "kept_by_source": dict(sorted(kept_by_source.items())),
        "skipped_by_source": dict(sorted(skipped_by_source.items())),
    }
    summary_json = args.summary_json.expanduser().resolve()
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(
        f"merged issues={len(merged)} status_counts={dict(sorted(status_counts.items()))} "
        f"out_jsonl={out_jsonl}"
    )


if __name__ == "__main__":
    main()
