#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path


DEFAULT_CORPUS_ROOT = Path(
    "/Users/saulrichardson/Dropbox/Inclusionary Zoning/Historical Analysis/raw_data/newspaper_ordinances"
)
DEFAULT_OUTPUT_DIR = Path("artifacts/reports/full_ordinance_missing_page_audit_20260313")

MISSING_TEXT_RE = re.compile(r"\[MISSING TEXT[^\]]*\]", re.I)
CONTINUED_FROM_RE = re.compile(r"continued\s+from\s+page\s+\d+", re.I)
CONTINUED_ON_RE = re.compile(r"continued\s+(?:on|to)\s+page\s+\d+", re.I)

# These phrases explicitly point to unattached / unavailable continuation pages
# or other page-level evidence gaps rather than generic unrecoverable text.
PAGE_GAP_HINT_RE = re.compile(
    r"\bpage\b|\bpages\b|next newspaper page|earlier pages|unavailable|not attached|"
    r"attached issue page|supplied page images|available issue pages|intervening page|"
    r"continued on page|continuation from page|continuation on missing newspaper page|"
    r"not present on the attached issue page",
    re.I,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit the full-ordinance Dropbox corpus for clues that parsed ordinance "
            "markdown is missing text because continuation pages were not part of the "
            "parsing pipeline."
        )
    )
    parser.add_argument("--corpus-root", type=Path, default=DEFAULT_CORPUS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--edge-window-chars",
        type=int,
        default=500,
        help="Treat continued-from/on markers within this many characters of file start/end as edge clues.",
    )
    parser.add_argument(
        "--edge-window-pct",
        type=float,
        default=0.05,
        help="Treat continued-from/on markers within this proportion of file start/end as edge clues.",
    )
    return parser.parse_args()


def is_near_start(position: int, text_len: int, edge_chars: int, edge_pct: float) -> bool:
    return position <= edge_chars or position / max(1, text_len) <= edge_pct


def is_near_end(position: int, text_len: int, edge_chars: int, edge_pct: float) -> bool:
    remaining = text_len - position
    return remaining <= edge_chars or remaining / max(1, text_len) <= edge_pct


def build_issue_record(
    row: dict[str, str],
    *,
    corpus_root: Path,
    edge_chars: int,
    edge_pct: float,
) -> dict[str, object]:
    parsed_path = corpus_root / row["parsed_relpath"]
    text = parsed_path.read_text(encoding="utf-8", errors="replace")
    text_len = len(text)

    placeholders = MISSING_TEXT_RE.findall(text)
    placeholder_pagey = [value for value in placeholders if PAGE_GAP_HINT_RE.search(value)]

    continued_from = list(CONTINUED_FROM_RE.finditer(text))
    continued_on = list(CONTINUED_ON_RE.finditer(text))
    continued_from_edge = [
        match.group(0)
        for match in continued_from
        if is_near_start(match.start(), text_len, edge_chars, edge_pct)
    ]
    continued_on_edge = [
        match.group(0)
        for match in continued_on
        if is_near_end(match.end(), text_len, edge_chars, edge_pct)
    ]

    high_conf_missing_other_pages = bool(placeholder_pagey or continued_from_edge or continued_on_edge)
    incomplete_but_cause_unclear = bool(placeholders) and not high_conf_missing_other_pages
    any_direct_gap_clue = high_conf_missing_other_pages or bool(placeholders)

    if high_conf_missing_other_pages:
        audit_category = "high_confidence_missing_other_pages"
    elif incomplete_but_cause_unclear:
        audit_category = "incomplete_text_cause_unclear"
    else:
        audit_category = "no_direct_gap_clue"

    return {
        "issue_id": row["issue_id"],
        "issue_date": row.get("issue_date", ""),
        "newspaper_slug": row.get("newspaper_slug", ""),
        "page_count": row.get("page_count", ""),
        "raw_image_count": row.get("raw_image_count", ""),
        "parsed_relpath": row.get("parsed_relpath", ""),
        "audit_category": audit_category,
        "has_missing_text_placeholder": "1" if placeholders else "0",
        "missing_text_placeholder_count": str(len(placeholders)),
        "has_page_gap_placeholder": "1" if placeholder_pagey else "0",
        "page_gap_placeholder_count": str(len(placeholder_pagey)),
        "has_continued_from_edge": "1" if continued_from_edge else "0",
        "continued_from_edge_count": str(len(continued_from_edge)),
        "has_continued_on_edge": "1" if continued_on_edge else "0",
        "continued_on_edge_count": str(len(continued_on_edge)),
        "any_direct_gap_clue": "1" if any_direct_gap_clue else "0",
        "placeholder_examples_json": json.dumps(placeholders[:3], ensure_ascii=True),
        "page_gap_placeholder_examples_json": json.dumps(placeholder_pagey[:3], ensure_ascii=True),
        "continued_from_edge_examples_json": json.dumps(continued_from_edge[:3], ensure_ascii=True),
        "continued_on_edge_examples_json": json.dumps(continued_on_edge[:3], ensure_ascii=True),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise SystemExit("No rows to write")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_summary(rows: list[dict[str, object]]) -> dict[str, object]:
    category_counts = Counter(str(row["audit_category"]) for row in rows)
    direct_gap_count = sum(1 for row in rows if row["any_direct_gap_clue"] == "1")
    missing_text_count = sum(1 for row in rows if row["has_missing_text_placeholder"] == "1")
    page_gap_count = sum(1 for row in rows if row["has_page_gap_placeholder"] == "1")
    continued_from_edge_count = sum(1 for row in rows if row["has_continued_from_edge"] == "1")
    continued_on_edge_count = sum(1 for row in rows if row["has_continued_on_edge"] == "1")
    return {
        "total_issues": len(rows),
        "audit_category_counts": dict(category_counts),
        "issues_with_any_direct_gap_clue": direct_gap_count,
        "issues_with_missing_text_placeholder": missing_text_count,
        "issues_with_page_gap_placeholder": page_gap_count,
        "issues_with_continued_from_edge": continued_from_edge_count,
        "issues_with_continued_on_edge": continued_on_edge_count,
    }


def main() -> None:
    args = parse_args()
    corpus_root = args.corpus_root.expanduser().resolve()
    metadata_path = corpus_root / "metadata.csv"
    if not metadata_path.is_file():
        raise SystemExit(f"Missing metadata.csv: {metadata_path}")

    with metadata_path.open("r", encoding="utf-8", newline="") as handle:
        metadata_rows = list(csv.DictReader(handle))

    records = [
        build_issue_record(
            row,
            corpus_root=corpus_root,
            edge_chars=args.edge_window_chars,
            edge_pct=args.edge_window_pct,
        )
        for row in metadata_rows
    ]

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "audit.csv"
    json_path = output_dir / "summary.json"
    write_csv(csv_path, records)
    summary = build_summary(records)
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps({"csv_path": str(csv_path), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
