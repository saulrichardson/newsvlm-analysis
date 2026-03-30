#!/usr/bin/env python3
"""Build tiny-batch worklists for matched but not yet strictly confirmed issue dates."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--queue-csv",
        type=Path,
        default=Path("artifacts/reports/newspapers_com_page_inventory_audit_20260318/issue_page_inventory_queue.csv"),
    )
    parser.add_argument(
        "--confirmed-csv",
        type=Path,
        default=Path(
            "artifacts/reports/newspapers_com_availability_full_20260318/confirmed_exact_issue_dates_from_slow_retries.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/reports/newspapers_com_availability_full_20260319_matched_unconfirmed"),
    )
    parser.add_argument("--batch-size", type=int, default=5)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        raise ValueError("No rows to write")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    queue_rows = read_csv(args.queue_csv)
    confirmed_rows = read_csv(args.confirmed_csv)
    confirmed_ids = {row["issue_id"] for row in confirmed_rows}

    matched_unconfirmed = [
        row
        for row in queue_rows
        if row["newspapers_com_match_status"] == "matched" and row["issue_id"] not in confirmed_ids
    ]

    # queue CSV is already priority-sorted; preserve that order and add batch metadata.
    output_rows = []
    for idx, row in enumerate(matched_unconfirmed):
        enriched = dict(row)
        enriched["batch_number"] = str(idx // args.batch_size + 1)
        enriched["batch_index_within_batch"] = str(idx % args.batch_size)
        enriched["global_matched_unconfirmed_index"] = str(idx)
        output_rows.append(enriched)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    queue_out = args.output_dir / "matched_unconfirmed_exact_issue_queue.csv"
    write_csv(queue_out, output_rows)

    # compact operator sheet
    operator_rows = []
    for row in output_rows:
        operator_rows.append(
            {
                "batch_number": row["batch_number"],
                "priority_rank": row["priority_rank"],
                "issue_id": row["issue_id"],
                "issue_date": row["issue_date"],
                "newspaper_display_name": row["newspaper_display_name"],
                "attached_raw_page_numbers_csv": row["attached_raw_page_numbers_csv"],
                "priority_group": row["priority_group"],
                "heuristic_audit_category": row["heuristic_audit_category"],
                "model_completeness_label": row["model_completeness_label"],
                "estimated_missing_share_0_to_1": row["estimated_missing_share_0_to_1"],
                "newspapers_com_matched_paper_title": row["newspapers_com_matched_paper_title"],
                "newspapers_com_exact_issue_url": row["newspapers_com_exact_issue_url"],
            }
        )
    operator_out = args.output_dir / "matched_unconfirmed_exact_issue_operator_sheet.csv"
    write_csv(operator_out, operator_rows)

    print(f"Wrote {queue_out}")
    print(f"Wrote {operator_out}")
    print(f"Matched but not strictly confirmed rows: {len(output_rows)}")


if __name__ == "__main__":
    main()
