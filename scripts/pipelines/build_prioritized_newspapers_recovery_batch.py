#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pandas as pd


DEFAULT_METADATA_CSV = Path(
    "/Users/saulrichardson/Dropbox/Inclusionary Zoning/Historical Analysis/raw_data/"
    "newspaper_ordinances/metadata.csv"
)
DEFAULT_RULE_PLAN_CSV = Path(
    "artifacts/reports/newspapers_com_rule_only_fetch_plan_20260321_from279/"
    "confirmed_issue_rule_only_fetch_plan.csv"
)
DEFAULT_PREFERRED_MANIFEST_CSV = Path(
    "artifacts/reports/newspapers_com_issue_page_catalog_20260321_from279_core/"
    "target_page_image_manifest_preferred_only.csv"
)
DEFAULT_OUTPUT_DIR = Path("artifacts/reports/newspapers_com_recovery_priority_20260321")

SEVERITY_RANK = {
    "definitely_incomplete": 0,
    "likely_incomplete": 1,
    "uncertain": 2,
    "complete_or_nearly_complete": 3,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build issue- and page-level prioritized Newspapers.com recovery "
            "manifests from the confirmed issue set and completeness metadata."
        )
    )
    parser.add_argument("--metadata-csv", type=Path, default=DEFAULT_METADATA_CSV)
    parser.add_argument("--rule-plan-csv", type=Path, default=DEFAULT_RULE_PLAN_CSV)
    parser.add_argument(
        "--preferred-manifest-csv", type=Path, default=DEFAULT_PREFERRED_MANIFEST_CSV
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--page-budget",
        type=int,
        default=20,
        help="Maximum number of pages for the first issue-complete batch.",
    )
    parser.add_argument(
        "--include-complete-or-nearly-complete",
        action="store_true",
        help="Include issues labeled complete_or_nearly_complete in the prioritized outputs.",
    )
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write to {path}")
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metadata = pd.read_csv(
        args.metadata_csv,
        usecols=[
            "issue_id",
            "issue_date",
            "full_ordinance_origin",
            "raw_image_count",
            "ordinance_artifact_completeness_label",
            "ordinance_artifact_estimated_missing_share_0_to_1",
            "ordinance_artifact_missing_share_band",
            "ordinance_artifact_completeness_confidence_0_to_1",
        ],
    )
    plan = pd.read_csv(
        args.rule_plan_csv,
        usecols=[
            "issue_id",
            "issue_date",
            "newspaper_display_name",
            "matched_paper_url",
            "exact_issue_url",
            "attached_page_count",
            "attached_pages",
            "core_fetch_page_count",
            "core_fetch_pages",
        ],
    )
    preferred = pd.read_csv(args.preferred_manifest_csv)

    required_preferred = {
        "issue_id",
        "issue_date",
        "page_num",
        "preferred_image_id",
        "preferred_image_page_url",
    }
    missing = required_preferred - set(preferred.columns)
    if missing:
        raise SystemExit(
            f"{args.preferred_manifest_csv} is missing required columns: {sorted(missing)}"
        )

    preferred_counts = preferred.groupby("issue_id").size().rename("matched_preferred_count")

    issues = (
        metadata.merge(plan, on=["issue_id", "issue_date"], how="inner")
        .merge(preferred_counts, on="issue_id", how="left")
        .fillna({"matched_preferred_count": 0})
    )
    issues["matched_preferred_count"] = issues["matched_preferred_count"].astype(int)
    issues["core_fetch_page_count"] = issues["core_fetch_page_count"].astype(int)
    issues["attached_page_count"] = issues["attached_page_count"].astype(int)
    issues["full_core_coverage"] = (
        issues["matched_preferred_count"] == issues["core_fetch_page_count"]
    )
    issues["severity_rank"] = issues["ordinance_artifact_completeness_label"].map(SEVERITY_RANK)

    if issues["severity_rank"].isna().any():
        bad = issues.loc[issues["severity_rank"].isna(), "ordinance_artifact_completeness_label"].unique()
        raise SystemExit(f"Unexpected completeness labels in metadata: {sorted(bad)}")

    if not args.include_complete_or_nearly_complete:
        issues = issues[
            issues["ordinance_artifact_completeness_label"] != "complete_or_nearly_complete"
        ].copy()

    prioritized_issues = issues[issues["full_core_coverage"]].copy()
    prioritized_issues = prioritized_issues.sort_values(
        by=[
            "severity_rank",
            "ordinance_artifact_estimated_missing_share_0_to_1",
            "core_fetch_page_count",
            "issue_date",
            "issue_id",
        ],
        ascending=[True, False, True, True, True],
        na_position="last",
    )

    issue_summary_rows = prioritized_issues[
        [
            "issue_id",
            "issue_date",
            "newspaper_display_name",
            "full_ordinance_origin",
            "raw_image_count",
            "attached_page_count",
            "attached_pages",
            "core_fetch_page_count",
            "core_fetch_pages",
            "matched_preferred_count",
            "ordinance_artifact_completeness_label",
            "ordinance_artifact_estimated_missing_share_0_to_1",
            "ordinance_artifact_missing_share_band",
            "ordinance_artifact_completeness_confidence_0_to_1",
            "matched_paper_url",
            "exact_issue_url",
        ]
    ].to_dict(orient="records")

    ranked_issue_ids = prioritized_issues["issue_id"].tolist()
    rank_lookup = {issue_id: rank for rank, issue_id in enumerate(ranked_issue_ids, start=1)}

    page_rows = preferred[preferred["issue_id"].isin(rank_lookup)].copy()
    page_rows["issue_rank"] = page_rows["issue_id"].map(rank_lookup)
    page_rows["page_num_int"] = page_rows["page_num"].astype(int)
    page_rows = page_rows.sort_values(by=["issue_rank", "page_num_int", "issue_id"])
    page_rows = page_rows.drop(columns=["page_num_int"])

    prioritized_page_rows = page_rows.to_dict(orient="records")

    first_batch_rows: list[dict[str, object]] = []
    running_pages = 0
    for issue_id in ranked_issue_ids:
        issue_page_rows = [row for row in prioritized_page_rows if row["issue_id"] == issue_id]
        if not issue_page_rows:
            continue
        if first_batch_rows and running_pages + len(issue_page_rows) > args.page_budget:
            break
        if not first_batch_rows and len(issue_page_rows) > args.page_budget:
            first_batch_rows.extend(issue_page_rows)
            running_pages += len(issue_page_rows)
            break
        first_batch_rows.extend(issue_page_rows)
        running_pages += len(issue_page_rows)

    issue_summary_path = args.output_dir / "prioritized_issue_summary.csv"
    page_manifest_path = args.output_dir / "prioritized_page_manifest.csv"
    first_batch_path = args.output_dir / "first_batch_page_manifest.csv"

    write_csv(issue_summary_path, issue_summary_rows)
    write_csv(page_manifest_path, prioritized_page_rows)
    write_csv(first_batch_path, first_batch_rows)

    summary = {
        "metadata_csv": str(args.metadata_csv),
        "rule_plan_csv": str(args.rule_plan_csv),
        "preferred_manifest_csv": str(args.preferred_manifest_csv),
        "page_budget": args.page_budget,
        "include_complete_or_nearly_complete": args.include_complete_or_nearly_complete,
        "prioritized_issue_count": len(issue_summary_rows),
        "prioritized_page_count": len(prioritized_page_rows),
        "first_batch_page_count": len(first_batch_rows),
        "first_batch_issue_count": len({row["issue_id"] for row in first_batch_rows}),
        "issue_summary_path": str(issue_summary_path),
        "page_manifest_path": str(page_manifest_path),
        "first_batch_path": str(first_batch_path),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
