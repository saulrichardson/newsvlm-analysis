#!/usr/bin/env python3
"""
Classification hardening QC checks.

Purpose:
- Validate that strict classification invariants hold after issue classification,
  dedup, event-status adjudication, and page-bucket export.
- Emit machine-readable QC summary plus violation tables.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


ALLOWED_ISSUE_CLASSES = {
    "full_code_comprehensive",
    "full_code_recodification",
    "full_code_partial_enacted",
    "full_code_partial_proposed",
    "full_code_proposed",
    "amendment_substantial",
    "amendment_targeted",
    "amendment_proposed",
    "zoning_notice_adoption",
    "zoning_notice_procedural",
    "building_code_non_zoning",
    "not_verbatim_law_text",
    "non_zoning",
}

PROPOSED_STATUSES = {"proposed_full_text", "proposed_partial_installment"}
ENACTED_STATUSES = {"enacted_complete", "enacted_partial_installment"}


def _norm(x: Any) -> str:
    return str(x or "").strip()


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run classification hardening QC checks.")
    ap.add_argument(
        "--pipeline-root",
        required=True,
        help="Pipeline root containing classification/, events_status/, and page_classification/.",
    )
    ap.add_argument(
        "--issue-assignments-csv",
        default="",
        help="Override issue assignments CSV (default: events_status/issue_event_assignments_status_v3.csv).",
    )
    ap.add_argument(
        "--page-classification-csv",
        default="",
        help="Override page classification CSV (default: page_classification/page_classification_index_v1.csv).",
    )
    ap.add_argument(
        "--output-dir",
        default="",
        help="Output directory (default: <pipeline-root>/qc).",
    )
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    root = Path(args.pipeline_root).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve() if _norm(args.output_dir) else root / "qc"
    out_dir.mkdir(parents=True, exist_ok=True)

    cls_path = root / "classification" / "issue_labels_v3.csv"
    asn_path = (
        Path(args.issue_assignments_csv).expanduser().resolve()
        if _norm(args.issue_assignments_csv)
        else root / "events_status" / "issue_event_assignments_status_v3.csv"
    )
    page_path = (
        Path(args.page_classification_csv).expanduser().resolve()
        if _norm(args.page_classification_csv)
        else root / "page_classification" / "page_classification_index_v1.csv"
    )

    if not cls_path.is_file():
        raise SystemExit(f"Missing classification CSV: {cls_path}")
    if not asn_path.is_file():
        raise SystemExit(f"Missing event-status assignments CSV: {asn_path}")
    if not page_path.is_file():
        raise SystemExit(f"Missing page classification CSV: {page_path}")

    cls = pd.read_csv(cls_path)
    asn = pd.read_csv(asn_path)
    page = pd.read_csv(page_path)

    for df in (cls, asn, page):
        for col in (
            "issue_id",
            "newspaper_slug",
            "issue_class",
            "event_type",
            "event_status",
            "qualification_decision",
            "jurisdiction_match_city",
            "page_class_bucket",
        ):
            if col in df.columns:
                df[col] = df[col].astype(str)

    if "jurisdiction_match_city" not in page.columns:
        page["jurisdiction_match_city"] = ""
    if "qualification_decision" not in page.columns:
        page["qualification_decision"] = ""
    if "event_type" not in page.columns:
        page["event_type"] = ""
    if "event_status" not in page.columns:
        page["event_status"] = ""

    violations: dict[str, pd.DataFrame] = {}

    bad_issue_class = cls[~cls["issue_class"].isin(sorted(ALLOWED_ISSUE_CLASSES))].copy()
    violations["unknown_issue_class"] = bad_issue_class

    full_prop = asn[
        (asn["event_type"] == "full_ordinance")
        & (asn["event_status"].isin(sorted(PROPOSED_STATUSES)))
    ].copy()
    violations["full_with_proposed_status"] = full_prop

    amend_prop = asn[
        (asn["event_type"] == "amendment")
        & (asn["event_status"].isin(sorted(PROPOSED_STATUSES)))
    ].copy()
    violations["amend_with_proposed_status"] = amend_prop

    full_bucket_bad = page[
        (page["page_class_bucket"] == "full_ordinance")
        & (
            (page["event_type"] != "full_ordinance")
            | (page["event_status"] != "enacted_complete")
            | (page["qualification_decision"].str.lower() != "pass")
            | (~page["jurisdiction_match_city"].str.lower().isin(["yes", "true", "1"]))
        )
    ].copy()
    violations["full_bucket_invariant_fail"] = full_bucket_bad

    amend_bucket_bad = page[
        (page["page_class_bucket"] == "amendment")
        & (
            (page["event_type"] != "amendment")
            | (~page["event_status"].isin(sorted(ENACTED_STATUSES)))
            | (page["qualification_decision"].str.lower() != "pass")
            | (~page["jurisdiction_match_city"].str.lower().isin(["yes", "true", "1"]))
        )
    ].copy()
    violations["amend_bucket_invariant_fail"] = amend_bucket_bad

    other_jur_law = page[page["page_class_bucket"] == "other_jurisdiction_law"].copy()
    violations["other_jurisdiction_law_pages"] = other_jur_law

    for name, df in violations.items():
        df.to_csv(out_dir / f"{name}.csv", index=False)

    summary = {
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "pipeline_root": str(root),
        "issue_rows": int(cls.shape[0]),
        "event_assignment_rows": int(asn.shape[0]),
        "page_rows": int(page.shape[0]),
        "page_bucket_counts": {k: int(v) for k, v in page["page_class_bucket"].value_counts().to_dict().items()},
        "checks": {
            "unknown_issue_class_count": int(bad_issue_class.shape[0]),
            "full_with_proposed_status_count": int(full_prop.shape[0]),
            "amend_with_proposed_status_count": int(amend_prop.shape[0]),
            "full_bucket_invariant_fail_count": int(full_bucket_bad.shape[0]),
            "amend_bucket_invariant_fail_count": int(amend_bucket_bad.shape[0]),
            "other_jurisdiction_law_page_count": int(other_jur_law.shape[0]),
        },
        "passes_strict_core_invariants": bool(
            bad_issue_class.empty
            and full_prop.empty
            and amend_prop.empty
            and full_bucket_bad.empty
            and amend_bucket_bad.empty
        ),
    }
    summary["script_path"] = str(Path(__file__).resolve())
    summary["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (out_dir / "classification_hardening_qc_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"done_qc output_dir={out_dir}")
    print(json.dumps(summary["checks"], indent=2, sort_keys=True))
    print(f"passes_strict_core_invariants={summary['passes_strict_core_invariants']}")


if __name__ == "__main__":
    main()
