#!/usr/bin/env python3
"""
Build page-level classification inventory from hardened issue/event outputs.

Goal:
- Assign every page_id a final bucket based on issue classification + event status.
- Avoid pseudo page text segmentation; page labels are propagated from issue/event
  decisions that used full issue-date context.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


ENACTED_STATUSES = {"enacted_complete", "enacted_partial_installment"}
PROPOSED_STATUSES = {"proposed_full_text", "proposed_partial_installment"}
LAWLIKE_ISSUE_CLASSES = {
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
}


def _norm(x: Any) -> str:
    return str(x or "").strip()


def _read_issue_pages(source_run_dir: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    panel_root = source_run_dir / "panels"
    if not panel_root.is_dir():
        return out
    for city_dir in sorted(panel_root.iterdir()):
        p = city_dir / "issue_texts.jsonl"
        if not city_dir.is_dir() or not p.is_file():
            continue
        with p.open("r", encoding="utf-8") as f:
            for raw in f:
                s = raw.strip()
                if not s:
                    continue
                try:
                    obj = json.loads(s)
                except Exception:
                    continue
                issue_id = _norm(obj.get("issue_id"))
                if not issue_id:
                    continue
                page_ids = obj.get("page_ids") if isinstance(obj.get("page_ids"), list) else []
                page_ids = [_norm(x) for x in page_ids if _norm(x)]
                if not page_ids:
                    n = max(1, int(float(obj.get("page_count") or 1)))
                    page_ids = [f"{issue_id}__page_{i+1}" for i in range(n)]
                out[issue_id] = {
                    "newspaper_slug": _norm(obj.get("pub_slug")).lower(),
                    "city_key": _norm(obj.get("city_key")),
                    "city_name": _norm(obj.get("city_name")),
                    "state_abbr": _norm(obj.get("state_abbr")).upper(),
                    "issue_date": _norm(obj.get("issue_date")),
                    "page_count": int(len(page_ids)),
                    "page_ids": page_ids,
                }
    return out


def _fallback_bucket(issue_class: str) -> str:
    ic = _norm(issue_class)
    if ic in {"full_code_comprehensive", "full_code_recodification"}:
        return "full_ordinance"
    if ic == "full_code_partial_enacted":
        return "partial_ordinance_enacted"
    if ic in {"full_code_proposed", "full_code_partial_proposed"}:
        return "proposal_full_text"
    if ic in {"amendment_substantial", "amendment_targeted"}:
        return "amendment"
    if ic == "amendment_proposed":
        return "proposal_amendment"
    if ic in {"zoning_notice_adoption", "zoning_notice_procedural"}:
        return "notice"
    if ic == "building_code_non_zoning":
        return "building_code_non_zoning"
    if ic == "non_zoning":
        return "non_zoning"
    if ic == "not_verbatim_law_text":
        return "narrative_or_nonverbatim"
    return "uncertain_other"


def _bucket(
    issue_class: str,
    event_type: str,
    event_status: str,
    *,
    qualification_decision: str,
    jurisdiction_match_city: str,
) -> str:
    et = _norm(event_type)
    es = _norm(event_status)
    qd = _norm(qualification_decision).lower()
    jmc = _norm(jurisdiction_match_city).lower()

    # Final qualification veto: anything not passing strict qualification cannot
    # enter enacted core buckets in page-level outputs.
    if qd and qd != "pass":
        return "uncertain_other"

    # Jurisdiction mismatch is surfaced as a dedicated bucket so city-level
    # analyses can exclude it without losing the page-level record.
    if jmc and jmc not in {"yes", "true", "1"} and (
        et in {"full_ordinance", "amendment", "proposal"} or _norm(issue_class) in LAWLIKE_ISSUE_CLASSES
    ):
        return "other_jurisdiction_law"

    if et == "building_code":
        return "building_code_non_zoning"
    if et == "partial_ordinance_enacted":
        return "partial_ordinance_enacted"
    if et == "full_ordinance":
        if es == "enacted_complete":
            return "full_ordinance"
        if es == "enacted_partial_installment":
            return "partial_ordinance_enacted"
        if es in PROPOSED_STATUSES:
            return "proposal_full_text"
        return "uncertain_other"
    if et == "amendment":
        if es in ENACTED_STATUSES:
            return "amendment"
        if es in PROPOSED_STATUSES:
            return "proposal_amendment"
        return "uncertain_other"
    if et == "proposal":
        return "proposal_full_text"
    if et == "notice":
        return "notice"
    if et in {"non_law", "other", "mixed", "uncertain"}:
        return _fallback_bucket(issue_class)
    return _fallback_bucket(issue_class)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build page-level classification inventory.")
    ap.add_argument(
        "--source-run-dir",
        default="/Users/saulrichardson/projects/newspapers/newspaper-analysis/reports/runs/pi_v1_run60_fullnewspaper",
        help="Source run directory containing panels/<city_key>/issue_texts.jsonl.",
    )
    ap.add_argument(
        "--pipeline-root",
        required=True,
        help="Pipeline output root containing classification and events_status artifacts.",
    )
    ap.add_argument(
        "--classification-csv",
        default="",
        help="Override issue classification CSV path.",
    )
    ap.add_argument(
        "--event-assignments-csv",
        default="",
        help="Override issue-event assignments CSV path.",
    )
    ap.add_argument(
        "--output-dir",
        default="",
        help="Output directory (default: <pipeline-root>/page_classification).",
    )
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    source_run_dir = Path(args.source_run_dir).expanduser().resolve()
    pipeline_root = Path(args.pipeline_root).expanduser().resolve()
    out_dir = (
        Path(args.output_dir).expanduser().resolve()
        if _norm(args.output_dir)
        else pipeline_root / "page_classification"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    cls_path = (
        Path(args.classification_csv).expanduser().resolve()
        if _norm(args.classification_csv)
        else pipeline_root / "classification" / "issue_labels_v3.csv"
    )
    asn_path = (
        Path(args.event_assignments_csv).expanduser().resolve()
        if _norm(args.event_assignments_csv)
        else pipeline_root / "events_status" / "issue_event_assignments_status_v3.csv"
    )
    if not asn_path.is_file():
        asn_path = pipeline_root / "events" / "issue_event_assignments_v3.csv"

    if not cls_path.is_file():
        raise SystemExit(f"classification CSV not found: {cls_path}")
    if not asn_path.is_file():
        raise SystemExit(f"event assignments CSV not found: {asn_path}")

    cls = pd.read_csv(cls_path)
    asn = pd.read_csv(asn_path)
    page_meta = _read_issue_pages(source_run_dir)

    for df in (cls, asn):
        for c in (
            "newspaper_slug",
            "city_key",
            "city_name",
            "state_abbr",
            "issue_id",
            "issue_date",
            "issue_class",
            "event_type",
            "event_status",
            "qualification_decision",
            "publication_role",
            "ordinance_event_id",
        ):
            if c in df.columns:
                df[c] = df[c].astype(str)

    asn_map = {str(r.issue_id): r for r in asn.itertuples(index=False) if _norm(getattr(r, "issue_id", ""))}

    rows: list[dict[str, Any]] = []
    for r in cls.itertuples(index=False):
        issue_id = _norm(getattr(r, "issue_id", ""))
        if not issue_id:
            continue
        m = page_meta.get(issue_id, {})
        page_ids = m.get("page_ids") if isinstance(m.get("page_ids"), list) else []
        if not page_ids:
            n = max(1, int(float(getattr(r, "available_page_count", 1) if hasattr(r, "available_page_count") else 1)))
            page_ids = [f"{issue_id}__page_{i+1}" for i in range(n)]

        a = asn_map.get(issue_id)
        event_type = _norm(getattr(a, "event_type", "")) if a is not None else ""
        event_status = _norm(getattr(a, "event_status", "")) if a is not None else ""
        ordinance_event_id = _norm(getattr(a, "ordinance_event_id", "")) if a is not None else ""
        publication_role = _norm(getattr(a, "publication_role", "")) if a is not None else ""
        qualification_decision = _norm(getattr(a, "qualification_decision", "")) if a is not None else ""

        issue_class = _norm(getattr(r, "issue_class", ""))
        final_bucket = _bucket(
            issue_class,
            event_type,
            event_status,
            qualification_decision=qualification_decision,
            jurisdiction_match_city=_norm(getattr(r, "jurisdiction_match_city", "")),
        )
        page_count = int(len(page_ids))
        evidence_mode = "single_page_issue_context" if page_count <= 1 else "multi_page_issue_context_propagated_to_pages"

        for i, page_id in enumerate(page_ids, start=1):
            rows.append(
                {
                    "newspaper_slug": _norm(getattr(r, "newspaper_slug", "")) or _norm(m.get("newspaper_slug")),
                    "city_key": _norm(getattr(r, "city_key", "")) or _norm(m.get("city_key")),
                    "city_name": _norm(getattr(r, "city_name", "")) or _norm(m.get("city_name")),
                    "state_abbr": _norm(getattr(r, "state_abbr", "")).upper() or _norm(m.get("state_abbr")).upper(),
                    "issue_id": issue_id,
                    "issue_date": _norm(getattr(r, "issue_date", "")) or _norm(m.get("issue_date")),
                    "page_id": _norm(page_id),
                    "page_num_in_issue": int(i),
                    "page_count_in_issue": int(page_count),
                    "issue_class": issue_class,
                    "issue_completeness": _norm(getattr(r, "issue_completeness", "")),
                    "possible_missing_pages": _norm(getattr(r, "possible_missing_pages", "")),
                    "contains_verbatim_zoning_law": _norm(getattr(r, "contains_verbatim_zoning_law", "")),
                    "law_text_sufficiency": _norm(getattr(r, "law_text_sufficiency", "")),
                    "legal_status": _norm(getattr(r, "legal_status", "")),
                    "jurisdiction_match_city": _norm(getattr(r, "jurisdiction_match_city", "")),
                    "enactment_signal": _norm(getattr(r, "enactment_signal", "")),
                    "proposal_signal": _norm(getattr(r, "proposal_signal", "")),
                    "issue_confidence_0_to_1": float(getattr(r, "confidence_0_to_1", 0.0) or 0.0),
                    "ordinance_event_id": ordinance_event_id,
                    "event_type": event_type,
                    "event_status": event_status,
                    "qualification_decision": qualification_decision,
                    "publication_role": publication_role,
                    "page_class_bucket": final_bucket,
                    "page_class_confidence_0_to_1": float(getattr(r, "confidence_0_to_1", 0.0) or 0.0),
                    "page_class_evidence_mode": evidence_mode,
                }
            )

    page_df = pd.DataFrame.from_records(rows).sort_values(
        ["newspaper_slug", "issue_date", "issue_id", "page_num_in_issue"]
    ).reset_index(drop=True)
    page_df.to_csv(out_dir / "page_classification_index_v1.csv", index=False, quoting=csv.QUOTE_MINIMAL)

    by_bucket = (
        page_df.groupby(["newspaper_slug", "page_class_bucket"], dropna=False)["page_id"]
        .count()
        .reset_index(name="page_count")
        .sort_values(["newspaper_slug", "page_count"], ascending=[True, False])
        .reset_index(drop=True)
    )
    by_bucket.to_csv(out_dir / "page_classification_summary_by_newspaper.csv", index=False, quoting=csv.QUOTE_MINIMAL)

    by_event = (
        page_df.groupby(["event_type", "event_status", "page_class_bucket"], dropna=False)["page_id"]
        .count()
        .reset_index(name="page_count")
        .sort_values(["page_count"], ascending=False)
        .reset_index(drop=True)
    )
    by_event.to_csv(out_dir / "page_classification_summary_by_event_status.csv", index=False, quoting=csv.QUOTE_MINIMAL)

    prov = {
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "script_path": str(Path(__file__).resolve()),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_run_dir": str(source_run_dir),
        "pipeline_root": str(pipeline_root),
        "classification_csv": str(cls_path),
        "event_assignments_csv": str(asn_path),
        "output_dir": str(out_dir),
        "issue_rows": int(cls.shape[0]),
        "page_rows": int(page_df.shape[0]),
    }
    (out_dir / "provenance.json").write_text(json.dumps(prov, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"done_page_classification_index_v1 output_dir={out_dir}")


if __name__ == "__main__":
    main()
