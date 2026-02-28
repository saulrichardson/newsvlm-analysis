#!/usr/bin/env python3
"""
Precision-first consistency gate for event-level outputs.

This applies deterministic integrity gates on top of LLM outputs to prevent
mixed/proposed leakage into enacted `full_ordinance` and `amendment` buckets.
It can also enforce strict full-ordinance semantics by moving enacted partial
installments out of `full_ordinance` into a separate `partial_ordinance_enacted`
bucket.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


FULL_ENACTED_CLASSES = {
    "full_code_comprehensive",
    "full_code_recodification",
    "full_code_partial_enacted",
}
FULL_PROPOSED_CLASSES = {
    "full_code_proposed",
    "full_code_partial_proposed",
}
AMEND_ENACTED_CLASSES = {
    "amendment_substantial",
    "amendment_targeted",
}
AMEND_PROPOSED_CLASSES = {
    "amendment_proposed",
}
NON_CORE_CLASSES = {
    "zoning_notice_adoption",
    "zoning_notice_procedural",
    "not_verbatim_law_text",
    "non_zoning",
    "building_code_non_zoning",
}
YES_SET = {"yes", "true", "1"}
ENACTED_COMPLETE = {"enacted_complete"}
ENACTED_PARTIAL = {"enacted_partial_installment"}
PROPOSED_STATUSES = {"proposed_full_text", "proposed_partial_installment"}


def _norm(x: Any) -> str:
    return str(x or "").strip()


def _is_yes(x: Any) -> bool:
    return _norm(x).lower() in YES_SET


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Enforce precision consistency gates on event outputs.")
    ap.add_argument(
        "--pipeline-root",
        required=True,
        help="Pipeline root containing events_status/*.csv",
    )
    ap.add_argument(
        "--issue-assignments-csv",
        default="",
        help="Override issue assignments CSV (default: events_status/issue_event_assignments_status_v3.csv)",
    )
    ap.add_argument(
        "--event-clusters-csv",
        default="",
        help="Override event clusters CSV (default: events_status/ordinance_event_clusters_status_v3.csv)",
    )
    ap.add_argument(
        "--output-dir",
        default="",
        help="Output directory (default: <pipeline-root>/events_status/precision_v1)",
    )
    ap.add_argument(
        "--strict-full-complete-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "If enabled, only enacted_complete stays in full_ordinance. "
            "enacted_partial_installment is moved to partial_ordinance_enacted."
        ),
    )
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    pipeline_root = Path(args.pipeline_root).expanduser().resolve()
    asn_path = (
        Path(args.issue_assignments_csv).expanduser().resolve()
        if _norm(args.issue_assignments_csv)
        else pipeline_root / "events_status" / "issue_event_assignments_status_v3.csv"
    )
    cl_path = (
        Path(args.event_clusters_csv).expanduser().resolve()
        if _norm(args.event_clusters_csv)
        else pipeline_root / "events_status" / "ordinance_event_clusters_status_v3.csv"
    )
    out_dir = (
        Path(args.output_dir).expanduser().resolve()
        if _norm(args.output_dir)
        else pipeline_root / "events_status" / "precision_v1"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    if not asn_path.is_file():
        raise SystemExit(f"Missing issue assignments csv: {asn_path}")
    if not cl_path.is_file():
        raise SystemExit(f"Missing event clusters csv: {cl_path}")

    asn = pd.read_csv(asn_path)
    clusters = pd.read_csv(cl_path)

    for df in (asn, clusters):
        for c in (
            "newspaper_slug",
            "ordinance_event_id",
            "event_type",
            "event_status",
            "qualification_decision",
            "issue_class",
            "issue_jurisdiction_match_city",
            "jurisdiction_match_city",
        ):
            if c in df.columns:
                df[c] = df[c].astype(str)

    demotions: list[dict[str, Any]] = []
    remaps: list[dict[str, Any]] = []

    key_cols = ["newspaper_slug", "ordinance_event_id"]
    grouped = asn.groupby(key_cols, dropna=False)

    # Mutable copies for output
    asn_out = asn.copy()
    cl_out = clusters.copy()

    for (slug, eid), g in grouped:
        event_type = _norm(g["event_type"].iloc[0]).lower()
        if event_type not in {"full_ordinance", "amendment"}:
            continue
        event_status_mode = _norm(g["event_status"].iloc[0]).lower()

        issue_classes = {_norm(x) for x in g["issue_class"].tolist()}
        has_mismatch = any(
            not _is_yes(x)
            for x in g.get("issue_jurisdiction_match_city", pd.Series(["yes"] * len(g))).tolist()
        ) or any(
            not _is_yes(x)
            for x in g.get("jurisdiction_match_city", pd.Series(["yes"] * len(g))).tolist()
        )

        reasons: list[str] = []
        if has_mismatch:
            reasons.append("jurisdiction_mismatch_in_members")

        if event_type == "full_ordinance":
            has_full_enacted = bool(issue_classes & FULL_ENACTED_CLASSES)
            has_full_proposed = bool(issue_classes & FULL_PROPOSED_CLASSES)
            has_amend = bool(issue_classes & (AMEND_ENACTED_CLASSES | AMEND_PROPOSED_CLASSES))
            has_non_core = bool(issue_classes & NON_CORE_CLASSES)
            if not has_full_enacted:
                reasons.append("no_enacted_full_member_class")
            if has_full_proposed:
                reasons.append("contains_proposed_full_member")
            if has_amend:
                reasons.append("contains_amendment_member")
            if has_non_core:
                reasons.append("contains_noncore_member")
            # Precision-first strict full policy: full_ordinance means enacted complete only.
            if args.strict_full_complete_only:
                if event_status_mode in ENACTED_PARTIAL:
                    # Not a failure; move to explicit partial bucket.
                    m = (asn_out["newspaper_slug"] == slug) & (asn_out["ordinance_event_id"] == eid)
                    asn_out.loc[m, "event_type"] = "partial_ordinance_enacted"
                    asn_out.loc[m, "precision_gate_demoted"] = "no"
                    asn_out.loc[m, "precision_gate_reasons"] = "strict_full_complete_only"

                    m2 = (cl_out["newspaper_slug"] == slug) & (cl_out["ordinance_event_id"] == eid)
                    cl_out.loc[m2, "event_type"] = "partial_ordinance_enacted"
                    cl_out.loc[m2, "precision_gate_demoted"] = "no"
                    cl_out.loc[m2, "precision_gate_reasons"] = "strict_full_complete_only"

                    remaps.append(
                        {
                            "newspaper_slug": slug,
                            "ordinance_event_id": eid,
                            "original_event_type": "full_ordinance",
                            "original_event_status": event_status_mode,
                            "new_event_type": "partial_ordinance_enacted",
                            "new_event_status": event_status_mode,
                            "reason": "strict_full_complete_only",
                        }
                    )
                elif event_status_mode in PROPOSED_STATUSES:
                    reasons.append("full_event_has_proposed_status")
                elif event_status_mode and event_status_mode not in ENACTED_COMPLETE:
                    reasons.append("full_event_not_enacted_complete")
        elif event_type == "amendment":
            has_amend_enacted = bool(issue_classes & AMEND_ENACTED_CLASSES)
            has_amend_proposed = bool(issue_classes & AMEND_PROPOSED_CLASSES)
            has_full = bool(issue_classes & (FULL_ENACTED_CLASSES | FULL_PROPOSED_CLASSES))
            has_non_core = bool(issue_classes & NON_CORE_CLASSES)
            if not has_amend_enacted:
                reasons.append("no_enacted_amendment_member_class")
            if has_amend_proposed:
                reasons.append("contains_proposed_amendment_member")
            if has_full:
                reasons.append("contains_fullcode_member")
            if has_non_core:
                reasons.append("contains_noncore_member")

        if reasons:
            # Demote event to uncertain for precision-first outputs.
            m = (asn_out["newspaper_slug"] == slug) & (asn_out["ordinance_event_id"] == eid)
            asn_out.loc[m, "event_type"] = "uncertain"
            asn_out.loc[m, "event_status"] = "uncertain"
            asn_out.loc[m, "qualification_decision"] = "fail"
            asn_out.loc[m, "precision_gate_demoted"] = "yes"
            asn_out.loc[m, "precision_gate_reasons"] = "|".join(sorted(reasons))

            m2 = (cl_out["newspaper_slug"] == slug) & (cl_out["ordinance_event_id"] == eid)
            cl_out.loc[m2, "event_type"] = "uncertain"
            cl_out.loc[m2, "event_status"] = "uncertain"
            cl_out.loc[m2, "qualification_decision"] = "fail"
            cl_out.loc[m2, "precision_gate_demoted"] = "yes"
            cl_out.loc[m2, "precision_gate_reasons"] = "|".join(sorted(reasons))

            demotions.append(
                {
                    "newspaper_slug": slug,
                    "ordinance_event_id": eid,
                    "original_event_type": event_type,
                    "member_issue_count": int(g.shape[0]),
                    "member_issue_classes": "|".join(sorted(issue_classes)),
                    "demotion_reasons": "|".join(sorted(reasons)),
                }
            )

    dem_df = pd.DataFrame.from_records(demotions).sort_values(
        ["newspaper_slug", "ordinance_event_id"]
    ) if demotions else pd.DataFrame(
        columns=[
            "newspaper_slug",
            "ordinance_event_id",
            "original_event_type",
            "member_issue_count",
            "member_issue_classes",
            "demotion_reasons",
        ]
    )
    remap_df = pd.DataFrame.from_records(remaps).sort_values(
        ["newspaper_slug", "ordinance_event_id"]
    ) if remaps else pd.DataFrame(
        columns=[
            "newspaper_slug",
            "ordinance_event_id",
            "original_event_type",
            "original_event_status",
            "new_event_type",
            "new_event_status",
            "reason",
        ]
    )

    asn_out.to_csv(out_dir / "issue_event_assignments_status_v3_precision.csv", index=False)
    cl_out.to_csv(out_dir / "ordinance_event_clusters_status_v3_precision.csv", index=False)
    dem_df.to_csv(out_dir / "precision_demotions.csv", index=False)
    remap_df.to_csv(out_dir / "precision_remaps.csv", index=False)

    summary = {
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "script_path": str(Path(__file__).resolve()),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "pipeline_root": str(pipeline_root),
        "issue_assignments_csv": str(asn_path),
        "event_clusters_csv": str(cl_path),
        "output_dir": str(out_dir),
        "input_issue_rows": int(asn.shape[0]),
        "input_event_rows": int(clusters.shape[0]),
        "demoted_events": int(dem_df.shape[0]),
        "remapped_events": int(remap_df.shape[0]),
        "strict_full_complete_only": bool(args.strict_full_complete_only),
        "output_event_type_counts_issue_rows": {
            k: int(v) for k, v in asn_out["event_type"].value_counts().to_dict().items()
        },
    }
    (out_dir / "provenance.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"done_enforce_precision output_dir={out_dir} demoted_events={summary['demoted_events']}")


if __name__ == "__main__":
    main()
