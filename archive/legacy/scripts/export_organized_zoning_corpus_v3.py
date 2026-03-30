#!/usr/bin/env python3
"""
Export an organized zoning-law corpus from a fullnewspaper v3 run.

Primary goal:
- Produce a deduplicated, structured set of legal event bundles with strict buckets:
  full_ordinance (enacted complete only), partial_ordinance_enacted,
  amendment, proposal_full_text, notice, building_code,
  out_of_jurisdiction_law,
  and uncertain_other.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Any

import pandas as pd


LAW_ISSUE_CLASSES = {
    "full_code_comprehensive",
    "full_code_recodification",
    "full_code_partial_enacted",
    "full_code_partial_proposed",
    "full_code_proposed",
    "amendment_substantial",
    "amendment_targeted",
    "amendment_proposed",
    "zoning_notice_procedural",
    "zoning_notice_adoption",
}

EXPORT_BUCKETS = [
    "full_ordinance",
    "partial_ordinance_enacted",
    "amendment",
    "proposal_full_text",
    "notice",
    "building_code",
    "out_of_jurisdiction_law",
    "uncertain_other",
]


def _norm(x: Any) -> str:
    return str(x or "").strip()


def _slugify(s: str) -> str:
    t = _norm(s).lower()
    t = re.sub(r"[^a-z0-9]+", "-", t)
    t = re.sub(r"-{2,}", "-", t).strip("-")
    return t or "na"


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _bucket_for_event(
    event_type: str,
    event_status: str,
    qualification_decision: str,
    jurisdiction_match_city: str = "",
    valid_zoning_law_event: str = "",
) -> str:
    et = _norm(event_type)
    es = _norm(event_status)
    qd = _norm(qualification_decision).lower()
    proposed = {"proposed_full_text", "proposed_partial_installment"}
    jm = _norm(jurisdiction_match_city).lower()
    ve = _norm(valid_zoning_law_event).lower()

    if qd and qd != "pass":
        return "uncertain_other"

    # Preserve valid legal objects that are out of target-city jurisdiction.
    # These should remain available for audit and reuse but excluded from city-target buckets.
    if jm == "no" and ve != "no":
        if es in {"enacted_complete", "enacted_partial_installment", "proposed_full_text", "proposed_partial_installment", "notice"}:
            return "out_of_jurisdiction_law"

    if et == "full_ordinance":
        if es == "enacted_complete" or not es:
            return "full_ordinance"
        if es == "enacted_partial_installment":
            return "partial_ordinance_enacted"
        if es in proposed:
            return "proposal_full_text"
        return "uncertain_other"
    if et == "amendment":
        if es in proposed:
            return "proposal_full_text"
        return "amendment"
    if et == "proposal":
        return "proposal_full_text"
    if et == "notice":
        return "notice"
    if et == "building_code":
        return "building_code"
    return "uncertain_other"


def _derive_event_subtype(bucket: str, member_classes: list[str]) -> str:
    vals = {_norm(x) for x in member_classes if _norm(x)}
    if bucket in {"full_ordinance", "partial_ordinance_enacted"}:
        has_comp = "full_code_comprehensive" in vals
        has_reco = "full_code_recodification" in vals
        has_partial = "full_code_partial_enacted" in vals
        if has_comp and has_reco:
            return "mixed_full"
        if has_partial and (has_comp or has_reco):
            return "mixed_full_with_partial_installment"
        if has_comp:
            return "full_code_comprehensive"
        if has_reco:
            return "full_code_recodification"
        if has_partial:
            return "full_code_partial_enacted"
        return "full_unspecified"
    if bucket == "amendment":
        has_target = "amendment_targeted" in vals
        has_sub = "amendment_substantial" in vals
        if has_target and has_sub:
            return "mixed_amendment"
        if has_target:
            return "amendment_targeted"
        if has_sub:
            return "amendment_substantial"
        return "amendment_unspecified"
    if bucket == "proposal_full_text":
        has_full_prop = "full_code_proposed" in vals
        has_full_partial_prop = "full_code_partial_proposed" in vals
        has_amend_prop = "amendment_proposed" in vals
        if (has_full_prop or has_full_partial_prop) and has_amend_prop:
            return "mixed_proposal"
        if has_full_prop:
            return "full_code_proposed"
        if has_full_partial_prop:
            return "full_code_partial_proposed"
        if has_amend_prop:
            return "amendment_proposed"
        return "proposal_unspecified"
    if bucket == "notice":
        if "zoning_notice_adoption" in vals:
            return "zoning_notice_adoption"
        if "zoning_notice_procedural" in vals:
            return "zoning_notice_procedural"
        return "notice_unspecified"
    if bucket == "building_code":
        if "building_code_non_zoning" in vals:
            return "building_code_non_zoning"
        return "building_code_unspecified"
    if bucket == "out_of_jurisdiction_law":
        has_full = any(v in {"full_code_comprehensive", "full_code_recodification", "full_code_partial_enacted"} for v in vals)
        has_amend = any(v in {"amendment_substantial", "amendment_targeted"} for v in vals)
        has_prop = any(v in {"full_code_proposed", "full_code_partial_proposed", "amendment_proposed"} for v in vals)
        has_notice = any(v in {"zoning_notice_adoption", "zoning_notice_procedural"} for v in vals)
        has_building = any(v in {"building_code_non_zoning"} for v in vals)
        if has_full:
            return "out_of_jurisdiction_full"
        if has_amend:
            return "out_of_jurisdiction_amendment"
        if has_prop:
            return "out_of_jurisdiction_proposal"
        if has_notice:
            return "out_of_jurisdiction_notice"
        if has_building:
            return "out_of_jurisdiction_building_code"
        return "out_of_jurisdiction_unspecified"
    return "other_or_uncertain"


def _build_event_markdown(
    *,
    newspaper_slug: str,
    city_key: str,
    event_key: str,
    ordinance_event_id: str,
    event_type: str,
    event_status: str,
    qualification_decision: str,
    jurisdiction_match_city: str,
    governing_jurisdiction_name: str,
    governing_jurisdiction_type: str,
    governing_jurisdiction_state_abbr: str,
    export_bucket: str,
    event_subtype: str,
    first_issue_date: str,
    last_issue_date: str,
    issue_count: int,
    publications: pd.DataFrame,
    combined_text: str,
) -> str:
    lines: list[str] = []
    title = f"{newspaper_slug} | {export_bucket} | {ordinance_event_id}"
    lines.append(f"# {title}")
    lines.append("")
    lines.append("## Metadata")
    lines.append("")
    lines.append("| field | value |")
    lines.append("|---|---|")
    lines.append(f"| newspaper_slug | `{newspaper_slug}` |")
    lines.append(f"| city_key | `{city_key}` |")
    lines.append(f"| event_key | `{event_key}` |")
    lines.append(f"| ordinance_event_id | `{ordinance_event_id}` |")
    lines.append(f"| event_type | `{event_type}` |")
    lines.append(f"| event_status | `{event_status}` |")
    lines.append(f"| qualification_decision | `{qualification_decision}` |")
    lines.append(f"| jurisdiction_match_city | `{jurisdiction_match_city}` |")
    lines.append(f"| governing_jurisdiction_name | `{governing_jurisdiction_name}` |")
    lines.append(f"| governing_jurisdiction_type | `{governing_jurisdiction_type}` |")
    lines.append(f"| governing_jurisdiction_state_abbr | `{governing_jurisdiction_state_abbr}` |")
    lines.append(f"| export_bucket | `{export_bucket}` |")
    lines.append(f"| event_subtype | `{event_subtype}` |")
    lines.append(f"| issue_count | {issue_count} |")
    lines.append(f"| first_issue_date | `{first_issue_date}` |")
    lines.append(f"| last_issue_date | `{last_issue_date}` |")
    lines.append("")
    lines.append("## Publication Members")
    lines.append("")
    lines.append("| issue_date | issue_id | publication_role | issue_class | clean_chars |")
    lines.append("|---|---|---|---|---:|")
    for _, r in publications.iterrows():
        issue_class = _norm(r.get("issue_class_label")) or _norm(r.get("issue_class"))
        lines.append(
            "| "
            + f"{_norm(r.get('issue_date'))} | `{_norm(r.get('issue_id'))}` | "
            + f"{_norm(r.get('publication_role'))} | {issue_class} | "
            + f"{int(float(r.get('clean_chars') or 0))} |"
        )
    lines.append("")
    lines.append("## Self-Contained Legal Text")
    lines.append("")
    lines.append("```text")
    lines.append(combined_text.rstrip("\n"))
    lines.append("```")
    lines.append("")
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Export organized zoning corpus from fullnewspaper v3 outputs.")
    ap.add_argument(
        "--run-root",
        default="/Users/saulrichardson/projects/newspapers/newspaper-analysis/reports/runs/pi_v1_run60_fullnewspaper/fullnewspaper_v3_run_20260220_021657",
        help="Path to fullnewspaper_v3 run root.",
    )
    ap.add_argument(
        "--output-dir",
        default="",
        help="Destination directory (default: <run-root>/organized_corpus_v3).",
    )
    ap.add_argument(
        "--classification-csv",
        default="",
        help="Optional override path for classification CSV.",
    )
    ap.add_argument(
        "--clean-csv",
        default="",
        help="Optional override path for clean-zoning CSV.",
    )
    ap.add_argument(
        "--event-assignments-csv",
        default="",
        help="Optional override path for issue-event assignments CSV.",
    )
    ap.add_argument(
        "--event-clusters-csv",
        default="",
        help="Optional override path for event clusters CSV.",
    )
    return ap.parse_args()


def _resolve_input(path_arg: str, fallback: Path) -> Path:
    if _norm(path_arg):
        return Path(path_arg).expanduser().resolve()
    return fallback


def main() -> None:
    args = _parse_args()
    run_root = Path(args.run_root).expanduser().resolve()
    out_root = Path(args.output_dir).expanduser().resolve() if _norm(args.output_dir) else run_root / "organized_corpus_v3"
    out_root.mkdir(parents=True, exist_ok=True)
    summary_dir = out_root / "summary"
    by_news_dir = out_root / "by_newspaper"

    # Rebuild export tree from scratch each run.
    if summary_dir.is_dir():
        shutil.rmtree(summary_dir)
    if by_news_dir.is_dir():
        shutil.rmtree(by_news_dir)
    summary_dir.mkdir(parents=True, exist_ok=True)
    by_news_dir.mkdir(parents=True, exist_ok=True)

    p_labels = _resolve_input(args.classification_csv, run_root / "classification" / "issue_labels_v3.csv")
    p_clean = _resolve_input(args.clean_csv, run_root / "cleaning" / "clean_zoning_text_flat.csv")
    p_assign = _resolve_input(args.event_assignments_csv, run_root / "events" / "issue_event_assignments_v3.csv")
    p_events = _resolve_input(args.event_clusters_csv, run_root / "events" / "ordinance_event_clusters_v3.csv")

    labels = _safe_read_csv(p_labels)
    clean = _safe_read_csv(p_clean)
    assign = _safe_read_csv(p_assign)
    events = _safe_read_csv(p_events)

    if labels.empty or clean.empty or assign.empty or events.empty:
        raise SystemExit("Missing required inputs. Ensure classification/cleaning/events outputs exist.")

    labels = labels.copy()
    clean = clean.copy()
    assign = assign.copy()
    events = events.copy()

    for df in (labels, clean, assign, events):
        for col in ("issue_id", "newspaper_slug", "city_key"):
            if col in df.columns:
                df[col] = df[col].map(_norm)

    issue_cols = [
        c
        for c in [
            "newspaper_slug",
            "city_key",
            "issue_id",
            "issue_date",
            "issue_class",
            "legal_status",
            "enactment_signal",
            "proposal_signal",
            "contains_verbatim_zoning_law",
            "law_text_sufficiency",
            "confidence_0_to_1",
        ]
        if c in labels.columns
    ]
    clean_cols = [
        c
        for c in [
            "issue_id",
            "clean_zoning_text",
            "clean_chars",
            "kept_anchor_count",
            "anchor_count",
            "variant",
        ]
        if c in clean.columns
    ]
    issue_rows = (
        labels[issue_cols]
        .merge(clean[clean_cols], on="issue_id", how="left")
        .sort_values(["newspaper_slug", "issue_date", "issue_id"])
        .reset_index(drop=True)
    )
    issue_rows["event_key"] = ""
    issue_rows.to_csv(summary_dir / "issue_rows_all.csv", index=False)

    law_rows = issue_rows[issue_rows.get("issue_class", pd.Series(dtype=str)).astype(str).isin(LAW_ISSUE_CLASSES)].copy()
    law_rows.to_csv(summary_dir / "issue_rows_law_only.csv", index=False)
    law_rows[law_rows["issue_class"].astype(str).str.startswith("full_code")].to_csv(
        summary_dir / "full_ordinance_issue_rows.csv", index=False
    )
    law_rows[law_rows["issue_class"].astype(str).str.startswith("amendment")].to_csv(
        summary_dir / "amendment_issue_rows.csv", index=False
    )

    member_cols = [
        c
        for c in [
            "newspaper_slug",
            "city_key",
            "issue_id",
            "issue_date",
            "event_type",
            "event_status",
            "ordinance_event_id",
            "publication_role",
            "issue_class",
        ]
        if c in assign.columns
    ]
    members = assign[member_cols].copy().sort_values(["newspaper_slug", "issue_date", "issue_id"])
    if "event_status" not in members.columns:
        members["event_status"] = ""
    members["event_type"] = members["event_type"].map(_norm)
    members["ordinance_event_id"] = members["ordinance_event_id"].map(_norm)
    members["event_key"] = members["newspaper_slug"].map(_norm) + "::" + members["ordinance_event_id"].map(_norm)

    merge_issue_cols = [
        c
        for c in [
            "issue_id",
            "issue_class",
            "legal_status",
            "enactment_signal",
            "proposal_signal",
            "contains_verbatim_zoning_law",
            "law_text_sufficiency",
            "clean_chars",
            "clean_zoning_text",
        ]
        if c in issue_rows.columns
    ]
    members = members.merge(issue_rows[merge_issue_cols], on="issue_id", how="left", suffixes=("", "_label"))
    members.to_csv(summary_dir / "event_publication_members.csv", index=False)

    event_rows = events.copy()
    event_rows["event_type"] = event_rows["event_type"].map(_norm)
    event_rows["ordinance_event_id"] = event_rows["ordinance_event_id"].map(_norm)
    event_rows["newspaper_slug"] = event_rows["newspaper_slug"].map(_norm)
    if "event_status" not in event_rows.columns:
        event_rows["event_status"] = ""
    event_rows["event_status"] = event_rows["event_status"].map(_norm)
    if "qualification_decision" not in event_rows.columns:
        event_rows["qualification_decision"] = ""
    event_rows["qualification_decision"] = event_rows["qualification_decision"].map(_norm)
    event_rows["event_key"] = event_rows["newspaper_slug"] + "::" + event_rows["ordinance_event_id"]
    if "jurisdiction_match_city" not in event_rows.columns:
        event_rows["jurisdiction_match_city"] = ""
    if "valid_zoning_law_event" not in event_rows.columns:
        event_rows["valid_zoning_law_event"] = ""
    event_rows["export_bucket"] = event_rows.apply(
        lambda r: _bucket_for_event(
            _norm(r.get("event_type")),
            _norm(r.get("event_status")),
            _norm(r.get("qualification_decision")),
            _norm(r.get("jurisdiction_match_city")),
            _norm(r.get("valid_zoning_law_event")),
        ),
        axis=1,
    )

    subtype_by_event = {}
    for ekey, grp in members.groupby("event_key", dropna=False):
        ev_meta = event_rows[event_rows["event_key"] == ekey]
        bucket = _norm(ev_meta["export_bucket"].iloc[0]) if not ev_meta.empty else "uncertain_other"
        cls_col = "issue_class_label" if "issue_class_label" in grp.columns else "issue_class"
        subtype_by_event[ekey] = _derive_event_subtype(bucket, grp.get(cls_col, pd.Series(dtype=str)).astype(str).tolist())
    event_rows["event_subtype"] = event_rows["event_key"].map(lambda k: subtype_by_event.get(k, "other_or_uncertain"))

    event_rows = event_rows.sort_values(["newspaper_slug", "first_issue_date", "ordinance_event_id"]).reset_index(drop=True)
    event_rows.to_csv(summary_dir / "event_rows_all.csv", index=False)
    for bucket in EXPORT_BUCKETS:
        event_rows[event_rows["export_bucket"] == bucket].to_csv(summary_dir / f"{bucket}_events.csv", index=False)

    if "first_issue_date" in event_rows.columns:
        tmp = event_rows[event_rows["export_bucket"].isin(EXPORT_BUCKETS)].copy()
        tmp["year"] = pd.to_datetime(tmp["first_issue_date"], errors="coerce").dt.year
        timeline = (
            tmp[tmp["year"].notna()]
            .groupby(["newspaper_slug", "year", "export_bucket"], dropna=False)["event_key"]
            .nunique()
            .reset_index(name="event_count")
            .sort_values(["newspaper_slug", "year", "export_bucket"])
            .reset_index(drop=True)
        )
    else:
        timeline = pd.DataFrame(columns=["newspaper_slug", "year", "export_bucket", "event_count"])
    timeline.to_csv(summary_dir / "city_timeline_events.csv", index=False)

    top_index_lines = [
        "# Organized Zoning Corpus v3",
        "",
        "This directory contains deduplicated, self-contained zoning-law event bundles.",
        "",
        "## Newspapers",
        "",
        "| newspaper_slug | full_ordinance | partial_ordinance_enacted | amendment | proposal_full_text | notice | building_code | out_of_jurisdiction_law | uncertain_other |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for slug, grp_events in event_rows.groupby("newspaper_slug", dropna=False):
        slug_dir = by_news_dir / _slugify(slug)
        bucket_dirs = {b: slug_dir / b for b in EXPORT_BUCKETS}
        for b in EXPORT_BUCKETS:
            bucket_dirs[b].mkdir(parents=True, exist_ok=True)

        bucket_counts = {b: int((grp_events["export_bucket"] == b).sum()) for b in EXPORT_BUCKETS}
        top_index_lines.append(
            "| `{}` | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                slug,
                bucket_counts["full_ordinance"],
                bucket_counts["partial_ordinance_enacted"],
                bucket_counts["amendment"],
                bucket_counts["proposal_full_text"],
                bucket_counts["notice"],
                bucket_counts["building_code"],
                bucket_counts["out_of_jurisdiction_law"],
                bucket_counts["uncertain_other"],
            )
        )

        mem_slug = members[members["newspaper_slug"] == slug].copy()
        bucket_indexes: dict[str, list[str]] = {}
        for b in EXPORT_BUCKETS:
            bucket_indexes[b] = [
                f"# {slug} | {b} events",
                "",
                "| ordinance_event_id | first_issue_date | last_issue_date | issue_count | event_subtype | event_status | markdown |",
                "|---|---|---|---:|---|---|---|",
            ]

        for _, ev in grp_events.iterrows():
            bucket = _norm(ev.get("export_bucket"))
            eid = _norm(ev.get("ordinance_event_id"))
            ekey = _norm(ev.get("event_key"))
            if not eid or bucket not in bucket_dirs:
                continue

            ev_dir = bucket_dirs[bucket] / _slugify(eid)
            pub_dir = ev_dir / "publications"
            pub_dir.mkdir(parents=True, exist_ok=True)

            mem = mem_slug[mem_slug["event_key"] == ekey].copy().sort_values(["issue_date", "issue_id"])
            keep_cols = [
                c
                for c in [
                    "issue_id",
                    "issue_date",
                    "publication_role",
                    "issue_class_label",
                    "issue_class",
                    "legal_status",
                    "enactment_signal",
                    "proposal_signal",
                    "contains_verbatim_zoning_law",
                    "law_text_sufficiency",
                    "clean_chars",
                ]
                if c in mem.columns
            ]
            mem[keep_cols].to_csv(ev_dir / "publications.csv", index=False)

            combined_parts: list[str] = []
            for _, r in mem.iterrows():
                iid = _norm(r.get("issue_id"))
                dt = _norm(r.get("issue_date"))
                role = _norm(r.get("publication_role"))
                txt = _norm(r.get("clean_zoning_text"))
                header = f"[ISSUE] {iid} | date={dt} | role={role}\n"
                body = txt + "\n"
                combined_parts.append(header + body)
                fname = f"{_slugify(dt)}__{_slugify(iid)}.txt"
                (pub_dir / fname).write_text(header + body, encoding="utf-8")

            combined_text = "\n".join(combined_parts).strip() + "\n"
            (ev_dir / "clean_text_combined.txt").write_text(combined_text, encoding="utf-8")

            manifest = {
                "newspaper_slug": slug,
                "city_key": _norm(ev.get("city_key")),
                "target_city_name": _norm(ev.get("target_city_name")),
                "target_state_abbr": _norm(ev.get("target_state_abbr")),
                "event_key": ekey,
                "ordinance_event_id": eid,
                "event_type": _norm(ev.get("event_type")),
                "event_status": _norm(ev.get("event_status")),
                "qualification_decision": _norm(ev.get("qualification_decision")),
                "jurisdiction_match_city": _norm(ev.get("jurisdiction_match_city")),
                "governing_jurisdiction_name": _norm(ev.get("governing_jurisdiction_name")),
                "governing_jurisdiction_type": _norm(ev.get("governing_jurisdiction_type")),
                "governing_jurisdiction_state_abbr": _norm(ev.get("governing_jurisdiction_state_abbr")),
                "valid_zoning_law_event": _norm(ev.get("valid_zoning_law_event")),
                "export_bucket": bucket,
                "event_subtype": _norm(ev.get("event_subtype")),
                "issue_count": int(float(ev.get("issue_count") or 0)),
                "first_issue_date": _norm(ev.get("first_issue_date")),
                "last_issue_date": _norm(ev.get("last_issue_date")),
                "publication_ids": mem["issue_id"].astype(str).tolist() if "issue_id" in mem.columns else [],
            }
            (ev_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            md = _build_event_markdown(
                newspaper_slug=slug,
                city_key=_norm(ev.get("city_key")),
                event_key=ekey,
                ordinance_event_id=eid,
                event_type=_norm(ev.get("event_type")),
                event_status=_norm(ev.get("event_status")),
                qualification_decision=_norm(ev.get("qualification_decision")),
                jurisdiction_match_city=_norm(ev.get("jurisdiction_match_city")),
                governing_jurisdiction_name=_norm(ev.get("governing_jurisdiction_name")),
                governing_jurisdiction_type=_norm(ev.get("governing_jurisdiction_type")),
                governing_jurisdiction_state_abbr=_norm(ev.get("governing_jurisdiction_state_abbr")),
                export_bucket=bucket,
                event_subtype=_norm(ev.get("event_subtype")),
                first_issue_date=_norm(ev.get("first_issue_date")),
                last_issue_date=_norm(ev.get("last_issue_date")),
                issue_count=int(float(ev.get("issue_count") or 0)),
                publications=mem,
                combined_text=combined_text,
            )
            (ev_dir / "event.md").write_text(md, encoding="utf-8")

            idx_line = (
                f"| `{eid}` | `{_norm(ev.get('first_issue_date'))}` | `{_norm(ev.get('last_issue_date'))}` | "
                f"{int(float(ev.get('issue_count') or 0))} | {_norm(ev.get('event_subtype'))} | "
                f"{_norm(ev.get('event_status'))} | "
                f"[event.md]({_slugify(eid)}/event.md) |"
            )
            bucket_indexes[bucket].append(idx_line)

        for b in EXPORT_BUCKETS:
            grp_bucket = grp_events[grp_events["export_bucket"] == b].copy()
            grp_bucket.to_csv(bucket_dirs[b] / "events.csv", index=False)
            (bucket_dirs[b] / "index.md").write_text("\n".join(bucket_indexes[b]).strip() + "\n", encoding="utf-8")

        slug_readme = [
            f"# {slug}",
            "",
            "- [Full ordinances](full_ordinance/index.md)",
            "- [Partial enacted ordinances](partial_ordinance_enacted/index.md)",
            "- [Amendments](amendment/index.md)",
            "- [Proposal full text](proposal_full_text/index.md)",
            "- [Notices](notice/index.md)",
            "- [Building-code legal text](building_code/index.md)",
            "- [Out-of-jurisdiction law](out_of_jurisdiction_law/index.md)",
            "- [Uncertain / other](uncertain_other/index.md)",
            "",
        ]
        (slug_dir / "README.md").write_text("\n".join(slug_readme), encoding="utf-8")

    inv = {
        "run_root": str(run_root),
        "output_dir": str(out_root),
        "issue_rows_total": int(len(issue_rows)),
        "issue_rows_law_only": int(len(law_rows)),
        "event_rows_total": int(len(event_rows)),
        "event_rows_full_ordinance": int((event_rows["export_bucket"] == "full_ordinance").sum()),
        "event_rows_partial_ordinance_enacted": int((event_rows["export_bucket"] == "partial_ordinance_enacted").sum()),
        "event_rows_amendment": int((event_rows["export_bucket"] == "amendment").sum()),
        "event_rows_proposal_full_text": int((event_rows["export_bucket"] == "proposal_full_text").sum()),
        "event_rows_notice": int((event_rows["export_bucket"] == "notice").sum()),
        "event_rows_building_code": int((event_rows["export_bucket"] == "building_code").sum()),
        "event_rows_out_of_jurisdiction_law": int((event_rows["export_bucket"] == "out_of_jurisdiction_law").sum()),
        "event_rows_uncertain_other": int((event_rows["export_bucket"] == "uncertain_other").sum()),
        "newspaper_count": int(event_rows["newspaper_slug"].astype(str).nunique() if "newspaper_slug" in event_rows.columns else 0),
        "input_paths": {
            "classification_csv": str(p_labels),
            "clean_csv": str(p_clean),
            "event_assignments_csv": str(p_assign),
            "event_clusters_csv": str(p_events),
        },
    }
    (summary_dir / "export_inventory.json").write_text(json.dumps(inv, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out_root / "README.md").write_text("\n".join(top_index_lines).strip() + "\n", encoding="utf-8")

    print(f"done_organized_corpus output_dir={out_root}")
    print(json.dumps(inv, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
