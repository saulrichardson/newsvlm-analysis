#!/usr/bin/env python3
"""
Build a precision-recovery memo and an exhaustive manual validation packet.

The packet is event-centric and targets full-ordinance + amendment events from the
winner run. For each event, it writes:

- event metadata and review template
- publication members with review fields
- per-issue transcript + clean text
- per-page transcript snippets and page PNGs (kept-anchor pages first; fallback pages if none)

This is designed to support exhaustive human review of whether outputs are:

1) zoning-law text (not narrative bleed),
2) correctly typed (full / amendment / proposal / other),
3) correctly deduplicated.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import re
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


REVIEW_COLS = [
    "review_label_correct",
    "review_status_correct",
    "review_completeness_correct",
    "review_dedup_correct",
    "review_clean_text_fidelity",
    "review_notes",
]


def _norm(x: Any) -> str:
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    t = str(x).strip()
    if t.lower() == "nan":
        return ""
    return t


def _slug(s: str) -> str:
    t = _norm(s).lower()
    t = re.sub(r"[^a-z0-9._-]+", "-", t)
    t = re.sub(r"-{2,}", "-", t).strip("-")
    return t or "na"


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _iter_jsonl(path: Path):
    if not path.is_file():
        return
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            s = raw.strip()
            if not s:
                continue
            try:
                yield json.loads(s)
            except Exception:
                continue


def _read_issue_pages_from_panels(panel_root: Path) -> tuple[dict[str, list[str]], dict[str, str], dict[str, str], dict[str, str]]:
    issue_page_ids: dict[str, list[str]] = {}
    issue_text: dict[str, str] = {}
    issue_date: dict[str, str] = {}
    issue_city: dict[str, str] = {}

    if not panel_root.is_dir():
        return issue_page_ids, issue_text, issue_date, issue_city

    for city_dir in sorted(panel_root.iterdir()):
        if not city_dir.is_dir():
            continue
        p = city_dir / "issue_texts.jsonl"
        if not p.is_file():
            continue
        for obj in _iter_jsonl(p):
            iid = _norm(obj.get("issue_id"))
            if not iid:
                continue
            pids = obj.get("page_ids") if isinstance(obj.get("page_ids"), list) else []
            issue_page_ids[iid] = [_norm(x) for x in pids if _norm(x)]
            issue_text[iid] = _norm(obj.get("text"))
            issue_date[iid] = _norm(obj.get("issue_date"))
            issue_city[iid] = _norm(obj.get("city_key"))
    return issue_page_ids, issue_text, issue_date, issue_city


def _read_page_text_from_page_tags(path: Path) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = defaultdict(dict)
    for obj in _iter_jsonl(path):
        iid = _norm(obj.get("issue_id"))
        if not iid:
            continue
        segs = obj.get("page_segments")
        if not isinstance(segs, list):
            continue
        for seg in segs:
            if not isinstance(seg, dict):
                continue
            pid = _norm(seg.get("page_id"))
            txt = _norm(seg.get("text"))
            if pid:
                out[iid][pid] = txt
    return out


def _read_anchor_decisions(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for obj in _iter_jsonl(path):
        iid = _norm(obj.get("issue_id"))
        if not iid:
            continue
        out[iid] = obj if isinstance(obj, dict) else {}
    return out


def _extract_kept_pages(anchor_obj: dict[str, Any]) -> tuple[list[str], dict[str, list[str]]]:
    anchors = anchor_obj.get("anchors") if isinstance(anchor_obj.get("anchors"), list) else []
    keep_ids = set()
    if isinstance(anchor_obj.get("kept_anchor_ids"), list):
        keep_ids = {_norm(x) for x in anchor_obj.get("kept_anchor_ids", []) if _norm(x)}

    page_order: list[str] = []
    page_to_snips: dict[str, list[str]] = defaultdict(list)
    for a in anchors:
        if not isinstance(a, dict):
            continue
        aid = _norm(a.get("anchor_id"))
        if aid and keep_ids and aid not in keep_ids:
            continue
        keep_flag = _norm(a.get("keep_for_law_corpus")).lower()
        if keep_ids:
            include = aid in keep_ids
        else:
            include = keep_flag in {"yes", "true", "1"}
        if not include:
            continue
        pid = _norm(a.get("page_id"))
        txt = _norm(a.get("text"))
        if not pid:
            continue
        if pid not in page_to_snips:
            page_order.append(pid)
        if txt:
            page_to_snips[pid].append(txt)
    return page_order, page_to_snips


def _find_local_png(page_id: str, cache_dirs: list[Path]) -> tuple[bool, str]:
    filename = f"{page_id}.png"
    for root in cache_dirs:
        if not root.is_dir():
            continue
        cand = root / filename
        if cand.is_file():
            return True, str(cand)
    return False, "missing"


def _fetch_png_from_torch(page_id: str, torch_host: str, dest_png: Path) -> tuple[bool, str]:
    filename = f"{page_id}.png"
    remote_unique = f"/scratch/sxr203/newspaper-downloads/dedupe-webp/unique_png/{filename}"
    try:
        rc = subprocess.run(
            ["scp", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", f"{torch_host}:{remote_unique}", str(dest_png)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=50,
        )
        if rc.returncode == 0 and dest_png.is_file():
            return True, remote_unique
    except Exception:
        pass
    return False, "missing"


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        pd.DataFrame().to_csv(path, index=False)
        return
    pd.DataFrame.from_records(rows).to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)


def _build_memo(
    *,
    memo_path: Path,
    precision_root: Path,
    winner_root: Path,
    output_dir: Path,
    pilot_summary: dict[str, Any],
    scoreboard: pd.DataFrame,
    export_inv: dict[str, Any],
    event_queue: pd.DataFrame,
) -> None:
    top_round2 = scoreboard[scoreboard["round"].astype(str) == "round2"].sort_values("score", ascending=False).copy()
    full_df = event_queue[event_queue["event_type"] == "full_ordinance"].copy()
    amend_df = event_queue[event_queue["event_type"] == "amendment"].copy()

    full_flagged = full_df[
        (full_df["risk_proposal_terms"] == 1)
        | (full_df["risk_nonzoning_terms"] == 1)
        | (full_df["risk_low_clean_chars"] == 1)
    ].copy()

    lines: list[str] = []
    lines.append("# Precision-Recovery Memo: Newspapers to Clean Zoning Ordinance Objects")
    lines.append("")
    lines.append(f"- Generated: {dt.datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"- Precision run root: `{precision_root}`")
    lines.append(f"- Winner run root: `{winner_root}`")
    lines.append(f"- Validation packet root: `{output_dir}`")
    lines.append("")
    lines.append("## Objective")
    lines.append("")
    lines.append(
        "The objective is to convert newspaper transcripts into clean, structured zoning-law objects "
        "that are explicitly typed as enacted full ordinance, amendment, proposal full text, notice, or uncertain/other, "
        "with event deduplication and publication lineage."
    )
    lines.append("")
    lines.append("## Run Outcome")
    lines.append("")
    lines.append(
        f"- Winner selected by pilot: `{_norm((pilot_summary.get('winner_round2') or {}).get('variant_label'))}`"
    )
    lines.append(
        f"- Winner score: `{_norm((pilot_summary.get('winner_round2') or {}).get('score'))}`"
    )
    lines.append(
        f"- Winner config: cleaning=`{_norm((pilot_summary.get('selected_config') or {}).get('cleaning_variant'))}`, "
        f"classification=`{_norm((pilot_summary.get('selected_config') or {}).get('classification_variant'))}`, "
        f"dedup=`{_norm((pilot_summary.get('selected_config') or {}).get('dedup_variant'))}`"
    )
    lines.append("")
    lines.append("Event inventory from winner export:")
    lines.append(
        f"- Full ordinance events: `{int(export_inv.get('event_rows_full_ordinance', 0) or 0)}`"
    )
    lines.append(
        f"- Amendment events: `{int(export_inv.get('event_rows_amendment', 0) or 0)}`"
    )
    lines.append(
        f"- Proposal full-text events: `{int(export_inv.get('event_rows_proposal_full_text', 0) or 0)}`"
    )
    lines.append(
        f"- Notice events: `{int(export_inv.get('event_rows_notice', 0) or 0)}`"
    )
    lines.append("")
    lines.append("## Exhaustive Validation Packet")
    lines.append("")
    lines.append(
        "The packet is exhaustive for full-ordinance and amendment events in the winner run."
    )
    lines.append("")
    lines.append(
        f"- Event rows in queue: `{len(event_queue)}` (full=`{len(full_df)}`, amendment=`{len(amend_df)}`)"
    )
    lines.append(
        f"- Full events with risk flags (proposal/non-zoning/low-clean): `{len(full_flagged)}`"
    )
    lines.append("")
    lines.append("Packet files:")
    lines.append(f"- `{output_dir / 'event_validation_queue.csv'}`")
    lines.append(f"- `{output_dir / 'issue_validation_queue.csv'}`")
    lines.append(f"- `{output_dir / 'page_asset_inventory.csv'}`")
    lines.append(f"- `{output_dir / 'summary' / 'packet_summary.json'}`")
    lines.append(f"- `{output_dir / 'by_newspaper'}`")
    lines.append("")
    lines.append("## What This Already Tells Us")
    lines.append("")
    lines.append(
        "The pipeline is now avoiding proposal-as-full leakage in the winner metrics, "
        "but manual review is still required to confirm semantic fit of `full_ordinance` and `amendment` buckets "
        "against original newspaper pages and full transcripts."
    )
    lines.append("")
    lines.append(
        "In preliminary inspection of full-ordinance objects, some rows appear to include substantial non-zoning legal material "
        "(for example, building-code style sections). Those should be treated as high-priority review targets."
    )
    lines.append("")
    if not full_flagged.empty:
        lines.append("## High-Priority Full-Event Review Targets")
        lines.append("")
        show_cols = [
            "newspaper_slug",
            "ordinance_event_id",
            "first_issue_date",
            "risk_proposal_terms",
            "risk_nonzoning_terms",
            "risk_low_clean_chars",
            "member_clean_chars_total",
        ]
        ff = full_flagged.sort_values(
            ["risk_nonzoning_terms", "risk_proposal_terms", "member_clean_chars_total"],
            ascending=[False, False, True],
        ).head(12)
        lines.append("| newspaper | event_id | first_date | proposal_flag | nonzoning_flag | low_clean_flag | clean_chars |")
        lines.append("|---|---|---:|---:|---:|---:|---:|")
        for _, r in ff.iterrows():
            lines.append(
                f"| {_norm(r['newspaper_slug'])} | {_norm(r['ordinance_event_id'])} | {_norm(r['first_issue_date'])} | "
                f"{int(r['risk_proposal_terms'])} | {int(r['risk_nonzoning_terms'])} | {int(r['risk_low_clean_chars'])} | "
                f"{int(float(r['member_clean_chars_total'] or 0))} |"
            )
        lines.append("")
    lines.append("## Recommended Next Step")
    lines.append("")
    lines.append(
        "Complete event-by-event human review in the packet (all full + all amendment), fill verdict columns, "
        "then use failure clusters to tighten ontology boundaries between zoning-law text and other legal text "
        "(especially building-code, utility-assessment, and notice-heavy items)."
    )
    lines.append("")
    lines.append("## Round-2 Candidate Score Snapshot")
    lines.append("")
    if top_round2.empty:
        lines.append("No round2 rows found in scoreboard.")
    else:
        lines.append("| candidate | score | full_events | amend_events | proposal_as_full | dedup_collision |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for _, r in top_round2.head(10).iterrows():
            lines.append(
                f"| {_norm(r['variant_label'])} | {float(r['score']):.6f} | "
                f"{int(float(r['full_event_count'] or 0))} | {int(float(r['amend_event_count'] or 0))} | "
                f"{int(float(r['proposal_as_full_count'] or 0))} | {float(r['dedup_collision_rate']):.6f} |"
            )
    memo_path.parent.mkdir(parents=True, exist_ok=True)
    memo_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build precision-recovery memo + exhaustive manual validation packet.")
    ap.add_argument(
        "--precision-run-root",
        default="/Users/saulrichardson/projects/newspapers/newspaper-analysis/reports/runs/pi_v1_run60_fullnewspaper/precision_recovery_pilot_v4_20260222_gemini",
        help="Precision pilot run root containing pilot_summary.json and shared/page_tagging.",
    )
    ap.add_argument(
        "--winner-run-root",
        default="",
        help="Winner run root. If empty, read winner_round2.run_root from pilot_summary.json.",
    )
    ap.add_argument(
        "--panel-run-root",
        default="/Users/saulrichardson/projects/newspapers/newspaper-analysis/reports/runs/pi_v1_run60_fullnewspaper",
        help="Run root that contains panels/<city_key>/issue_texts.jsonl.",
    )
    ap.add_argument(
        "--output-dir",
        default="",
        help="Packet output directory (default: <winner-run-root>/manual_validation/exhaustive_packet_v2).",
    )
    ap.add_argument(
        "--memo-path",
        default="",
        help="Memo output path (default: <winner-run-root>/manual_validation/precision_recovery_memo.md).",
    )
    ap.add_argument(
        "--event-types",
        default="full_ordinance,amendment",
        help="Comma-separated event types to include in exhaustive packet.",
    )
    ap.add_argument("--max-pages-per-issue", type=int, default=0, help="Optional cap for copied pages per issue (0=no cap).")
    ap.add_argument("--torch-host", default="torch", help="SSH host for PNG fetch fallback.")
    ap.add_argument("--no-torch-fetch", action="store_true", help="Disable torch PNG fetch fallback.")
    return ap.parse_args()


def main() -> None:
    args = _parse_args()

    precision_root = Path(args.precision_run_root).expanduser().resolve()
    pilot_summary_path = precision_root / "pilot_summary.json"
    if not pilot_summary_path.is_file():
        raise SystemExit(f"Missing pilot_summary.json: {pilot_summary_path}")
    pilot_summary = json.loads(pilot_summary_path.read_text(encoding="utf-8"))

    if _norm(args.winner_run_root):
        winner_root = Path(args.winner_run_root).expanduser().resolve()
    else:
        winner_root = Path(_norm((pilot_summary.get("winner_round2") or {}).get("run_root"))).expanduser().resolve()
    if not winner_root.is_dir():
        raise SystemExit(f"Winner run root not found: {winner_root}")

    panel_root = Path(args.panel_run_root).expanduser().resolve() / "panels"
    page_tags_path = precision_root / "shared" / "page_tagging" / "page_tags.jsonl"

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if _norm(args.output_dir)
        else winner_root / "manual_validation" / "exhaustive_packet_v2"
    )
    memo_path = (
        Path(args.memo_path).expanduser().resolve()
        if _norm(args.memo_path)
        else winner_root / "manual_validation" / "precision_recovery_memo.md"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    include_types = {_norm(x) for x in _norm(args.event_types).split(",") if _norm(x)}

    events = _safe_read_csv(winner_root / "events_status" / "ordinance_event_clusters_status_v3.csv")
    assign = _safe_read_csv(winner_root / "events_status" / "issue_event_assignments_status_v3.csv")
    labels = _safe_read_csv(winner_root / "classification" / "issue_labels_v3.csv")
    clean = _safe_read_csv(winner_root / "cleaning" / "clean_zoning_text_flat.csv")
    if events.empty or assign.empty:
        raise SystemExit("Missing event-status tables in winner run.")

    events = events[events["event_type"].astype(str).isin(include_types)].copy()
    events = events.sort_values(["newspaper_slug", "event_type", "first_issue_date", "ordinance_event_id"]).reset_index(drop=True)
    if events.empty:
        raise SystemExit(f"No events found for event_types={sorted(include_types)}.")

    clean_map: dict[str, dict[str, Any]] = {}
    if not clean.empty and "issue_id" in clean.columns:
        for _, r in clean.iterrows():
            iid = _norm(r.get("issue_id"))
            if not iid:
                continue
            clean_map[iid] = {
                "clean_text": _norm(r.get("clean_zoning_text")),
                "clean_chars": int(float(r.get("clean_chars") or 0)),
                "kept_anchor_count": int(float(r.get("kept_anchor_count") or 0)),
                "anchor_count": int(float(r.get("anchor_count") or 0)),
            }

    label_map: dict[str, dict[str, Any]] = {}
    if not labels.empty and "issue_id" in labels.columns:
        for _, r in labels.iterrows():
            iid = _norm(r.get("issue_id"))
            if iid:
                label_map[iid] = dict(r)

    issue_page_ids, issue_text_map, issue_date_map, issue_city_map = _read_issue_pages_from_panels(panel_root)
    page_text_map = _read_page_text_from_page_tags(page_tags_path)
    anchor_by_issue = _read_anchor_decisions(winner_root / "cleaning" / "anchor_decisions.jsonl")

    prev_log = _safe_read_csv(winner_root / "manual_validation" / "exhaustive_full_amend_validation_log.csv")
    prev_reviews: dict[tuple[str, str], dict[str, Any]] = {}
    if not prev_log.empty:
        for _, r in prev_log.iterrows():
            k = (_norm(r.get("newspaper_slug")), _norm(r.get("ordinance_event_id")))
            prev_reviews[k] = {c: _norm(r.get(c)) for c in REVIEW_COLS}

    cache_dirs = [
        winner_root / "reports" / "fullnewspaper_v3" / "figures" / "appendix_pages",
        winner_root / "reports" / "fullnewspaper_v3" / "figures" / "evidence_pages",
        precision_root / "reports" / "fullnewspaper_v3" / "figures" / "appendix_pages",
        precision_root / "reports" / "fullnewspaper_v3" / "figures" / "evidence_pages",
        Path("/Users/saulrichardson/projects/newspapers/old-ocr/newspaper-parsing-local/data/unique_png"),
        Path("/Users/saulrichardson/projects/newspapers/old-ocr/newspaper-parsing-local/data/issue_zoning_samples_20260125_223032"),
        Path("/Users/saulrichardson/projects/newspapers/old-ocr/newspaper-parsing-local/data/issue_zoning_samples_diverse_20260126_031245"),
        Path("/Users/saulrichardson/projects/newspapers/old-ocr/newspaper-parsing-local/data/issue_zoning_samples_diverse_fullissue_min2_20260126_032321"),
    ]

    event_rows: list[dict[str, Any]] = []
    issue_rows: list[dict[str, Any]] = []
    page_rows: list[dict[str, Any]] = []

    packet_root = output_dir / "by_newspaper"
    packet_root.mkdir(parents=True, exist_ok=True)

    proposal_terms = ["proposed", "proposal", "hearing", "recommend", "introduced", "draft"]
    nonzoning_terms = ["curb", "gutter", "sidewalk", "sewer", "assessment", "paving", "sanitary code", "building code"]

    for _, er in events.iterrows():
        slug = _norm(er.get("newspaper_slug"))
        city_key = _norm(er.get("city_key"))
        event_type = _norm(er.get("event_type"))
        event_status = _norm(er.get("event_status"))
        event_id = _norm(er.get("ordinance_event_id"))
        event_key = f"{slug}::{event_id}" if slug and event_id else ""

        mem = assign[
            (assign["newspaper_slug"].astype(str) == slug)
            & (assign["ordinance_event_id"].astype(str) == event_id)
        ].copy()
        mem = mem.sort_values(["issue_date", "issue_id"]).reset_index(drop=True)
        member_issue_ids = [_norm(x) for x in mem["issue_id"].astype(str).tolist()]

        rel_event_dir = Path("by_newspaper") / _slug(slug) / _slug(event_type) / _slug(event_id)
        event_dir = output_dir / rel_event_dir
        event_dir.mkdir(parents=True, exist_ok=True)

        member_clean_chars_total = 0
        member_issue_classes: list[str] = []
        selected_page_total = 0
        png_ok = 0
        png_missing = 0
        combined_clean_parts: list[str] = []

        for _, mr in mem.iterrows():
            iid = _norm(mr.get("issue_id"))
            issue_class = _norm(mr.get("issue_class"))
            member_issue_classes.append(issue_class)

            ce = clean_map.get(iid, {})
            clean_text = _norm(ce.get("clean_text"))
            member_clean_chars_total += int(ce.get("clean_chars") or len(clean_text))
            if clean_text:
                combined_clean_parts.append(clean_text)

        combined_clean_text = "\n\n".join([x for x in combined_clean_parts if x]).strip()
        ltxt = combined_clean_text.lower()
        risk_proposal = 1 if any(t in ltxt for t in proposal_terms) else 0
        risk_nonzoning = 1 if any(t in ltxt for t in nonzoning_terms) else 0
        risk_low_clean = 1 if member_clean_chars_total < 1200 else 0

        review_seed = prev_reviews.get((slug, event_id), {})

        event_row: dict[str, Any] = {
            "newspaper_slug": slug,
            "city_key": city_key,
            "event_type": event_type,
            "event_status": event_status,
            "qualification_decision": _norm(er.get("qualification_decision")),
            "ordinance_event_id": event_id,
            "event_key": event_key,
            "issue_count": int(float(er.get("issue_count") or len(mem))),
            "first_issue_date": _norm(er.get("first_issue_date")),
            "last_issue_date": _norm(er.get("last_issue_date")),
            "member_issue_ids": "|".join(member_issue_ids),
            "member_issue_classes": "|".join([x for x in member_issue_classes if x]),
            "member_clean_chars_total": int(member_clean_chars_total),
            "risk_proposal_terms": int(risk_proposal),
            "risk_nonzoning_terms": int(risk_nonzoning),
            "risk_low_clean_chars": int(risk_low_clean),
            "selected_page_count": 0,
            "png_ok": 0,
            "png_missing": 0,
            "event_folder_rel": str(rel_event_dir),
        }
        for c in REVIEW_COLS:
            event_row[c] = _norm(review_seed.get(c))

        publications_csv_rows: list[dict[str, Any]] = []
        pubs_root = event_dir / "publications"
        pubs_root.mkdir(parents=True, exist_ok=True)

        for _, mr in mem.iterrows():
            iid = _norm(mr.get("issue_id"))
            issue_date = _norm(mr.get("issue_date")) or issue_date_map.get(iid, "")
            issue_class = _norm(mr.get("issue_class"))
            pub_role = _norm(mr.get("publication_role"))
            label_row = label_map.get(iid, {})
            ce = clean_map.get(iid, {})
            clean_text = _norm(ce.get("clean_text"))
            issue_transcript = issue_text_map.get(iid, "")

            issue_anchor_obj = anchor_by_issue.get(iid, {})
            kept_pages, page_to_snips = _extract_kept_pages(issue_anchor_obj)
            if not kept_pages:
                kept_pages = [_norm(x) for x in issue_page_ids.get(iid, []) if _norm(x)][:3]
            if int(args.max_pages_per_issue) > 0 and len(kept_pages) > int(args.max_pages_per_issue):
                kept_pages = kept_pages[: int(args.max_pages_per_issue)]

            selected_page_total += len(kept_pages)
            issue_dir = pubs_root / _slug(iid)
            issue_dir.mkdir(parents=True, exist_ok=True)

            issue_meta = {
                "newspaper_slug": slug,
                "city_key": city_key,
                "event_type": event_type,
                "event_status": event_status,
                "ordinance_event_id": event_id,
                "issue_id": iid,
                "issue_date": issue_date,
                "publication_role": pub_role,
                "issue_class": issue_class,
                "legal_status": _norm(label_row.get("legal_status")),
                "enactment_signal": _norm(label_row.get("enactment_signal")),
                "proposal_signal": _norm(label_row.get("proposal_signal")),
                "contains_verbatim_zoning_law": _norm(label_row.get("contains_verbatim_zoning_law")),
                "law_text_sufficiency": _norm(label_row.get("law_text_sufficiency")),
                "clean_chars": int(ce.get("clean_chars") or 0),
                "kept_anchor_count": int(ce.get("kept_anchor_count") or 0),
                "selected_page_ids": kept_pages,
            }
            _write_json(issue_dir / "issue_metadata.json", issue_meta)
            (issue_dir / "issue_transcript_full.txt").write_text(issue_transcript + ("\n" if issue_transcript else ""), encoding="utf-8")
            (issue_dir / "clean_zoning_text.txt").write_text(clean_text + ("\n" if clean_text else ""), encoding="utf-8")

            snip_lines: list[str] = []
            for pid in kept_pages:
                for i, txt in enumerate(page_to_snips.get(pid, []), start=1):
                    snip_lines.extend([f"[{pid}] kept_anchor_snippet_{i}", txt, ""])
            (issue_dir / "kept_anchor_snippets.txt").write_text("\n".join(snip_lines).strip() + ("\n" if snip_lines else ""), encoding="utf-8")

            issue_png_ok = 0
            issue_png_missing = 0
            for idx, pid in enumerate(kept_pages, start=1):
                stem = f"page_{idx:03d}__{_slug(pid)}"
                txt_file = issue_dir / f"{stem}.txt"
                png_file = issue_dir / f"{stem}.png"
                page_txt = _norm(page_text_map.get(iid, {}).get(pid))
                if not page_txt:
                    joined = "\n\n".join(page_to_snips.get(pid, []))
                    page_txt = joined
                txt_file.write_text(page_txt + ("\n" if page_txt else ""), encoding="utf-8")

                ok, src = _find_local_png(pid, cache_dirs)
                if ok:
                    shutil.copy2(src, png_file)
                    png_ok += 1
                    issue_png_ok += 1
                    png_state = "ok"
                else:
                    if not bool(args.no_torch_fetch):
                        ok2, src2 = _fetch_png_from_torch(pid, str(args.torch_host), png_file)
                        if ok2:
                            png_ok += 1
                            issue_png_ok += 1
                            png_state = "ok"
                            src = src2
                        else:
                            png_missing += 1
                            issue_png_missing += 1
                            png_state = "missing"
                            src = "missing"
                    else:
                        png_missing += 1
                        issue_png_missing += 1
                        png_state = "missing"
                        src = "missing"

                page_rows.append(
                    {
                        "newspaper_slug": slug,
                        "event_type": event_type,
                        "ordinance_event_id": event_id,
                        "issue_id": iid,
                        "issue_date": issue_date,
                        "page_id": pid,
                        "txt_file_rel": str(txt_file.relative_to(output_dir)),
                        "png_file_rel": str(png_file.relative_to(output_dir)),
                        "png_status": png_state,
                        "png_source": src,
                    }
                )

            issue_rows.append(
                {
                    "newspaper_slug": slug,
                    "city_key": city_key,
                    "event_type": event_type,
                    "event_status": event_status,
                    "ordinance_event_id": event_id,
                    "issue_id": iid,
                    "issue_date": issue_date,
                    "publication_role": pub_role,
                    "issue_class": issue_class,
                    "legal_status": _norm(label_row.get("legal_status")),
                    "enactment_signal": _norm(label_row.get("enactment_signal")),
                    "proposal_signal": _norm(label_row.get("proposal_signal")),
                    "contains_verbatim_zoning_law": _norm(label_row.get("contains_verbatim_zoning_law")),
                    "law_text_sufficiency": _norm(label_row.get("law_text_sufficiency")),
                    "clean_chars": int(ce.get("clean_chars") or 0),
                    "kept_anchor_count": int(ce.get("kept_anchor_count") or 0),
                    "selected_page_count": len(kept_pages),
                    "png_ok": issue_png_ok,
                    "png_missing": issue_png_missing,
                    "issue_folder_rel": str(issue_dir.relative_to(output_dir)),
                }
            )

            publications_csv_rows.append(
                {
                    "issue_id": iid,
                    "issue_date": issue_date,
                    "publication_role": pub_role,
                    "issue_class": issue_class,
                    "legal_status": _norm(label_row.get("legal_status")),
                    "enactment_signal": _norm(label_row.get("enactment_signal")),
                    "proposal_signal": _norm(label_row.get("proposal_signal")),
                    "contains_verbatim_zoning_law": _norm(label_row.get("contains_verbatim_zoning_law")),
                    "law_text_sufficiency": _norm(label_row.get("law_text_sufficiency")),
                    "clean_chars": int(ce.get("clean_chars") or 0),
                    "selected_page_count": len(kept_pages),
                    "issue_folder_rel": str(issue_dir.relative_to(output_dir)),
                }
            )

        event_row["selected_page_count"] = int(selected_page_total)
        event_row["png_ok"] = int(png_ok)
        event_row["png_missing"] = int(png_missing)
        event_rows.append(event_row)

        _write_json(
            event_dir / "event_manifest.json",
            {
                **{k: v for k, v in event_row.items() if not k.startswith("review_")},
                "event_status_evidence_quotes": _norm(er.get("event_status_evidence_quotes")),
                "qualification_reasons": _norm(er.get("qualification_reasons")),
                "status_notes": _norm(er.get("status_notes")),
            },
        )
        _write_csv(event_dir / "publications.csv", publications_csv_rows)
        (event_dir / "clean_text_combined.txt").write_text(combined_clean_text + ("\n" if combined_clean_text else ""), encoding="utf-8")

        event_md = [
            f"# Event Validation: {slug} | {event_type} | {event_id}",
            "",
            "## Event Metadata",
            "",
            f"- city_key: `{city_key}`",
            f"- event_status: `{event_status}`",
            f"- qualification_decision: `{_norm(er.get('qualification_decision'))}`",
            f"- issue_count: `{int(float(er.get('issue_count') or len(mem)))}`",
            f"- first_issue_date: `{_norm(er.get('first_issue_date'))}`",
            f"- last_issue_date: `{_norm(er.get('last_issue_date'))}`",
            "",
            "## Risk Flags (for reviewer triage)",
            "",
            f"- proposal_terms_in_clean_text: `{int(risk_proposal)}`",
            f"- nonzoning_terms_in_clean_text: `{int(risk_nonzoning)}`",
            f"- low_clean_chars: `{int(risk_low_clean)}`",
            "",
            "## Reviewer Verdict Fields",
            "",
            f"- review_label_correct: `{_norm(event_row.get('review_label_correct'))}`",
            f"- review_status_correct: `{_norm(event_row.get('review_status_correct'))}`",
            f"- review_completeness_correct: `{_norm(event_row.get('review_completeness_correct'))}`",
            f"- review_dedup_correct: `{_norm(event_row.get('review_dedup_correct'))}`",
            f"- review_clean_text_fidelity: `{_norm(event_row.get('review_clean_text_fidelity'))}`",
            f"- review_notes: `{_norm(event_row.get('review_notes'))}`",
            "",
            "## Evidence Paths",
            "",
            "- `publications.csv`",
            "- `clean_text_combined.txt`",
            "- `publications/<issue_id>/issue_transcript_full.txt`",
            "- `publications/<issue_id>/clean_zoning_text.txt`",
            "- `publications/<issue_id>/page_###__<page_id>.txt` and `.png`",
            "",
        ]
        (event_dir / "event_validation.md").write_text("\n".join(event_md).strip() + "\n", encoding="utf-8")

    _write_csv(output_dir / "event_validation_queue.csv", event_rows)
    _write_csv(output_dir / "issue_validation_queue.csv", issue_rows)
    _write_csv(output_dir / "page_asset_inventory.csv", page_rows)

    e_df = pd.DataFrame.from_records(event_rows) if event_rows else pd.DataFrame()
    i_df = pd.DataFrame.from_records(issue_rows) if issue_rows else pd.DataFrame()
    p_df = pd.DataFrame.from_records(page_rows) if page_rows else pd.DataFrame()

    summary_dir = output_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    if not e_df.empty:
        by_event = (
            e_df.groupby(["newspaper_slug", "event_type"], dropna=False)
            .agg(
                event_count=("ordinance_event_id", "count"),
                selected_page_count=("selected_page_count", "sum"),
                png_ok=("png_ok", "sum"),
                png_missing=("png_missing", "sum"),
                risk_proposal=("risk_proposal_terms", "sum"),
                risk_nonzoning=("risk_nonzoning_terms", "sum"),
                risk_low_clean=("risk_low_clean_chars", "sum"),
            )
            .reset_index()
            .sort_values(["newspaper_slug", "event_type"])
            .reset_index(drop=True)
        )
        by_event.to_csv(summary_dir / "by_newspaper_eventtype.csv", index=False)
    else:
        by_event = pd.DataFrame()

    scoreboard = _safe_read_csv(precision_root / "pilot_scoreboard.csv")
    export_inv_path = winner_root / "organized_corpus_v3" / "summary" / "export_inventory.json"
    export_inv = json.loads(export_inv_path.read_text(encoding="utf-8")) if export_inv_path.is_file() else {}

    summary_obj = {
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "precision_run_root": str(precision_root),
        "winner_run_root": str(winner_root),
        "panel_root": str(panel_root),
        "page_tags_path": str(page_tags_path),
        "event_types_included": sorted(include_types),
        "event_count": int(len(e_df)),
        "issue_count": int(len(i_df)),
        "page_rows": int(len(p_df)),
        "png_ok": int((p_df["png_status"] == "ok").sum()) if not p_df.empty else 0,
        "png_missing": int((p_df["png_status"] != "ok").sum()) if not p_df.empty else 0,
        "winner_variant_label": _norm((pilot_summary.get("winner_round2") or {}).get("variant_label")),
        "winner_score": _norm((pilot_summary.get("winner_round2") or {}).get("score")),
        "winner_config": pilot_summary.get("selected_config") or {},
    }
    _write_json(summary_dir / "packet_summary.json", summary_obj)

    readme_lines = [
        "# Exhaustive Manual Validation Packet (Full + Amendment Events)",
        "",
        "This packet is exhaustive for the selected event types in the winner run.",
        "",
        "## Included Files",
        "",
        "- `event_validation_queue.csv`: one row per event with reviewer verdict columns.",
        "- `issue_validation_queue.csv`: one row per publication member issue.",
        "- `page_asset_inventory.csv`: page-level transcript/PNG assets used for review.",
        "- `summary/by_newspaper_eventtype.csv`: coverage and risk-flag rollup.",
        "- `summary/packet_summary.json`: packet metadata and totals.",
        "- `by_newspaper/<slug>/<event_type>/<event_id>/...`: event review folders.",
        "",
        "## Reviewer Workflow",
        "",
        "1. Start with `event_validation_queue.csv`.",
        "2. Open each event folder and review publication members + page evidence.",
        "3. Fill verdict columns in `event_validation_queue.csv`.",
        "4. Use `issue_validation_queue.csv` and `page_asset_inventory.csv` for trace-back.",
        "",
        "Verdict columns:",
        "- `review_label_correct`",
        "- `review_status_correct`",
        "- `review_completeness_correct`",
        "- `review_dedup_correct`",
        "- `review_clean_text_fidelity`",
        "- `review_notes`",
        "",
    ]
    (output_dir / "README.md").write_text("\n".join(readme_lines).strip() + "\n", encoding="utf-8")

    _build_memo(
        memo_path=memo_path,
        precision_root=precision_root,
        winner_root=winner_root,
        output_dir=output_dir,
        pilot_summary=pilot_summary,
        scoreboard=scoreboard,
        export_inv=export_inv,
        event_queue=e_df,
    )

    print(json.dumps(
        {
            "output_dir": str(output_dir),
            "memo_path": str(memo_path),
            "event_count": int(len(e_df)),
            "issue_count": int(len(i_df)),
            "page_rows": int(len(p_df)),
            "png_ok": int((p_df["png_status"] == "ok").sum()) if not p_df.empty else 0,
            "png_missing": int((p_df["png_status"] != "ok").sum()) if not p_df.empty else 0,
        },
        indent=2,
        sort_keys=True,
    ))


if __name__ == "__main__":
    main()
