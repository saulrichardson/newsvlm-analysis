#!/usr/bin/env python3
"""
Build a precision-first manual audit packet for enacted ordinance classes.

This does not change semantic labels. It assembles a reviewer-friendly packet:
- event inventory (full/amend/partial-enacted),
- issue-level rows with risk flags,
- per-event markdown folders with full transcript text for each member issue.
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


TARGET_EVENT_TYPES = {"full_ordinance", "partial_ordinance_enacted", "amendment"}
YES_SET = {"yes", "true", "1"}


def _norm(x: Any) -> str:
    return str(x or "").strip()


def _to_bool_yes(x: Any) -> bool:
    return _norm(x).lower() in YES_SET


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build enacted-class precision audit packet.")
    ap.add_argument(
        "--source-run-dir",
        required=True,
        help="Run root containing panels/<city_key>/issue_texts.jsonl",
    )
    ap.add_argument(
        "--pipeline-root",
        required=True,
        help="Pipeline output root containing classification + events_status",
    )
    ap.add_argument(
        "--output-dir",
        default="",
        help="Output dir (default: <pipeline-root>/manual_validation_precision_v1)",
    )
    ap.add_argument(
        "--issue-assignments-csv",
        default="",
        help="Override issue assignments CSV (default: events_status/issue_event_assignments_status_v3.csv).",
    )
    ap.add_argument(
        "--event-clusters-csv",
        default="",
        help="Override event clusters CSV (default: events_status/ordinance_event_clusters_status_v3.csv).",
    )
    ap.add_argument(
        "--max-issue-preview-chars",
        type=int,
        default=3000,
        help="Preview chars shown in markdown before full transcript file reference.",
    )
    return ap.parse_args()


def _load_issue_texts(source_run_dir: Path) -> dict[str, dict[str, Any]]:
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
                text = _norm(obj.get("text"))
                page_ids = obj.get("page_ids") if isinstance(obj.get("page_ids"), list) else []
                page_ids = [_norm(x) for x in page_ids if _norm(x)]
                if not page_ids:
                    n = int(obj.get("page_count") or 1)
                    page_ids = [f"{issue_id}__page_{i+1}" for i in range(max(1, n))]
                slug = _norm(obj.get("pub_slug")).lower()
                if not slug:
                    slug = issue_id.rsplit("__", 1)[0].lower() if "__" in issue_id else issue_id.lower()
                out[issue_id] = {
                    "newspaper_slug": slug,
                    "city_key": _norm(obj.get("city_key")),
                    "city_name": _norm(obj.get("city_name")),
                    "state_abbr": _norm(obj.get("state_abbr")).upper(),
                    "issue_date": _norm(obj.get("issue_date")),
                    "text": text,
                    "text_chars": int(len(text)),
                    "page_ids": page_ids,
                    "page_count": int(len(page_ids)),
                }
    return out


def _risk_flags(row: pd.Series) -> dict[str, Any]:
    proposal_signal = _to_bool_yes(row.get("proposal_signal"))
    jurisdiction_mismatch = not _to_bool_yes(row.get("jurisdiction_match_city"))
    incomplete = _norm(row.get("issue_completeness")).lower() not in {"complete"} or _to_bool_yes(row.get("possible_missing_pages"))
    low_conf = _safe_float(row.get("confidence_0_to_1"), 0.0) < 0.65
    uncertain_status = _norm(row.get("event_status")).lower() in {"uncertain"}
    risk_score = int(proposal_signal) + int(jurisdiction_mismatch) + int(incomplete) + int(low_conf) + int(uncertain_status)
    return {
        "risk_proposal_signal": int(proposal_signal),
        "risk_jurisdiction_mismatch": int(jurisdiction_mismatch),
        "risk_issue_incomplete_or_missing_pages": int(incomplete),
        "risk_low_confidence": int(low_conf),
        "risk_uncertain_event_status": int(uncertain_status),
        "risk_score": int(risk_score),
    }


def main() -> None:
    args = _parse_args()
    source_run_dir = Path(args.source_run_dir).expanduser().resolve()
    pipeline_root = Path(args.pipeline_root).expanduser().resolve()
    out_dir = (
        Path(args.output_dir).expanduser().resolve()
        if _norm(args.output_dir)
        else pipeline_root / "manual_validation_precision_v1"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    cls_path = pipeline_root / "classification" / "issue_labels_v3.csv"
    asn_path = (
        Path(args.issue_assignments_csv).expanduser().resolve()
        if _norm(args.issue_assignments_csv)
        else pipeline_root / "events_status" / "issue_event_assignments_status_v3.csv"
    )
    clusters_path = (
        Path(args.event_clusters_csv).expanduser().resolve()
        if _norm(args.event_clusters_csv)
        else pipeline_root / "events_status" / "ordinance_event_clusters_status_v3.csv"
    )
    if not cls_path.is_file():
        raise SystemExit(f"Missing classification csv: {cls_path}")
    if not asn_path.is_file():
        raise SystemExit(f"Missing event assignments csv: {asn_path}")
    if not clusters_path.is_file():
        raise SystemExit(f"Missing clusters csv: {clusters_path}")

    cls = pd.read_csv(cls_path)
    asn = pd.read_csv(asn_path)
    clusters = pd.read_csv(clusters_path)
    issue_meta = _load_issue_texts(source_run_dir)

    for df in (cls, asn, clusters):
        for c in (
            "newspaper_slug",
            "issue_id",
            "issue_date",
            "issue_class",
            "event_type",
            "event_status",
            "qualification_decision",
            "ordinance_event_id",
            "legal_status",
            "enactment_signal",
            "proposal_signal",
            "issue_completeness",
            "possible_missing_pages",
            "jurisdiction_match_city",
        ):
            if c in df.columns:
                df[c] = df[c].astype(str)

    target_asn = asn[asn["event_type"].isin(sorted(TARGET_EVENT_TYPES))].copy()
    if target_asn.empty:
        (out_dir / "README.txt").write_text(
            "No full_ordinance / partial_ordinance_enacted / amendment rows found in event assignments.\n",
            encoding="utf-8",
        )
        print(f"done_audit_packet output_dir={out_dir} (empty)")
        return

    merged = target_asn.merge(
        cls[
            [
                "newspaper_slug",
                "issue_id",
                "issue_class",
                "confidence_0_to_1",
                "legal_status",
                "enactment_signal",
                "proposal_signal",
                "issue_completeness",
                "possible_missing_pages",
                "jurisdiction_match_city",
                "evidence_quotes",
                "status_evidence_quotes",
                "clean_chars",
            ]
        ],
        on=["newspaper_slug", "issue_id"],
        how="left",
        suffixes=("", "_cls"),
    )

    # Normalize quote fields if they arrived as stringified JSON
    for c in ("evidence_quotes", "status_evidence_quotes"):
        if c in merged.columns:
            def _parse_list(v: Any) -> list[str]:
                if isinstance(v, list):
                    return [_norm(x) for x in v if _norm(x)]
                s = _norm(v)
                if not s:
                    return []
                try:
                    obj = json.loads(s)
                    if isinstance(obj, list):
                        return [_norm(x) for x in obj if _norm(x)]
                except Exception:
                    pass
                return [s]
            merged[c] = merged[c].map(_parse_list)
        else:
            merged[c] = [[] for _ in range(len(merged))]

    # Append issue metadata and risk flags.
    records: list[dict[str, Any]] = []
    for r in merged.itertuples(index=False):
        issue_id = _norm(getattr(r, "issue_id", ""))
        m = issue_meta.get(issue_id, {})
        row = {
            "newspaper_slug": _norm(getattr(r, "newspaper_slug", "")),
            "ordinance_event_id": _norm(getattr(r, "ordinance_event_id", "")),
            "event_type": _norm(getattr(r, "event_type", "")),
            "event_status": _norm(getattr(r, "event_status", "")),
            "qualification_decision": _norm(getattr(r, "qualification_decision", "")),
            "publication_role": _norm(getattr(r, "publication_role", "")),
            "issue_id": issue_id,
            "issue_date": _norm(getattr(r, "issue_date", "")),
            "issue_class": _norm(getattr(r, "issue_class", "")),
            "legal_status": _norm(getattr(r, "legal_status", "")),
            "enactment_signal": _norm(getattr(r, "enactment_signal", "")),
            "proposal_signal": _norm(getattr(r, "proposal_signal", "")),
            "issue_completeness": _norm(getattr(r, "issue_completeness", "")),
            "possible_missing_pages": _norm(getattr(r, "possible_missing_pages", "")),
            "jurisdiction_match_city": _norm(getattr(r, "jurisdiction_match_city", "")),
            "confidence_0_to_1": _safe_float(getattr(r, "confidence_0_to_1", 0.0), 0.0),
            "clean_chars": int(_safe_float(getattr(r, "clean_chars", 0.0), 0.0)),
            "city_key": _norm(m.get("city_key")),
            "city_name": _norm(m.get("city_name")),
            "state_abbr": _norm(m.get("state_abbr")),
            "page_count": int(m.get("page_count") or 0),
            "page_ids": "|".join(m.get("page_ids") if isinstance(m.get("page_ids"), list) else []),
            "raw_text_chars": int(m.get("text_chars") or 0),
            "evidence_quotes": " | ".join(getattr(r, "evidence_quotes", []) if isinstance(getattr(r, "evidence_quotes", []), list) else []),
            "status_evidence_quotes": " | ".join(getattr(r, "status_evidence_quotes", []) if isinstance(getattr(r, "status_evidence_quotes", []), list) else []),
        }
        row.update(_risk_flags(pd.Series(row)))
        records.append(row)

    issue_df = pd.DataFrame.from_records(records).sort_values(
        ["event_type", "newspaper_slug", "issue_date", "issue_id"]
    )
    issue_df.to_csv(out_dir / "full_amend_issue_audit_packet.csv", index=False, quoting=csv.QUOTE_MINIMAL)

    # Event-level inventory.
    # Use (newspaper_slug, ordinance_event_id) as event key.
    event_rows: list[dict[str, Any]] = []
    grouped = issue_df.groupby(["newspaper_slug", "ordinance_event_id"], dropna=False)
    for (slug, eid), g in grouped:
        r0 = g.iloc[0]
        event_rows.append(
            {
                "newspaper_slug": slug,
                "ordinance_event_id": eid,
                "event_type": _norm(r0.get("event_type")),
                "event_status": _norm(r0.get("event_status")),
                "qualification_decision": _norm(r0.get("qualification_decision")),
                "issue_count": int(g["issue_id"].nunique()),
                "first_issue_date": _norm(g["issue_date"].min()),
                "last_issue_date": _norm(g["issue_date"].max()),
                "max_risk_score": int(g["risk_score"].max()),
                "mean_risk_score": float(g["risk_score"].mean()),
                "proposal_signal_rows": int(g["risk_proposal_signal"].sum()),
                "jurisdiction_mismatch_rows": int(g["risk_jurisdiction_mismatch"].sum()),
                "incomplete_rows": int(g["risk_issue_incomplete_or_missing_pages"].sum()),
                "member_issue_ids": "|".join(g["issue_id"].astype(str).tolist()),
            }
        )
    event_df = pd.DataFrame.from_records(event_rows).sort_values(
        ["event_type", "newspaper_slug", "first_issue_date", "ordinance_event_id"]
    )
    event_df.to_csv(out_dir / "full_amend_event_inventory.csv", index=False, quoting=csv.QUOTE_MINIMAL)

    # Per-event markdown folders with full issue transcripts.
    event_root = out_dir / "events"
    event_root.mkdir(parents=True, exist_ok=True)
    for er in event_df.itertuples(index=False):
        slug = _norm(getattr(er, "newspaper_slug", ""))
        eid = _norm(getattr(er, "ordinance_event_id", ""))
        folder = event_root / f"{slug}__{eid}"
        folder.mkdir(parents=True, exist_ok=True)
        g = issue_df[(issue_df["newspaper_slug"] == slug) & (issue_df["ordinance_event_id"] == eid)].sort_values(
            ["issue_date", "issue_id"]
        )
        lines: list[str] = []
        lines.append(f"# {slug} | {eid}")
        lines.append("")
        lines.append(f"- event_type: {_norm(getattr(er, 'event_type', ''))}")
        lines.append(f"- event_status: {_norm(getattr(er, 'event_status', ''))}")
        lines.append(f"- qualification_decision: {_norm(getattr(er, 'qualification_decision', ''))}")
        lines.append(f"- issue_count: {int(getattr(er, 'issue_count', 0) or 0)}")
        lines.append(f"- first_issue_date: {_norm(getattr(er, 'first_issue_date', ''))}")
        lines.append(f"- last_issue_date: {_norm(getattr(er, 'last_issue_date', ''))}")
        lines.append(f"- max_risk_score: {int(getattr(er, 'max_risk_score', 0) or 0)}")
        lines.append("")
        lines.append("## Member issues")
        lines.append("")
        for ir in g.itertuples(index=False):
            issue_id = _norm(getattr(ir, "issue_id", ""))
            im = issue_meta.get(issue_id, {})
            full_text = _norm(im.get("text"))
            preview = full_text[: max(0, int(args.max_issue_preview_chars))]
            issue_file = f"{issue_id}.txt"
            (folder / issue_file).write_text(full_text + ("\n" if not full_text.endswith("\n") else ""), encoding="utf-8")
            lines.append(f"### {issue_id}")
            lines.append(f"- issue_date: {_norm(getattr(ir, 'issue_date', ''))}")
            lines.append(f"- publication_role: {_norm(getattr(ir, 'publication_role', ''))}")
            lines.append(f"- issue_class: {_norm(getattr(ir, 'issue_class', ''))}")
            lines.append(f"- legal_status: {_norm(getattr(ir, 'legal_status', ''))}")
            lines.append(f"- enactment_signal: {_norm(getattr(ir, 'enactment_signal', ''))}")
            lines.append(f"- proposal_signal: {_norm(getattr(ir, 'proposal_signal', ''))}")
            lines.append(f"- issue_completeness: {_norm(getattr(ir, 'issue_completeness', ''))}")
            lines.append(f"- possible_missing_pages: {_norm(getattr(ir, 'possible_missing_pages', ''))}")
            lines.append(f"- jurisdiction_match_city: {_norm(getattr(ir, 'jurisdiction_match_city', ''))}")
            lines.append(f"- risk_score: {int(getattr(ir, 'risk_score', 0) or 0)}")
            lines.append(f"- page_ids: {_norm(getattr(ir, 'page_ids', ''))}")
            lines.append(f"- transcript_file: {issue_file}")
            eq = _norm(getattr(ir, "evidence_quotes", ""))
            sq = _norm(getattr(ir, "status_evidence_quotes", ""))
            if eq:
                lines.append(f"- evidence_quotes: {eq}")
            if sq:
                lines.append(f"- status_evidence_quotes: {sq}")
            lines.append("")
            if preview:
                lines.append("```text")
                lines.append(preview)
                if len(full_text) > len(preview):
                    lines.append("\n[... preview truncated, see transcript_file for full text ...]")
                lines.append("```")
                lines.append("")
        (folder / "event_review.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")

    # Summary note.
    summary = {
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "script_path": str(Path(__file__).resolve()),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_run_dir": str(source_run_dir),
        "pipeline_root": str(pipeline_root),
        "output_dir": str(out_dir),
        "target_event_types": sorted(TARGET_EVENT_TYPES),
        "issue_rows_in_packet": int(issue_df.shape[0]),
        "event_rows_in_packet": int(event_df.shape[0]),
        "high_risk_issue_rows_ge3": int((issue_df["risk_score"] >= 3).sum()),
        "full_rows": int((issue_df["event_type"] == "full_ordinance").sum()),
        "partial_enacted_rows": int((issue_df["event_type"] == "partial_ordinance_enacted").sum()),
        "amend_rows": int((issue_df["event_type"] == "amendment").sum()),
    }
    (out_dir / "provenance.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"done_precision_audit_packet output_dir={out_dir}")


if __name__ == "__main__":
    main()
