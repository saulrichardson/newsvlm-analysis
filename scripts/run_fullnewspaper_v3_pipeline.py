#!/usr/bin/env python3
"""
Orchestrate full-newspaper v3 pipeline:
  1) page tagging
  2) calibration A/B
  3) production cleaning/classification/events/longitudinal
  4) analysis tables + manual validation packet
  5) organized corpus export (full ordinances / amendments)
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd


LAW_CLASSES = {
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


def _norm(x: Any) -> str:
    return str(x or "").strip()


def _run(cmd: list[str], cwd: Path) -> None:
    print("[run] " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run full-newspaper v3 pipeline.")
    ap.add_argument(
        "--run-dir",
        default="/Users/saulrichardson/projects/newspapers/newspaper-analysis/reports/runs/pi_v1_run60_fullnewspaper",
        help="Source run dir containing panels/<city_key>/issue_texts.jsonl.",
    )
    ap.add_argument(
        "--output-dir",
        default="",
        help="Pipeline output root (default: <run-dir>/fullnewspaper_v3_run_<timestamp>).",
    )
    ap.add_argument(
        "--newspapers",
        default="auto",
        help='Comma-separated slugs or "auto" to discover all newspapers in selected panel manifest.',
    )
    ap.add_argument("--model-primary", default="openai:gpt-5")
    ap.add_argument("--model-secondary", default="gemini:gemini-2.5-flash")
    ap.add_argument("--calibration-issue-limit", type=int, default=40)
    ap.add_argument("--run-calibration", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--cleaning-variants", default="single_pass,two_pass,three_pass")
    ap.add_argument("--classification-variants", default="single_model,dual_model_with_llm_adjudicator")
    ap.add_argument(
        "--classification-input-mode",
        default="raw_only",
        choices=["raw_plus_clean", "raw_only"],
        help="raw_only classifies from full issue text as primary evidence; raw_plus_clean includes cleaned text as support.",
    )
    ap.add_argument(
        "--require-clean-for-law",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If enabled, empty cleaned text blocks law classes. Default false.",
    )
    ap.add_argument("--dedup-variants", default="direct_cluster,pairwise_then_cluster")
    ap.add_argument("--manual-qa-per-paper", type=int, default=3, help="Legacy minimum per-paper sample floor.")
    ap.add_argument("--manual-qa-full-share", type=float, default=0.20)
    ap.add_argument("--manual-qa-amend-share", type=float, default=0.10)
    ap.add_argument("--manual-qa-narrative-per-paper", type=int, default=2)
    ap.add_argument("--concurrency", type=int, default=3)
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument(
        "--max-panel-chars",
        type=int,
        default=0,
        help="Optional hard cap passed to panel longitudinal stage. 0 means no local cap (no clipping). If a positive cap is exceeded, the stage fails.",
    )
    ap.add_argument(
        "--gateway-runner",
        default="/Users/saulrichardson/projects/newspapers/old-ocr/experimental/scripts/run_openai_requests_via_gateway.py",
    )
    ap.add_argument(
        "--gateway-pythonpath",
        default="/Users/saulrichardson/projects/newspapers/old-ocr/newspaper-parsing-local/agent-gateway/src",
    )
    ap.add_argument("--gov-env-path", default=".env")
    ap.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


def _discover_newspapers(run_dir: Path) -> list[str]:
    manifest = run_dir / "panels" / "selected_panel_issues.csv"
    if manifest.is_file():
        try:
            d = pd.read_csv(manifest)
        except Exception:
            d = pd.DataFrame()
        if not d.empty and "issue_id" in d.columns:
            slugs = (
                d["issue_id"]
                .astype(str)
                .str.replace(r"__.*$", "", regex=True)
                .str.strip()
                .str.lower()
            )
            vals = sorted({s for s in slugs.tolist() if s})
            if vals:
                return vals

    out: set[str] = set()
    panel_root = run_dir / "panels"
    if panel_root.is_dir():
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
                    slug = _norm(obj.get("pub_slug")).lower()
                    if slug:
                        out.add(slug)
    return sorted(out)


def _score_cleaning_classification(cls_csv: Path) -> dict[str, float]:
    if not cls_csv.is_file():
        return {"score": -1.0, "law_rows": 0.0, "mean_clean_chars": 0.0, "narrative_bleed": 1.0}
    d = pd.read_csv(cls_csv)
    if d.empty:
        return {"score": -1.0, "law_rows": 0.0, "mean_clean_chars": 0.0, "narrative_bleed": 1.0}
    d["is_law"] = d["issue_class"].astype(str).isin(LAW_CLASSES).astype(int)
    law_rows = float(d["is_law"].sum())
    mean_clean_chars = float(pd.to_numeric(d.get("clean_chars", 0), errors="coerce").fillna(0).mean())
    # bleed proxy: law class but narrative flag yes
    bleed = d[(d["is_law"] == 1) & (d["non_law_zoning_narrative_present"].astype(str).str.lower() == "yes")]
    narrative_bleed = float(len(bleed) / max(1, int(law_rows))) if law_rows > 0 else 0.0
    score = (law_rows / max(1.0, float(len(d)))) + 0.00001 * mean_clean_chars - 2.0 * narrative_bleed
    return {
        "score": float(score),
        "law_rows": float(law_rows),
        "mean_clean_chars": float(mean_clean_chars),
        "narrative_bleed": float(narrative_bleed),
    }


def _score_dedup(assign_csv: Path) -> dict[str, float]:
    if not assign_csv.is_file():
        return {"score": -1.0, "singleton_share": 1.0, "event_count": 0.0}
    d = pd.read_csv(assign_csv)
    if d.empty:
        return {"score": -1.0, "singleton_share": 1.0, "event_count": 0.0}
    c = d.groupby("ordinance_event_id", dropna=False)["issue_id"].nunique().reset_index(name="n")
    singleton_share = float((c["n"] == 1).mean()) if not c.empty else 1.0
    event_count = float(len(c))
    # prefer less singleton noise but not collapse everything
    score = (1.0 - singleton_share) + 0.001 * min(event_count, 1000.0)
    return {"score": float(score), "singleton_share": singleton_share, "event_count": event_count}


def _build_analysis_tables(
    output_root: Path,
    *,
    manual_qa_per_paper: int,
    manual_qa_full_share: float,
    manual_qa_amend_share: float,
    manual_qa_narrative_per_paper: int,
) -> None:
    cls_path = output_root / "classification" / "issue_labels_v3.csv"
    ev_assign_path = output_root / "events_status" / "issue_event_assignments_status_v3.csv"
    ev_cluster_path = output_root / "events_status" / "ordinance_event_clusters_status_v3.csv"
    if not ev_assign_path.is_file():
        ev_assign_path = output_root / "events" / "issue_event_assignments_v3.csv"
    if not ev_cluster_path.is_file():
        ev_cluster_path = output_root / "events" / "ordinance_event_clusters_v3.csv"

    long_dir = output_root / "longitudinal"
    out_dir = output_root / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    cls = pd.read_csv(cls_path) if cls_path.is_file() else pd.DataFrame()
    ev_assign = pd.read_csv(ev_assign_path) if ev_assign_path.is_file() else pd.DataFrame()
    ev_cluster = pd.read_csv(ev_cluster_path) if ev_cluster_path.is_file() else pd.DataFrame()

    # Full vs amendment by year from deduplicated event objects (not raw issue rows).
    if not ev_cluster.empty:
        d = ev_cluster.copy()
        d["year"] = pd.to_datetime(d.get("first_issue_date"), errors="coerce").dt.year
        d = d[d["year"].notna()].copy()
        d["year"] = d["year"].astype(int)
        if "event_status" in d.columns:
            enacted_mask = d["event_status"].astype(str).isin({"enacted_complete", "enacted_partial_installment"})
            d["is_full"] = ((d["event_type"].astype(str) == "full_ordinance") & enacted_mask).astype(int)
        else:
            d["is_full"] = (d["event_type"].astype(str) == "full_ordinance").astype(int)
        d["is_amend"] = (d["event_type"].astype(str) == "amendment").astype(int)
        d["events"] = 1
        by_year = (
            d.groupby("year", dropna=False)
            .agg(full_ordinance=("is_full", "sum"), amendment=("is_amend", "sum"), events=("events", "sum"))
            .reset_index()
            .sort_values("year")
        )
    else:
        by_year = pd.DataFrame(columns=["year", "full_ordinance", "amendment", "events"])
    _write_csv(out_dir / "full_vs_amendment_by_year.csv", by_year)

    # Category trajectory means.
    cat_path = long_dir / "city_category_trajectory.csv"
    cat = pd.read_csv(cat_path) if cat_path.is_file() else pd.DataFrame()
    if not cat.empty:
        by_cat = (
            cat.groupby("category", dropna=False)
            .agg(
                early=("early_share", "mean"),
                mid=("mid_share", "mean"),
                late=("late_share", "mean"),
            )
            .reset_index()
        )
        by_cat["late_minus_early_pp"] = (by_cat["late"] - by_cat["early"]) * 100.0
    else:
        by_cat = pd.DataFrame(columns=["category", "early", "mid", "late", "late_minus_early_pp"])
    _write_csv(out_dir / "category_trajectory_stage_means.csv", by_cat)

    # Complexity trend summary.
    cpx_path = long_dir / "city_complexity_trajectory.csv"
    cpx = pd.read_csv(cpx_path) if cpx_path.is_file() else pd.DataFrame()
    cpx_rows: list[dict[str, Any]] = []
    if not cpx.empty:
        for metric in ["regulatory_complexity_trend", "procedural_intensity_trend", "dimensional_specificity_trend"]:
            tmp = cpx[metric].astype(str).value_counts(dropna=False).rename_axis("direction").reset_index(name="count")
            tmp["metric"] = metric
            cpx_rows.extend(tmp.to_dict(orient="records"))
    cpx_out = pd.DataFrame.from_records(cpx_rows)
    _write_csv(out_dir / "complexity_trajectory_summary.csv", cpx_out)

    # Numeric quote catalog from longitudinal output.
    num_path = long_dir / "city_numeric_trajectory.csv"
    num = pd.read_csv(num_path) if num_path.is_file() else pd.DataFrame()
    _write_csv(out_dir / "numeric_quote_catalog.csv", num)

    # Manual validation candidate sheet.
    validation_dir = output_root / "manual_validation"
    validation_dir.mkdir(parents=True, exist_ok=True)
    candidates: list[dict[str, Any]] = []

    # Build event-level candidate pools from deduped event table.
    full_events = pd.DataFrame()
    amend_events = pd.DataFrame()
    if not ev_cluster.empty:
        e = ev_cluster.copy()
        if "event_status" in e.columns:
            full_events = e[
                (e["event_type"].astype(str) == "full_ordinance")
                & (e["event_status"].astype(str).isin({"enacted_complete", "enacted_partial_installment"}))
            ].copy()
        else:
            full_events = e[e["event_type"].astype(str) == "full_ordinance"].copy()
        amend_events = e[e["event_type"].astype(str) == "amendment"].copy()

    # Risk score for targeted oversampling (proposal cues, uncertainty, multi-issue).
    def _risk_score(df: pd.DataFrame) -> pd.Series:
        if df.empty:
            return pd.Series(dtype=float)
        s = pd.Series(0.0, index=df.index)
        if "event_status" in df.columns:
            s = s + df["event_status"].astype(str).isin({"proposed_full_text", "proposed_partial_installment", "uncertain"}).astype(float) * 5.0
        if "summary" in df.columns:
            s = s + df["summary"].astype(str).str.contains("proposed|hearing|draft|installment|continued", case=False, regex=True).astype(float) * 3.0
        if "issue_count" in df.columns:
            s = s + (pd.to_numeric(df["issue_count"], errors="coerce").fillna(0) > 1).astype(float) * 2.0
        return s

    def _sample_events(df: pd.DataFrame, share: float, label: str) -> list[dict[str, Any]]:
        if df.empty:
            return []
        d = df.copy()
        d["risk_score"] = _risk_score(d)
        target = max(1, int(math.ceil(float(max(0.0, share)) * len(d))))
        # Keep per-paper floor.
        if "newspaper_slug" in d.columns:
            floor = max(1, int(manual_qa_per_paper))
            target = max(target, min(len(d), floor))
        d = d.sort_values(
            ["risk_score", "issue_count", "first_issue_date", "ordinance_event_id"],
            ascending=[False, False, True, True],
        )
        sampled = d.head(target).copy()
        out: list[dict[str, Any]] = []
        for r in sampled.itertuples(index=False):
            out.append(
                {
                    "newspaper_slug": _norm(getattr(r, "newspaper_slug", "")),
                    "city_key": _norm(getattr(r, "city_key", "")),
                    "validation_type": label,
                    "ordinance_event_id": _norm(getattr(r, "ordinance_event_id", "")),
                    "issue_id": "",
                    "issue_date": _norm(getattr(r, "first_issue_date", "")),
                    "expected_label": _norm(getattr(r, "event_type", "")),
                    "event_status": _norm(getattr(r, "event_status", "")),
                    "event_type": _norm(getattr(r, "event_type", "")),
                    "issue_count": int(float(getattr(r, "issue_count", 0) or 0)),
                    "sample_reason": f"risk_score={float(getattr(r, 'risk_score', 0.0)):.2f}",
                    "pass_fail": "",
                    "notes": "",
                    "evidence_excerpt": "",
                }
            )
        return out

    candidates.extend(_sample_events(full_events, manual_qa_full_share, "full_ordinance_event"))
    candidates.extend(_sample_events(amend_events, manual_qa_amend_share, "amendment_event"))

    # Narrative exclusions for bleed checks.
    if not cls.empty:
        nar = cls[cls["issue_class"].astype(str).isin(["not_verbatim_law_text", "non_zoning"])].copy()
        for slug, g in nar.groupby("newspaper_slug", dropna=False):
            top = g.sort_values(["clean_chars", "issue_date"], ascending=[False, True]).head(max(1, int(manual_qa_narrative_per_paper)))
            for r in top.itertuples(index=False):
                candidates.append(
                    {
                        "newspaper_slug": _norm(slug),
                        "city_key": _norm(getattr(r, "city_key", "")),
                        "validation_type": "narrative_exclusion",
                        "ordinance_event_id": "",
                        "issue_id": _norm(getattr(r, "issue_id", "")),
                        "issue_date": _norm(getattr(r, "issue_date", "")),
                        "expected_label": _norm(getattr(r, "issue_class", "")),
                        "event_status": "",
                        "event_type": "",
                        "issue_count": 1,
                        "sample_reason": "narrative_bleed_check",
                        "pass_fail": "",
                        "notes": "",
                        "evidence_excerpt": "",
                    }
                )

    cand_df = pd.DataFrame.from_records(candidates).drop_duplicates(
        subset=["validation_type", "newspaper_slug", "ordinance_event_id", "issue_id"],
        keep="first",
    )

    # Attach a representative issue row for event audits.
    if not cand_df.empty and not ev_assign.empty:
        ev_assign_sorted = ev_assign.sort_values(["issue_date", "issue_id"])
        reps = (
            ev_assign_sorted.groupby(["newspaper_slug", "ordinance_event_id"], dropna=False)
            .first()
            .reset_index()[["newspaper_slug", "ordinance_event_id", "issue_id", "issue_date"]]
            .rename(columns={"issue_id": "representative_issue_id", "issue_date": "representative_issue_date"})
        )
        cand_df = cand_df.merge(reps, on=["newspaper_slug", "ordinance_event_id"], how="left")

    _write_csv(validation_dir / "validation_candidates.csv", cand_df)
    _write_csv(validation_dir / "validation_log.csv", cand_df.copy())


def main() -> None:
    args = _parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    run_dir = Path(args.run_dir).expanduser().resolve()

    if _norm(args.output_dir):
        out_root = Path(args.output_dir).expanduser().resolve()
    else:
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_root = run_dir / f"fullnewspaper_v3_run_{stamp}"
    out_root.mkdir(parents=True, exist_ok=True)

    if _norm(args.newspapers).lower() in {"", "auto", "all"}:
        newspapers = _discover_newspapers(run_dir)
    else:
        newspapers = [x.strip().lower() for x in _norm(args.newspapers).split(",") if x.strip()]
    if not newspapers:
        raise SystemExit("No newspapers provided.")

    py = sys.executable
    shared = [
        "--run-dir",
        str(run_dir),
        "--newspapers",
        ",".join(newspapers),
        "--concurrency",
        str(int(args.concurrency)),
        "--timeout",
        str(float(args.timeout)),
        "--gateway-runner",
        str(Path(args.gateway_runner).expanduser().resolve()),
        "--gateway-pythonpath",
        str(Path(args.gateway_pythonpath).expanduser().resolve()),
        "--gov-env-path",
        str(Path(args.gov_env_path).expanduser().resolve()),
    ]
    if bool(args.skip_existing):
        shared.append("--skip-existing")
    if bool(args.dry_run):
        shared.append("--dry-run")

    # 1) Page tagging (single production pass; reused by calibration and final runs).
    _run(
        [
            py,
            str(repo_root / "scripts" / "llm_page_tag_v3.py"),
            *shared,
            "--output-dir",
            str(out_root / "page_tagging"),
            "--model",
            str(args.model_primary),
        ],
        cwd=repo_root,
    )

    cal_rows: list[dict[str, Any]] = []
    selected_clean = "three_pass"
    selected_cls = "single_model"
    selected_dedup = "direct_cluster"

    if bool(args.run_calibration):
        clean_variants = [x.strip() for x in _norm(args.cleaning_variants).split(",") if x.strip()]
        cls_variants = [x.strip() for x in _norm(args.classification_variants).split(",") if x.strip()]
        dedup_variants = [x.strip() for x in _norm(args.dedup_variants).split(",") if x.strip()]

        # 2a) Cleaning variant calibration using single-model classification as scorer.
        clean_scores: list[dict[str, Any]] = []
        for v in clean_variants:
            c_out = out_root / "calibration" / f"cleaning_{v}"
            _run(
                [
                    py,
                    str(repo_root / "scripts" / "llm_anchor_clean_v3.py"),
                    *shared,
                    "--page-tags-jsonl",
                    str(out_root / "page_tagging" / "page_tags.jsonl"),
                    "--output-dir",
                    str(c_out),
                    "--variant",
                    v,
                    "--model",
                    str(args.model_primary),
                    "--limit-issues",
                    str(int(args.calibration_issue_limit)),
                ],
                cwd=repo_root,
            )
            cls_out = out_root / "calibration" / f"classification_clean_{v}"
            _run(
                [
                    py,
                    str(repo_root / "scripts" / "llm_issue_classify_v3.py"),
                    *shared,
                    "--clean-jsonl",
                    str(c_out / "clean_zoning_text.jsonl"),
                    "--output-dir",
                    str(cls_out),
                    "--variant",
                    "single_model",
                    "--model-primary",
                    str(args.model_primary),
                    "--classification-input-mode",
                    str(args.classification_input_mode),
                    *([] if not bool(args.require_clean_for_law) else ["--require-clean-for-law"]),
                    "--limit-issues",
                    str(int(args.calibration_issue_limit)),
                ],
                cwd=repo_root,
            )
            m = _score_cleaning_classification(cls_out / "issue_labels_v3.csv")
            m["step"] = "cleaning_variant"
            m["variant"] = v
            clean_scores.append(m)
            cal_rows.append(m)
        if clean_scores:
            selected_clean = sorted(clean_scores, key=lambda x: (float(x["score"]), float(x["law_rows"])), reverse=True)[0]["variant"]

        # 2b) Classification variant calibration on selected cleaning.
        base_clean = out_root / "calibration" / f"cleaning_{selected_clean}" / "clean_zoning_text.jsonl"
        cls_scores: list[dict[str, Any]] = []
        for v in cls_variants:
            cls_out = out_root / "calibration" / f"classification_variant_{v}"
            _run(
                [
                    py,
                    str(repo_root / "scripts" / "llm_issue_classify_v3.py"),
                    *shared,
                    "--clean-jsonl",
                    str(base_clean),
                    "--output-dir",
                    str(cls_out),
                    "--variant",
                    v,
                    "--model-primary",
                    str(args.model_primary),
                    "--model-secondary",
                    str(args.model_secondary),
                    "--classification-input-mode",
                    str(args.classification_input_mode),
                    *([] if not bool(args.require_clean_for_law) else ["--require-clean-for-law"]),
                    "--limit-issues",
                    str(int(args.calibration_issue_limit)),
                ],
                cwd=repo_root,
            )
            m = _score_cleaning_classification(cls_out / "issue_labels_v3.csv")
            m["step"] = "classification_variant"
            m["variant"] = v
            cls_scores.append(m)
            cal_rows.append(m)
        if cls_scores:
            selected_cls = sorted(cls_scores, key=lambda x: (float(x["score"]), float(x["law_rows"])), reverse=True)[0]["variant"]

        # 2c) Dedup variant calibration.
        cls_selected_path = out_root / "calibration" / f"classification_variant_{selected_cls}" / "issue_labels_v3.jsonl"
        dedup_scores: list[dict[str, Any]] = []
        for v in dedup_variants:
            ev_out = out_root / "calibration" / f"events_{v}"
            _run(
                [
                    py,
                    str(repo_root / "scripts" / "llm_event_cluster_v3.py"),
                    "--classification-jsonl",
                    str(cls_selected_path),
                    "--clean-jsonl",
                    str(base_clean),
                    "--output-dir",
                    str(ev_out),
                    "--variant",
                    v,
                    "--model",
                    str(args.model_primary),
                    "--concurrency",
                    str(int(args.concurrency)),
                    "--timeout",
                    str(float(args.timeout)),
                    "--gateway-runner",
                    str(Path(args.gateway_runner).expanduser().resolve()),
                    "--gateway-pythonpath",
                    str(Path(args.gateway_pythonpath).expanduser().resolve()),
                    "--gov-env-path",
                    str(Path(args.gov_env_path).expanduser().resolve()),
                    *([] if not bool(args.skip_existing) else ["--skip-existing"]),
                    *([] if not bool(args.dry_run) else ["--dry-run"]),
                ],
                cwd=repo_root,
            )
            m = _score_dedup(ev_out / "issue_event_assignments_v3.csv")
            m["step"] = "dedup_variant"
            m["variant"] = v
            dedup_scores.append(m)
            cal_rows.append(m)
        if dedup_scores:
            selected_dedup = sorted(dedup_scores, key=lambda x: float(x["score"]), reverse=True)[0]["variant"]

    # Write calibration results and selections.
    cal_df = pd.DataFrame.from_records(cal_rows)
    if not cal_df.empty:
        _write_csv(out_root / "calibration" / "ab_scores.csv", cal_df)
    decision = {
        "selected_cleaning_variant": selected_clean,
        "selected_classification_variant": selected_cls,
        "selected_dedup_variant": selected_dedup,
        "classification_input_mode": str(args.classification_input_mode),
        "require_clean_for_law": bool(args.require_clean_for_law),
    }
    (out_root / "calibration" / "selected_variants.json").parent.mkdir(parents=True, exist_ok=True)
    (out_root / "calibration" / "selected_variants.json").write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # 3) Production runs with selected variants.
    _run(
        [
            py,
            str(repo_root / "scripts" / "llm_anchor_clean_v3.py"),
            *shared,
            "--page-tags-jsonl",
            str(out_root / "page_tagging" / "page_tags.jsonl"),
            "--output-dir",
            str(out_root / "cleaning"),
            "--variant",
            selected_clean,
            "--model",
            str(args.model_primary),
        ],
        cwd=repo_root,
    )

    _run(
        [
            py,
            str(repo_root / "scripts" / "llm_issue_classify_v3.py"),
            *shared,
            "--clean-jsonl",
            str(out_root / "cleaning" / "clean_zoning_text.jsonl"),
            "--output-dir",
            str(out_root / "classification"),
            "--variant",
            selected_cls,
            "--model-primary",
            str(args.model_primary),
            "--model-secondary",
            str(args.model_secondary),
            "--classification-input-mode",
            str(args.classification_input_mode),
            *([] if not bool(args.require_clean_for_law) else ["--require-clean-for-law"]),
        ],
        cwd=repo_root,
    )

    _run(
        [
            py,
            str(repo_root / "scripts" / "llm_event_cluster_v3.py"),
            "--classification-jsonl",
            str(out_root / "classification" / "issue_labels_v3.jsonl"),
            "--clean-jsonl",
            str(out_root / "cleaning" / "clean_zoning_text.jsonl"),
            "--output-dir",
            str(out_root / "events"),
            "--variant",
            selected_dedup,
            "--model",
            str(args.model_primary),
            "--concurrency",
            str(int(args.concurrency)),
            "--timeout",
            str(float(args.timeout)),
            "--gateway-runner",
            str(Path(args.gateway_runner).expanduser().resolve()),
            "--gateway-pythonpath",
            str(Path(args.gateway_pythonpath).expanduser().resolve()),
            "--gov-env-path",
            str(Path(args.gov_env_path).expanduser().resolve()),
            *([] if not bool(args.skip_existing) else ["--skip-existing"]),
            *([] if not bool(args.dry_run) else ["--dry-run"]),
        ],
        cwd=repo_root,
    )

    # 3b) Event-level status adjudication for strict enacted/proposed handling.
    _run(
        [
            py,
            str(repo_root / "scripts" / "llm_event_status_adjudicate_v3.py"),
            "--classification-jsonl",
            str(out_root / "classification" / "issue_labels_v3.jsonl"),
            "--clean-jsonl",
            str(out_root / "cleaning" / "clean_zoning_text.jsonl"),
            "--event-clusters-csv",
            str(out_root / "events" / "ordinance_event_clusters_v3.csv"),
            "--event-assignments-csv",
            str(out_root / "events" / "issue_event_assignments_v3.csv"),
            "--output-dir",
            str(out_root / "events_status"),
            "--model",
            str(args.model_primary),
            "--concurrency",
            str(int(args.concurrency)),
            "--timeout",
            str(float(args.timeout)),
            "--gateway-runner",
            str(Path(args.gateway_runner).expanduser().resolve()),
            "--gateway-pythonpath",
            str(Path(args.gateway_pythonpath).expanduser().resolve()),
            "--gov-env-path",
            str(Path(args.gov_env_path).expanduser().resolve()),
            *([] if not bool(args.skip_existing) else ["--skip-existing"]),
            *([] if not bool(args.dry_run) else ["--dry-run"]),
        ],
        cwd=repo_root,
    )

    # 3c) Precision-first consistency gate (demote mixed/proposed leakage in core buckets).
    _run(
        [
            py,
            str(repo_root / "scripts" / "enforce_precision_event_consistency_v1.py"),
            "--pipeline-root",
            str(out_root),
        ],
        cwd=repo_root,
    )

    # Use precision-gated event assignments when available.
    event_assign_path_for_long = out_root / "events_status" / "precision_v1" / "issue_event_assignments_status_v3_precision.csv"
    if not event_assign_path_for_long.is_file():
        event_assign_path_for_long = out_root / "events_status" / "issue_event_assignments_status_v3.csv"
    if not event_assign_path_for_long.is_file():
        event_assign_path_for_long = out_root / "events" / "issue_event_assignments_v3.csv"

    event_cluster_path_for_export = out_root / "events_status" / "precision_v1" / "ordinance_event_clusters_status_v3_precision.csv"
    if not event_cluster_path_for_export.is_file():
        event_cluster_path_for_export = out_root / "events_status" / "ordinance_event_clusters_status_v3.csv"
    if not event_cluster_path_for_export.is_file():
        event_cluster_path_for_export = out_root / "events" / "ordinance_event_clusters_v3.csv"

    # 3d) Canonical page-level classification index (page_id -> final bucket).
    _run(
        [
            py,
            str(repo_root / "scripts" / "build_page_classification_index_v1.py"),
            "--source-run-dir",
            str(run_dir),
            "--pipeline-root",
            str(out_root),
            "--event-assignments-csv",
            str(event_assign_path_for_long),
            "--output-dir",
            str(out_root / "page_classification_precision_v1"),
        ],
        cwd=repo_root,
    )

    # 3e) Classification invariants/QC checks.
    _run(
        [
            py,
            str(repo_root / "scripts" / "check_classification_hardening_v1.py"),
            "--pipeline-root",
            str(out_root),
            "--issue-assignments-csv",
            str(event_assign_path_for_long),
            "--page-classification-csv",
            str(out_root / "page_classification_precision_v1" / "page_classification_index_v1.csv"),
            "--output-dir",
            str(out_root / "qc_precision_v1"),
        ],
        cwd=repo_root,
    )

    _run(
        [
            py,
            str(repo_root / "scripts" / "llm_panel_longitudinal_v3.py"),
            "--run-dir",
            str(run_dir),
            "--classification-jsonl",
            str(out_root / "classification" / "issue_labels_v3.jsonl"),
            "--clean-jsonl",
            str(out_root / "cleaning" / "clean_zoning_text.jsonl"),
            "--event-assignments-csv",
            str(event_assign_path_for_long),
            "--output-dir",
            str(out_root / "longitudinal"),
            "--model",
            str(args.model_primary),
            "--max-panel-chars",
            str(int(args.max_panel_chars)),
            "--concurrency",
            str(max(1, int(args.concurrency) - 1)),
            "--timeout",
            str(float(args.timeout)),
            "--gateway-runner",
            str(Path(args.gateway_runner).expanduser().resolve()),
            "--gateway-pythonpath",
            str(Path(args.gateway_pythonpath).expanduser().resolve()),
            "--gov-env-path",
            str(Path(args.gov_env_path).expanduser().resolve()),
            *([] if not bool(args.skip_existing) else ["--skip-existing"]),
            *([] if not bool(args.dry_run) else ["--dry-run"]),
        ],
        cwd=repo_root,
    )

    # 4) Build analysis and manual validation sheets.
    _build_analysis_tables(
        out_root,
        manual_qa_per_paper=int(args.manual_qa_per_paper),
        manual_qa_full_share=float(args.manual_qa_full_share),
        manual_qa_amend_share=float(args.manual_qa_amend_share),
        manual_qa_narrative_per_paper=int(args.manual_qa_narrative_per_paper),
    )

    # 5) Organized corpus export.
    _run(
        [
            py,
            str(repo_root / "scripts" / "export_organized_zoning_corpus_v3.py"),
            "--run-root",
            str(out_root),
            "--classification-csv",
            str(out_root / "classification" / "issue_labels_v3.csv"),
            "--clean-csv",
            str(out_root / "cleaning" / "clean_zoning_text_flat.csv"),
            "--event-assignments-csv",
            str(event_assign_path_for_long),
            "--event-clusters-csv",
            str(event_cluster_path_for_export),
        ],
        cwd=repo_root,
    )

    # 6) Report.
    _run(
        [
            py,
            str(repo_root / "scripts" / "build_fullnewspaper_v3_report.py"),
            "--run-root",
            str(out_root),
            "--compile-pdf",
        ],
        cwd=repo_root,
    )

    prov = {
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "script_path": str(Path(__file__).resolve()),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "run_dir": str(run_dir),
        "output_dir": str(out_root),
        "newspapers": newspapers,
        "selected_cleaning_variant": selected_clean,
        "selected_classification_variant": selected_cls,
        "selected_dedup_variant": selected_dedup,
        "classification_input_mode": str(args.classification_input_mode),
        "require_clean_for_law": bool(args.require_clean_for_law),
        "model_primary": str(args.model_primary),
        "model_secondary": str(args.model_secondary),
    }
    (out_root / "provenance.json").write_text(json.dumps(prov, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"done_fullnewspaper_v3_pipeline output_dir={out_root}")


if __name__ == "__main__":
    main()
