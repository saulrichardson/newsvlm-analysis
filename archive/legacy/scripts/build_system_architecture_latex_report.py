#!/usr/bin/env python3
"""
Build a self-contained LaTeX walkthrough of the end-to-end newspaper zoning pipeline.

The report is architecture-first and example-driven:
1) ingest newspaper issues,
2) identify legal zoning text,
3) classify issue type,
4) deduplicate repeated publications into events,
5) run timeline-level longitudinal extraction,
6) export an organized ordinance/amendment corpus.
"""

from __future__ import annotations

import argparse
import ast
import datetime as dt
import json
import subprocess
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch


FULL_CLASSES = {"full_code_comprehensive", "full_code_recodification"}
AMEND_CLASSES = {"amendment_substantial", "amendment_targeted"}

MONTH_ABBR = {
    1: "jan",
    2: "feb",
    3: "mar",
    4: "apr",
    5: "may",
    6: "jun",
    7: "jul",
    8: "aug",
    9: "sep",
    10: "oct",
    11: "nov",
    12: "dec",
}


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _safe_read_jsonl(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    try:
        return pd.read_json(path, lines=True)
    except Exception:
        return pd.DataFrame()


def _safe_read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _clip_text(value: Any, n: int = 280) -> str:
    s = str(value or "").replace("\n", " ").strip()
    if len(s) <= n:
        return s
    return s[: max(0, n - 3)] + "..."


def _tex_escape(value: Any) -> str:
    s = str(value or "")
    s = s.replace("\\", "\\textbackslash{}")
    s = s.replace("&", "\\&")
    s = s.replace("%", "\\%")
    s = s.replace("$", "\\$")
    s = s.replace("#", "\\#")
    s = s.replace("_", "\\_")
    s = s.replace("{", "\\{")
    s = s.replace("}", "\\}")
    s = s.replace("^", "\\textasciicircum{}")
    s = s.replace("~", "\\textasciitilde{}")
    return s


def _write_text(path: Path, text: str) -> None:
    _ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")


def _to_latex_table(df: pd.DataFrame, out_path: Path, *, index: bool = False, float_fmt: str = "{:.3f}") -> None:
    _ensure_dir(out_path.parent)
    if df.empty:
        out_path.write_text("\\begin{tabular}{l}\n\\toprule\nNo data available.\\\\\n\\bottomrule\n\\end{tabular}\n", encoding="utf-8")
        return

    def _fmt(x: Any) -> str:
        if isinstance(x, float):
            return float_fmt.format(x)
        return str(x)

    tex = df.to_latex(index=index, escape=True, na_rep="", float_format=lambda x: _fmt(x))
    out_path.write_text(tex, encoding="utf-8")


def _issue_png_candidates(fig_dir: Path, issue_id: str) -> list[Path]:
    if "__" not in issue_id:
        return []
    slug, date_part = issue_id.split("__", 1)
    try:
        d = dt.date.fromisoformat(date_part)
    except Exception:
        return []
    prefix = f"{slug}-{MONTH_ABBR[d.month]}-{d.day:02d}-{d.year}-p-"
    return sorted(fig_dir.glob(prefix + "*.png"))


def _pick_issue_with_png(issue_df: pd.DataFrame, png_dir: Path) -> tuple[str, Path | None]:
    if issue_df.empty:
        return "", None
    for _, row in issue_df.iterrows():
        issue_id = str(row.get("issue_id", ""))
        candidates = _issue_png_candidates(png_dir, issue_id)
        if candidates:
            return issue_id, candidates[0]
    # fallback to first row without png
    return str(issue_df.iloc[0].get("issue_id", "")), None


def _extract_page_tag_rows(page_tags_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in page_tags_df.iterrows():
        tags = row.get("page_tags", [])
        if not isinstance(tags, list):
            continue
        for tag_obj in tags:
            if not isinstance(tag_obj, dict):
                continue
            rows.append(
                {
                    "newspaper_slug": row.get("newspaper_slug", ""),
                    "issue_id": row.get("issue_id", ""),
                    "issue_date": row.get("issue_date", ""),
                    "page_id": tag_obj.get("page_id", ""),
                    "page_tag": tag_obj.get("tag", "missing"),
                    "confidence_0_to_1": tag_obj.get("confidence_0_to_1", np.nan),
                    "evidence": tag_obj.get("evidence", ""),
                }
            )
    return pd.DataFrame(rows)


def _draw_node(ax: Any, x: float, y: float, w: float, h: float, label: str, color: str) -> None:
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.01,rounding_size=0.02",
        linewidth=1.2,
        edgecolor="#3e3e3e",
        facecolor=color,
    )
    ax.add_patch(box)
    ax.text(x + w / 2.0, y + h / 2.0, label, ha="center", va="center", fontsize=10)


def _draw_arrow(ax: Any, x1: float, y1: float, x2: float, y2: float) -> None:
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", lw=1.4, color="#4a4a4a"))


def _plot_architecture_diagram(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(13.6, 4.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    y = 0.55
    w = 0.145
    h = 0.24
    xs = [0.02, 0.185, 0.35, 0.515, 0.68, 0.845]
    labels = [
        "Newspaper issues\n(full transcripts)",
        "Legal-text extraction\n(LLM page+anchor pass)",
        "Issue classification\n(LLM ontology labels)",
        "Event deduplication\n(cluster repeat publications)",
        "Timeline extraction\n(first/full + evolution)",
        "Organized outputs\n(full + amend bundles)",
    ]
    colors = ["#bcd7ea", "#c8e6c9", "#c8e6c9", "#c8e6c9", "#c8e6c9", "#f3e2bd"]

    for x, label, color in zip(xs, labels, colors):
        _draw_node(ax, x, y, w, h, label, color)

    for i in range(len(xs) - 1):
        _draw_arrow(ax, xs[i] + w, y + h / 2.0, xs[i + 1], y + h / 2.0)

    # Secondary output branch to report
    _draw_node(ax, 0.43, 0.10, 0.22, 0.22, "Descriptive report\n(tables, figures, examples)", "#f3e2bd")
    _draw_arrow(ax, xs[5], y + h * 0.35, 0.65, 0.21)
    _draw_arrow(ax, xs[4] + w * 0.5, y, 0.54, 0.32)

    fig.tight_layout()
    _ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _plot_dedup_funnel(raw_rows: int, dedup_events: int, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    labels = ["Issue rows\n(full+amend)", "Unique deduped\nevents"]
    vals = [raw_rows, dedup_events]
    bars = ax.bar(labels, vals, color=["#4c78a8", "#f58518"])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2.0, v + 0.3, f"{v}", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("Count")
    ax.set_title("Deduplication impact")
    ax.set_ylim(0, max(vals + [1]) * 1.25)
    fig.tight_layout()
    _ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _plot_issue_class_counts(issue_labels: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    if issue_labels.empty:
        ax.text(0.5, 0.5, "No issue labels available", ha="center", va="center")
        ax.axis("off")
    else:
        d = issue_labels.groupby("issue_class", dropna=False)["issue_id"].count().reset_index(name="count")
        d = d.sort_values("count", ascending=False)
        ax.barh(d["issue_class"], d["count"], color="#54a24b")
        ax.invert_yaxis()
        ax.set_xlabel("Issue rows")
        ax.set_title("Issue-level ontology class counts")
    fig.tight_layout()
    _ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _compile_pdf(output_dir: Path) -> bool:
    tex_path = output_dir / "report.tex"
    if not tex_path.is_file():
        return False
    try:
        subprocess.run(
            ["latexmk", "-pdf", "-interaction=nonstopmode", "report.tex"],
            cwd=str(output_dir),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        return True
    except Exception:
        return False


def build_report(run_dir: Path, output_dir: Path, compile_pdf: bool = True) -> dict[str, Any]:
    page_tags_path = run_dir / "page_tagging" / "page_tags.jsonl"
    anchor_decisions_path = run_dir / "cleaning" / "anchor_decisions.jsonl"
    clean_text_path = run_dir / "cleaning" / "clean_zoning_text_flat.csv"
    issue_labels_path = run_dir / "classification" / "issue_labels_v3.csv"
    assign_path = run_dir / "events" / "issue_event_assignments_v3.csv"
    clusters_path = run_dir / "events" / "ordinance_event_clusters_v3.csv"
    city_summary_path = run_dir / "longitudinal" / "city_longitudinal_summary.csv"
    city_cat_path = run_dir / "longitudinal" / "city_category_trajectory.csv"
    city_complexity_path = run_dir / "longitudinal" / "city_complexity_trajectory.csv"
    panel_outputs_path = run_dir / "longitudinal" / "panel_outputs_v3.jsonl"
    export_inventory_path = run_dir / "organized_corpus_v3" / "summary" / "export_inventory.json"
    organized_full_events_path = run_dir / "organized_corpus_v3" / "summary" / "full_ordinance_events.csv"
    organized_amend_events_path = run_dir / "organized_corpus_v3" / "summary" / "amendment_events.csv"

    page_tags = _safe_read_jsonl(page_tags_path)
    anchor_decisions = _safe_read_jsonl(anchor_decisions_path)
    clean_text = _safe_read_csv(clean_text_path)
    issue_labels = _safe_read_csv(issue_labels_path)
    assign = _safe_read_csv(assign_path)
    clusters = _safe_read_csv(clusters_path)
    city_summary = _safe_read_csv(city_summary_path)
    city_cat = _safe_read_csv(city_cat_path)
    city_complexity = _safe_read_csv(city_complexity_path)
    panel_outputs = _safe_read_jsonl(panel_outputs_path)
    export_inventory = _safe_read_json(export_inventory_path)
    organized_full_events = _safe_read_csv(organized_full_events_path)
    organized_amend_events = _safe_read_csv(organized_amend_events_path)

    _ensure_dir(output_dir)
    fig_dir = output_dir / "figures"
    tbl_dir = output_dir / "tables"
    _ensure_dir(fig_dir)
    _ensure_dir(tbl_dir)

    # Choose one full-ordinance and one amendment exemplar with available PNG.
    png_search_dirs = [
        run_dir / "reports" / "fullnewspaper_v3" / "figures" / "appendix_pages",
        run_dir / "reports" / "fullnewspaper_v3" / "figures" / "evidence_pages",
    ]
    png_dir = next((d for d in png_search_dirs if d.is_dir()), run_dir)

    full_candidates = issue_labels[issue_labels["issue_class"].isin(sorted(FULL_CLASSES))].copy()
    full_candidates = full_candidates.sort_values(["clean_chars", "issue_date"], ascending=[False, True])
    full_issue_id, full_png = _pick_issue_with_png(full_candidates, png_dir)

    amend_candidates = issue_labels[issue_labels["issue_class"].isin(sorted(AMEND_CLASSES))].copy()
    if not amend_candidates.empty:
        amend_candidates = amend_candidates.merge(
            clean_text[["issue_id", "clean_zoning_text"]], on="issue_id", how="left", suffixes=("", "_clean")
        )
        combined = (
            amend_candidates["evidence_quotes"].fillna("").astype(str).str.lower()
            + " "
            + amend_candidates["clean_zoning_text"].fillna("").astype(str).str.lower()
        )
        amend_candidates["zoning_example_score"] = (
            combined.str.contains("zoning").astype(int) * 3
            + combined.str.contains("district").astype(int) * 2
            + combined.str.contains("amend").astype(int) * 1
            - combined.str.contains("water course|treasurer|tax").astype(int) * 3
        )
    else:
        amend_candidates["zoning_example_score"] = 0
    amend_candidates = amend_candidates.sort_values(
        ["zoning_example_score", "clean_chars", "issue_date"],
        ascending=[False, False, True],
    )
    amend_issue_id, amend_png = _pick_issue_with_png(amend_candidates, png_dir)

    # Stage-level contract table.
    stage_contracts = pd.DataFrame(
        [
            ["Issue ingestion", "Issue transcripts with page segmentation", "Chronological issue panel", "Issue (newspaper-date)"],
            ["Legal-text extraction", "Page segments + anchorized text", "Clean zoning legal text", "Anchor + issue"],
            ["Issue classification", "Clean legal text", "Ontology label + category shares", "Issue"],
            ["Event deduplication", "Issue labels over time", "Clustered legal events", "Event"],
            ["Longitudinal extraction", "Full newspaper timeline", "First/full timing + content trajectories", "Timeline (city/newspaper)"],
            ["Organized corpus export", "Deduped events + issue members", "Full/amend bundles for downstream analysis", "Event + publication member"],
        ],
        columns=["Stage", "Primary input", "Primary output", "Unit of analysis"],
    )
    _to_latex_table(stage_contracts, tbl_dir / "stage_contracts.tex", index=False, float_fmt="{:.0f}")

    # Coverage summary.
    issue_count = int(issue_labels["issue_id"].nunique()) if not issue_labels.empty else 0
    newspaper_count = int(issue_labels["newspaper_slug"].nunique()) if not issue_labels.empty else 0
    page_rows = _extract_page_tag_rows(page_tags)
    page_count = int(page_rows["page_id"].nunique()) if not page_rows.empty else 0
    clean_nonempty = int((clean_text["clean_chars"] > 0).sum()) if not clean_text.empty else 0
    raw_full_amend_rows = 0
    dedup_full_amend_events = 0
    if not assign.empty:
        raw_full_amend_rows = int(assign["event_type"].isin(["full_ordinance", "amendment"]).sum())
    if not clusters.empty:
        dedup_full_amend_events = int(clusters["event_type"].isin(["full_ordinance", "amendment"]).sum())
    reduction_pct = 0.0
    if raw_full_amend_rows > 0:
        reduction_pct = (1.0 - (dedup_full_amend_events / raw_full_amend_rows)) * 100.0
    timeline_count = int(city_summary["newspaper_slug"].nunique()) if not city_summary.empty else 0

    coverage_summary = pd.DataFrame(
        [
            ["Newspapers", newspaper_count],
            ["Issue rows", issue_count],
            ["Tagged pages", page_count],
            ["Issue rows with non-empty clean legal text", clean_nonempty],
            ["Issue rows labeled full/amend (pre-dedup)", raw_full_amend_rows],
            ["Unique full/amend events (post-dedup)", dedup_full_amend_events],
            ["Dedup reduction (percent)", round(reduction_pct, 1)],
            ["Longitudinal timelines", timeline_count],
        ],
        columns=["Metric", "Value"],
    )
    _to_latex_table(coverage_summary, tbl_dir / "coverage_summary.tex", index=False, float_fmt="{:.1f}")

    # Page-tag mix table.
    page_tag_mix = pd.DataFrame(columns=["page_tag", "page_count", "share_pct"])
    if not page_rows.empty:
        page_tag_mix = page_rows.groupby("page_tag", dropna=False)["page_id"].nunique().reset_index(name="page_count")
        total_pages = max(1, int(page_tag_mix["page_count"].sum()))
        page_tag_mix["share_pct"] = (page_tag_mix["page_count"] / total_pages * 100.0).round(1)
        page_tag_mix = page_tag_mix.sort_values("page_count", ascending=False)
    _to_latex_table(page_tag_mix, tbl_dir / "page_tag_mix.tex", index=False, float_fmt="{:.1f}")

    # Ontology counts.
    ontology_counts = pd.DataFrame(columns=["issue_class", "issue_rows"])
    if not issue_labels.empty:
        ontology_counts = issue_labels.groupby("issue_class", dropna=False)["issue_id"].count().reset_index(name="issue_rows")
        ontology_counts = ontology_counts.sort_values("issue_rows", ascending=False)
    _to_latex_table(ontology_counts, tbl_dir / "ontology_counts.tex", index=False, float_fmt="{:.0f}")

    # Exemplar issue trace table.
    example_rows: list[dict[str, Any]] = []
    for tag, issue_id in [("Full-ordinance example", full_issue_id), ("Amendment example", amend_issue_id)]:
        if not issue_id:
            continue
        irow = issue_labels[issue_labels["issue_id"] == issue_id]
        crow = clean_text[clean_text["issue_id"] == issue_id]
        arow = assign[assign["issue_id"] == issue_id]
        if irow.empty:
            continue
        irow0 = irow.iloc[0]
        crow0 = crow.iloc[0] if not crow.empty else {}
        arow0 = arow.iloc[0] if not arow.empty else {}
        example_rows.append(
            {
                "example": tag,
                "issue_id": issue_id,
                "issue_date": irow0.get("issue_date", ""),
                "issue_class": irow0.get("issue_class", ""),
                "verbatim_law": irow0.get("contains_verbatim_zoning_law", ""),
                "law_sufficiency": irow0.get("law_text_sufficiency", ""),
                "clean_chars": int(crow0.get("clean_chars", irow0.get("clean_chars", 0)) or 0),
                "event_type": arow0.get("event_type", ""),
                "event_id": arow0.get("ordinance_event_id", ""),
            }
        )
    example_trace = pd.DataFrame(example_rows)
    _to_latex_table(example_trace, tbl_dir / "example_issue_trace.tex", index=False, float_fmt="{:.0f}")

    # Dedup cluster example.
    dedup_example = pd.DataFrame(columns=["issue_id", "issue_date", "issue_class", "publication_role"])
    dedup_header = {"newspaper_slug": "", "ordinance_event_id": "", "event_type": "", "issue_count": 0}
    if not clusters.empty and not assign.empty:
        c = clusters[clusters["event_type"].isin(["full_ordinance", "amendment"]) & (clusters["issue_count"] > 1)].copy()
        if not c.empty:
            c = c.sort_values(["issue_count", "first_issue_date"], ascending=[False, True])
            c0 = c.iloc[0]
            dedup_header = {
                "newspaper_slug": c0.get("newspaper_slug", ""),
                "ordinance_event_id": c0.get("ordinance_event_id", ""),
                "event_type": c0.get("event_type", ""),
                "issue_count": int(c0.get("issue_count", 0) or 0),
            }
            dedup_example = assign[
                (assign["newspaper_slug"] == c0.get("newspaper_slug", ""))
                & (assign["ordinance_event_id"] == c0.get("ordinance_event_id", ""))
            ][["issue_id", "issue_date", "issue_class", "publication_role"]].sort_values("issue_date")
    _to_latex_table(dedup_example, tbl_dir / "dedup_example_cluster.tex", index=False, float_fmt="{:.0f}")

    # Longitudinal summary + category means.
    first_timing = pd.DataFrame(columns=["newspaper_slug", "first_signal_date", "first_signal_type", "first_full_date", "first_full_status"])
    if not city_summary.empty:
        cols = ["newspaper_slug", "first_signal_date", "first_signal_type", "first_full_date", "first_full_status"]
        first_timing = city_summary[cols].copy()
    _to_latex_table(first_timing, tbl_dir / "first_timing.tex", index=False, float_fmt="{:.0f}")

    category_means = pd.DataFrame(columns=["category", "early", "mid", "late", "late_minus_early_pp"])
    if not city_cat.empty:
        g = city_cat.groupby("category", dropna=False)[["early_share", "mid_share", "late_share"]].mean().reset_index()
        g["late_minus_early_pp"] = (g["late_share"] - g["early_share"]) * 100.0
        category_means = g.rename(columns={"early_share": "early", "mid_share": "mid", "late_share": "late"})
        category_means = category_means.sort_values("late", ascending=False)
    _to_latex_table(category_means, tbl_dir / "category_means.tex", index=False, float_fmt="{:.3f}")

    complexity_counts = pd.DataFrame(columns=["metric", "up", "flat", "down", "mixed_or_uncertain"])
    if not city_complexity.empty:
        rows = []
        metric_map = {
            "regulatory_complexity_trend": "regulatory_complexity",
            "procedural_intensity_trend": "procedural_intensity",
            "dimensional_specificity_trend": "dimensional_specificity",
        }
        for col, metric in metric_map.items():
            vals = city_complexity[col].fillna("uncertain").astype(str).str.lower()
            rows.append(
                {
                    "metric": metric,
                    "up": int((vals == "up").sum()),
                    "flat": int((vals == "flat").sum()),
                    "down": int((vals == "down").sum()),
                    "mixed_or_uncertain": int(((vals == "mixed") | (vals == "uncertain")).sum()),
                }
            )
        complexity_counts = pd.DataFrame(rows)
    _to_latex_table(complexity_counts, tbl_dir / "complexity_counts.tex", index=False, float_fmt="{:.0f}")

    # Longitudinal narrative example from parsed json.
    timeline_example_slug = ""
    timeline_example_text = ""
    timeline_first_signal = {}
    timeline_first_full = {}
    if not panel_outputs.empty:
        # Pick the same newspaper as the full example when possible.
        if full_issue_id and "__" in full_issue_id:
            timeline_example_slug = full_issue_id.split("__", 1)[0]
        p = panel_outputs.copy()
        if timeline_example_slug:
            p = p[p["newspaper_slug"] == timeline_example_slug]
        if p.empty:
            p = panel_outputs
        row = p.iloc[0]
        parsed = row.get("parsed_json", {}) if isinstance(row.get("parsed_json", {}), dict) else {}
        timeline_example_slug = str(row.get("newspaper_slug", timeline_example_slug))
        timeline_example_text = _clip_text(parsed.get("narrative_summary", ""), n=900)
        timeline_first_signal = parsed.get("first_zoning_signal", {}) if isinstance(parsed.get("first_zoning_signal", {}), dict) else {}
        timeline_first_full = parsed.get("first_full_ordinance", {}) if isinstance(parsed.get("first_full_ordinance", {}), dict) else {}

    # Organized corpus summary.
    organized_summary_rows = [
        ["Full ordinance events", int(export_inventory.get("full_ordinance_event_count", 0) or 0)],
        ["Amendment events", int(export_inventory.get("amendment_event_count", 0) or 0)],
        ["Law-only issue rows", int(export_inventory.get("law_issue_row_count", 0) or 0)],
        ["Total issue rows", int(export_inventory.get("issue_row_count", 0) or 0)],
        ["Newspapers", int(export_inventory.get("newspaper_count", 0) or 0)],
    ]
    organized_summary = pd.DataFrame(organized_summary_rows, columns=["Output artifact", "Count"])
    _to_latex_table(organized_summary, tbl_dir / "organized_summary.tex", index=False, float_fmt="{:.0f}")

    # Newspaper-level event table from organized outputs.
    events_by_paper = pd.DataFrame(columns=["newspaper_slug", "full_events", "amendment_events"])
    if not organized_full_events.empty or not organized_amend_events.empty:
        full_counts = (
            organized_full_events.groupby("newspaper_slug", dropna=False)["ordinance_event_id"]
            .nunique()
            .reset_index(name="full_events")
            if not organized_full_events.empty
            else pd.DataFrame(columns=["newspaper_slug", "full_events"])
        )
        amend_counts = (
            organized_amend_events.groupby("newspaper_slug", dropna=False)["ordinance_event_id"]
            .nunique()
            .reset_index(name="amendment_events")
            if not organized_amend_events.empty
            else pd.DataFrame(columns=["newspaper_slug", "amendment_events"])
        )
        events_by_paper = full_counts.merge(amend_counts, on="newspaper_slug", how="outer").fillna(0)
        events_by_paper["full_events"] = events_by_paper["full_events"].astype(int)
        events_by_paper["amendment_events"] = events_by_paper["amendment_events"].astype(int)
        events_by_paper = events_by_paper.sort_values("newspaper_slug")
    _to_latex_table(events_by_paper, tbl_dir / "events_by_paper.tex", index=False, float_fmt="{:.0f}")

    # Figures.
    architecture_fig = fig_dir / "architecture_6stage.png"
    dedup_funnel_fig = fig_dir / "dedup_funnel.png"
    class_counts_fig = fig_dir / "ontology_counts.png"
    _plot_architecture_diagram(architecture_fig)
    _plot_dedup_funnel(raw_full_amend_rows, dedup_full_amend_events, dedup_funnel_fig)
    _plot_issue_class_counts(issue_labels, class_counts_fig)

    # Reuse selected existing figures if present.
    full_report_fig_dir = run_dir / "reports" / "fullnewspaper_v3" / "figures"
    category_fig = full_report_fig_dir / "category_trajectory_stage_means.png"
    complexity_fig = full_report_fig_dir / "complexity_heatmap.png"
    dedup_year_fig = full_report_fig_dir / "full_vs_amendment_by_year_dedup.png"

    # Copy exemplar page PNGs into report figures.
    full_page_copy = None
    amend_page_copy = None
    if full_png and full_png.is_file():
        full_page_copy = fig_dir / full_png.name
        full_page_copy.write_bytes(full_png.read_bytes())
    if amend_png and amend_png.is_file():
        amend_page_copy = fig_dir / amend_png.name
        amend_page_copy.write_bytes(amend_png.read_bytes())

    # Pull evidence text snippets for exemplars.
    full_evidence_quote = ""
    amend_evidence_quote = ""
    full_clean_snippet = ""
    amend_clean_snippet = ""
    if full_issue_id:
        full_row = issue_labels[issue_labels["issue_id"] == full_issue_id]
        if not full_row.empty:
            ev = full_row.iloc[0].get("evidence_quotes", "")
            try:
                parsed = ast.literal_eval(ev) if isinstance(ev, str) else ev
                if isinstance(parsed, list) and parsed:
                    full_evidence_quote = _clip_text(parsed[0], 360)
            except Exception:
                full_evidence_quote = _clip_text(ev, 360)
        full_clean = clean_text[clean_text["issue_id"] == full_issue_id]
        if not full_clean.empty:
            full_clean_snippet = _clip_text(full_clean.iloc[0].get("clean_zoning_text", ""), 750)
    if amend_issue_id:
        amend_row = issue_labels[issue_labels["issue_id"] == amend_issue_id]
        if not amend_row.empty:
            ev = amend_row.iloc[0].get("evidence_quotes", "")
            try:
                parsed = ast.literal_eval(ev) if isinstance(ev, str) else ev
                if isinstance(parsed, list) and parsed:
                    amend_evidence_quote = _clip_text(parsed[0], 360)
            except Exception:
                amend_evidence_quote = _clip_text(ev, 360)
        amend_clean = clean_text[clean_text["issue_id"] == amend_issue_id]
        if not amend_clean.empty:
            amend_clean_snippet = _clip_text(amend_clean.iloc[0].get("clean_zoning_text", ""), 750)

    # Anchor-level detail table for full exemplar.
    anchor_example_rows = pd.DataFrame(columns=["anchor_id", "keep", "anchor_role", "confidence", "text_excerpt"])
    if full_issue_id and not anchor_decisions.empty:
        arow = anchor_decisions[anchor_decisions["issue_id"] == full_issue_id]
        if not arow.empty:
            anchors = arow.iloc[0].get("anchors", [])
            if isinstance(anchors, list):
                kept = [a for a in anchors if isinstance(a, dict) and a.get("keep_for_law_corpus") == 1][:3]
                dropped = [a for a in anchors if isinstance(a, dict) and a.get("keep_for_law_corpus") == 0][:3]
                rows = []
                for a in kept + dropped:
                    rows.append(
                        {
                            "anchor_id": a.get("anchor_id", ""),
                            "keep": "yes" if a.get("keep_for_law_corpus") == 1 else "no",
                            "anchor_role": a.get("anchor_role", ""),
                            "confidence": float(a.get("confidence_0_to_1", 0.0) or 0.0),
                            "text_excerpt": _clip_text(a.get("text", ""), 130),
                        }
                    )
                anchor_example_rows = pd.DataFrame(rows)
    _to_latex_table(anchor_example_rows, tbl_dir / "anchor_example.tex", index=False, float_fmt="{:.2f}")

    # Report tex.
    date_str = dt.date.today().isoformat()
    tex = f"""
\\documentclass[11pt]{{article}}
\\usepackage[margin=1in]{{geometry}}
\\usepackage{{booktabs}}
\\usepackage{{graphicx}}
\\usepackage{{float}}
\\usepackage{{hyperref}}
\\usepackage{{xurl}}
\\usepackage{{longtable}}
\\setlength{{\\parskip}}{{0.5em}}
\\setlength{{\\parindent}}{{0pt}}
\\setcounter{{tocdepth}}{{2}}
\\newcommand{{\\fignote}}[1]{{\\vspace{{0.1em}}\\begin{{minipage}}{{0.98\\textwidth}}\\footnotesize\\textit{{Detail.}} #1\\end{{minipage}}}}
\\title{{System Architecture Walkthrough: Newspaper-to-Zoning Pipeline}}
\\author{{Automated architecture report}}
\\date{{{_tex_escape(date_str)}}}
\\begin{{document}}
\\maketitle
\\tableofcontents
\\clearpage

\\section{{End-to-end architecture}}
The production workflow starts with complete newspaper issue transcripts and ends with an organized legal corpus that separates full ordinances, amendments, and non-event material.
Every stage writes structured outputs that feed the next stage, so the same issue can be traced from raw page text all the way to timeline-level policy summaries.

\\begin{{figure}}[H]
\\centering
\\includegraphics[width=0.98\\textwidth]{{figures/architecture_6stage.png}}
\\caption{{Six-stage pipeline from issue ingestion to organized legal outputs.}}
\\fignote{{This diagram is generated from the implemented v3 workflow. The core semantic decisions are legal-text extraction, issue ontology classification, event deduplication, and full-timeline longitudinal extraction.}}
\\end{{figure}}

\\begin{{table}}[H]
\\centering
\\caption{{Stage-by-stage data contract}}
\\resizebox{{\\textwidth}}{{!}}{{\\input{{tables/stage_contracts.tex}}}}
\\end{{table}}
\\fignote{{Each stage has a distinct analysis unit. The unit changes from issue-level rows to deduplicated legal events and then to timeline-level summaries.}}

\\begin{{table}}[H]
\\centering
\\caption{{Run coverage snapshot}}
\\resizebox{{0.72\\textwidth}}{{!}}{{\\input{{tables/coverage_summary.tex}}}}
\\end{{table}}
\\fignote{{Counts summarize the latest five-newspaper production run. Pre-dedup legal rows and post-dedup event counts are both shown so repeated newspaper publications are separated from unique legal actions.}}

\\section{{Legal-text extraction from mixed newspaper pages}}
Issue text often mixes legal language, reporting, advertisements, and unrelated notices in the same page.
The extraction stage uses page-level context tags and anchor-level keep/drop decisions to isolate verbatim zoning law text.
Page tags improve auditability, while anchor decisions are the final inclusion mechanism.

\\begin{{table}}[H]
\\centering
\\caption{{Page context tags across the run}}
\\resizebox{{0.70\\textwidth}}{{!}}{{\\input{{tables/page_tag_mix.tex}}}}
\\end{{table}}
\\fignote{{A page tag is an LLM context label at page scope. Final clean text is still decided at anchor scope, so legal clauses on mixed pages can be retained even when nearby content is non-legal.}}

\\begin{{table}}[H]
\\centering
\\caption{{Anchor keep/drop sample from one full-ordinance issue}}
\\resizebox{{\\textwidth}}{{!}}{{\\input{{tables/anchor_example.tex}}}}
\\end{{table}}
\\fignote{{The same issue can contain both dropped non-legal anchors and kept legal anchors. The retained anchors are concatenated in original order into clean zoning text for downstream classification.}}

\\begin{{table}}[H]
\\centering
\\caption{{Worked issue trace through extraction and labeling}}
\\resizebox{{\\textwidth}}{{!}}{{\\input{{tables/example_issue_trace.tex}}}}
\\end{{table}}
\\fignote{{The full-ordinance and amendment rows show how one issue moves through legal sufficiency checks, receives an ontology class, and is attached to an event identifier.}}

\\subsection{{Full-ordinance example page}}
\\begin{{figure}}[H]
\\centering
{f"\\includegraphics[width=0.78\\textwidth]{{figures/{_tex_escape(full_page_copy.name)}}}" if full_page_copy else "\\fbox{No full-ordinance example image available}"}
\\caption{{Example source page classified into the full-ordinance stream.}}
\\fignote{{Issue: \\texttt{{{_tex_escape(full_issue_id)}}}. The clean legal text extracted from this issue is used for ontology classification and event assignment.}}
\\end{{figure}}

\\begin{{quote}}\\small\\textbf{{Example evidence quote:}} {_tex_escape(full_evidence_quote or "No evidence quote available.")}\\end{{quote}}
\\begin{{quote}}\\small\\textbf{{Extracted clean legal text snippet:}} {_tex_escape(full_clean_snippet or "No clean text snippet available.")}\\end{{quote}}

\\subsection{{Amendment example page}}
\\begin{{figure}}[H]
\\centering
{f"\\includegraphics[width=0.78\\textwidth]{{figures/{_tex_escape(amend_page_copy.name)}}}" if amend_page_copy else "\\fbox{No amendment example image available}"}
\\caption{{Example source page classified into the amendment stream.}}
\\fignote{{Issue: \\texttt{{{_tex_escape(amend_issue_id)}}}. This row is processed through the same extraction and classification path as full-ordinance rows.}}
\\end{{figure}}
\\begin{{quote}}\\small\\textbf{{Example evidence quote:}} {_tex_escape(amend_evidence_quote or "No evidence quote available.")}\\end{{quote}}
\\begin{{quote}}\\small\\textbf{{Extracted clean legal text snippet:}} {_tex_escape(amend_clean_snippet or "No clean text snippet available.")}\\end{{quote}}

\\section{{Issue ontology and event structure}}
Issue-level ontology labels are assigned after clean legal text extraction.
Those labels are then deduplicated into event clusters so repeated publications are counted once per underlying legal action.

\\begin{{figure}}[H]
\\centering
\\includegraphics[width=0.86\\textwidth]{{figures/ontology_counts.png}}
\\caption{{Issue-level ontology class counts.}}
\\fignote{{Bars count issue rows after extraction and issue-level classification. This view is pre-dedup and can include repeated publication of the same legal action.}}
\\end{{figure}}

\\begin{{table}}[H]
\\centering
\\caption{{Issue ontology counts}}
\\resizebox{{0.66\\textwidth}}{{!}}{{\\input{{tables/ontology_counts.tex}}}}
\\end{{table}}
\\fignote{{Counts are issue-level rows. Event-level deduplication happens in the next stage.}}

\\begin{{figure}}[H]
\\centering
\\includegraphics[width=0.62\\textwidth]{{figures/dedup_funnel.png}}
\\caption{{From issue-level legal rows to deduplicated legal events.}}
\\fignote{{The left bar is all issue rows labeled full/amend. The right bar counts unique event identifiers after clustering repeated publications within newspaper timelines.}}
\\end{{figure}}

\\begin{{table}}[H]
\\centering
\\caption{{Concrete dedup cluster example}}
\\resizebox{{\\textwidth}}{{!}}{{\\input{{tables/dedup_example_cluster.tex}}}}
\\end{{table}}
\\fignote{{This example cluster is event \\texttt{{{_tex_escape(dedup_header.get("ordinance_event_id", ""))}}} in {_tex_escape(dedup_header.get("newspaper_slug", ""))}. It is typed as {_tex_escape(dedup_header.get("event_type", ""))} and groups {_tex_escape(dedup_header.get("issue_count", 0))} issue publications into one event.}}

\\begin{{figure}}[H]
\\centering
{f"\\includegraphics[width=0.90\\textwidth]{{figures/{_tex_escape(dedup_year_fig.name)}}}" if dedup_year_fig.is_file() else "\\fbox{No deduplicated yearly figure available}"}
\\caption{{Full-ordinance and amendment counts over time after deduplication.}}
\\fignote{{Each yearly point counts unique event identifiers by first publication year, not raw issue rows.}}
\\end{{figure}}

\\section{{Longitudinal timeline extraction}}
After event structuring, each newspaper timeline is evaluated as one chronological panel.
The timeline extractor returns first-signal timing, first-full timing, phase/wave summaries, category trajectories, complexity direction, and numeric trajectories.

\\begin{{table}}[H]
\\centering
\\caption{{Timeline-level first signal and first full ordinance}}
\\resizebox{{\\textwidth}}{{!}}{{\\input{{tables/first_timing.tex}}}}
\\end{{table}}
\\fignote{{Each row is one timeline output. These are timeline-level fields, not issue-level counts.}}

\\begin{{table}}[H]
\\centering
\\caption{{Timeline-level category means (early, mid, late)}}
\\resizebox{{\\textwidth}}{{!}}{{\\input{{tables/category_means.tex}}}}
\\end{{table}}
\\fignote{{For each newspaper, the issue sequence is partitioned into early/mid/late thirds. Stage shares are then averaged across newspapers.}}

\\begin{{figure}}[H]
\\centering
{f"\\includegraphics[width=0.94\\textwidth]{{figures/{_tex_escape(category_fig.name)}}}" if category_fig.is_file() else "\\fbox{No category trajectory figure available}"}
\\caption{{Category trajectory profile from timeline extraction.}}
\\fignote{{Lines are mean category shares by stage. Values reflect composition shifts over timeline thirds.}}
\\end{{figure}}

\\begin{{table}}[H]
\\centering
\\caption{{Complexity-direction counts by metric}}
\\resizebox{{0.72\\textwidth}}{{!}}{{\\input{{tables/complexity_counts.tex}}}}
\\end{{table}}
\\fignote{{Each metric receives one direction label per newspaper timeline. Counts summarize how often each direction appears across timelines.}}

\\begin{{figure}}[H]
\\centering
{f"\\includegraphics[width=0.82\\textwidth]{{figures/{_tex_escape(complexity_fig.name)}}}" if complexity_fig.is_file() else "\\fbox{No complexity figure available}"}
\\caption{{Complexity-direction heatmap from timeline extraction.}}
\\fignote{{Rows are complexity metrics and columns are direction classes. Cell values are shares of timelines in each direction.}}
\\end{{figure}}

\\begin{{quote}}\\small
\\textbf{{Timeline narrative example ({_tex_escape(timeline_example_slug)})}}: {_tex_escape(timeline_example_text or "No timeline narrative available.")}
\\end{{quote}}
\\begin{{quote}}\\small
\\textbf{{First-signal example field}}: {_tex_escape(json.dumps(timeline_first_signal, ensure_ascii=True))}
\\end{{quote}}
\\begin{{quote}}\\small
\\textbf{{First-full example field}}: {_tex_escape(json.dumps(timeline_first_full, ensure_ascii=True))}
\\end{{quote}}

\\section{{Organized outputs for downstream use}}
The final export packages results into an organized legal corpus so downstream work can consume full ordinances and amendments directly without re-running extraction logic.

\\begin{{table}}[H]
\\centering
\\caption{{Organized output inventory}}
\\resizebox{{0.62\\textwidth}}{{!}}{{\\input{{tables/organized_summary.tex}}}}
\\end{{table}}
\\fignote{{These counts are taken from the final organized export inventory and represent packaged outputs ready for analysis.}}

\\begin{{table}}[H]
\\centering
\\caption{{Deduplicated full/amend event counts by newspaper}}
\\resizebox{{0.78\\textwidth}}{{!}}{{\\input{{tables/events_by_paper.tex}}}}
\\end{{table}}
\\fignote{{Counts are unique event identifiers in the exported full-ordinance and amendment bundles.}}

\\end{{document}}
"""
    _write_text(output_dir / "report.tex", tex.strip() + "\n")

    # Copy reusable run figures into report output for self-contained paths.
    for src in [category_fig, complexity_fig, dedup_year_fig]:
        if src.is_file():
            dst = fig_dir / src.name
            if dst.resolve() != src.resolve():
                dst.write_bytes(src.read_bytes())

    provenance = {
        "generated_at": dt.datetime.now().isoformat(),
        "run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "inputs": {
            "page_tags": str(page_tags_path),
            "anchor_decisions": str(anchor_decisions_path),
            "clean_text": str(clean_text_path),
            "issue_labels": str(issue_labels_path),
            "issue_event_assignments": str(assign_path),
            "event_clusters": str(clusters_path),
            "city_longitudinal_summary": str(city_summary_path),
            "city_category_trajectory": str(city_cat_path),
            "city_complexity_trajectory": str(city_complexity_path),
            "panel_outputs": str(panel_outputs_path),
            "organized_export_inventory": str(export_inventory_path),
        },
        "selected_examples": {
            "full_issue_id": full_issue_id,
            "full_page_png": str(full_page_copy) if full_page_copy else "",
            "amend_issue_id": amend_issue_id,
            "amend_page_png": str(amend_page_copy) if amend_page_copy else "",
            "dedup_example": dedup_header,
            "timeline_example_slug": timeline_example_slug,
        },
    }
    _write_text(output_dir / "provenance.json", json.dumps(provenance, indent=2))

    compiled = False
    if compile_pdf:
        compiled = _compile_pdf(output_dir)

    return {
        "compiled_pdf": compiled,
        "output_dir": str(output_dir),
        "report_tex": str(output_dir / "report.tex"),
        "report_pdf": str(output_dir / "report.pdf"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a system architecture LaTeX report for the fullnewspaper pipeline.")
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Run directory containing page_tagging/, cleaning/, classification/, events/, longitudinal/, organized_corpus_v3/.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for report artifacts. Default: <run-dir>/reports/system_architecture",
    )
    parser.add_argument(
        "--no-compile-pdf",
        action="store_true",
        help="Write report.tex and assets but skip LaTeX compilation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    output_dir = args.output_dir.resolve() if args.output_dir else (run_dir / "reports" / "system_architecture")
    result = build_report(run_dir, output_dir, compile_pdf=not args.no_compile_pdf)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
