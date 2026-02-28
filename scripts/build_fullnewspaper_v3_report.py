#!/usr/bin/env python3
"""
Build an expanded self-contained LaTeX report for full-newspaper v3 outputs.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch


LAW_FULL_CLASSES = {"full_code_comprehensive", "full_code_recodification", "full_code_partial_enacted"}
LAW_AMEND_CLASSES = {"amendment_substantial", "amendment_targeted"}


def _draw_node(ax, x: float, y: float, w: float, h: float, text: str, face: str, edge: str = "#444444", fs: int = 9) -> None:
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.01,rounding_size=0.012",
        linewidth=1.2,
        edgecolor=edge,
        facecolor=face,
    )
    ax.add_patch(box)
    ax.text(x + w / 2.0, y + h / 2.0, text, ha="center", va="center", fontsize=fs)


def _draw_arrow(ax, x1: float, y1: float, x2: float, y2: float, color: str = "#555555") -> None:
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", lw=1.4, color=color))


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


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _to_latex_table(df: pd.DataFrame, out_path: Path, index: bool = False, float_format: str = "{:.3f}") -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if df.empty:
        out_path.write_text("\\begin{tabular}{l}No data\\\\\\end{tabular}\n", encoding="utf-8")
        return
    tex = df.to_latex(index=index, escape=True, na_rep="", float_format=lambda x: float_format.format(x))
    out_path.write_text(tex, encoding="utf-8")


def _slug_label(slug: str) -> str:
    return _norm(slug).replace("-", " ").title()


def _tex_escape(s: Any) -> str:
    t = _norm(s)
    return (
        t.replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("$", "\\$")
        .replace("#", "\\#")
        .replace("_", "\\_")
        .replace("{", "\\{")
        .replace("}", "\\}")
    )


def _plot_full_vs_amend(
    by_year: pd.DataFrame,
    out: Path,
    *,
    title: str = "Full ordinances vs amendments over time",
    ylabel: str = "Issue-event rows",
) -> None:
    plt.style.use("ggplot")
    fig, ax = plt.subplots(figsize=(10.2, 5.2))
    if by_year.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
    else:
        d = by_year.sort_values("year").copy()
        ax.plot(d["year"], d["full_ordinance"], marker="o", label="Full ordinance")
        ax.plot(d["year"], d["amendment"], marker="o", label="Amendment")
        ax.set_xlabel("Year")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220)
    plt.close(fig)


def _build_dedup_full_amend_by_year(event_clusters: pd.DataFrame) -> pd.DataFrame:
    if event_clusters.empty:
        return pd.DataFrame(columns=["year", "full_ordinance", "amendment"])
    req = {"event_type", "ordinance_event_id", "first_issue_date"}
    if not req.issubset(event_clusters.columns):
        return pd.DataFrame(columns=["year", "full_ordinance", "amendment"])
    d = event_clusters.copy()
    d = d[d["event_type"].astype(str).isin(["full_ordinance", "amendment"])].copy()
    if d.empty:
        return pd.DataFrame(columns=["year", "full_ordinance", "amendment"])
    d["year"] = pd.to_datetime(d["first_issue_date"], errors="coerce").dt.year
    d["event_key"] = d["ordinance_event_id"].astype(str)
    if "newspaper_slug" in d.columns:
        d["event_key"] = d["newspaper_slug"].astype(str) + "::" + d["ordinance_event_id"].astype(str)
    d = d[d["year"].notna()].copy()
    if d.empty:
        return pd.DataFrame(columns=["year", "full_ordinance", "amendment"])
    by = (
        d.groupby(["year", "event_type"], dropna=False)["event_key"]
        .nunique()
        .reset_index(name="event_count")
    )
    p = by.pivot(index="year", columns="event_type", values="event_count").fillna(0).reset_index()
    for c in ["full_ordinance", "amendment"]:
        if c not in p.columns:
            p[c] = 0
    p["year"] = p["year"].astype(int)
    p["full_ordinance"] = p["full_ordinance"].astype(int)
    p["amendment"] = p["amendment"].astype(int)
    return p.sort_values("year").reset_index(drop=True)


def _plot_issue_volume_by_newspaper(issue_labels: pd.DataFrame, out: Path) -> None:
    plt.style.use("ggplot")
    fig, ax = plt.subplots(figsize=(9.6, 4.8))
    if issue_labels.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
    else:
        d = (
            issue_labels.groupby("newspaper_slug", dropna=False)["issue_id"]
            .nunique()
            .reset_index(name="issue_count")
            .sort_values("issue_count", ascending=False)
        )
        ax.bar(d["newspaper_slug"], d["issue_count"], color="#2c7fb8")
        ax.set_xlabel("Newspaper")
        ax.set_ylabel("Issue count")
        ax.set_title("Issue volume by newspaper")
        ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220)
    plt.close(fig)


def _plot_issue_class_stacked(issue_labels: pd.DataFrame, out: Path) -> None:
    plt.style.use("ggplot")
    fig, ax = plt.subplots(figsize=(11.2, 5.2))
    if issue_labels.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
    else:
        p = pd.pivot_table(
            issue_labels,
            index="newspaper_slug",
            columns="issue_class",
            values="issue_id",
            aggfunc="count",
            fill_value=0,
        )
        p = p.sort_index()
        bottom = np.zeros(len(p))
        for c in p.columns:
            ax.bar(p.index, p[c].values, bottom=bottom, label=c)
            bottom = bottom + p[c].values
        ax.set_ylabel("Issue count")
        ax.set_xlabel("Newspaper")
        ax.set_title("Issue-class composition by newspaper")
        ax.tick_params(axis="x", rotation=25)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.23), ncol=3, fontsize=8)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220)
    plt.close(fig)


def _plot_issue_classes(issue_class_summary: pd.DataFrame, out: Path) -> None:
    plt.style.use("ggplot")
    fig, ax = plt.subplots(figsize=(10, 5))
    if issue_class_summary.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
    else:
        d = issue_class_summary.copy()
        d = d.groupby("issue_class", dropna=False)["issue_count"].sum().reset_index().sort_values("issue_count", ascending=False)
        ax.barh(d["issue_class"], d["issue_count"], color="#4daf4a")
        ax.invert_yaxis()
        ax.set_xlabel("Issue count")
        ax.set_title("Issue class distribution")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220)
    plt.close(fig)


def _plot_event_type_stacked(events: pd.DataFrame, out: Path) -> None:
    plt.style.use("ggplot")
    fig, ax = plt.subplots(figsize=(11.2, 5.2))
    if events.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
    else:
        p = pd.pivot_table(
            events,
            index="newspaper_slug",
            columns="event_type",
            values="ordinance_event_id",
            aggfunc="nunique",
            fill_value=0,
        )
        p = p.sort_index()
        bottom = np.zeros(len(p))
        for c in p.columns:
            ax.bar(p.index, p[c].values, bottom=bottom, label=c)
            bottom = bottom + p[c].values
        ax.set_ylabel("Deduped event count")
        ax.set_xlabel("Newspaper")
        ax.set_title("Deduplicated event types by newspaper")
        ax.tick_params(axis="x", rotation=25)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=4, fontsize=8)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220)
    plt.close(fig)


def _plot_full_amend_by_newspaper(assign: pd.DataFrame, out: Path) -> None:
    plt.style.use("ggplot")
    fig, ax = plt.subplots(figsize=(10.4, 5.0))
    if assign.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
    else:
        d = assign[assign["event_type"].isin(["full_ordinance", "amendment"])].copy()
        p = (
            d.groupby(["newspaper_slug", "event_type"], dropna=False)["issue_id"]
            .nunique()
            .reset_index(name="issue_event_rows")
        )
        if p.empty:
            ax.text(0.5, 0.5, "No full/amendment rows", ha="center", va="center")
        else:
            cats = sorted(p["newspaper_slug"].unique().tolist())
            x = np.arange(len(cats))
            w = 0.38
            full = np.array([int(p[(p["newspaper_slug"] == c) & (p["event_type"] == "full_ordinance")]["issue_event_rows"].sum()) for c in cats])
            amend = np.array([int(p[(p["newspaper_slug"] == c) & (p["event_type"] == "amendment")]["issue_event_rows"].sum()) for c in cats])
            ax.bar(x - w / 2, full, width=w, label="Full ordinance rows")
            ax.bar(x + w / 2, amend, width=w, label="Amendment rows")
            ax.set_xticks(x)
            ax.set_xticklabels(cats, rotation=25)
            ax.set_ylabel("Issue-event rows")
            ax.set_xlabel("Newspaper")
            ax.set_title("Full vs amendment rows by newspaper")
            ax.legend()
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220)
    plt.close(fig)


def _plot_dedup_impact(issue_labels: pd.DataFrame, events: pd.DataFrame, out: Path) -> None:
    plt.style.use("ggplot")
    fig, ax = plt.subplots(figsize=(11.5, 5.2))
    if issue_labels.empty or events.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
    else:
        slugs = sorted(issue_labels["newspaper_slug"].dropna().unique().tolist())
        raw_full = issue_labels[issue_labels["issue_class"].isin(LAW_FULL_CLASSES)].groupby("newspaper_slug")["issue_id"].nunique()
        dedup_full = events[events["event_type"] == "full_ordinance"].groupby("newspaper_slug")["ordinance_event_id"].nunique()
        raw_amend = issue_labels[issue_labels["issue_class"].isin(LAW_AMEND_CLASSES)].groupby("newspaper_slug")["issue_id"].nunique()
        dedup_amend = events[events["event_type"] == "amendment"].groupby("newspaper_slug")["ordinance_event_id"].nunique()

        x = np.arange(len(slugs))
        w = 0.2
        ax.bar(x - 1.5 * w, [int(raw_full.get(s, 0)) for s in slugs], width=w, label="Raw full rows")
        ax.bar(x - 0.5 * w, [int(dedup_full.get(s, 0)) for s in slugs], width=w, label="Dedup full events")
        ax.bar(x + 0.5 * w, [int(raw_amend.get(s, 0)) for s in slugs], width=w, label="Raw amendment rows")
        ax.bar(x + 1.5 * w, [int(dedup_amend.get(s, 0)) for s in slugs], width=w, label="Dedup amendment events")
        ax.set_xticks(x)
        ax.set_xticklabels(slugs, rotation=25)
        ax.set_ylabel("Count")
        ax.set_xlabel("Newspaper")
        ax.set_title("Deduplication impact: raw issue rows vs deduped events")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, fontsize=9)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220)
    plt.close(fig)


def _plot_category_trajectory(cat: pd.DataFrame, out: Path) -> None:
    plt.style.use("ggplot")
    fig, ax = plt.subplots(figsize=(12.0, 5.2))
    if cat.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
    else:
        d = cat.copy().sort_values("late", ascending=False)
        x = np.arange(len(d))
        ax.plot(x, d["early"] * 100.0, marker="o", label="Early")
        ax.plot(x, d["mid"] * 100.0, marker="o", label="Mid")
        ax.plot(x, d["late"] * 100.0, marker="o", label="Late")
        ax.set_xticks(x)
        ax.set_xticklabels(d["category"], rotation=35, ha="right")
        ax.set_ylabel("Mean share (%)")
        ax.set_title("Category trajectory means (early/mid/late)")
        ax.legend()
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220)
    plt.close(fig)


def _plot_complexity_heatmap(cpx: pd.DataFrame, out: Path) -> None:
    plt.style.use("ggplot")
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    if cpx.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
    else:
        wanted = ["up", "flat", "down", "mixed", "uncertain"]
        metrics = sorted(cpx["metric"].astype(str).unique().tolist())
        mat = np.zeros((len(metrics), len(wanted)))
        for i, m in enumerate(metrics):
            sub = cpx[cpx["metric"].astype(str) == m]
            total = float(sub["count"].sum())
            for j, w in enumerate(wanted):
                cnt = float(sub.loc[sub["direction"].astype(str) == w, "count"].sum())
                mat[i, j] = (100.0 * cnt / total) if total > 0 else 0.0
        im = ax.imshow(mat, aspect="auto", cmap="YlGnBu", vmin=0, vmax=100)
        ax.set_yticks(np.arange(len(metrics)))
        ax.set_yticklabels(metrics)
        ax.set_xticks(np.arange(len(wanted)))
        ax.set_xticklabels(wanted)
        ax.set_title("Complexity trend direction by metric (% of newspaper panels)")
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(j, i, f"{mat[i, j]:.0f}%", ha="center", va="center", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="%")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220)
    plt.close(fig)


def _plot_phase_label_counts(phase_df: pd.DataFrame, out: Path) -> None:
    plt.style.use("ggplot")
    fig, ax = plt.subplots(figsize=(9.6, 4.8))
    if phase_df.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
    else:
        d = phase_df.groupby("phase_label", dropna=False).size().reset_index(name="count").sort_values("count", ascending=False)
        ax.barh(d["phase_label"], d["count"], color="#3b8bc2")
        ax.invert_yaxis()
        ax.set_xlabel("Phase row count")
        ax.set_title("Evolution phase label counts")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220)
    plt.close(fig)


def _plot_wave_type_counts(wave_df: pd.DataFrame, out: Path) -> None:
    plt.style.use("ggplot")
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    if wave_df.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
    else:
        d = wave_df.groupby("change_type", dropna=False).size().reset_index(name="count").sort_values("count", ascending=False)
        ax.bar(d["change_type"], d["count"], color="#7cb342")
        ax.set_xlabel("Amendment wave type")
        ax.set_ylabel("Wave row count")
        ax.set_title("Amendment wave type counts")
        ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220)
    plt.close(fig)


def _plot_numeric_direction_counts(num_df: pd.DataFrame, out: Path) -> None:
    plt.style.use("ggplot")
    fig, ax = plt.subplots(figsize=(10.8, 5.0))
    if num_df.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
    else:
        p = pd.pivot_table(num_df, index="metric", columns="direction", values="newspaper_slug", aggfunc="count", fill_value=0)
        p = p.sort_index()
        bottom = np.zeros(len(p))
        for c in p.columns:
            ax.bar(p.index, p[c].values, bottom=bottom, label=c)
            bottom = bottom + p[c].values
        ax.set_ylabel("Row count")
        ax.set_xlabel("Numeric metric")
        ax.set_title("Numeric trajectory directions")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=4, fontsize=8)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220)
    plt.close(fig)


def _plot_first_signal_lag(first_events: pd.DataFrame, out: Path) -> None:
    plt.style.use("ggplot")
    fig, ax = plt.subplots(figsize=(10.0, 4.8))
    if first_events.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
    else:
        d = first_events.copy()
        d["first_signal_date"] = pd.to_datetime(d["first_signal_date"], errors="coerce")
        d["first_full_date"] = pd.to_datetime(d["first_full_date"], errors="coerce")
        d = d[d["first_full_status"].astype(str) == "found"].copy()
        d["lag_years"] = (d["first_full_date"] - d["first_signal_date"]).dt.days / 365.25
        if d.empty:
            ax.text(0.5, 0.5, "No found first-full rows", ha="center", va="center")
        else:
            d = d.sort_values("lag_years", ascending=False)
            ax.bar(d["newspaper_slug"], d["lag_years"], color="#ef6c00")
            ax.set_ylabel("Years from first signal to first full")
            ax.set_xlabel("Newspaper")
            ax.set_title("Time from first zoning signal to first full ordinance")
            ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220)
    plt.close(fig)


def _plot_pipeline_overview(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(14.2, 7.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    data_face = "#d9ecff"
    clean_face = "#dff5df"
    llm_face = "#e8f7e8"
    out_face = "#ffeac9"
    # Row 1: consolidated cleaning flow from raw newspaper to clean zoning text.
    w, h = 0.26, 0.12
    x_clean = [0.03, 0.37, 0.71]
    y_clean = 0.72
    clean_labels = [
        "Newspaper ingestion\n(all issues,\ntranscripts + page ids/images)",
        "LLM legal-text extraction\n(page context + anchor selection\n+ recall/adjudication)",
        "Clean zoning text\n(verbatim law language,\nkept in original order)",
    ]
    clean_faces = [data_face, clean_face, out_face]
    for x, label, face in zip(x_clean, clean_labels, clean_faces):
        _draw_node(ax, x, y_clean, w, h, label, face)
    for i in range(len(x_clean) - 1):
        _draw_arrow(ax, x_clean[i] + w, y_clean + h / 2, x_clean[i + 1], y_clean + h / 2)

    # Row 2: downstream interpretation + final outputs (collapsed to one row).
    w2, h2 = 0.19, 0.11
    y_mid = 0.42
    x_mid = [0.02, 0.26, 0.50, 0.74]
    mid_labels = [
        "Issue ontology\n(LLM)\nfull ordinance /\namendment / other",
        "Event deduplication\n(LLM)\ncluster repeat\npublications",
        "Panel longitudinal\n(LLM)\nfirst/full timing +\ncontent evolution",
        "Descriptive report\n(tables, figures,\nPNG evidence appendix)",
    ]
    mid_faces = [llm_face, llm_face, llm_face, out_face]
    for x, label, face in zip(x_mid, mid_labels, mid_faces):
        _draw_node(ax, x, y_mid, w2, h2, label, face)
    _draw_arrow(ax, x_clean[-1] + w / 2, y_clean, x_mid[0] + 0.10, y_mid + h2)
    _draw_arrow(ax, x_mid[0] + w2, y_mid + h2 / 2, x_mid[1], y_mid + h2 / 2)
    _draw_arrow(ax, x_mid[1] + w2, y_mid + h2 / 2, x_mid[2], y_mid + h2 / 2)
    _draw_arrow(ax, x_mid[2] + w2, y_mid + h2 / 2, x_mid[3], y_mid + h2 / 2)

    ax.text(0.01, 0.95, "Pipeline overview: from newspaper pages to clean zoning timeline", fontsize=14, fontweight="bold", ha="left")
    ax.text(
        0.01,
        0.05,
        "Blue=input sources, green=cleaning stages, light-green=LLM interpretation stages, orange=final reporting outputs.",
        fontsize=10,
        ha="left",
    )
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220)
    plt.close(fig)


def _plot_pipeline_artifact_lineage(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(13.2, 7.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    x_in, x_mid, x_out = 0.05, 0.38, 0.72
    w, h = 0.24, 0.10

    in_nodes = [
        (0.78, "Issue transcripts\npanels/*/issue_texts.jsonl"),
        (0.63, "Raw page images\n(unique PNG sources)"),
        (0.48, "Run configuration\n(selected newspapers,\nmodel variants)"),
    ]
    mid_nodes = [
        (0.82, "page_tags.jsonl"),
        (0.68, "anchor_decisions.jsonl"),
        (0.54, "clean_zoning_text.jsonl"),
        (0.40, "issue_labels_v3.csv/jsonl"),
        (0.26, "issue_event_assignments_v3.csv\nordinance_event_clusters_v3.csv"),
    ]
    out_nodes = [
        (0.78, "longitudinal/*.csv\n(first/full, phases,\ncategory/complexity/numeric)"),
        (0.58, "analysis/*.csv\n(full-vs-amend, stage means,\ncomplexity summaries)"),
        (0.38, "report tables/figures\n+ evidence galleries"),
        (0.18, "final report.pdf"),
    ]

    for y, t in in_nodes:
        _draw_node(ax, x_in, y, w, h, t, "#d9ecff")
    for y, t in mid_nodes:
        _draw_node(ax, x_mid, y, w, h, t, "#dff5df")
    for y, t in out_nodes:
        _draw_node(ax, x_out, y, w, h, t, "#ffeac9")

    # Connections input -> mid
    _draw_arrow(ax, x_in + w, 0.83, x_mid, 0.87)
    _draw_arrow(ax, x_in + w, 0.83, x_mid, 0.73)
    _draw_arrow(ax, x_in + w, 0.68, x_mid, 0.73)
    _draw_arrow(ax, x_in + w, 0.68, x_mid, 0.59)
    _draw_arrow(ax, x_in + w, 0.53, x_mid, 0.45)

    # Mid chain
    _draw_arrow(ax, x_mid + w, 0.87, x_out, 0.83)
    _draw_arrow(ax, x_mid + w, 0.73, x_out, 0.83)
    _draw_arrow(ax, x_mid + w, 0.59, x_out, 0.63)
    _draw_arrow(ax, x_mid + w, 0.45, x_out, 0.63)
    _draw_arrow(ax, x_mid + w, 0.31, x_out, 0.43)
    _draw_arrow(ax, x_out + w / 2, 0.38, x_out + w / 2, 0.28)

    ax.text(0.01, 0.95, "Artifact lineage map (what feeds what)", fontsize=14, fontweight="bold", ha="left")
    ax.text(0.01, 0.04, "Left: source inputs. Middle: pipeline artifacts. Right: analysis/report outputs.", fontsize=10, ha="left")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220)
    plt.close(fig)


def _plot_classification_gate_flow(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(12.8, 6.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    _draw_node(ax, 0.03, 0.58, 0.16, 0.12, "Issue text\n(whole issue)", "#d9ecff")
    _draw_node(ax, 0.25, 0.58, 0.20, 0.12, "Contains verbatim\nzoning law text?\n(LLM gate 1)", "#dff5df")
    _draw_node(ax, 0.51, 0.58, 0.20, 0.12, "Law text sufficiency\nfor legal-event\nclassification?\n(LLM gate 2)", "#dff5df")
    _draw_node(ax, 0.77, 0.72, 0.20, 0.12, "Full classes:\nfull_code_comprehensive\nfull_code_recodification", "#ffeac9")
    _draw_node(ax, 0.77, 0.52, 0.20, 0.12, "Amendment classes:\namendment_substantial\namendment_targeted", "#ffeac9")
    _draw_node(ax, 0.77, 0.32, 0.20, 0.12, "Other law class:\nzoning_notice_procedural", "#ffeac9")
    _draw_node(ax, 0.51, 0.30, 0.20, 0.12, "law_fragment /\nnot_verbatim_law_text\n(non-event bucket)", "#f8dbe6")
    _draw_node(ax, 0.25, 0.30, 0.20, 0.12, "non_zoning\nor not_verbatim\n(non-event)", "#f8dbe6")

    _draw_arrow(ax, 0.19, 0.64, 0.25, 0.64)
    _draw_arrow(ax, 0.45, 0.64, 0.51, 0.64)
    _draw_arrow(ax, 0.71, 0.66, 0.77, 0.78)
    _draw_arrow(ax, 0.71, 0.64, 0.77, 0.58)
    _draw_arrow(ax, 0.71, 0.62, 0.77, 0.38)
    _draw_arrow(ax, 0.61, 0.58, 0.61, 0.42)
    _draw_arrow(ax, 0.35, 0.58, 0.35, 0.42)

    ax.text(0.01, 0.94, "Classification and gating logic (LLM semantic decisions)", fontsize=14, fontweight="bold", ha="left")
    ax.text(
        0.01,
        0.05,
        "Only issues passing both gates are counted as legal-event classes (full/amend/procedural). Others remain non-event outputs.",
        fontsize=10,
        ha="left",
    )
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220)
    plt.close(fig)


def _load_issue_page_ids_map(run_root: Path) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    source_run = run_root.parent
    panel_dir = source_run / "panels"
    if not panel_dir.is_dir():
        return out
    for city in sorted(panel_dir.iterdir()):
        if not city.is_dir():
            continue
        p = city / "issue_texts.jsonl"
        if not p.is_file():
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
                iid = _norm(obj.get("issue_id"))
                pids = obj.get("page_ids") if isinstance(obj.get("page_ids"), list) else []
                if iid and pids:
                    vals = [_norm(x) for x in pids if _norm(x)]
                    if vals:
                        out[iid] = vals
    return out


def _fetch_first_available_png(page_ids: list[str], fig_dir: Path) -> tuple[str, str, str, str]:
    if not page_ids:
        return "", "", "missing", ""

    for pid in page_ids:
        filename = f"{pid}.png"
        local_png = fig_dir / filename
        fetch_status, source = _try_fetch_image(local_png, filename)
        if fetch_status != "missing":
            return pid, filename, fetch_status, source

    first = page_ids[0]
    return first, f"{first}.png", "missing", ""


def _try_fetch_image(local_png: Path, filename: str) -> tuple[str, str]:
    if local_png.is_file():
        return "existing", str(local_png)

    local_roots = [
        Path("/Users/saulrichardson/projects/newspapers/old-ocr/newspaper-parsing-local/data/unique_png"),
        Path("/Users/saulrichardson/projects/newspapers/old-ocr/newspaper-parsing-local/data/issue_zoning_samples_20260125_223032"),
        Path("/Users/saulrichardson/projects/newspapers/old-ocr/newspaper-parsing-local/data/issue_zoning_samples_diverse_20260126_031245"),
        Path("/Users/saulrichardson/projects/newspapers/old-ocr/newspaper-parsing-local/data/issue_zoning_samples_diverse_fullissue_min2_20260126_032321"),
    ]
    for root in local_roots:
        cand = root / filename
        if cand.is_file():
            shutil.copy2(cand, local_png)
            return "local_copy", str(cand)
    for root in local_roots[1:]:
        try:
            for cand in root.rglob(filename):
                if cand.is_file():
                    shutil.copy2(cand, local_png)
                    return "local_copy", str(cand)
        except Exception:
            continue

    remote_unique = f"/scratch/sxr203/newspaper-downloads/dedupe-webp/unique_png/{filename}"
    try:
        rc = subprocess.run(
            ["scp", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", f"torch:{remote_unique}", str(local_png)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=35,
        )
        if rc.returncode == 0 and local_png.is_file():
            return "torch_unique_png", remote_unique
    except Exception:
        pass

    try:
        find_cmd = f"find /scratch/sxr203/newspaper-downloads -type f -name {json.dumps(filename)} | head -n 1"
        rs = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "torch", find_cmd],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=35,
        )
        rp = rs.stdout.strip().splitlines()[0] if rs.returncode == 0 and rs.stdout.strip() else ""
        if rp:
            rc2 = subprocess.run(
                ["scp", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", f"torch:{rp}", str(local_png)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=35,
            )
            if rc2.returncode == 0 and local_png.is_file():
                return "torch_find_copy", rp
    except Exception:
        pass
    return "missing", ""


def _ensure_evidence_page_index(run_root: Path, out_dir: Path) -> pd.DataFrame:
    val = _safe_read_csv(run_root / "manual_validation" / "validation_log.csv")
    if val.empty:
        return pd.DataFrame()

    issue_pages = _load_issue_page_ids_map(run_root)
    fig_dir = out_dir / "figures" / "evidence_pages"
    fig_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for r in val.to_dict(orient="records"):
        issue_id = _norm(r.get("issue_id"))
        page_candidates = issue_pages.get(issue_id, [])
        page_id, filename, fetch_status, source = _fetch_first_available_png(page_candidates, fig_dir)
        rows.append(
            {
                "newspaper_slug": _norm(r.get("newspaper_slug")),
                "validation_type": _norm(r.get("validation_type")),
                "ordinance_event_id": _norm(r.get("ordinance_event_id")),
                "issue_id": issue_id,
                "issue_date": _norm(r.get("issue_date")),
                "pass_fail": _norm(r.get("pass_fail")),
                "page_id": page_id,
                "filename": filename,
                "figure_relpath": f"figures/evidence_pages/{filename}" if filename else "",
                "image_fetch_status": fetch_status,
                "image_source": source,
            }
        )
    idx = pd.DataFrame.from_records(rows)
    idx.to_csv(out_dir / "tables" / "evidence_page_index.csv", index=False)
    return idx


def _select_evidence_examples(val_log: pd.DataFrame, evidence_idx: pd.DataFrame) -> pd.DataFrame:
    if val_log.empty or evidence_idx.empty:
        return pd.DataFrame()
    merged = val_log.merge(
        evidence_idx[
            [
                "newspaper_slug",
                "validation_type",
                "issue_id",
                "page_id",
                "filename",
                "figure_relpath",
                "image_fetch_status",
            ]
        ],
        on=["newspaper_slug", "validation_type", "issue_id"],
        how="left",
    )
    merged = merged[merged["image_fetch_status"].astype(str) != "missing"].copy()
    if merged.empty:
        return merged

    out_rows: list[dict[str, Any]] = []
    for slug, g in merged.groupby("newspaper_slug", dropna=False):
        g = g.sort_values("issue_date")
        gf = g[g["validation_type"].astype(str) == "full_ordinance_event"].copy()
        gn = g[g["validation_type"].astype(str) == "narrative_exclusion"].copy()

        row_full = pd.DataFrame()
        if not gf.empty:
            fails = gf[gf["pass_fail"].astype(str).str.lower() == "fail"]
            row_full = fails.head(1) if not fails.empty else gf.head(1)
        row_narr = gn.head(1) if not gn.empty else pd.DataFrame()

        for pick in [row_full, row_narr]:
            if pick.empty:
                continue
            out_rows.extend(pick.to_dict(orient="records"))

    ex = pd.DataFrame.from_records(out_rows).reset_index(drop=True)
    if not ex.empty:
        ex["example_id"] = [f"EX{i + 1:02d}" for i in range(len(ex))]
    return ex


def _sample_remaining_issue_rows(issue_labels: pd.DataFrame, per_class: int = 4) -> pd.DataFrame:
    if issue_labels.empty:
        return pd.DataFrame()
    remaining = issue_labels[
        ~issue_labels["issue_class"].isin(sorted(LAW_FULL_CLASSES | {"amendment_substantial"}))
    ].copy()
    if remaining.empty:
        return remaining
    remaining = remaining.sort_values(["issue_class", "newspaper_slug", "issue_date", "issue_id"]).reset_index(drop=True)

    picks: list[pd.DataFrame] = []
    for _, g in remaining.groupby("issue_class", dropna=False):
        g = g.reset_index(drop=True)
        take = min(per_class, len(g))
        if take <= 0:
            continue
        if len(g) <= take:
            picks.append(g)
            continue
        idx = np.linspace(0, len(g) - 1, num=take, dtype=int)
        picks.append(g.iloc[idx].copy())
    if not picks:
        return pd.DataFrame()
    return pd.concat(picks, ignore_index=True)


def _build_appendix_png_index(run_root: Path, out_dir: Path, issue_labels: pd.DataFrame) -> pd.DataFrame:
    if issue_labels.empty:
        return pd.DataFrame()

    cls = issue_labels.copy()
    keep_cols = [c for c in ["newspaper_slug", "city_key", "issue_id", "issue_date", "issue_class"] if c in cls.columns]
    cls = cls[keep_cols].copy()

    full_rows = cls[cls["issue_class"].isin(sorted(LAW_FULL_CLASSES))].copy()
    full_rows["appendix_group"] = "full_all"

    sub_rows = cls[cls["issue_class"].astype(str) == "amendment_substantial"].copy()
    sub_rows["appendix_group"] = "amendment_substantial_all"

    rem_rows = _sample_remaining_issue_rows(cls, per_class=4)
    if not rem_rows.empty:
        rem_rows["appendix_group"] = "remaining_sample"

    selected = pd.concat([full_rows, sub_rows, rem_rows], ignore_index=True)
    if selected.empty:
        return pd.DataFrame()

    selected = selected.sort_values(["appendix_group", "newspaper_slug", "issue_date", "issue_id"]).drop_duplicates(
        subset=["appendix_group", "issue_id"]
    )

    issue_pages = _load_issue_page_ids_map(run_root)
    fig_dir = out_dir / "figures" / "appendix_pages"
    fig_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for r in selected.to_dict(orient="records"):
        issue_id = _norm(r.get("issue_id"))
        page_candidates = issue_pages.get(issue_id, [])
        page_id, filename, fetch_status, source = _fetch_first_available_png(page_candidates, fig_dir)
        rows.append(
            {
                "appendix_group": _norm(r.get("appendix_group")),
                "newspaper_slug": _norm(r.get("newspaper_slug")),
                "city_key": _norm(r.get("city_key")),
                "issue_id": issue_id,
                "issue_date": _norm(r.get("issue_date")),
                "issue_class": _norm(r.get("issue_class")),
                "page_id": page_id,
                "filename": filename,
                "figure_relpath": f"figures/appendix_pages/{filename}" if filename else "",
                "image_fetch_status": fetch_status,
                "image_source": source,
            }
        )

    idx = pd.DataFrame.from_records(rows)
    idx.to_csv(out_dir / "tables" / "appendix_classified_png_index.csv", index=False)
    return idx


def _build_appendix_gallery_block(df: pd.DataFrame, section_label: str) -> str:
    if df.empty:
        return "No pages available for this section."
    lines: list[str] = []
    dd = df[df["image_fetch_status"].astype(str) != "missing"].copy()
    if dd.empty:
        return "No fetchable PNG pages available for this section."
    dd = dd.sort_values(["newspaper_slug", "issue_date", "issue_id"]).reset_index(drop=True)

    # Per-page review annotations for known edge cases surfaced during manual review.
    page_review_notes = {
        "bountiful-davis-county-clipper-nov-17-1961-p-7": (
            "Partial failure: this appears to be a proposed ordinance publication, with no clear evidence "
            "on this page that the code was already in effect. Earlier classification gates may not have "
            "fully captured proposal-versus-enacted status."
        )
    }

    for i, r in dd.iterrows():
        fig_rel = _norm(r.get("figure_relpath"))
        if not fig_rel:
            continue
        slug = _slug_label(_norm(r.get("newspaper_slug")))
        issue_class = _norm(r.get("issue_class"))
        issue_date = _norm(r.get("issue_date"))
        issue_id = _norm(r.get("issue_id"))
        page_id = _norm(r.get("page_id"))
        review_note = page_review_notes.get(page_id, "")
        caption = (
            f"{section_label}: {slug} | {issue_class} | {issue_date}. "
            "Produced by taking the classified issue/page output and rendering the linked newspaper page image."
        )

        lines.extend(
            [
                "\\clearpage" if i > 0 else "",
                "\\begin{figure}[H]",
                "\\centering",
                f"\\includegraphics[width=0.92\\textwidth,height=0.62\\textheight,keepaspectratio]{{{_tex_escape(fig_rel)}}}",
                f"\\caption{{{_tex_escape(caption)}}}",
                "\\vspace{0.25em}",
                "\\begin{minipage}{0.94\\textwidth}",
                "\\footnotesize",
                f"\\textit{{Issue/page:}} {_tex_escape(issue_id)}; {_tex_escape(page_id)}.\\\\",
                f"\\textit{{Class:}} {_tex_escape(issue_class)}."
                + (
                    "\\\\\n\\textit{Review note:} " + _tex_escape(review_note)
                    if review_note
                    else ""
                ),
                "\\end{minipage}",
                "\\end{figure}",
                "",
            ]
        )
    return "\n".join(lines).strip()


def _build_appendix_sampled_by_class_block(df: pd.DataFrame) -> str:
    if df.empty:
        return "No sampled pages available for this section."
    dd = df[df["image_fetch_status"].astype(str) != "missing"].copy()
    if dd.empty:
        return "No fetchable PNG pages available for this section."

    dd["issue_class_norm"] = dd["issue_class"].astype(str).map(_norm)
    dd = dd.sort_values(["issue_class_norm", "newspaper_slug", "issue_date", "issue_id"]).reset_index(drop=True)

    blocks: list[str] = []
    for cls in sorted([c for c in dd["issue_class_norm"].unique().tolist() if c]):
        cls_df = dd[dd["issue_class_norm"] == cls].copy()
        cls_n = int(len(cls_df))
        cls_label = cls.replace("_", " ")
        gallery = _build_appendix_gallery_block(cls_df, f"Classified as {cls_label} (sampled display)")
        blocks.extend(
            [
                f"\\subsubsection{{Classified as {_tex_escape(cls_label)}}}",
                "Display choice: this class is shown as a sample in the appendix for readability.",
                f"Pages shown: {cls_n}.",
                gallery,
                "",
            ]
        )
    return "\n".join(blocks).strip()


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build fullnewspaper v3 report.")
    ap.add_argument(
        "--run-root",
        default="/Users/saulrichardson/projects/newspapers/newspaper-analysis/reports/runs/pi_v1_run60_fullnewspaper/fullnewspaper_v3_run_latest",
        help="Root directory produced by run_fullnewspaper_v3_pipeline.py",
    )
    ap.add_argument("--output-dir", default="", help="Report output dir (default: <run-root>/reports/fullnewspaper_v3)")
    ap.add_argument("--compile-pdf", action=argparse.BooleanOptionalAction, default=True)
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    run_root = Path(args.run_root).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve() if _norm(args.output_dir) else run_root / "reports" / "fullnewspaper_v3"
    fig_dir = out_dir / "figures"
    tbl_dir = out_dir / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    tbl_dir.mkdir(parents=True, exist_ok=True)

    analysis_dir = run_root / "analysis"
    cls_dir = run_root / "classification"
    long_dir = run_root / "longitudinal"
    calibration_dir = run_root / "calibration"
    validation_dir = run_root / "manual_validation"
    events_status_dir = run_root / "events_status"
    events_dir = run_root / "events"
    geo_dir_candidates = [run_root / "geo_backbone", run_root.parent / "geo_backbone"]
    geo_dir = next((p for p in geo_dir_candidates if p.is_dir()), geo_dir_candidates[-1])

    by_year = _safe_read_csv(analysis_dir / "full_vs_amendment_by_year.csv")
    cat = _safe_read_csv(analysis_dir / "category_trajectory_stage_means.csv")
    cpx = _safe_read_csv(analysis_dir / "complexity_trajectory_summary.csv")
    num = _safe_read_csv(long_dir / "city_numeric_trajectory.csv")
    issue_class_summary = _safe_read_csv(cls_dir / "issue_class_summary.csv")
    issue_labels = _safe_read_csv(cls_dir / "issue_labels_v3.csv")
    coverage = _safe_read_csv(long_dir / "city_longitudinal_summary.csv")
    city_geo = _safe_read_csv(geo_dir / "city_geo_backbone.csv")
    first_events = _safe_read_csv(long_dir / "city_first_zoning_events.csv")
    phase_df = _safe_read_csv(long_dir / "city_evolution_phases.csv")
    wave_df = _safe_read_csv(long_dir / "city_amendment_waves.csv")
    event_clusters_path = (
        events_status_dir / "ordinance_event_clusters_status_v3.csv"
        if (events_status_dir / "ordinance_event_clusters_status_v3.csv").is_file()
        else events_dir / "ordinance_event_clusters_v3.csv"
    )
    assign_path = (
        events_status_dir / "issue_event_assignments_status_v3.csv"
        if (events_status_dir / "issue_event_assignments_status_v3.csv").is_file()
        else events_dir / "issue_event_assignments_v3.csv"
    )
    event_clusters = _safe_read_csv(event_clusters_path)
    assign = _safe_read_csv(assign_path)
    ab_scores = _safe_read_csv(calibration_dir / "ab_scores.csv")
    val_log = _safe_read_csv(validation_dir / "validation_log.csv")
    val_summary = _safe_read_csv(validation_dir / "validation_summary.csv")

    selected_variants = {}
    p_sel = calibration_dir / "selected_variants.json"
    if p_sel.is_file():
        try:
            selected_variants = json.loads(p_sel.read_text(encoding="utf-8"))
        except Exception:
            selected_variants = {}

    # Ensure evidence PNG artifacts are present and indexed.
    evidence_idx = _ensure_evidence_page_index(run_root, out_dir)
    evidence_examples = _select_evidence_examples(val_log, evidence_idx)
    appendix_png_idx = _build_appendix_png_index(run_root, out_dir, issue_labels)

    # Explicit per-user exclusions for pages that should not be displayed in report galleries.
    excluded_page_ids = {"abilene-reporter-news-nov-30-1945-p-23"}
    excluded_issue_ids = {"abilene-reporter-news__1945-11-30"}
    if not evidence_idx.empty:
        keep_mask = pd.Series(True, index=evidence_idx.index)
        if "page_id" in evidence_idx.columns:
            keep_mask &= ~evidence_idx["page_id"].astype(str).isin(excluded_page_ids)
        if "issue_id" in evidence_idx.columns:
            keep_mask &= ~evidence_idx["issue_id"].astype(str).isin(excluded_issue_ids)
        evidence_idx = evidence_idx[keep_mask].copy()
    if not evidence_examples.empty:
        keep_mask = pd.Series(True, index=evidence_examples.index)
        if "page_id" in evidence_examples.columns:
            keep_mask &= ~evidence_examples["page_id"].astype(str).isin(excluded_page_ids)
        if "issue_id" in evidence_examples.columns:
            keep_mask &= ~evidence_examples["issue_id"].astype(str).isin(excluded_issue_ids)
        evidence_examples = evidence_examples[keep_mask].copy()
    if not appendix_png_idx.empty:
        keep_mask = pd.Series(True, index=appendix_png_idx.index)
        if "page_id" in appendix_png_idx.columns:
            keep_mask &= ~appendix_png_idx["page_id"].astype(str).isin(excluded_page_ids)
        if "issue_id" in appendix_png_idx.columns:
            keep_mask &= ~appendix_png_idx["issue_id"].astype(str).isin(excluded_issue_ids)
        appendix_png_idx = appendix_png_idx[keep_mask].copy()

    # Figures
    by_year_dedup = _build_dedup_full_amend_by_year(event_clusters)
    _plot_pipeline_overview(fig_dir / "pipeline_overview.png")
    _plot_classification_gate_flow(fig_dir / "classification_gate_flow.png")
    _plot_full_vs_amend(by_year, fig_dir / "full_vs_amendment_by_year.png")
    _plot_full_vs_amend(
        by_year_dedup,
        fig_dir / "full_vs_amendment_by_year_dedup.png",
        title="Full ordinances vs amendments over time (deduplicated events)",
        ylabel="Deduplicated event count",
    )
    _plot_issue_volume_by_newspaper(issue_labels, fig_dir / "issue_volume_by_newspaper.png")
    _plot_issue_class_stacked(issue_labels, fig_dir / "issue_class_stacked_by_newspaper.png")
    _plot_issue_classes(issue_class_summary, fig_dir / "issue_class_distribution.png")
    _plot_event_type_stacked(event_clusters, fig_dir / "event_type_stacked_by_newspaper.png")
    _plot_full_amend_by_newspaper(assign, fig_dir / "full_amend_by_newspaper.png")
    _plot_dedup_impact(issue_labels, event_clusters, fig_dir / "dedup_impact_raw_vs_events.png")
    _plot_first_signal_lag(first_events, fig_dir / "first_signal_to_full_lag.png")
    _plot_category_trajectory(cat, fig_dir / "category_trajectory_stage_means.png")
    _plot_complexity_heatmap(cpx, fig_dir / "complexity_heatmap.png")
    _plot_phase_label_counts(phase_df, fig_dir / "phase_label_counts.png")
    _plot_wave_type_counts(wave_df, fig_dir / "wave_type_counts.png")
    _plot_numeric_direction_counts(num, fig_dir / "numeric_direction_counts.png")

    # Tables
    if not coverage.empty:
        keep_cov = [c for c in ["newspaper_slug", "city_key", "issue_count"] if c in coverage.columns]
        cov_tbl = coverage[keep_cov].copy().sort_values(["issue_count", "newspaper_slug"], ascending=[False, True]).reset_index(drop=True)
    else:
        cov_tbl = pd.DataFrame(columns=["newspaper_slug", "city_key", "issue_count"])

    census_tbl = pd.DataFrame(
        columns=[
            "newspaper",
            "city",
            "place_geoid",
            "pop_2010",
            "tracts_2020",
            "counties_2020",
            "pumas_2020",
            "primary_county_share_pct",
        ]
    )
    if not coverage.empty and not city_geo.empty and {"newspaper_slug", "city_key"}.issubset(coverage.columns) and "city_key" in city_geo.columns:
        city_map = (
            coverage[["newspaper_slug", "city_key"]]
            .drop_duplicates()
            .rename(columns={"newspaper_slug": "newspaper"})
        )
        d_geo = city_geo.copy()

        def _pad_geoid7(x: Any) -> str:
            s = "".join(ch for ch in _norm(x) if ch.isdigit())
            return s.zfill(7) if s else ""

        d_geo["place_geoid"] = d_geo.get("place20_geoid", "").map(_pad_geoid7)
        d_geo["pop_2010"] = pd.to_numeric(d_geo.get("place_pop10"), errors="coerce").round(0)
        d_geo["tracts_2020"] = pd.to_numeric(d_geo.get("tract20_count"), errors="coerce").round(0)
        d_geo["counties_2020"] = pd.to_numeric(d_geo.get("county20_count"), errors="coerce").round(0)
        d_geo["pumas_2020"] = pd.to_numeric(d_geo.get("puma20_count"), errors="coerce").round(0)
        d_geo["primary_county_share_pct"] = (
            100.0 * pd.to_numeric(d_geo.get("county20_primary_weight"), errors="coerce")
        ).round(1)

        d_geo = d_geo.rename(columns={"city_name": "city"})
        keep_geo = [
            c
            for c in [
                "city_key",
                "city",
                "place_geoid",
                "pop_2010",
                "tracts_2020",
                "counties_2020",
                "pumas_2020",
                "primary_county_share_pct",
            ]
            if c in d_geo.columns
        ]
        census_tbl = city_map.merge(d_geo[keep_geo], on="city_key", how="left")
        keep_out = [
            c
            for c in [
                "newspaper",
                "city",
                "place_geoid",
                "pop_2010",
                "tracts_2020",
                "counties_2020",
                "pumas_2020",
                "primary_county_share_pct",
            ]
            if c in census_tbl.columns
        ]
        census_tbl = census_tbl[keep_out].copy()
        for c in ["pop_2010", "tracts_2020", "counties_2020", "pumas_2020"]:
            if c in census_tbl.columns:
                census_tbl[c] = pd.to_numeric(census_tbl[c], errors="coerce").astype("Int64")
        census_tbl = census_tbl.sort_values(["pop_2010", "newspaper"], ascending=[False, True], na_position="last").reset_index(drop=True)

    if not first_events.empty:
        keep_first = [c for c in ["newspaper_slug", "city_key", "first_signal_date", "first_full_date"] if c in first_events.columns]
        first_tbl = first_events[keep_first].copy().sort_values(["newspaper_slug", "first_signal_date"], ascending=[True, True]).reset_index(drop=True)
    else:
        first_tbl = pd.DataFrame(columns=["newspaper_slug", "city_key", "first_signal_date", "first_full_date"])

    if not issue_labels.empty:
        issue_counts = (
            issue_labels.groupby("newspaper_slug", dropna=False)["issue_id"]
            .nunique()
            .reset_index(name="issue_count")
            .sort_values("issue_count", ascending=False)
        )
    else:
        issue_counts = pd.DataFrame(columns=["newspaper_slug", "issue_count"])

    full_comp_newspaper_n = 0
    full_comp_newspaper_total = 0
    if not issue_labels.empty and {"issue_class", "newspaper_slug"}.issubset(issue_labels.columns):
        full_comp_rows = issue_labels[issue_labels["issue_class"].astype(str) == "full_code_comprehensive"].copy()
        full_comp_newspaper_n = int(full_comp_rows["newspaper_slug"].astype(str).nunique())
        full_comp_newspaper_total = int(issue_labels["newspaper_slug"].astype(str).nunique())
    if full_comp_newspaper_total > 0:
        full_comp_note = (
            f"In this run, the full code comprehensive label appears in "
            f"{full_comp_newspaper_n} of {full_comp_newspaper_total} newspapers (issue-level labels)."
        )
    else:
        full_comp_note = "In this run, newspaper coverage for the full code comprehensive label is not available."

    if not event_clusters.empty:
        ev_tbl = event_clusters[
            [c for c in ["newspaper_slug", "event_type", "issue_count", "first_issue_date", "last_issue_date"] if c in event_clusters.columns]
        ].copy()
        ev_tbl = ev_tbl.sort_values(["newspaper_slug", "issue_count"], ascending=[True, False]).reset_index(drop=True)
        ev_counts = (
            event_clusters.groupby(["newspaper_slug", "event_type"], dropna=False)["ordinance_event_id"]
            .nunique()
            .reset_index(name="event_count")
            .sort_values(["newspaper_slug", "event_count"], ascending=[True, False])
            .reset_index(drop=True)
        )
    else:
        ev_tbl = pd.DataFrame(columns=["newspaper_slug", "event_type", "issue_count", "first_issue_date", "last_issue_date"])
        ev_counts = pd.DataFrame(columns=["newspaper_slug", "event_type", "event_count"])

    if not assign.empty:
        law_mix = assign[assign["event_type"].isin(["full_ordinance", "amendment"])].copy()
        law_mix_tbl = (
            law_mix.groupby(["newspaper_slug", "event_type"], dropna=False)["issue_id"]
            .nunique()
            .reset_index(name="issue_event_rows")
            .sort_values(["newspaper_slug", "event_type"])
        )
    else:
        law_mix_tbl = pd.DataFrame(columns=["newspaper_slug", "event_type", "issue_event_rows"])

    dedup_issue_rows = 0
    dedup_event_count = 0
    dedup_reduction_pct = math.nan
    if not assign.empty and {"event_type", "issue_id", "ordinance_event_id"}.issubset(assign.columns):
        d_dedup = assign[assign["event_type"].astype(str).isin(["full_ordinance", "amendment"])].copy()
        d_dedup["event_key"] = d_dedup["ordinance_event_id"].astype(str)
        if "newspaper_slug" in d_dedup.columns:
            d_dedup["event_key"] = d_dedup["newspaper_slug"].astype(str) + "::" + d_dedup["ordinance_event_id"].astype(str)
        dedup_issue_rows = int(len(d_dedup))
        dedup_event_count = int(d_dedup["event_key"].astype(str).nunique())
        if dedup_issue_rows > 0:
            dedup_reduction_pct = 100.0 * (1.0 - (dedup_event_count / float(dedup_issue_rows)))

    dedup_example_sentence = "Illustrative example is not available for this run."
    if not event_clusters.empty and {"newspaper_slug", "ordinance_event_id", "event_type", "issue_count", "first_issue_date", "last_issue_date"}.issubset(event_clusters.columns):
        ex = event_clusters[event_clusters["event_type"].astype(str).isin(["full_ordinance", "amendment"])].copy()
        if not ex.empty:
            ex = ex.sort_values(["issue_count", "newspaper_slug"], ascending=[False, True]).reset_index(drop=True)
            r0 = ex.iloc[0]
            ex_slug = _slug_label(_norm(r0.get("newspaper_slug")))
            ex_event_type = "full ordinance" if _norm(r0.get("event_type")) == "full_ordinance" else "amendment"
            ex_issue_n = int(float(r0.get("issue_count") or 0))
            ex_first = _norm(r0.get("first_issue_date"))
            ex_last = _norm(r0.get("last_issue_date"))
            role_note = ""
            if not assign.empty and {"newspaper_slug", "ordinance_event_id", "publication_role"}.issubset(assign.columns):
                rr = assign[
                    (assign["newspaper_slug"].astype(str) == _norm(r0.get("newspaper_slug")))
                    & (assign["ordinance_event_id"].astype(str) == _norm(r0.get("ordinance_event_id")))
                ].copy()
                if not rr.empty:
                    rc = rr["publication_role"].astype(str).value_counts()
                    first_pub_n = int(rc.get("first_publication", 0))
                    repub_n = int(rc.get("republication", 0))
                    follow_n = int(rc.get("followup_notice", 0))
                    parts = []
                    if first_pub_n > 0:
                        parts.append(f"{first_pub_n} first publication")
                    if repub_n > 0:
                        parts.append(f"{repub_n} republication")
                    if follow_n > 0:
                        parts.append(f"{follow_n} follow-up notice")
                    if parts:
                        role_note = " Role split: " + ", ".join(parts) + "."
            dedup_example_sentence = (
                f"For example, in {ex_slug}, one {ex_event_type} appears in {ex_issue_n} newspaper issues "
                f"between {ex_first} and {ex_last}; deduplication treats these as one underlying legal event."
                f"{role_note}"
            )

    if not cat.empty:
        cat_tbl = cat.copy().sort_values("late", ascending=False).reset_index(drop=True)
        for col in ["early", "mid", "late", "late_minus_early_pp"]:
            if col in cat_tbl.columns:
                cat_tbl[col] = pd.to_numeric(cat_tbl[col], errors="coerce")
    else:
        cat_tbl = pd.DataFrame(columns=["category", "early", "mid", "late", "late_minus_early_pp"])

    if not val_log.empty:
        val_tbl = val_log[
            [c for c in ["newspaper_slug", "validation_type", "ordinance_event_id", "issue_id", "expected_label", "pass_fail", "notes"] if c in val_log.columns]
        ].copy()
    else:
        val_tbl = pd.DataFrame(columns=["newspaper_slug", "validation_type", "ordinance_event_id", "issue_id", "expected_label", "pass_fail", "notes"])

    full_audit_n = 0
    full_audit_precision = math.nan
    narrative_audit_n = 0
    narrative_misclassified_n = 0
    law_fragment_audit_n = 0
    law_fragment_misclassified_n = 0
    if not val_summary.empty and {"metric", "value"}.issubset(val_summary.columns):
        m = {_norm(k): _norm(v) for k, v in zip(val_summary["metric"].tolist(), val_summary["value"].tolist())}
        try:
            full_audit_n = int(float(m.get("full_ordinance_audited_n", "0") or 0))
        except Exception:
            full_audit_n = 0
        try:
            full_audit_precision = float(m.get("full_ordinance_precision", "nan"))
        except Exception:
            full_audit_precision = math.nan
        try:
            narrative_audit_n = int(float(m.get("narrative_exclusion_audited_n", "0") or 0))
        except Exception:
            narrative_audit_n = 0
        try:
            narrative_misclassified_n = int(float(m.get("narrative_exclusion_misclassified_n", "0") or 0))
        except Exception:
            narrative_misclassified_n = 0
        try:
            law_fragment_audit_n = int(float(m.get("law_fragment_exclusion_audited_n", "0") or 0))
        except Exception:
            law_fragment_audit_n = 0
        try:
            law_fragment_misclassified_n = int(float(m.get("law_fragment_exclusion_misclassified_n", "0") or 0))
        except Exception:
            law_fragment_misclassified_n = 0
    full_audit_precision_pct = f"{full_audit_precision * 100.0:.1f}\\%" if math.isfinite(full_audit_precision) else "n/a"

    fail_rows = val_tbl[val_tbl["pass_fail"].astype(str).str.lower() == "fail"].copy() if not val_tbl.empty else pd.DataFrame()
    if fail_rows.empty:
        fail_table_block = "No manual-validation failures in the current audited set."
    else:
        fail_table_block = (
            "\\begin{table}[H]\n"
            "\\centering\n"
            "\\caption{Manual validation failures}\n"
            "\\resizebox{\\textwidth}{!}{\\input{tables/validation_failures.tex}}\n"
            "\\end{table}"
        )

    # Artifact index for easy backtracking.
    artifact_rows = []
    for rel in [
        "page_tagging/page_tags.jsonl",
        "cleaning/anchor_decisions.jsonl",
        "cleaning/clean_zoning_text.jsonl",
        "classification/issue_labels_v3.jsonl",
        "classification/issue_labels_v3.csv",
        "events/ordinance_event_clusters_v3.csv",
        "events/issue_event_assignments_v3.csv",
        "longitudinal/panel_outputs_v3.jsonl",
        "longitudinal/city_category_trajectory.csv",
        "longitudinal/city_complexity_trajectory.csv",
        "analysis/full_vs_amendment_by_year.csv",
        "analysis/category_trajectory_stage_means.csv",
        "analysis/complexity_trajectory_summary.csv",
        "analysis/numeric_quote_catalog.csv",
        "manual_validation/validation_log.csv",
        "manual_validation/validation_summary.csv",
        "reports/fullnewspaper_v3/tables/evidence_page_index.csv",
        "reports/fullnewspaper_v3/tables/appendix_classified_png_index.csv",
    ]:
        p = run_root / rel
        if p.is_file():
            row_count = ""
            if p.suffix.lower() == ".csv":
                try:
                    row_count = int(len(pd.read_csv(p)))
                except Exception:
                    row_count = ""
            elif p.suffix.lower() == ".jsonl":
                try:
                    row_count = int(sum(1 for _ in p.open("r", encoding="utf-8")))
                except Exception:
                    row_count = ""
            artifact_rows.append({"artifact": rel, "rows": row_count, "used_in_report": "yes"})
    artifact_tbl = pd.DataFrame.from_records(artifact_rows)

    _to_latex_table(cov_tbl, tbl_dir / "coverage_summary.tex", index=False)
    _to_latex_table(census_tbl, tbl_dir / "census_city_profile.tex", index=False, float_format="{:.1f}")
    _to_latex_table(first_tbl, tbl_dir / "first_events.tex", index=False)
    _to_latex_table(issue_counts, tbl_dir / "issue_counts_by_newspaper.tex", index=False)
    _to_latex_table(law_mix_tbl, tbl_dir / "full_amend_rows_by_newspaper.tex", index=False)
    _to_latex_table(ev_counts, tbl_dir / "event_type_by_newspaper.tex", index=False)
    _to_latex_table(ev_tbl.head(25), tbl_dir / "event_clusters.tex", index=False)
    _to_latex_table(cat_tbl, tbl_dir / "category_means.tex", index=False)
    _to_latex_table(val_tbl.head(25), tbl_dir / "validation_log.tex", index=False)
    _to_latex_table(fail_rows, tbl_dir / "validation_failures.tex", index=False)
    _to_latex_table(artifact_tbl, tbl_dir / "artifact_index.tex", index=False)

    evidence_tbl = evidence_idx.copy()
    if not evidence_tbl.empty:
        keep_evidence = [
            c
            for c in [
                "newspaper_slug",
                "validation_type",
                "ordinance_event_id",
                "issue_id",
                "issue_date",
                "page_id",
            ]
            if c in evidence_tbl.columns
        ]
        evidence_tbl = evidence_tbl[keep_evidence].copy()
    _to_latex_table(evidence_tbl, tbl_dir / "evidence_page_index.tex", index=False)

    appendix_tbl = appendix_png_idx.copy()
    if not appendix_tbl.empty:
        keep_appendix = [
            c
            for c in [
                "appendix_group",
                "newspaper_slug",
                "issue_id",
                "issue_date",
                "issue_class",
                "page_id",
            ]
            if c in appendix_tbl.columns
        ]
        appendix_tbl = appendix_tbl[keep_appendix].copy()
    _to_latex_table(appendix_tbl, tbl_dir / "appendix_classified_png_index.tex", index=False)

    selected_clean = _norm(selected_variants.get("selected_cleaning_variant"))
    selected_cls = _norm(selected_variants.get("selected_classification_variant"))
    selected_dedup = _norm(selected_variants.get("selected_dedup_variant"))

    def _tested_variants_for_step(step_key: str) -> list[str]:
        if ab_scores.empty or "step" not in ab_scores.columns or "variant" not in ab_scores.columns:
            return []
        d = ab_scores[ab_scores["step"].astype(str) == str(step_key)].copy()
        if d.empty:
            return []
        if "score" in d.columns:
            d["score_num"] = pd.to_numeric(d["score"], errors="coerce")
            d = d.sort_values(["score_num", "variant"], ascending=[False, True], na_position="last")
        else:
            d = d.sort_values(["variant"], ascending=[True])
        out: list[str] = []
        seen: set[str] = set()
        for v in d["variant"].astype(str).tolist():
            vv = _norm(v)
            if vv and vv not in seen:
                seen.add(vv)
                out.append(vv)
        return out

    def _fmt_variant_list(vs: list[str]) -> str:
        if not vs:
            return "n/a"
        return ", ".join([_tex_escape(v).replace("_", " ") for v in vs])

    clean_tested = _fmt_variant_list(_tested_variants_for_step("cleaning_variant"))
    cls_tested = _fmt_variant_list(_tested_variants_for_step("classification_variant"))
    dedup_tested = _fmt_variant_list(_tested_variants_for_step("dedup_variant"))
    selected_clean_tex = _tex_escape(selected_clean).replace("_", " ") if selected_clean else "n/a"
    selected_cls_tex = _tex_escape(selected_cls).replace("_", " ") if selected_cls else "n/a"
    selected_dedup_tex = _tex_escape(selected_dedup).replace("_", " ") if selected_dedup else "n/a"

    # Evidence gallery LaTeX block
    gallery_lines: list[str] = []
    if evidence_examples.empty:
        gallery_lines.append("No evidence PNG examples available.")
    else:
        for i, (_, r) in enumerate(evidence_examples.iterrows()):
            fig_rel = _norm(r.get("figure_relpath"))
            if not fig_rel:
                continue
            label = f"{_slug_label(_norm(r.get('newspaper_slug')))} | {_norm(r.get('validation_type'))} | {_norm(r.get('pass_fail')).upper()}"
            issue_id_raw = _norm(r.get("issue_id"))
            event_id_raw = _norm(r.get("ordinance_event_id")) or "n/a"
            page_id_raw = _norm(r.get("page_id"))
            notes = _tex_escape(r.get("notes"))
            notes = notes[:220] + ("..." if len(notes) > 220 else "")
            fig_rel_tex = _tex_escape(fig_rel)
            gallery_lines.extend(
                [
                    "\\clearpage" if i > 0 else "",
                    "\\begin{figure}[H]",
                    "\\centering",
                    f"\\includegraphics[width=0.92\\textwidth,height=0.62\\textheight,keepaspectratio]{{{fig_rel_tex}}}",
                    f"\\caption{{Evidence page: {_tex_escape(label)}. Produced from sampled manual-validation rows by linking issue/page identifiers to the source newspaper image.}}",
                    "\\vspace{0.35em}",
                    "\\begin{minipage}{0.94\\textwidth}",
                    "\\footnotesize",
                    f"\\textit{{Issue/event/page:}} {_tex_escape(issue_id_raw)}; {_tex_escape(event_id_raw)}; {_tex_escape(page_id_raw)}.\\\\",
                    "Displayed page corresponds to the sampled validation row.\\\\",
                    f"\\textit{{Validation note:}} {notes if notes else 'n/a'}.",
                    "\\end{minipage}",
                    "\\end{figure}",
                    "",
                ]
            )
    gallery_block = "\n".join(gallery_lines)

    appendix_full_df = pd.DataFrame()
    appendix_sub_df = pd.DataFrame()
    appendix_rest_df = pd.DataFrame()
    if not appendix_png_idx.empty:
        appendix_full_df = appendix_png_idx[appendix_png_idx["appendix_group"].astype(str) == "full_all"].copy()
        appendix_sub_df = appendix_png_idx[
            appendix_png_idx["appendix_group"].astype(str) == "amendment_substantial_all"
        ].copy()
        appendix_rest_df = appendix_png_idx[appendix_png_idx["appendix_group"].astype(str) == "remaining_sample"].copy()

    appendix_full_n = int((appendix_full_df["image_fetch_status"].astype(str) != "missing").sum()) if not appendix_full_df.empty else 0
    appendix_sub_n = int((appendix_sub_df["image_fetch_status"].astype(str) != "missing").sum()) if not appendix_sub_df.empty else 0
    appendix_rest_n = int((appendix_rest_df["image_fetch_status"].astype(str) != "missing").sum()) if not appendix_rest_df.empty else 0

    appendix_full_block = _build_appendix_gallery_block(appendix_full_df, "All full-classified pages")
    appendix_sub_block = _build_appendix_gallery_block(appendix_sub_df, "All substantial-amendment pages")
    appendix_rest_block = _build_appendix_sampled_by_class_block(appendix_rest_df)

    if not census_tbl.empty:
        census_table_block = (
            "\\begin{table}[H]\n"
            "\\centering\n"
            "\\caption{Census-derived descriptive profile for newspaper cities}\n"
            "\\resizebox{\\textwidth}{!}{\\input{tables/census_city_profile.tex}}\n"
            "\\end{table}\n"
            "\\objnote{This table links each newspaper city to its matched Census place geography and summarizes selected Census-derived descriptors: place population (2010), number of intersecting 2020 tracts/counties/PUMAs, and the primary-county land-share weight. Larger tract/county/PUMA counts indicate more geospatially fragmented city footprints for downstream merges.}\n"
        )
    else:
        census_table_block = ""

    tex = f"""
\\documentclass[11pt]{{article}}
\\usepackage[margin=1in]{{geometry}}
\\usepackage{{booktabs}}
\\usepackage{{graphicx}}
\\usepackage{{float}}
\\usepackage{{longtable}}
\\usepackage{{hyperref}}
\\usepackage{{xurl}}
\\setlength{{\\parskip}}{{0.5em}}
\\setlength{{\\parindent}}{{0pt}}
\\setlength{{\\emergencystretch}}{{2em}}
\\setcounter{{tocdepth}}{{2}}
\\newcommand{{\\objnote}}[1]{{\\vspace{{0.2em}}\\begin{{minipage}}{{0.97\\textwidth}}\\footnotesize\\textit{{Notes.}} #1\\end{{minipage}}}}
\\title{{Five-Newspaper Zoning Pipeline: Expanded Descriptive Report}}
\\author{{Automated v3 pipeline}}
\\date{{{dt.date.today().isoformat()}}}
\\begin{{document}}
\\maketitle
\n\\tableofcontents
\\clearpage

\\section{{High-level walkthrough}}
This report tracks the complete pipeline across five newspapers and is organized so each major result can be traced back to concrete artifacts.
The core stages are:
\\begin{{itemize}}
\\item gather each newspaper timeline in full date order,
\\item isolate the actual zoning-law language from surrounding news coverage,
\\item classify each legal publication into broad types (full code, amendment, or other legal material),
\\item collapse repeat printings so counts reflect underlying legal actions rather than reprints,
\\item summarize when zoning first appears and how content shifts over time.
\\end{{itemize}}
Early page screening is used for organization, but final inclusion comes from direct review of legal text passages so zoning clauses on mixed-content pages are not dropped.

\\begin{{figure}}[H]
\\centering
\\includegraphics[width=0.97\\textwidth]{{figures/pipeline_overview.png}}
\\caption{{End-to-end pipeline overview from raw issues to final report outputs. Produced from the finalized production workflow stages used in this run.}}
\\end{{figure}}
\\objnote{{This diagram is constructed from the finalized production workflow used in this run. Boxes are process stages and arrows are artifact handoffs. It is a structural map, not a statistical result.}}
The pipeline runs in sequence from raw newspapers to clean legal text, then to legal-action classification, and finally to timeline summaries. The first stage is shown as one consolidated extraction block to keep the workflow readable while preserving the full review steps behind it.

\\begin{{figure}}[H]
\\centering
\\includegraphics[width=0.97\\textwidth]{{figures/classification_gate_flow.png}}
\\caption{{Classification gate logic used before counting legal events. Produced by diagramming the two required issue-level gates: verbatim-law presence and law-text sufficiency.}}
\\end{{figure}}
\\objnote{{The gate is evaluated at issue level. An issue is counted in legal-event totals only if both conditions are satisfied: verbatim zoning-law text is present and extracted law text is sufficient for ontology labeling. Issues failing either condition remain in context outputs but are excluded from full/amend/procedural event counts.}}
Event counts are protected from inflation by zoning-related narrative coverage. An issue is counted as a legal event only when both conditions are met: (1) verbatim zoning-law text is present and (2) legal text is sufficient for classification. Issues that fail either condition are retained for context but excluded from full/amend/procedural event totals.

\\textbf{{Boundary note on full-code labels.}} {full_comp_note} Important caveat: issue-level full code comprehensive labels can include repeated publications or installment-style republication of the same legal text. A truly new citywide full code is unlikely to occur regularly; repeated appearances are often reprints. This is handled downstream via later post-processing in the cleaning pipeline and event-level clustering so repeated printings are collapsed before substantive interpretation.

Classification improvement recommended from manual review: split full-code outcomes into two explicit classes at issue level before event counting: (1) full ordinance comprehensive and (2) full ordinance installment fragment. Promote an installment fragment to comprehensive only when the stitched issue sequence covers broad ordinance structure (district framework, use rules, dimensional standards, procedures, and definitions) across contiguous installments; otherwise keep it as partial pass evidence and exclude it from comprehensive full-code totals.

Pipeline tuning choices (options tested and finalized):
\\begin{{itemize}}
\\item \\textbf{{Cleaning stage.}} Options tested: {clean_tested}. Finalized: {selected_clean_tex}.
\\item \\textbf{{Classification stage.}} Options tested: {cls_tested}. Finalized: {selected_cls_tex}.
\\item \\textbf{{Deduplication stage.}} Options tested: {dedup_tested}. Finalized: {selected_dedup_tex}.
\\end{{itemize}}

\\section{{Coverage and corpus composition}}
Coverage is summarized at newspaper level, then broken into issue classes to show where legal-text extraction concentrated. All counts in this section are issue-level publication rows and are not deduplicated.

\\begin{{table}}[H]
\\centering
\\caption{{Panel coverage summary}}
\\resizebox{{\\textwidth}}{{!}}{{\\input{{tables/coverage_summary.tex}}}}
\\end{{table}}
\\objnote{{One row equals one newspaper timeline in this five-paper run. Issue counts reflect loaded issue records for analysis and are not deduplicated legal events.}}

{census_table_block}

\\begin{{figure}}[H]
\\centering
\\includegraphics[width=0.90\\textwidth]{{figures/issue_volume_by_newspaper.png}}
\\caption{{Issue volume by newspaper. Produced as unique issue counts loaded for each newspaper timeline in the five-paper run.}}
\\end{{figure}}
\\objnote{{Bars use unique issue counts per newspaper timeline. This is an input-volume view and does not apply full/amend event filtering or deduplication.}}

\\begin{{figure}}[H]
\\centering
\\includegraphics[width=0.96\\textwidth]{{figures/issue_class_stacked_by_newspaper.png}}
\\caption{{Issue-class composition by newspaper. Produced from issue-level ontology labels returned by the classification pass and stacked by class within newspaper.}}
\\end{{figure}}
\\objnote{{Each issue receives one ontology label after legal-text extraction. Stacks summarize issue-level class counts by newspaper; repeated publications are still separate rows here.}}

\\begin{{figure}}[H]
\\centering
\\includegraphics[width=0.86\\textwidth]{{figures/issue_class_distribution.png}}
\\caption{{Overall issue-class distribution across all five newspapers. Produced by summing the same issue-level ontology labels across newspapers.}}
\\end{{figure}}
\\objnote{{This aggregates the same issue-level ontology labels across all newspapers in the run. It shows composition of classified issue rows, not deduplicated legal events.}}

Full-ordinance and amendment publication rows are shown next before repeat publications are collapsed into deduplicated events.

\\begin{{table}}[H]
\\centering
\\caption{{Full/amendment publication rows by newspaper (pre-dedup).}}
\\resizebox{{0.78\\textwidth}}{{!}}{{\\input{{tables/full_amend_rows_by_newspaper.tex}}}}
\\end{{table}}
\\objnote{{Rows are issue-level publication counts where event type is full ordinance or amendment before clustering. A single underlying legal action can appear multiple times due to reprints or follow-up publications.}}

\\begin{{figure}}[H]
\\centering
\\includegraphics[width=0.94\\textwidth]{{figures/full_vs_amendment_by_year.png}}
\\caption{{Full-ordinance and amendment publication rows over time (pre-dedup). Produced from yearly counts of issue-level publication rows before event clustering.}}
\\end{{figure}}
\\objnote{{Each point uses publication-row counts by year, before deduplication. Spikes can reflect repeated publication of the same legal action within a year.}}

\\begin{{figure}}[H]
\\centering
\\includegraphics[width=0.94\\textwidth]{{figures/full_amend_by_newspaper.png}}
\\caption{{Full vs amendment publication rows by newspaper (pre-dedup). Produced from issue-level publication-row counts by newspaper before deduplication.}}
\\end{{figure}}
\\objnote{{Bars compare raw publication-row totals by newspaper before clustering repeated printings into unique events.}}

\\section{{Event structure and deduplication}}
The starting unit is a publication row: one newspaper issue that contains legal zoning text. Many legal actions appear in multiple issues because newspapers reprint the same legal text or publish follow-up notices. Deduplication groups repeated publications into one underlying legal event and counts that event once.
Except for the raw-vs-dedup comparison chart, outputs in this section are deduplicated event-level counts.
\\begin{{itemize}}
\\item Before deduplication: {dedup_issue_rows} full/amendment publication rows.
\\item After deduplication: {dedup_event_count} distinct underlying legal events.
\\item Net reduction from collapsing repeats: {f"{dedup_reduction_pct:.1f}" if math.isfinite(dedup_reduction_pct) else "n/a"}\\%.
\\end{{itemize}}
{dedup_example_sentence}

\\begin{{figure}}[H]
\\centering
\\includegraphics[width=0.96\\textwidth]{{figures/dedup_impact_raw_vs_events.png}}
\\caption{{Dedup impact: raw issue rows vs deduplicated events (full + amendment). Produced by comparing pre-dedup issue rows with post-clustering event counts.}}
\\end{{figure}}
\\objnote{{For each newspaper, raw full/amend publication rows are compared with unique clustered event counts keyed by event identifier. The gap quantifies repeated publication collapsed by deduplication.}}

\\begin{{figure}}[H]
\\centering
\\includegraphics[width=0.94\\textwidth]{{figures/full_vs_amendment_by_year_dedup.png}}
\\caption{{Full-ordinance and amendment counts over time after deduplication. Produced by counting unique clustered legal events by first publication year and event type.}}
\\end{{figure}}
\\objnote{{Each point is a count of unique event identifiers that first appear in that year. Repeated publication rows are already collapsed before yearly counting.}}

\\begin{{figure}}[H]
\\centering
\\includegraphics[width=0.96\\textwidth]{{figures/event_type_stacked_by_newspaper.png}}
\\caption{{Deduplicated event-type mix by newspaper. Produced from one row per clustered event, grouped by event type within newspaper.}}
\\end{{figure}}
\\objnote{{Unit is one deduplicated legal event. Stacks show event-type composition after repeated publications are collapsed.}}

\\begin{{table}}[H]
\\centering
\\caption{{Deduplicated event-type counts by newspaper}}
\\resizebox{{\\textwidth}}{{!}}{{\\input{{tables/event_type_by_newspaper.tex}}}}
\\end{{table}}
\\objnote{{Each cell is a count of unique clustered event identifiers by newspaper and event type. Publication-row duplicates are removed before counting.}}

\\begin{{table}}[H]
\\centering
\\caption{{Illustrative deduplicated event clusters (top rows)}}
\\resizebox{{\\textwidth}}{{!}}{{\\input{{tables/event_clusters.tex}}}}
\\end{{table}}
\\objnote{{Each row is one clustered legal event with number of linked issue publications and first/last publication dates. Rows shown are the largest clusters for inspection.}}

\\section{{First-signal and first-full timing}}
The timing table and lag plot summarize when zoning first appears in each timeline and when a full ordinance first appears. These are timeline-level extraction outputs and not publication-row or deduplicated-event counts.
\\begin{{table}}[H]
\\centering
\\caption{{First zoning signal and first full ordinance}}
\\resizebox{{\\textwidth}}{{!}}{{\\input{{tables/first_events.tex}}}}
\\end{{table}}
\\objnote{{One row per newspaper timeline from longitudinal extraction. First-signal and first-full dates are timeline-level outputs derived from the full chronological panel, not event-row tallies.}}

\\begin{{figure}}[H]
\\centering
\\includegraphics[width=0.88\\textwidth]{{figures/first_signal_to_full_lag.png}}
\\caption{{Years from first zoning signal to first full ordinance (found cases). Produced as timeline-level date differences between extracted first-signal and first-full events.}}
\\end{{figure}}
\\objnote{{Lag is computed as first-full date minus first-signal date in years and plotted only for timelines where first-full status is found.}}

\\section{{Longitudinal content and complexity}}
Each newspaper timeline is read in chronological order and split into three within-newspaper stages: early (first third of issues), mid (middle third), and late (final third). The extraction returns stage-level topic shares for the fixed zoning categories, so each stage is a composition of zoning content rather than a raw count of articles.

The category figure and table aggregate those stage-level outputs across newspapers. For each category and stage, the reported value is the mean share across newspapers. The late-minus-early column reports the change in percentage points between the late and early stage means, which makes directional shifts easier to read than raw stage values alone.

Complexity outputs are produced as timeline-direction labels for regulatory complexity, procedural intensity, and dimensional specificity. The heatmap converts those labels into row-wise shares of newspapers by direction (up, flat, down, mixed/uncertain), so each row describes how common each trend direction is for one complexity metric.

Numeric trajectory counts summarize direction labels for lot size, height, setback, and parking, using the same timeline-level direction output returned by the longitudinal extraction.
\\begin{{figure}}[H]
\\centering
\\includegraphics[width=0.96\\textwidth]{{figures/category_trajectory_stage_means.png}}
\\caption{{Timeline-level category trajectory means (early/mid/late). Produced by splitting each newspaper timeline into early/mid/late thirds and averaging stage shares across newspapers.}}
\\end{{figure}}
\\objnote{{Each newspaper timeline is split by issue order into early, mid, and late thirds. For each stage, category shares come from longitudinal extraction and are compositional within newspaper-stage. Lines show cross-newspaper mean stage shares.}}

\\begin{{table}}[H]
\\centering
\\caption{{Timeline-level category trajectory means (shares and late-minus-early)}}
\\resizebox{{\\textwidth}}{{!}}{{\\input{{tables/category_means.tex}}}}
\\end{{table}}
\\objnote{{Early, mid, and late columns are cross-newspaper means of stage shares for each category. The late-minus-early column is reported in percentage points and equals (late share minus early share) multiplied by 100. Positive values indicate rising share over the timeline; negative values indicate declining share.}}

\\begin{{figure}}[H]
\\centering
\\includegraphics[width=0.86\\textwidth]{{figures/complexity_heatmap.png}}
\\caption{{Timeline-level complexity trend direction by metric. Produced by aggregating extracted direction labels (up/flat/down/mixed) into row-wise shares by metric.}}
\\end{{figure}}
\\objnote{{For each complexity metric, each newspaper contributes one direction label from longitudinal extraction. Cells are shares of newspapers by direction within metric; rows sum to approximately 100\\% subject to rounding.}}

\\begin{{figure}}[H]
\\centering
\\includegraphics[width=0.92\\textwidth]{{figures/numeric_direction_counts.png}}
\\caption{{Timeline-level numeric trajectory direction counts (lot size, height, setback, parking). Produced from extracted timeline-direction labels for each numeric-rule category.}}
\\end{{figure}}
\\objnote{{For each numeric-rule category, each timeline contributes one direction label when available. Bars show label counts across timelines and therefore summarize directional tendency rather than magnitude changes.}}

\\section{{Manual validation and error profile}}
Manual checks compare sampled classified rows to the newspaper evidence pages and transcript context.
\\begin{{itemize}}
\\item Audited full-ordinance rows: {full_audit_n}; precision: {full_audit_precision_pct}.
\\item Audited narrative exclusions: {narrative_audit_n}; misclassified as full/amendment: {narrative_misclassified_n}.
\\item Audited law-fragment exclusions: {law_fragment_audit_n}; misclassified as full/amendment: {law_fragment_misclassified_n}.
\\end{{itemize}}
{fail_table_block}

\\begin{{table}}[H]
\\centering
\\caption{{Manual validation log (top rows)}}
\\resizebox{{\\textwidth}}{{!}}{{\\input{{tables/validation_log.tex}}}}
\\end{{table}}
\\objnote{{Rows are sampled manual audit records used to compare classification output against newspaper evidence pages and transcript context. Display is truncated to top rows for readability.}}

\\section{{Evidence PNG gallery}}
The pages below are the evidence images used for sampled validation rows.
{gallery_block}

\\appendix
\\section{{Classified PNG galleries}}
This appendix includes all PNG pages classified as full ordinance, all pages classified as substantial amendment, and sampled pages from the remaining classes.

\\subsection{{All pages classified as full ordinance}}
Total pages shown: {appendix_full_n}.
{appendix_full_block}

\\subsection{{All pages classified as substantial amendment}}
Total pages shown: {appendix_sub_n}.
{appendix_sub_block}

\\subsection{{Other classified pages (sampled for display)}}
These pages are grouped by their assigned class labels. Sampling here is only a display choice to keep the appendix readable.
Total pages shown: {appendix_rest_n}.
{appendix_rest_block}

\\begin{{table}}[H]
\\centering
\\caption{{Appendix classified PNG index}}
\\resizebox{{\\textwidth}}{{!}}{{\\input{{tables/appendix_classified_png_index.tex}}}}
\\end{{table}}
\\objnote{{This is an index of displayed appendix pages and their assigned class labels in this report. It does not represent the full universe of pages in the corpus.}}

\\end{{document}}
"""
    _write_text(out_dir / "report.tex", tex.strip() + "\n")

    if bool(args.compile_pdf):
        cmd = ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "report.tex"]
        try:
            subprocess.run(cmd, cwd=str(out_dir), check=True)
        except Exception:
            pass

    provenance = {
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "run_root": str(run_root),
        "output_dir": str(out_dir),
        "compile_pdf": bool(args.compile_pdf),
    }
    _write_text(out_dir / "provenance.json", json.dumps(provenance, indent=2) + "\n")
    print(f"done_fullnewspaper_v3_report output_dir={out_dir}")


if __name__ == "__main__":
    main()
