#!/usr/bin/env python3
"""
Build a standalone LaTeX writeup for the full-coverage 20-city longitudinal run.

This report intentionally avoids any "previous approach" framing and describes
the current workflow and outputs as the primary analysis.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


_FIXED_CATEGORY_KEYS = [
    "land_use_restrictions",
    "bulk_dimensional_standards",
    "procedural_governance",
    "nonconforming_use",
    "definitions",
    "parking_transport",
    "environmental_protection",
    "historic_preservation",
    "flexible_zoning",
    "inclusionary_affordable_housing",
    "enforcement_penalties",
    "miscellaneous_site_rules",
]

_PHASE_LABEL_MEANING = {
    "initial_establishment": "Initial setup/adoption of zoning framework.",
    "formalization_enforcement": "Procedures, governance, permits, and enforcement structure.",
    "expansion_reclassification": "District expansion, rezoning, and reclassification activity.",
    "flexible_special_use": "Special/conditional/flexible-use mechanisms.",
    "infrastructure_environment": "Infrastructure and environmental zoning provisions.",
    "maturation_implementation": "Ongoing implementation and incremental adjustment.",
    "other_unclear": "Ambiguous or mixed content not clearly mapped above.",
}


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _tex_escape(s: Any) -> str:
    return (
        str(s)
        .replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("$", "\\$")
        .replace("#", "\\#")
        .replace("_", "\\_")
        .replace("{", "\\{")
        .replace("}", "\\}")
        .replace("~", "\\textasciitilde{}")
        .replace("^", "\\textasciicircum{}")
    )


def _to_year(s: Any) -> float:
    txt = str(s or "").strip()
    if not txt:
        return math.nan
    try:
        return float(pd.to_datetime(txt, errors="coerce").year)
    except Exception:
        return math.nan


def _write_table_tex(path: Path, df: pd.DataFrame, cols: list[str], float_cols: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if df.empty:
        path.write_text("\\begin{tabular}{l}\n\\toprule\nNo data\\\\\n\\bottomrule\n\\end{tabular}\n", encoding="utf-8")
        return

    keep = [c for c in cols if c in df.columns]
    if not keep:
        keep = list(df.columns[: min(8, len(df.columns))])

    sub = df[keep].copy()
    float_cols = float_cols or []
    def _fmt_num(v: Any) -> str:
        if pd.isna(v):
            return ""
        f = float(v)
        if abs(f - round(f)) < 1e-8:
            return str(int(round(f)))
        return f"{f:.3f}"

    for c in sub.columns:
        if c in float_cols or pd.api.types.is_float_dtype(sub[c]):
            sub[c] = pd.to_numeric(sub[c], errors="coerce").map(_fmt_num)

    lines: list[str] = []
    lines.append("\\begin{tabular}{" + "l" * len(sub.columns) + "}")
    lines.append("\\toprule")
    lines.append(" & ".join(_tex_escape(c) for c in sub.columns) + " \\\\")
    lines.append("\\midrule")
    for row in sub.itertuples(index=False):
        lines.append(" & ".join(_tex_escape(v) for v in row) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot_issue_coverage(selection: pd.DataFrame, out: Path) -> None:
    if selection.empty:
        return
    d = selection.copy()
    d["selected_issue_count"] = pd.to_numeric(d["selected_issue_count"], errors="coerce")
    d = d.dropna(subset=["selected_issue_count"]).sort_values("selected_issue_count", ascending=False)
    if d.empty:
        return
    d["label"] = d["city_name"].astype(str) + ", " + d["state_abbr"].astype(str).str.upper()
    fig, ax = plt.subplots(figsize=(10.6, 5.8))
    ax.barh(d["label"], d["selected_issue_count"], color="#4e79a7")
    ax.invert_yaxis()
    ax.set_xlabel("Issue count in panel")
    ax.set_ylabel("City")
    ax.set_title("Full-coverage panel sizes by city")
    ax.grid(axis="x", alpha=0.25, linestyle="--")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220)
    plt.close(fig)


def _plot_first_event_years(first_df: pd.DataFrame, out_path: Path) -> None:
    if first_df.empty:
        return
    d = first_df.copy()
    d["first_zoning_year"] = d["first_zoning_issue_date"].map(_to_year)
    d["first_full_year"] = d["first_full_issue_date"].map(_to_year)
    d = d[(d["first_zoning_year"].notna()) | (d["first_full_year"].notna())].copy()
    if d.empty:
        return

    z = d["first_zoning_year"].dropna().astype(int)
    f = d["first_full_year"].dropna().astype(int)
    if z.empty and f.empty:
        return

    # With N=20 and mostly unique years, year-by-year bars are not informative.
    # Use decade bins + cumulative coverage curves instead.
    z_dec = ((z // 10) * 10).astype(int)
    f_dec = ((f // 10) * 10).astype(int)
    decades = sorted(set(z_dec.tolist() + f_dec.tolist()))
    zc = pd.Series(z_dec).value_counts().reindex(decades, fill_value=0)
    fc = pd.Series(f_dec).value_counts().reindex(decades, fill_value=0)

    all_years = sorted(set(z.tolist() + f.tolist()))
    n_city = len(d)
    sig_series = d["first_zoning_year"]
    full_series = d["first_full_year"]
    sig_cum = [float((sig_series <= y).sum() / n_city) for y in all_years]
    full_cum = [float((full_series <= y).sum() / n_city) for y in all_years]

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.9))

    x = np.arange(len(decades))
    w = 0.42
    axes[0].bar(x - w / 2, zc.values, width=w, color="#4e79a7", label="First zoning signal")
    axes[0].bar(x + w / 2, fc.values, width=w, color="#f28e2b", label="First full ordinance")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"{y}s" for y in decades], rotation=30, ha="right")
    axes[0].set_ylabel("Cities")
    axes[0].set_xlabel("Decade")
    axes[0].set_title("First-event counts by decade")
    axes[0].grid(axis="y", alpha=0.25, linestyle="--")
    axes[0].legend(frameon=False, fontsize=9)

    axes[1].plot(all_years, sig_cum, marker="o", color="#4e79a7", label="First zoning signal")
    axes[1].plot(all_years, full_cum, marker="o", color="#f28e2b", label="First full ordinance")
    axes[1].set_xlabel("Year")
    axes[1].set_ylabel("Cumulative share of cities")
    axes[1].set_ylim(0, 1.02)
    axes[1].set_title("Cumulative timing coverage")
    axes[1].grid(axis="both", alpha=0.25, linestyle="--")
    axes[1].legend(frameon=False, fontsize=9)

    fig.suptitle("Timing of first zoning signal and first full ordinance", fontsize=13)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _plot_category_trajectory(cat_df: pd.DataFrame, out_path: Path) -> None:
    if cat_df.empty:
        return
    d = cat_df.copy()
    for c in ("early_share", "mid_share", "late_share"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d[d["category"].isin(_FIXED_CATEGORY_KEYS)]
    d = d[d[["early_share", "mid_share", "late_share"]].notna().any(axis=1)]
    if d.empty:
        return

    g = (
        d.groupby("category", dropna=False)[["early_share", "mid_share", "late_share"]]
        .mean()
        .reset_index()
    )
    g["avg"] = g[["early_share", "mid_share", "late_share"]].mean(axis=1)
    g = g.sort_values("avg", ascending=False).reset_index(drop=True)

    x = np.arange(len(g))
    fig, ax = plt.subplots(figsize=(11.2, 5.1))
    ax.plot(x, g["early_share"], marker="o", color="#4e79a7", label="Early")
    ax.plot(x, g["mid_share"], marker="o", color="#59a14f", label="Mid")
    ax.plot(x, g["late_share"], marker="o", color="#e15759", label="Late")
    ax.set_xticks(x)
    ax.set_xticklabels(g["category"].astype(str).tolist(), rotation=38, ha="right")
    ax.set_ylabel("Mean category share")
    ax.set_title("Category trajectory profiles (early/mid/late)")
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.legend(frameon=False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _normalize_category_composition(
    cat_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Normalize early/mid/late category vectors to sum to 1 within each city."""
    if cat_df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    d = cat_df.copy()
    d = d[d["category"].isin(_FIXED_CATEGORY_KEYS)].copy()
    if d.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    stages = ["early_share", "mid_share", "late_share"]
    for c in stages:
        d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0.0).clip(lower=0.0)

    city_cols = ["city_key"]
    if "city_name" in d.columns:
        city_cols.append("city_name")
    if "state_abbr" in d.columns:
        city_cols.append("state_abbr")

    raw_city = d.groupby(city_cols, dropna=False)[stages].sum().reset_index()

    dn = d.copy()
    for c in stages:
        denom = dn.groupby("city_key", dropna=False)[c].transform("sum")
        dn[c] = np.where(denom > 0, dn[c] / denom, np.nan)

    norm_city = (
        dn.groupby(city_cols, dropna=False)[stages]
        .agg(lambda s: s.sum(min_count=1))
        .reset_index()
    )

    def _stage_diag(stage: str, stage_pretty: str) -> dict[str, Any]:
        raw = pd.to_numeric(raw_city[stage], errors="coerce")
        norm = pd.to_numeric(norm_city[stage], errors="coerce")
        return {
            "stage": stage_pretty,
            "raw_mean_sum": float(raw.mean()) if not raw.empty else math.nan,
            "raw_p10_sum": float(raw.quantile(0.10)) if not raw.empty else math.nan,
            "raw_median_sum": float(raw.median()) if not raw.empty else math.nan,
            "raw_p90_sum": float(raw.quantile(0.90)) if not raw.empty else math.nan,
            "raw_min_sum": float(raw.min()) if not raw.empty else math.nan,
            "raw_max_sum": float(raw.max()) if not raw.empty else math.nan,
            "raw_within_pm_0p02_share": float(((raw - 1.0).abs() <= 0.02).mean()) if not raw.empty else math.nan,
            "norm_valid_city_n": int(norm.notna().sum()) if not norm.empty else 0,
            "norm_mean_sum": float(norm.mean()) if not norm.empty else math.nan,
            "norm_min_sum": float(norm.min()) if not norm.empty else math.nan,
            "norm_max_sum": float(norm.max()) if not norm.empty else math.nan,
        }

    simplex_diag = pd.DataFrame(
        [
            _stage_diag("early_share", "early"),
            _stage_diag("mid_share", "mid"),
            _stage_diag("late_share", "late"),
        ]
    )

    # Provide interpretable city-stage examples for the largest raw-sum deviations.
    raw_long = raw_city.melt(id_vars=city_cols, value_vars=stages, var_name="stage", value_name="raw_sum")
    norm_long = norm_city.melt(id_vars=city_cols, value_vars=stages, var_name="stage", value_name="norm_sum")
    ex = raw_long.merge(norm_long, on=city_cols + ["stage"], how="left")
    ex["stage"] = (
        ex["stage"]
        .astype(str)
        .str.replace("_share", "", regex=False)
        .str.replace("early", "early", regex=False)
        .str.replace("mid", "mid", regex=False)
        .str.replace("late", "late", regex=False)
    )
    ex["deviation_from_1"] = (pd.to_numeric(ex["raw_sum"], errors="coerce") - 1.0).abs()
    if "city_name" in ex.columns and "state_abbr" in ex.columns:
        ex["city"] = ex["city_name"].astype(str) + ", " + ex["state_abbr"].astype(str).str.upper()
    else:
        ex["city"] = ex["city_key"].astype(str)
    ex = ex.sort_values("deviation_from_1", ascending=False).head(8).reset_index(drop=True)
    ex_tbl = ex[["city", "stage", "raw_sum", "norm_sum", "deviation_from_1"]].copy()

    return dn, simplex_diag, ex_tbl


def _plot_phase_archetypes(phases: pd.DataFrame, out: Path) -> None:
    if phases.empty:
        return
    d = phases.copy()
    d["phase_label"] = d["phase_label"].astype(str).str.strip().str.lower()
    d = d[d["phase_label"] != ""]
    if d.empty:
        return
    counts = d["phase_label"].value_counts().head(12)
    pretty = {k: k.replace("_", " ") for k in counts.index.tolist()}
    fig, ax = plt.subplots(figsize=(10.8, 4.8))
    ax.barh([pretty[k] for k in counts.index.tolist()[::-1]], counts.values[::-1], color="#76b7b2")
    ax.set_xlabel("Count")
    ax.set_ylabel("Phase label")
    ax.set_title("Most common evolution phase labels")
    ax.grid(axis="x", alpha=0.25, linestyle="--")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220)
    plt.close(fig)


def _plot_wave_types(waves: pd.DataFrame, out: Path) -> None:
    if waves.empty:
        return
    d = waves.copy()
    d["change_type"] = d["change_type"].astype(str).str.strip().str.lower()
    if d.empty:
        return
    counts = d["change_type"].value_counts()
    fig, ax = plt.subplots(figsize=(8.4, 4.5))
    ax.bar(counts.index.tolist(), counts.values, color="#f28e2b")
    ax.set_xlabel("Wave change type")
    ax.set_ylabel("Count")
    ax.set_title("Amendment wave type counts")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220)
    plt.close(fig)


def _plot_complexity_trends(cpx: pd.DataFrame, out: Path) -> None:
    if cpx.empty:
        return
    keys = [
        "regulatory_complexity_trend",
        "procedural_intensity_trend",
        "dimensional_specificity_trend",
    ]
    pretty_metric = {
        "regulatory_complexity_trend": "Regulatory complexity",
        "procedural_intensity_trend": "Procedural intensity",
        "dimensional_specificity_trend": "Dimensional specificity",
    }
    rows = []
    for k in keys:
        if k not in cpx.columns:
            continue
        s = cpx[k].astype(str).str.strip().str.lower()
        n = int(len(s))
        up_n = int((s == "up").sum())
        flat_n = int((s == "flat").sum())
        down_n = int((s == "down").sum())
        other_n = int((~s.isin(["up", "flat", "down"])).sum())
        rows.append(
            {
                "metric": k,
                "metric_pretty": pretty_metric.get(k, k.replace("_", " ")),
                "n_cities": n,
                "up_n": up_n,
                "flat_n": flat_n,
                "down_n": down_n,
                "other_n": other_n,
                "up_pct": (100.0 * up_n / n) if n > 0 else 0.0,
                "flat_pct": (100.0 * flat_n / n) if n > 0 else 0.0,
                "down_pct": (100.0 * down_n / n) if n > 0 else 0.0,
                "other_pct": (100.0 * other_n / n) if n > 0 else 0.0,
            }
        )
    if not rows:
        return
    d = pd.DataFrame(rows)

    cols = ["up_pct", "flat_pct", "down_pct", "other_pct"]
    count_cols = ["up_n", "flat_n", "down_n", "other_n"]
    col_labels = ["Up", "Flat", "Down", "Mixed/uncertain"]
    mat = d[cols].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(10.4, 3.9))
    im = ax.imshow(mat, cmap="YlGnBu", vmin=0.0, vmax=100.0, aspect="auto")

    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels)
    ax.set_yticks(np.arange(len(d)))
    ax.set_yticklabels(d["metric_pretty"].tolist())
    ax.set_xlabel("Trend direction")
    ax.set_ylabel("Complexity metric")
    ax.set_title("Complexity trend direction by metric (share of cities)")

    # Cell annotations include both percent and count for direct interpretability.
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            pct = float(mat[i, j])
            n = int(d.iloc[i][count_cols[j]])
            txt = f"{pct:.0f}%\\n(n={n})"
            color = "white" if pct >= 55.0 else "black"
            ax.text(j, i, txt, ha="center", va="center", color=color, fontsize=9)

    cbar = fig.colorbar(im, ax=ax, shrink=0.95)
    cbar.set_label("% of cities")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220)
    plt.close(fig)


def _plot_full_vs_amendment_over_time(issue_df: pd.DataFrame, out: Path) -> None:
    if issue_df.empty:
        return
    d = issue_df.copy()
    d["year"] = pd.to_datetime(d["issue_date"], errors="coerce").dt.year
    d = d[d["year"].notna()].copy()
    if d.empty:
        return
    d["year"] = d["year"].astype(int)
    d["decade"] = (d["year"] // 10) * 10

    cls = d["classification_label"].astype(str).str.strip().str.lower()
    d["is_full"] = (cls == "full_ordinance").astype(int)
    d["is_amendment"] = cls.str.startswith("amendment").astype(int)

    by_dec = (
        d.groupby("decade", dropna=False)[["is_full", "is_amendment"]]
        .sum()
        .reset_index()
        .sort_values("decade")
    )
    if by_dec.empty:
        return
    by_dec["total"] = by_dec["is_full"] + by_dec["is_amendment"]
    by_dec["full_share"] = by_dec["is_full"] / by_dec["total"].replace(0, np.nan)
    by_dec["amend_share"] = by_dec["is_amendment"] / by_dec["total"].replace(0, np.nan)

    x = np.arange(len(by_dec))
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8))

    axes[0].bar(x, by_dec["is_amendment"], color="#4e79a7", label="Amendments")
    axes[0].bar(x, by_dec["is_full"], bottom=by_dec["is_amendment"], color="#f28e2b", label="Full ordinances")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"{int(d)}s" for d in by_dec["decade"]], rotation=30, ha="right")
    axes[0].set_xlabel("Decade")
    axes[0].set_ylabel("Issue count")
    axes[0].set_title("Counts by decade")
    axes[0].grid(axis="y", alpha=0.25, linestyle="--")
    axes[0].legend(frameon=False, fontsize=9)

    axes[1].plot(by_dec["decade"], by_dec["full_share"], marker="o", color="#f28e2b", label="Full share")
    axes[1].plot(by_dec["decade"], by_dec["amend_share"], marker="o", color="#4e79a7", label="Amendment share")
    axes[1].set_ylim(0, 1.02)
    axes[1].set_xlabel("Decade")
    axes[1].set_ylabel("Share of labeled issues")
    axes[1].set_title("Composition by decade")
    axes[1].grid(axis="both", alpha=0.25, linestyle="--")
    axes[1].legend(frameon=False, fontsize=9)

    fig.suptitle("Full ordinances vs amendments over time", fontsize=13)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220)
    plt.close(fig)


def _plot_first_full_status_by_region(first: pd.DataFrame, out: Path) -> None:
    if first.empty:
        return
    d = first.copy()
    d["region"] = d["region"].astype(str)
    d["first_full_status"] = d["first_full_status"].astype(str).str.strip().str.lower()
    pv = d.groupby(["region", "first_full_status"], dropna=False).size().unstack(fill_value=0)
    if pv.empty:
        return
    cols = [c for c in ["found", "not_found", "uncertain"] if c in pv.columns] + [c for c in pv.columns if c not in {"found", "not_found", "uncertain"}]
    pv = pv[cols]
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    pv.plot(kind="bar", stacked=True, ax=ax, colormap="tab20")
    ax.set_xlabel("Region")
    ax.set_ylabel("City count")
    ax.set_title("First full-ordinance status by region")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220)
    plt.close(fig)


def _plot_wave_types_by_group(waves: pd.DataFrame, out: Path) -> None:
    if waves.empty:
        return
    d = waves.copy()
    d["change_type"] = d["change_type"].astype(str).str.strip().str.lower()
    d["region"] = d["region"].astype(str)
    d["urbanicity_proxy"] = d["urbanicity_proxy"].astype(str)
    by_region = d.groupby(["region", "change_type"], dropna=False).size().unstack(fill_value=0)
    by_urb = d.groupby(["urbanicity_proxy", "change_type"], dropna=False).size().unstack(fill_value=0)
    if by_region.empty and by_urb.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.0))
    if not by_region.empty:
        by_region.plot(kind="bar", stacked=True, ax=axes[0], colormap="tab20")
        axes[0].set_title("Wave types by region")
        axes[0].set_xlabel("Region")
        axes[0].set_ylabel("Wave count")
        axes[0].tick_params(axis="x", rotation=30)
        axes[0].grid(axis="y", alpha=0.25, linestyle="--")
        axes[0].legend(frameon=False, fontsize=7)
    else:
        axes[0].axis("off")

    if not by_urb.empty:
        by_urb.plot(kind="bar", stacked=True, ax=axes[1], colormap="tab20")
        axes[1].set_title("Wave types by urbanicity")
        axes[1].set_xlabel("Urbanicity")
        axes[1].set_ylabel("Wave count")
        axes[1].tick_params(axis="x", rotation=30)
        axes[1].grid(axis="y", alpha=0.25, linestyle="--")
        axes[1].legend(frameon=False, fontsize=7)
    else:
        axes[1].axis("off")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220)
    plt.close(fig)


def _plot_panel_span(city_span: pd.DataFrame, out: Path) -> None:
    if city_span.empty or "span_years" not in city_span.columns:
        return
    d = city_span.copy()
    d["span_years"] = pd.to_numeric(d["span_years"], errors="coerce")
    d = d.dropna(subset=["span_years"])
    if d.empty:
        return
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    bins = np.linspace(0, max(1.0, float(d["span_years"].max())), 10)
    ax.hist(d["span_years"], bins=bins, color="#76b7b2", edgecolor="white")
    ax.set_xlabel("Panel span (years, min to max issue date)")
    ax.set_ylabel("Cities")
    ax.set_title("City panel time-span distribution")
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220)
    plt.close(fig)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build standalone full-newspaper longitudinal report.")
    ap.add_argument(
        "--run-dir",
        default="/Users/saulrichardson/projects/newspapers/newspaper-analysis/reports/runs/pi_v1_run60_fullnewspaper",
        help="Run directory containing longitudinal outputs.",
    )
    ap.add_argument(
        "--input-dir",
        default="",
        help="Longitudinal input directory (default: <run-dir>/longitudinal).",
    )
    ap.add_argument(
        "--output-dir",
        default="",
        help="Output report directory (default: <run-dir>/reports/fullnewspaper_standalone).",
    )
    ap.add_argument(
        "--title",
        default="Panel-Level Longitudinal Analysis of Municipal Zoning Text (20-City Full-Coverage Sample)",
    )
    ap.add_argument("--compile-pdf", action=argparse.BooleanOptionalAction, default=True)
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    in_dir = Path(args.input_dir).expanduser().resolve() if str(args.input_dir).strip() else (run_dir / "longitudinal")
    out_dir = Path(args.output_dir).expanduser().resolve() if str(args.output_dir).strip() else (run_dir / "reports" / "fullnewspaper_standalone")
    figs_dir = out_dir / "figures"
    tbls_dir = out_dir / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    figs_dir.mkdir(parents=True, exist_ok=True)
    tbls_dir.mkdir(parents=True, exist_ok=True)

    selection = _read_csv(in_dir / "pilot20_selection_quality.csv")
    summary = _read_csv(in_dir / "city_longitudinal_summary.csv")
    first = _read_csv(in_dir / "city_first_zoning_events.csv")
    phases = _read_csv(in_dir / "city_evolution_phases.csv")
    waves = _read_csv(in_dir / "city_amendment_waves.csv")
    cats = _read_csv(in_dir / "city_category_trajectory.csv")
    cpx = _read_csv(in_dir / "city_complexity_trajectory.csv")
    nums = _read_csv(in_dir / "city_numeric_trajectory.csv")
    selected_issues = _read_csv(run_dir / "panels" / "selected_panel_issues.csv")

    # Figures.
    fig_cov = figs_dir / "issue_coverage_by_city.png"
    fig_cat = figs_dir / "category_trajectory_profiles.png"
    fig_phase = figs_dir / "phase_label_counts.png"
    fig_wave = figs_dir / "amendment_wave_type_counts.png"
    fig_cpx = figs_dir / "complexity_trend_directions.png"
    fig_wave_group = figs_dir / "amendment_wave_types_by_region_urbanicity.png"
    fig_span = figs_dir / "panel_span_years_distribution.png"
    fig_mix_time = figs_dir / "full_vs_amendment_over_time.png"

    cats_norm, cat_simplex_diag_tbl, cat_norm_examples_tbl = _normalize_category_composition(cats)

    _plot_issue_coverage(selection, fig_cov)
    _plot_category_trajectory(cats_norm, fig_cat)
    _plot_phase_archetypes(phases, fig_phase)
    _plot_wave_types(waves, fig_wave)
    _plot_complexity_trends(cpx, fig_cpx)
    _plot_wave_types_by_group(waves, fig_wave_group)
    _plot_full_vs_amendment_over_time(selected_issues, fig_mix_time)

    # Tables.
    coverage_tbl = selection.copy()
    if not coverage_tbl.empty:
        coverage_tbl = coverage_tbl.sort_values("selected_issue_count", ascending=False)

    first_tbl = first.copy()
    if not first_tbl.empty:
        first_tbl = first_tbl[
            [
                "city_key",
                "region",
                "urbanicity_proxy",
                "first_zoning_issue_date",
                "first_zoning_signal_type",
                "first_full_issue_date",
                "first_full_status",
            ]
        ].sort_values(["first_zoning_issue_date", "city_key"], na_position="last")

    phase_tbl = pd.DataFrame()
    phase_detail_tbl = pd.DataFrame()
    phase_example_tbl = pd.DataFrame()
    if not phases.empty:
        p = phases.assign(phase_label=phases["phase_label"].astype(str).str.strip().str.lower()).query("phase_label != ''").copy()
        phase_tbl = (
            p.groupby("phase_label", dropna=False)
            .agg(
                phase_rows=("phase_label", "size"),
                city_count=("city_key", "nunique"),
            )
            .reset_index()
            .sort_values("phase_rows", ascending=False)
        )
        phase_total = float(phase_tbl["phase_rows"].sum()) if not phase_tbl.empty else float("nan")
        city_total = float(summary["city_key"].nunique()) if not summary.empty else float("nan")
        if phase_total and pd.notna(phase_total):
            phase_tbl["phase_row_share"] = phase_tbl["phase_rows"] / phase_total
        if city_total and pd.notna(city_total):
            phase_tbl["city_share"] = phase_tbl["city_count"] / city_total

        # Granular production diagnostics for phase labels.
        p["start_year"] = pd.to_datetime(p["start_issue_date"], errors="coerce").dt.year
        p["end_year"] = pd.to_datetime(p["end_issue_date"], errors="coerce").dt.year
        p["phase_order_num"] = pd.to_numeric(p["phase_order"], errors="coerce")

        def _token_count(v: Any) -> int:
            txt = str(v or "").strip()
            if not txt:
                return 0
            return len([x for x in txt.split("|") if str(x).strip()])

        def _first_token(v: Any) -> str:
            txt = str(v or "").strip()
            if not txt:
                return ""
            for tok in txt.split("|"):
                t = str(tok).strip()
                if t:
                    return t
            return ""

        p["dominant_cat_count"] = p["dominant_categories"].map(_token_count)
        p["evidence_issue_count"] = p["evidence_issue_ids"].map(_token_count)
        p["example_evidence_issue_id"] = p["evidence_issue_ids"].map(_first_token)
        p["has_evidence"] = (p["evidence_issue_count"] > 0).astype(float)

        phase_detail_tbl = (
            p.groupby("phase_label", dropna=False)
            .agg(
                phase_rows=("phase_label", "size"),
                cities_with_label=("city_key", "nunique"),
                mean_phase_order=("phase_order_num", "mean"),
                median_start_year=("start_year", "median"),
                median_end_year=("end_year", "median"),
                mean_dominant_category_count=("dominant_cat_count", "mean"),
                mean_evidence_issue_count=("evidence_issue_count", "mean"),
                evidence_present_share=("has_evidence", "mean"),
            )
            .reset_index()
            .sort_values("phase_rows", ascending=False)
        )

        # One illustrative phase segment per label with direct issue-id trace.
        phase_example_tbl = (
            p.sort_values(["phase_label", "city_key", "phase_order_num"], na_position="last")
            .groupby("phase_label", as_index=False)
            .head(1)
            .loc[
                :,
                [
                    "phase_label",
                    "city_key",
                    "phase_order",
                    "start_year",
                    "end_year",
                    "dominant_categories",
                    "example_evidence_issue_id",
                ],
            ]
            .rename(
                columns={
                    "city_key": "example_city",
                    "phase_order": "example_phase_order",
                    "dominant_categories": "example_dominant_categories",
                }
            )
            .sort_values("phase_label")
            .reset_index(drop=True)
        )

    phase_legend_tbl = pd.DataFrame(
        [{"phase_label": k, "meaning": v} for k, v in _PHASE_LABEL_MEANING.items()]
    )

    wave_tbl = pd.DataFrame()
    if not waves.empty:
        wave_tbl = (
            waves.assign(change_type=waves["change_type"].astype(str).str.strip().str.lower())
            .groupby(["change_type"], dropna=False)
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
        )

    cpx_tbl = pd.DataFrame()
    if not cpx.empty:
        metric_pretty = {
            "regulatory_complexity_trend": "regulatory_complexity",
            "procedural_intensity_trend": "procedural_intensity",
            "dimensional_specificity_trend": "dimensional_specificity",
        }
        rows = []
        for metric in [
            "regulatory_complexity_trend",
            "procedural_intensity_trend",
            "dimensional_specificity_trend",
        ]:
            if metric not in cpx.columns:
                continue
            s = cpx[metric].astype(str).str.strip().str.lower()
            n = int(len(s))
            up_n = int((s == "up").sum())
            flat_n = int((s == "flat").sum())
            down_n = int((s == "down").sum())
            other_n = int((~s.isin(["up", "flat", "down"])).sum())
            rows.append(
                {
                    "metric": metric_pretty.get(metric, metric),
                    "n_cities": n,
                    "up_n": up_n,
                    "flat_n": flat_n,
                    "down_n": down_n,
                    "other_n": other_n,
                    "up_pct": (100.0 * up_n / n) if n > 0 else 0.0,
                    "flat_pct": (100.0 * flat_n / n) if n > 0 else 0.0,
                    "down_pct": (100.0 * down_n / n) if n > 0 else 0.0,
                    "other_pct": (100.0 * other_n / n) if n > 0 else 0.0,
                }
            )
        cpx_tbl = pd.DataFrame(rows)

    num_tbl = pd.DataFrame()
    if not nums.empty:
        metric_order = ["lot_size", "height", "setback", "parking"]
        direction_order = ["up", "flat", "down", "mixed", "uncertain"]
        nraw = nums.copy()
        nraw["metric"] = nraw["metric"].astype(str).str.strip().str.lower()
        nraw["direction"] = nraw["direction"].astype(str).str.strip().str.lower()
        nraw = nraw[nraw["metric"].isin(metric_order)].copy()
        if not nraw.empty:
            metric_pretty = {
                "lot_size": "lot size",
                "height": "height",
                "setback": "setback",
                "parking": "parking",
            }
            ct = (
                nraw.groupby(["metric", "direction"], dropna=False)
                .size()
                .unstack(fill_value=0)
            )
            for d in direction_order:
                if d not in ct.columns:
                    ct[d] = 0
            ct = ct[direction_order]
            ct = ct.reindex(metric_order).fillna(0).astype(int)
            n_cities = nraw.groupby("metric").size().reindex(metric_order).fillna(0).astype(int)
            num_tbl = ct.reset_index()
            num_tbl["n_cities"] = num_tbl["metric"].map(n_cities.to_dict()).fillna(0).astype(int)
            num_tbl = num_tbl[
                ["metric", "n_cities", "up", "flat", "down", "mixed", "uncertain"]
            ]
            num_tbl["metric"] = num_tbl["metric"].map(metric_pretty).fillna(num_tbl["metric"])

    cat_tbl = pd.DataFrame()
    if not cats_norm.empty:
        tmp = cats_norm.copy()
        for c in ["early_share", "mid_share", "late_share"]:
            tmp[c] = pd.to_numeric(tmp[c], errors="coerce")
        cat_tbl = (
            tmp.groupby("category", dropna=False)[["early_share", "mid_share", "late_share"]]
            .mean()
            .reset_index()
            .assign(late_minus_early=lambda d: d["late_share"] - d["early_share"])
            .sort_values("late_share", ascending=False)
        )
        cat_tbl["early_pct"] = cat_tbl["early_share"] * 100.0
        cat_tbl["mid_pct"] = cat_tbl["mid_share"] * 100.0
        cat_tbl["late_pct"] = cat_tbl["late_share"] * 100.0
        cat_tbl["delta_pp"] = cat_tbl["late_minus_early"] * 100.0

    city_span = pd.DataFrame()
    if not selected_issues.empty:
        s = selected_issues.copy()
        s["issue_date_parsed"] = pd.to_datetime(s["issue_date"], errors="coerce")
        city_span = (
            s.groupby(["city_key", "city_name", "state_abbr", "region", "urbanicity_proxy"], dropna=False)
            .agg(
                issues=("issue_id", "count"),
                min_issue_date=("issue_date_parsed", "min"),
                max_issue_date=("issue_date_parsed", "max"),
            )
            .reset_index()
        )
        city_span["span_days"] = (city_span["max_issue_date"] - city_span["min_issue_date"]).dt.days
        city_span["span_years"] = city_span["span_days"] / 365.25
        city_span["min_year"] = city_span["min_issue_date"].dt.year
        city_span["max_year"] = city_span["max_issue_date"].dt.year
        _plot_panel_span(city_span, fig_span)

    region_urb_counts = pd.DataFrame()
    if not selection.empty:
        region_urb_counts = (
            selection.groupby(["region", "urbanicity_proxy"], dropna=False)
            .size()
            .reset_index(name="city_count")
            .sort_values(["region", "urbanicity_proxy"])
        )

    first_signal_region = pd.DataFrame()
    if not first.empty:
        first_signal_region = (
            first.assign(first_zoning_signal_type=first["first_zoning_signal_type"].astype(str).str.strip().str.lower())
            .groupby(["region", "first_zoning_signal_type"], dropna=False)
            .size()
            .reset_index(name="count")
            .sort_values(["region", "count"], ascending=[True, False])
        )

    first_full_region_urb = pd.DataFrame()
    if not first.empty:
        first_full_region_urb = (
            first.assign(first_full_status=first["first_full_status"].astype(str).str.strip().str.lower())
            .groupby(["region", "urbanicity_proxy", "first_full_status"], dropna=False)
            .size()
            .reset_index(name="count")
            .sort_values(["region", "urbanicity_proxy", "count"], ascending=[True, True, False])
        )

    wave_region_urb = pd.DataFrame()
    if not waves.empty:
        wave_region_urb = (
            waves.assign(change_type=waves["change_type"].astype(str).str.strip().str.lower())
            .groupby(["region", "urbanicity_proxy", "change_type"], dropna=False)
            .size()
            .reset_index(name="count")
            .sort_values(["region", "urbanicity_proxy", "count"], ascending=[True, True, False])
        )

    category_region_tbl = pd.DataFrame()
    if not cats_norm.empty:
        ctmp = cats_norm.copy()
        for c in ["early_share", "late_share"]:
            ctmp[c] = pd.to_numeric(ctmp[c], errors="coerce")
        category_region_tbl = (
            ctmp.groupby(["region", "category"], dropna=False)[["early_share", "late_share"]]
            .mean()
            .reset_index()
            .assign(late_minus_early=lambda d: d["late_share"] - d["early_share"])
            .sort_values(["region", "late_share"], ascending=[True, False])
        )

    full_amend_decade = pd.DataFrame()
    if not selected_issues.empty:
        s = selected_issues.copy()
        s["year"] = pd.to_datetime(s["issue_date"], errors="coerce").dt.year
        s = s[s["year"].notna()].copy()
        s["year"] = s["year"].astype(int)
        s["decade"] = (s["year"] // 10) * 10
        cls = s["classification_label"].astype(str).str.strip().str.lower()
        s["amendment_substantial"] = (cls == "amendment_substantial").astype(int)
        s["amendment_targeted"] = (cls == "amendment_targeted").astype(int)
        s["full_ordinance"] = (cls == "full_ordinance").astype(int)
        full_amend_decade = (
            s.groupby("decade", dropna=False)[["full_ordinance", "amendment_substantial", "amendment_targeted"]]
            .sum()
            .reset_index()
            .sort_values("decade")
        )
        full_amend_decade["amendments_total"] = full_amend_decade["amendment_substantial"] + full_amend_decade["amendment_targeted"]
        full_amend_decade["total"] = full_amend_decade["full_ordinance"] + full_amend_decade["amendments_total"]
        full_amend_decade["full_share"] = full_amend_decade["full_ordinance"] / full_amend_decade["total"].replace(0, np.nan)
        full_amend_decade["amend_share"] = full_amend_decade["amendments_total"] / full_amend_decade["total"].replace(0, np.nan)

    coverage_disp = coverage_tbl.rename(
        columns={
            "city_key": "city",
            "urbanicity_proxy": "urbanicity",
            "selected_issue_count": "issues",
            "median_issue_text_chars": "median_chars",
        }
    )
    _write_table_tex(
        tbls_dir / "coverage_by_city.tex",
        coverage_disp,
        ["city", "region", "urbanicity", "issues", "median_chars"],
        float_cols=["median_chars"],
    )
    first_disp = first_tbl.rename(
        columns={
            "city_key": "city",
            "urbanicity_proxy": "urbanicity",
            "first_zoning_issue_date": "first_signal_date",
            "first_zoning_signal_type": "first_signal_type",
            "first_full_issue_date": "first_full_date",
            "first_full_status": "first_full_status",
        }
    )
    _write_table_tex(
        tbls_dir / "first_events.tex",
        first_disp.head(20),
        [
            "city",
            "region",
            "urbanicity",
            "first_signal_date",
            "first_signal_type",
            "first_full_date",
            "first_full_status",
        ],
    )
    _write_table_tex(
        tbls_dir / "phase_label_counts.tex",
        phase_tbl.rename(columns={"phase_label": "phase"}),
        ["phase", "phase_rows", "city_count", "phase_row_share", "city_share"],
        float_cols=["phase_row_share", "city_share"],
    )
    _write_table_tex(
        tbls_dir / "phase_label_detail.tex",
        phase_detail_tbl,
        [
            "phase_label",
            "phase_rows",
            "cities_with_label",
            "mean_phase_order",
            "median_start_year",
            "median_end_year",
            "mean_dominant_category_count",
            "mean_evidence_issue_count",
            "evidence_present_share",
        ],
        float_cols=[
            "mean_phase_order",
            "median_start_year",
            "median_end_year",
            "mean_dominant_category_count",
            "mean_evidence_issue_count",
            "evidence_present_share",
        ],
    )
    _write_table_tex(
        tbls_dir / "phase_label_examples.tex",
        phase_example_tbl,
        [
            "phase_label",
            "example_city",
            "example_phase_order",
            "start_year",
            "end_year",
            "example_dominant_categories",
            "example_evidence_issue_id",
        ],
    )
    _write_table_tex(tbls_dir / "phase_label_legend.tex", phase_legend_tbl, ["phase_label", "meaning"])
    _write_table_tex(tbls_dir / "amendment_wave_counts.tex", wave_tbl.rename(columns={"change_type": "wave_type"}), ["wave_type", "count"])
    _write_table_tex(
        tbls_dir / "full_vs_amendment_decade.tex",
        full_amend_decade,
        [
            "decade",
            "full_ordinance",
            "amendment_substantial",
            "amendment_targeted",
            "amendments_total",
            "full_share",
            "amend_share",
        ],
        float_cols=["full_share", "amend_share"],
    )
    _write_table_tex(
        tbls_dir / "region_urbanicity_city_counts.tex",
        region_urb_counts.rename(columns={"urbanicity_proxy": "urbanicity"}),
        ["region", "urbanicity", "city_count"],
    )
    _write_table_tex(
        tbls_dir / "first_signal_by_region.tex",
        first_signal_region.rename(columns={"first_zoning_signal_type": "first_signal_type"}),
        ["region", "first_signal_type", "count"],
    )
    _write_table_tex(
        tbls_dir / "first_full_by_region_urbanicity.tex",
        first_full_region_urb.rename(columns={"urbanicity_proxy": "urbanicity"}),
        ["region", "urbanicity", "first_full_status", "count"],
    )
    _write_table_tex(
        tbls_dir / "wave_types_by_region_urbanicity.tex",
        wave_region_urb.rename(columns={"urbanicity_proxy": "urbanicity", "change_type": "wave_type"}).head(20),
        ["region", "urbanicity", "wave_type", "count"],
    )
    _write_table_tex(
        tbls_dir / "complexity_trend_shares.tex",
        cpx_tbl,
        ["metric", "n_cities", "up_n", "up_pct", "flat_n", "flat_pct", "down_n", "down_pct", "other_n", "other_pct"],
        float_cols=["up_pct", "flat_pct", "down_pct", "other_pct"],
    )
    _write_table_tex(
        tbls_dir / "numeric_direction_counts.tex",
        num_tbl,
        ["metric", "n_cities", "up", "flat", "down", "mixed", "uncertain"],
    )
    _write_table_tex(
        tbls_dir / "city_panel_spans.tex",
        city_span.rename(columns={"city_key": "city", "urbanicity_proxy": "urbanicity"}).sort_values("span_years", ascending=False)
        if not city_span.empty
        else city_span,
        ["city", "region", "urbanicity", "issues", "min_year", "max_year", "span_years"],
        float_cols=["span_years"],
    )
    _write_table_tex(
        tbls_dir / "category_by_region_late_share.tex",
        category_region_tbl.head(20),
        ["region", "category", "early_share", "late_share", "late_minus_early"],
        float_cols=["early_share", "late_share", "late_minus_early"],
    )
    _write_table_tex(
        tbls_dir / "category_means.tex",
        cat_tbl.head(12),
        ["category", "early_pct", "mid_pct", "late_pct", "delta_pp"],
        float_cols=["early_pct", "mid_pct", "late_pct", "delta_pp"],
    )
    _write_table_tex(
        tbls_dir / "category_simplex_diagnostics.tex",
        cat_simplex_diag_tbl,
        [
            "stage",
            "raw_mean_sum",
            "raw_p10_sum",
            "raw_median_sum",
            "raw_p90_sum",
            "raw_min_sum",
            "raw_max_sum",
            "raw_within_pm_0p02_share",
            "norm_valid_city_n",
            "norm_mean_sum",
            "norm_min_sum",
            "norm_max_sum",
        ],
        float_cols=[
            "raw_mean_sum",
            "raw_p10_sum",
            "raw_median_sum",
            "raw_p90_sum",
            "raw_min_sum",
            "raw_max_sum",
            "raw_within_pm_0p02_share",
            "norm_mean_sum",
            "norm_min_sum",
            "norm_max_sum",
        ],
    )
    _write_table_tex(
        tbls_dir / "category_simplex_examples.tex",
        cat_norm_examples_tbl,
        ["city", "stage", "raw_sum", "norm_sum", "deviation_from_1"],
        float_cols=["raw_sum", "norm_sum", "deviation_from_1"],
    )

    # Metrics for narrative.
    n_city = int(summary["city_key"].nunique()) if not summary.empty else 0
    total_issues = int(len(selected_issues)) if not selected_issues.empty else 0
    mean_issues_city = float(pd.to_numeric(selection.get("selected_issue_count"), errors="coerce").mean()) if not selection.empty else math.nan
    overflow_count = int(pd.to_numeric(summary.get("overflow_applied"), errors="coerce").fillna(0).sum()) if not summary.empty else 0
    clipped_issue_total = int(pd.to_numeric(summary.get("clipped_issue_count"), errors="coerce").fillna(0).sum()) if not summary.empty else 0
    mean_span_years = float(pd.to_numeric(city_span.get("span_years"), errors="coerce").mean()) if not city_span.empty else math.nan

    first_signal_types = first["first_zoning_signal_type"].astype(str).str.strip().str.lower().value_counts() if not first.empty else pd.Series(dtype=int)
    first_full_statuses = first["first_full_status"].astype(str).str.strip().str.lower().value_counts() if not first.empty else pd.Series(dtype=int)
    cat_stage_mean_sum_early = float(cat_tbl["early_share"].sum()) if not cat_tbl.empty and "early_share" in cat_tbl.columns else math.nan
    cat_stage_mean_sum_mid = float(cat_tbl["mid_share"].sum()) if not cat_tbl.empty and "mid_share" in cat_tbl.columns else math.nan
    cat_stage_mean_sum_late = float(cat_tbl["late_share"].sum()) if not cat_tbl.empty and "late_share" in cat_tbl.columns else math.nan

    # TeX document.
    report_date = dt.datetime.now().strftime("%Y-%m-%d")
    doc: list[str] = []
    doc.append("\\documentclass[11pt]{article}")
    doc.append("\\usepackage[margin=1in]{geometry}")
    doc.append("\\usepackage{graphicx}")
    doc.append("\\usepackage{booktabs}")
    doc.append("\\usepackage{float}")
    doc.append("\\usepackage{hyperref}")
    doc.append(f"\\title{{{_tex_escape(args.title)}}}")
    doc.append("\\author{Automated longitudinal pipeline}")
    doc.append(f"\\date{{{_tex_escape(report_date)}}}")
    doc.append("\\begin{document}")
    doc.append("\\maketitle")
    doc.append("\\tableofcontents")
    doc.append("\\clearpage")

    doc.append("\\section{High-level measurement strategy}")
    doc.append(
        "This report summarizes panel-level longitudinal zoning measurements for a 20-city sample built from full local transcript coverage."
    )
    doc.append("\\begin{enumerate}")
    doc.append(
        "\\item \\textbf{Panel assembly.} For each city, we assemble all locally available issue transcripts in chronological order."
    )
    doc.append(
        "\\item \\textbf{Single panel-level extraction.} We send one structured LLM request per city panel, so each extraction sees the full within-city sequence."
    )
    doc.append(
        "\\item \\textbf{Structured outputs.} Each city returns first zoning signal, first full ordinance status, evolution phases, amendment waves, category trajectory, complexity trajectory, and numeric trajectory."
    )
    doc.append(
        "\\item \\textbf{Traceability checks.} Phase and amendment-wave outputs include evidence issue IDs so claims remain tied to specific transcript entries."
    )
    doc.append("\\end{enumerate}")

    doc.append("\\section{Coverage and run integrity}")
    doc.append("\\begin{itemize}")
    doc.append(f"\\item Cities analyzed: {n_city}")
    doc.append(f"\\item Total issues in panels: {total_issues}")
    doc.append(f"\\item Mean issues per city: {mean_issues_city:.2f}" if pd.notna(mean_issues_city) else "\\item Mean issues per city: NA")
    doc.append(f"\\item Mean panel span (years): {mean_span_years:.2f}" if pd.notna(mean_span_years) else "\\item Mean panel span (years): NA")
    if overflow_count == 0:
        doc.append("\\item Context clipping triggered in 0 cities.")
    else:
        doc.append(f"\\item Context clipping triggered in {overflow_count} cities (clipped issues total: {clipped_issue_total}).")
    doc.append("\\end{itemize}")
    if fig_cov.is_file():
        doc.append(
            "\\textit{Method note.} This figure uses observed panel metadata only: one bar per city with the total number of issues in that city panel."
        )
        doc.append(
            "\\begin{figure}[H]\\centering\\includegraphics[width=0.90\\textwidth]{figures/issue_coverage_by_city.png}"
            "\\caption{Issue coverage per city panel.}\\end{figure}"
        )
    if fig_span.is_file():
        doc.append(
            "\\textit{Method note.} For each city, panel span is measured from the first issue date to the last issue date, then summarized as a distribution."
        )
        doc.append(
            "\\begin{figure}[H]\\centering\\includegraphics[width=0.78\\textwidth]{figures/panel_span_years_distribution.png}"
            "\\caption{Distribution of panel time spans by city.}\\end{figure}"
        )
    doc.append(
        "\\begin{table}[H]\\centering\\caption{City-level panel coverage}"
        "\\resizebox{\\textwidth}{!}{\\input{tables/coverage_by_city.tex}}\\end{table}"
    )
    doc.append(
        "\\begin{table}[H]\\centering\\caption{City counts by region and urbanicity}"
        "\\resizebox{0.78\\textwidth}{!}{\\input{tables/region_urbanicity_city_counts.tex}}\\end{table}"
    )
    doc.append(
        "\\begin{table}[H]\\centering\\caption{City panel spans}"
        "\\resizebox{\\textwidth}{!}{\\input{tables/city_panel_spans.tex}}\\end{table}"
    )

    doc.append("\\section{When zoning first appears}")
    doc.append(
        "\\textbf{How these results are produced.} For each city, the full chronological transcript panel is read as one sequence, "
        "then the extraction identifies the earliest zoning-related signal and earliest full-ordinance event with supporting evidence references. "
        "The figure and tables in this section aggregate those city-level first-event outputs."
    )
    mention_n = int(first_signal_types.get("mention", 0))
    amend_n = int(first_signal_types.get("amendment", 0))
    full_n = int(first_signal_types.get("full_ordinance", 0))
    found_n = int(first_full_statuses.get("found", 0))
    not_found_n = int(first_full_statuses.get("not_found", 0))
    uncertain_n = int(first_full_statuses.get("uncertain", 0))
    doc.append("\\begin{itemize}")
    doc.append(f"\\item First-signal type counts: mention={mention_n}, amendment={amend_n}, full\\_ordinance={full_n}.")
    doc.append(f"\\item First-full status counts: found={found_n}, not\\_found={not_found_n}, uncertain={uncertain_n}.")
    doc.append("\\end{itemize}")
    if fig_mix_time.is_file():
        doc.append(
            "\\textit{Method note.} This figure pools issue-level ordinance labels across all cities and bins them by decade to compare full ordinances versus amendments over time."
        )
        doc.append(
            "\\begin{figure}[H]\\centering\\includegraphics[width=0.94\\textwidth]{figures/full_vs_amendment_over_time.png}"
            "\\caption{Distribution over time of full ordinances versus amendments (decade bins).}\\end{figure}"
        )
    doc.append(
        "\\textit{Method note.} The next table lists one city per row, showing the earliest zoning signal and earliest full-ordinance event identified from that city's full timeline."
    )
    doc.append(
        "\\begin{table}[H]\\centering\\caption{Illustrative first-event rows}"
        "\\resizebox{\\textwidth}{!}{\\input{tables/first_events.tex}}\\end{table}"
    )
    doc.append(
        "\\textit{Method note.} First-signal types are then grouped by region to show how the first observed zoning-related event differs across regions."
    )
    doc.append(
        "\\begin{table}[H]\\centering\\caption{First-signal types by region}"
        "\\resizebox{0.74\\textwidth}{!}{\\input{tables/first_signal_by_region.tex}}\\end{table}"
    )
    doc.append(
        "\\textit{Method note.} The decade table uses the same pooled issue labels as the time figure, reported as counts and shares by decade."
    )
    doc.append(
        "\\begin{table}[H]\\centering\\caption{Full ordinances vs amendments by decade}"
        "\\resizebox{\\textwidth}{!}{\\input{tables/full_vs_amendment_decade.tex}}\\end{table}"
    )

    doc.append("\\section{How zoning content evolves}")
    doc.append(
        "\\textbf{How these results are produced.} Each city timeline is summarized at three points in time: early, mid, and late. "
        "For each point, the model reports how much of the zoning text is about each fixed topic. "
        "Those topic shares are then averaged across cities to produce the section-level patterns."
    )
    if fig_cat.is_file():
        doc.append(
            "\\textit{Method note.} For each topic, this line chart compares the average share in the early, mid, and late portions of city timelines."
        )
        doc.append(
            "\\begin{figure}[H]\\centering\\includegraphics[width=0.95\\textwidth]{figures/category_trajectory_profiles.png}"
            "\\caption{Mean category shares across early/mid/late trajectory points.}\\end{figure}"
        )
    doc.append(
        "\\textit{Method note.} This table reports the same topic shares numerically and adds a simple late-minus-early difference column."
    )
    doc.append(
        "\\begin{table}[H]\\centering\\caption{Mean category shares by stage (percent) and late-minus-early change (pp)}"
        "\\resizebox{0.84\\textwidth}{!}{\\input{tables/category_means.tex}}\\end{table}"
    )
    doc.append(
        "\\textit{Method note.} Region patterns are computed by averaging city-level late-stage topic shares within each region."
    )
    doc.append(
        "\\begin{table}[H]\\centering\\caption{Category late-share patterns by region (head)}"
        "\\resizebox{\\textwidth}{!}{\\input{tables/category_by_region_late_share.tex}}\\end{table}"
    )

    doc.append("\\section{Phases, amendment waves, and complexity direction}")
    doc.append(
        "Phase labels come from one panel-level extraction request per city (all city issues in chronological order). "
        "Each returned phase segment must choose from a fixed canonical label set; the full label definitions are in the legend table below."
    )
    doc.append(
        "Production steps used for this report: "
        "(1) the model emits an ordered list of phases for each city; "
        "(2) each phase is mapped to the canonical label set used in this report "
        "(anything outside the set is assigned to the catch-all unclear category); "
        "(3) each extracted phase segment contributes one row to the phase-summary tables; "
        "(4) Figure 6 counts those rows by label. "
        "So the unit in Figure 6 is extracted phase segments, not cities."
    )
    doc.append(
        "\\textbf{How amendment-wave results are produced.} Within each city timeline, the extraction marks amendment waves "
        "as periods of meaningful change and assigns each wave a change type (for example expansion, tightening, clarification, or reclassification). "
        "Wave figures and tables then aggregate those wave-level labels across cities and by region/urbanicity."
    )
    doc.append(
        "\\textbf{How complexity and numeric-direction results are produced.} For each city, the extraction assigns a direction "
        "(up, flat, down, or mixed/uncertain) for each complexity dimension and each numeric constraint type. "
        "The heatmap and direction tables summarize those city-level direction assignments."
    )
    if fig_phase.is_file():
        doc.append(
            "\\textit{Method note.} Each city timeline is split into ordered phase segments; each segment gets one canonical phase label. "
            "This chart counts those segments by label."
        )
        doc.append(
            "\\begin{figure}[H]\\centering\\includegraphics[width=0.88\\textwidth]{figures/phase_label_counts.png}"
            "\\caption{Most frequent evolution phase labels.}\\end{figure}"
        )
    if fig_wave.is_file():
        doc.append(
            "\\textit{Method note.} Each extracted amendment wave contributes one observation with a wave type; this chart counts wave types across all cities."
        )
        doc.append(
            "\\begin{figure}[H]\\centering\\includegraphics[width=0.78\\textwidth]{figures/amendment_wave_type_counts.png}"
            "\\caption{Amendment wave type counts.}\\end{figure}"
        )
    if fig_wave_group.is_file():
        doc.append(
            "\\textit{Method note.} This figure uses the same wave observations, grouped by region and urbanicity, to show where different change types are concentrated."
        )
        doc.append(
            "\\begin{figure}[H]\\centering\\includegraphics[width=0.97\\textwidth]{figures/amendment_wave_types_by_region_urbanicity.png}"
            "\\caption{Amendment wave types by region and urbanicity.}\\end{figure}"
        )
    if fig_cpx.is_file():
        doc.append(
            "\\textit{Method note.} For each city, each complexity metric is assigned a direction (up/flat/down/mixed-uncertain). "
            "The heatmap reports the share of cities in each direction cell."
        )
        doc.append(
            "\\begin{figure}[H]\\centering\\includegraphics[width=0.86\\textwidth]{figures/complexity_trend_directions.png}"
            "\\caption{Complexity-direction heatmap. Read across each row: share of cities classified up, flat, down, or mixed/uncertain (with cell counts).}\\end{figure}"
        )
    doc.append(
        "\\textit{Method note.} The next tables provide detailed counts, coverage shares, and examples of phase-segment labeling output."
    )
    doc.append(
        "\\begin{table}[H]\\centering\\caption{Phase label counts (unit = extracted phase segment)}"
        "\\resizebox{0.86\\textwidth}{!}{\\input{tables/phase_label_counts.tex}}\\end{table}"
    )
    doc.append(
        "\\begin{table}[H]\\centering\\caption{Phase-label production detail (timing, order, and evidence granularity)}"
        "\\resizebox{\\textwidth}{!}{\\input{tables/phase_label_detail.tex}}\\end{table}"
    )
    doc.append(
        "\\begin{table}[H]\\centering\\caption{One illustrative extracted phase row per label (with example evidence issue id)}"
        "\\resizebox{\\textwidth}{!}{\\input{tables/phase_label_examples.tex}}\\end{table}"
    )
    doc.append(
        "\\begin{table}[H]\\centering\\caption{Phase label legend}"
        "\\resizebox{\\textwidth}{!}{\\input{tables/phase_label_legend.tex}}\\end{table}"
    )
    doc.append(
        "\\begin{table}[H]\\centering\\caption{Amendment wave type frequencies}"
        "\\resizebox{0.62\\textwidth}{!}{\\input{tables/amendment_wave_counts.tex}}\\end{table}"
    )
    doc.append(
        "\\begin{table}[H]\\centering\\caption{Wave types by region and urbanicity (head)}"
        "\\resizebox{\\textwidth}{!}{\\input{tables/wave_types_by_region_urbanicity.tex}}\\end{table}"
    )
    doc.append(
        "\\textit{Method note.} Complexity and numeric-direction tables summarize city-level direction assignments with counts (and percentages for complexity)."
    )
    doc.append(
        "\\begin{table}[H]\\centering\\caption{Complexity trend directions: counts and percentages by metric}"
        "\\resizebox{0.86\\textwidth}{!}{\\input{tables/complexity_trend_shares.tex}}\\end{table}"
    )
    doc.append(
        "\\begin{table}[H]\\centering\\small\\caption{Numeric trajectory direction matrix (city counts)}"
        "\\input{tables/numeric_direction_counts.tex}\\end{table}"
    )

    doc.append("\\end{document}")

    tex_path = out_dir / "report.tex"
    tex_path.write_text("\n".join(doc) + "\n", encoding="utf-8")

    makefile = [
        "LATEXMK ?= latexmk",
        "",
        "all: report.pdf",
        "",
        "report.pdf: report.tex",
        "\t$(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error report.tex",
        "",
        "clean:",
        "\t$(LATEXMK) -C",
        "",
        ".PHONY: all clean",
    ]
    (out_dir / "Makefile").write_text("\n".join(makefile) + "\n", encoding="utf-8")

    compiled_pdf = False
    if bool(args.compile_pdf):
        latexmk = shutil.which("latexmk")
        if latexmk:
            try:
                subprocess.run(
                    [latexmk, "-pdf", "-interaction=nonstopmode", "-halt-on-error", "report.tex"],
                    cwd=str(out_dir),
                    check=True,
                )
                compiled_pdf = True
            except Exception:
                compiled_pdf = False

    prov = {
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "run_dir": str(run_dir),
        "input_dir": str(in_dir),
        "output_dir": str(out_dir),
        "compiled_pdf": bool(compiled_pdf),
        "n_city": n_city,
        "total_issues": total_issues,
        "overflow_count": overflow_count,
    }
    (out_dir / "provenance.json").write_text(json.dumps(prov, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(
        "Done. "
        f"report_tex={tex_path} "
        f"report_pdf={(out_dir / 'report.pdf')} "
        f"compiled_pdf={compiled_pdf}"
    )


if __name__ == "__main__":
    main()
