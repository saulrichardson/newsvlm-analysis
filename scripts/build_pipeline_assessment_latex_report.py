#!/usr/bin/env python3
"""
Build a concise LaTeX assessment report for the zoning pipeline iterations.

The report answers:
  - What pipeline should we move forward with?
  - What changed over iterations?
  - What evidence supports the recommendation?
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import subprocess
from pathlib import Path

import pandas as pd

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


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


def _to_latex_table(df: pd.DataFrame, path: Path, *, index: bool = False, float_fmt: str = "%.3f") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if df.empty:
        _write_text(path, "\\begin{tabular}{l}\n\\toprule\nNo data available.\\\\\n\\bottomrule\n\\end{tabular}\n")
        return
    tex = df.to_latex(index=index, escape=True, float_format=float_fmt.__mod__, na_rep="")
    _write_text(path, tex)


def _clip_text(x: object, n: int = 120) -> str:
    s = str(x or "").replace("\n", " ").strip()
    if len(s) <= n:
        return s
    return s[: max(0, n - 3)] + "..."


def _latex_escape(x: object) -> str:
    s = str(x or "")
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for k, v in repl.items():
        s = s.replace(k, v)
    return s


def _render_fig_issue_counts(issue_counts: pd.DataFrame, out_path: Path) -> bool:
    if plt is None:
        return False
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if issue_counts.empty:
        plt.figure(figsize=(8, 4))
        plt.text(0.5, 0.5, "No issue coverage data", ha="center", va="center")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(out_path, dpi=180)
        plt.close()
        return True
    top = issue_counts.sort_values("issue_count", ascending=False).head(20).copy()
    plt.figure(figsize=(10, 6))
    plt.barh(top["city_key"], top["issue_count"], color="#2a9d8f")
    plt.gca().invert_yaxis()
    plt.title("Issue coverage per city panel (all available local issues)")
    plt.xlabel("Issue count")
    plt.ylabel("City")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
    return True


def _render_fig_shift_bucket_counts(reaudit_all: pd.DataFrame, out_path: Path) -> bool:
    if plt is None:
        return False
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if reaudit_all.empty or "manual_bucket" not in reaudit_all.columns:
        plt.figure(figsize=(8, 4))
        plt.text(0.5, 0.5, "No shift-bucket data", ha="center", va="center")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(out_path, dpi=180)
        plt.close()
        return True
    counts = reaudit_all["manual_bucket"].fillna("missing").value_counts().sort_values(ascending=True)
    plt.figure(figsize=(10, 5))
    plt.barh(counts.index.tolist(), counts.values.tolist(), color="#8ab17d")
    plt.title("Re-audit outcomes for rows originally labeled full ordinance")
    plt.xlabel("Count of rows")
    plt.ylabel("Re-audit bucket")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
    return True


def _render_fig_shift_city_stacked(reaudit_city: pd.DataFrame, out_path: Path) -> bool:
    if plt is None:
        return False
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if reaudit_city.empty:
        plt.figure(figsize=(8, 4))
        plt.text(0.5, 0.5, "No city re-audit data", ha="center", va="center")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(out_path, dpi=180)
        plt.close()
        return True
    cols = [
        "likely_full_zoning_ordinance",
        "likely_zoning_amendment",
        "likely_zoning_notice_hearing",
        "likely_non_zoning_notice_or_ordinance",
        "unclear_zoning_ordinance_like",
        "unclear",
    ]
    keep = [c for c in cols if c in reaudit_city.columns]
    if not keep:
        plt.figure(figsize=(8, 4))
        plt.text(0.5, 0.5, "No bucket columns found", ha="center", va="center")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(out_path, dpi=180)
        plt.close()
        return True

    top = reaudit_city.sort_values("original_full_rows", ascending=False).head(10).copy()
    x = range(len(top))
    bottom = [0.0] * len(top)
    colors = ["#2a9d8f", "#e76f51", "#457b9d", "#6c757d", "#f4a261", "#bdbdbd"]

    plt.figure(figsize=(12, 5.5))
    for i, c in enumerate(keep):
        vals = top[c].fillna(0).astype(float).tolist()
        plt.bar(x, vals, bottom=bottom, label=c, color=colors[i % len(colors)])
        bottom = [b + v for b, v in zip(bottom, vals)]
    plt.xticks(list(x), top["city_key"].tolist(), rotation=35, ha="right")
    plt.ylabel("Count of rows originally labeled full ordinance")
    plt.xlabel("City")
    plt.title("City-level composition after re-audit (top cities by original full count)")
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
    return True


def _render_fig_reaudit_decade(reaudit_decade: pd.DataFrame, out_path: Path) -> bool:
    if plt is None:
        return False
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if reaudit_decade.empty:
        plt.figure(figsize=(8, 4))
        plt.text(0.5, 0.5, "No re-audit decade data", ha="center", va="center")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(out_path, dpi=180)
        plt.close()
        return True
    x = reaudit_decade["decade"].astype(int).tolist()
    y_orig = reaudit_decade["orig_full"].astype(float).tolist()
    y_rev = reaudit_decade["revised_full"].astype(float).tolist()
    plt.figure(figsize=(10, 5))
    plt.plot(x, y_orig, marker="o", label="Original full-ordinance count", color="#264653")
    plt.plot(x, y_rev, marker="o", label="Re-audited full-ordinance count", color="#e76f51")
    plt.title("Full-ordinance counts by decade: original vs re-audit")
    plt.xlabel("Decade")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
    return True


def _render_fig_first_signal(city_summary: pd.DataFrame, out_path: Path) -> bool:
    if plt is None:
        return False
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if city_summary.empty or "first_signal_type" not in city_summary.columns:
        plt.figure(figsize=(8, 4))
        plt.text(0.5, 0.5, "No first-signal data", ha="center", va="center")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(out_path, dpi=180)
        plt.close()
        return True
    counts = city_summary["first_signal_type"].fillna("missing").value_counts().sort_values(ascending=False)
    plt.figure(figsize=(8, 4))
    plt.bar(counts.index.tolist(), counts.values.tolist(), color="#457b9d")
    plt.title("Earliest observed zoning signal type across cities")
    plt.xlabel("First signal type")
    plt.ylabel("City count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
    return True


def _compile_pdf(output_dir: Path) -> bool:
    tex_path = output_dir / "report.tex"
    if not tex_path.is_file():
        return False
    cmd = ["latexmk", "-pdf", "-interaction=nonstopmode", "report.tex"]
    try:
        subprocess.run(cmd, cwd=str(output_dir), check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        return True
    except Exception:
        return False


def _select_and_copy_example_images(
    levittown_pages: pd.DataFrame,
    figures_dir: Path,
) -> list[dict[str, str]]:
    if levittown_pages.empty:
        return []
    required = {"manual_label", "issue_id", "issue_date", "evidence_excerpt", "og_image_path"}
    if not required.issubset(set(levittown_pages.columns)):
        return []

    preferred_labels = [
        "amendment_targeted",
        "amendment_substantial",
        "zoning_notice_or_summary",
        "non_zoning_ordinance_notice",
        "non_zoning_notice",
    ]

    selected_rows: list[pd.Series] = []
    seen_issue_ids: set[str] = set()
    for label in preferred_labels:
        sub = levittown_pages[levittown_pages["manual_label"].astype(str) == label]
        if sub.empty:
            continue
        for _, row in sub.sort_values(["issue_date", "issue_id"]).iterrows():
            iid = str(row.get("issue_id", ""))
            p = Path(str(row.get("og_image_path", "")))
            if iid and iid not in seen_issue_ids and p.is_file():
                selected_rows.append(row)
                seen_issue_ids.add(iid)
                break
        if len(selected_rows) >= 3:
            break

    if len(selected_rows) < 3:
        for _, row in levittown_pages.sort_values(["issue_date", "issue_id"]).iterrows():
            iid = str(row.get("issue_id", ""))
            p = Path(str(row.get("og_image_path", "")))
            if iid and iid not in seen_issue_ids and p.is_file():
                selected_rows.append(row)
                seen_issue_ids.add(iid)
            if len(selected_rows) >= 3:
                break

    out: list[dict[str, str]] = []
    figures_dir.mkdir(parents=True, exist_ok=True)
    for idx, row in enumerate(selected_rows, start=1):
        src = Path(str(row.get("og_image_path", "")))
        if not src.is_file():
            continue
        dst = figures_dir / f"class_shift_example_{idx}.png"
        try:
            shutil.copy2(src, dst)
        except Exception:
            continue
        out.append(
            {
                "file": f"figures/{dst.name}",
                "manual_label": str(row.get("manual_label", "")),
                "issue_id": str(row.get("issue_id", "")),
                "issue_date": str(row.get("issue_date", "")),
                "evidence_excerpt": _clip_text(row.get("evidence_excerpt", ""), 120),
            }
        )
    return out


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build pipeline assessment LaTeX report.")
    ap.add_argument(
        "--run-dir",
        default="/Users/saulrichardson/projects/newspapers/newspaper-analysis/reports/runs/pi_v1_run60_fullnewspaper",
        help="Run directory containing pipeline artifacts.",
    )
    ap.add_argument(
        "--output-dir",
        default="",
        help="Output directory for report (default: <run-dir>/reports/pipeline_assessment).",
    )
    ap.add_argument("--compile-pdf", action=argparse.BooleanOptionalAction, default=True)
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if str(args.output_dir).strip()
        else run_dir / "reports" / "pipeline_assessment"
    )
    figures_dir = output_dir / "figures"
    tables_dir = output_dir / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    selected = _safe_read_csv(run_dir / "panels" / "selected_panel_issues.csv")
    city_summary = _safe_read_csv(
        run_dir
        / "longitudinal_iter_v13_full20_refined"
        / "final"
        / "panel_lean_numeric_quote_v1"
        / "city_summary.csv"
    )
    stage_shares = _safe_read_csv(
        run_dir
        / "longitudinal_iter_v13_full20_refined"
        / "final"
        / "panel_lean_numeric_quote_v1"
        / "city_stage_category_shares.csv"
    )
    phase_segments = _safe_read_csv(
        run_dir
        / "longitudinal_iter_v13_full20_refined"
        / "final"
        / "panel_lean_numeric_quote_v1"
        / "city_phase_segments.csv"
    )
    waves = _safe_read_csv(
        run_dir
        / "longitudinal_iter_v13_full20_refined"
        / "final"
        / "panel_lean_numeric_quote_v1"
        / "city_amendment_waves.csv"
    )
    ontology_summary = _safe_read_csv(run_dir / "ontology_ab" / "ab_runs" / "v2_full243" / "ab_variant_summary.csv")
    reaudit_city = _safe_read_csv(
        run_dir / "longitudinal_iter_v13_full20_refined" / "full_ordinance_reaudit_city_summary_v2.csv"
    )
    reaudit_all = _safe_read_csv(
        run_dir / "longitudinal_iter_v13_full20_refined" / "full_ordinance_reaudit_all_cities_v2.csv"
    )
    reaudit_decade = _safe_read_csv(
        run_dir / "longitudinal_iter_v13_full20_refined" / "full_ordinance_reaudit_decade_impact.csv"
    )
    levittown_pages = _safe_read_csv(
        run_dir / "longitudinal_iter_v13_full20_refined" / "levittown_full_ordinance_manual_review_with_pages.csv"
    )
    single_city_pipeline = _safe_read_csv(
        run_dir / "single_city_pipeline" / "levittown_smoke_20260220" / "pipeline_summary.csv"
    )

    issue_counts = pd.DataFrame()
    if not selected.empty:
        issue_counts = (
            selected.groupby(["city_key"], as_index=False)
            .agg(issue_count=("issue_id", "nunique"))
            .sort_values(["issue_count", "city_key"], ascending=[False, True])
            .reset_index(drop=True)
        )

    coverage_tbl = pd.DataFrame()
    if not selected.empty:
        coverage_tbl = (
            selected.groupby(["city_key", "region", "urbanicity_proxy"], as_index=False)
            .agg(
                issue_count=("issue_id", "nunique"),
                min_issue_date=("issue_date", "min"),
                max_issue_date=("issue_date", "max"),
                median_text_chars=("text_chars", "median"),
            )
            .sort_values(["issue_count", "city_key"], ascending=[False, True])
            .reset_index(drop=True)
        )
        if "min_issue_date" in coverage_tbl.columns:
            coverage_tbl["min_issue_year"] = coverage_tbl["min_issue_date"].astype(str).str.slice(0, 4)
            coverage_tbl["max_issue_year"] = coverage_tbl["max_issue_date"].astype(str).str.slice(0, 4)
        coverage_tbl = coverage_tbl[
            [
                "city_key",
                "region",
                "urbanicity_proxy",
                "issue_count",
                "min_issue_year",
                "max_issue_year",
                "median_text_chars",
            ]
        ]
        coverage_tbl["median_text_chars"] = coverage_tbl["median_text_chars"].round(0).astype(int)

    recommendation_tbl = pd.DataFrame(
        [
            {
                "component": "Issue ontology / document typing",
                "recommended_choice": "Ontology v2 (two-stage presence + legal subtype)",
                "why": "Cuts false full-ordinance calls and separates notices/amendments",
            },
            {
                "component": "Zoning text extraction",
                "recommended_choice": "LLM-only extraction",
                "why": "Meets no-deterministic requirement; keeps full legal-language focus",
            },
            {
                "component": "Longitudinal panel extraction",
                "recommended_choice": "panel_lean_numeric_quote_v1",
                "why": "Most stable on long full-city panels; preserves evidence fields",
            },
            {
                "component": "Prompt scope",
                "recommended_choice": "One request per city panel with all available issues",
                "why": "Retains full city sequence context for event timing and evolution",
            },
            {
                "component": "Overflow handling",
                "recommended_choice": "Deterministic clipping fallback (rare)",
                "why": "Keeps single-shot structure when one city exceeds context budget",
            },
        ]
    )

    ontology_tbl = pd.DataFrame()
    if not ontology_summary.empty:
        keep_cols = [c for c in ["variant", "n_issues", "parse_ok_share", "pred_full_share", "pred_notice_share"] if c in ontology_summary.columns]
        ontology_tbl = ontology_summary[keep_cols].copy()
        for c in ["parse_ok_share", "pred_full_share", "pred_notice_share"]:
            if c in ontology_tbl.columns:
                ontology_tbl[c] = (ontology_tbl[c] * 100.0).round(1)
                ontology_tbl = ontology_tbl.rename(columns={c: c.replace("_share", "_pct")})

    reaudit_tbl = pd.DataFrame()
    if not reaudit_city.empty:
        cols = [
            "city_key",
            "original_full_rows",
            "likely_full_zoning_ordinance",
            "likely_zoning_amendment",
            "likely_zoning_notice_hearing",
            "likely_non_zoning_notice_or_ordinance",
            "unclear",
        ]
        cols = [c for c in cols if c in reaudit_city.columns]
        reaudit_tbl = reaudit_city[cols].copy().sort_values("original_full_rows", ascending=False).reset_index(drop=True)

    shift_bucket_tbl = pd.DataFrame()
    if not reaudit_all.empty and "manual_bucket" in reaudit_all.columns:
        shift_bucket_tbl = (
            reaudit_all["manual_bucket"]
            .fillna("missing")
            .value_counts(dropna=False)
            .rename_axis("manual_bucket")
            .reset_index(name="count")
        )
        shift_bucket_tbl["share_pct"] = (
            100.0 * shift_bucket_tbl["count"] / max(1, shift_bucket_tbl["count"].sum())
        ).round(1)

    amend_examples_tbl = pd.DataFrame()
    if not reaudit_all.empty:
        amend_rows = reaudit_all[reaudit_all["manual_bucket"].astype(str) == "likely_zoning_amendment"].copy()
        if not amend_rows.empty:
            amend_rows = amend_rows.sort_values(["city_key", "issue_date", "issue_id"])
            diverse = amend_rows.groupby("city_key", as_index=False).head(1)
            if len(diverse) < 10:
                remain = amend_rows[~amend_rows["issue_id"].isin(diverse["issue_id"])].head(10 - len(diverse))
                pick = pd.concat([diverse, remain], ignore_index=True).head(10)
            else:
                pick = diverse.head(10)
            amend_examples_tbl = pick[
                ["city_key", "issue_date", "issue_id", "manual_bucket", "evidence_excerpt"]
            ].copy()
            amend_examples_tbl["evidence_excerpt"] = amend_examples_tbl["evidence_excerpt"].map(
                lambda x: _clip_text(x, 95)
            )

    notice_examples_tbl = pd.DataFrame()
    if not reaudit_all.empty:
        notice_rows = reaudit_all[
            reaudit_all["manual_bucket"].astype(str).isin(
                ["likely_zoning_notice_hearing", "likely_non_zoning_notice_or_ordinance"]
            )
        ].copy()
        if not notice_rows.empty:
            notice_rows = notice_rows.sort_values(["city_key", "issue_date", "issue_id"])
            diverse = notice_rows.groupby("city_key", as_index=False).head(1)
            if len(diverse) < 10:
                remain = notice_rows[~notice_rows["issue_id"].isin(diverse["issue_id"])].head(10 - len(diverse))
                pick = pd.concat([diverse, remain], ignore_index=True).head(10)
            else:
                pick = diverse.head(10)
            notice_examples_tbl = pick[
                ["city_key", "issue_date", "issue_id", "manual_bucket", "evidence_excerpt"]
            ].copy()
            notice_examples_tbl["evidence_excerpt"] = notice_examples_tbl["evidence_excerpt"].map(
                lambda x: _clip_text(x, 95)
            )

    longitudinal_tbl = pd.DataFrame()
    if not city_summary.empty:
        cols = [
            "city_key",
            "issue_count",
            "first_signal_type",
            "first_signal_date",
            "first_full_status",
            "first_full_date",
            "full_ordinance_issues",
            "amendment_issues",
            "mention_only_issues",
            "overall_quality",
        ]
        cols = [c for c in cols if c in city_summary.columns]
        longitudinal_tbl = city_summary[cols].copy().sort_values("issue_count", ascending=False).reset_index(drop=True)
        if "first_signal_date" in longitudinal_tbl.columns:
            longitudinal_tbl["first_signal_year"] = longitudinal_tbl["first_signal_date"].astype(str).str.slice(0, 4)
        if "first_full_date" in longitudinal_tbl.columns:
            longitudinal_tbl["first_full_year"] = longitudinal_tbl["first_full_date"].astype(str).str.slice(0, 4)
        keep = [c for c in ["city_key", "issue_count", "first_signal_type", "first_signal_year", "first_full_status", "first_full_year", "full_ordinance_issues", "amendment_issues", "mention_only_issues", "overall_quality"] if c in longitudinal_tbl.columns]
        longitudinal_tbl = longitudinal_tbl[keep]
        if "overall_quality" in longitudinal_tbl.columns:
            longitudinal_tbl["overall_quality"] = longitudinal_tbl["overall_quality"].round(3)

    stage_tbl = pd.DataFrame()
    if not stage_shares.empty:
        stage_means = (
            stage_shares.groupby(["stage", "category"], as_index=False)["share"]
            .mean()
            .rename(columns={"share": "mean_share"})
        )
        pivot = stage_means.pivot(index="category", columns="stage", values="mean_share").reset_index()
        for c in ["early", "mid", "late"]:
            if c in pivot.columns:
                pivot[c] = (pivot[c] * 100.0).round(1)
        if all(c in pivot.columns for c in ["early", "late"]):
            pivot["late_minus_early_pp"] = (pivot["late"] - pivot["early"]).round(1)
        stage_tbl = pivot.sort_values("late", ascending=False) if "late" in pivot.columns else pivot

    phase_tbl = pd.DataFrame()
    if not phase_segments.empty:
        label_col = "phase_archetype" if "phase_archetype" in phase_segments.columns else ""
        if label_col:
            phase_tbl = (
                phase_segments.groupby([label_col], as_index=False)
                .size()
                .rename(columns={"size": "count"})
                .sort_values("count", ascending=False)
            )

    wave_tbl = pd.DataFrame()
    if not waves.empty and "change_type" in waves.columns:
        wave_tbl = (
            waves.groupby(["change_type"], as_index=False)
            .size()
            .rename(columns={"size": "count"})
            .sort_values("count", ascending=False)
        )

    extraction_tbl = pd.DataFrame()
    if not single_city_pipeline.empty:
        keep_cols = [
            "target",
            "winner_variant",
            "overall_quality",
            "text_keep_ratio",
            "full_ordinance_issues",
            "amendment_issues",
            "mention_only_issues",
            "non_zoning_or_uncertain_issues",
        ]
        keep_cols = [c for c in keep_cols if c in single_city_pipeline.columns]
        extraction_tbl = single_city_pipeline[keep_cols].copy()
        for c in ["overall_quality", "text_keep_ratio"]:
            if c in extraction_tbl.columns:
                extraction_tbl[c] = extraction_tbl[c].round(3)

    fig_issue_ok = _render_fig_issue_counts(issue_counts, figures_dir / "issue_coverage_top20.png")
    fig_shift_bucket_ok = _render_fig_shift_bucket_counts(reaudit_all, figures_dir / "full_shift_bucket_counts.png")
    fig_shift_city_ok = _render_fig_shift_city_stacked(reaudit_city, figures_dir / "full_shift_city_stacked.png")
    fig_reaudit_ok = _render_fig_reaudit_decade(reaudit_decade, figures_dir / "full_ordinance_reaudit_decade.png")
    fig_signal_ok = _render_fig_first_signal(city_summary, figures_dir / "first_signal_type_counts.png")
    example_images = _select_and_copy_example_images(levittown_pages, figures_dir)

    _to_latex_table(recommendation_tbl, tables_dir / "recommended_pipeline.tex", index=False)
    _to_latex_table(coverage_tbl, tables_dir / "coverage_by_city.tex", index=False)
    _to_latex_table(ontology_tbl, tables_dir / "ontology_ab_summary.tex", index=False)
    _to_latex_table(extraction_tbl, tables_dir / "single_city_extraction_compare.tex", index=False)
    _to_latex_table(reaudit_tbl, tables_dir / "full_ordinance_reaudit_city.tex", index=False)
    _to_latex_table(shift_bucket_tbl, tables_dir / "full_shift_bucket_summary.tex", index=False)
    _to_latex_table(amend_examples_tbl, tables_dir / "full_to_amend_examples.tex", index=False)
    _to_latex_table(notice_examples_tbl, tables_dir / "full_to_notice_examples.tex", index=False)
    _to_latex_table(longitudinal_tbl, tables_dir / "longitudinal_city_summary.tex", index=False)
    _to_latex_table(stage_tbl, tables_dir / "category_stage_means.tex", index=False)
    _to_latex_table(phase_tbl, tables_dir / "phase_counts.tex", index=False)
    _to_latex_table(wave_tbl, tables_dir / "wave_counts.tex", index=False)

    n_city = int(issue_counts["city_key"].nunique()) if not issue_counts.empty else 0
    n_issues = int(issue_counts["issue_count"].sum()) if not issue_counts.empty else 0
    min_issues = int(issue_counts["issue_count"].min()) if not issue_counts.empty else 0
    max_issues = int(issue_counts["issue_count"].max()) if not issue_counts.empty else 0
    median_issues = float(issue_counts["issue_count"].median()) if not issue_counts.empty else 0.0

    overflow_cities = 0
    clipped_issues = 0
    if not city_summary.empty:
        if "overflow_applied" in city_summary.columns:
            overflow_cities = int(city_summary["overflow_applied"].fillna(0).astype(int).sum())
        if "clipped_issue_count" in city_summary.columns:
            clipped_issues = int(city_summary["clipped_issue_count"].fillna(0).sum())

    phase_rows = int(len(phase_segments)) if not phase_segments.empty else 0
    wave_rows = int(len(waves)) if not waves.empty else 0

    issue_figure_block = r"""
\begin{figure}[H]
\centering
\includegraphics[width=0.95\textwidth]{figures/issue_coverage_top20.png}
\caption{Issue coverage per city panel.}
\end{figure}
""" if fig_issue_ok else r"""
\noindent\textit{Figure omitted in this build environment (plotting dependency unavailable).}
"""

    reaudit_figure_block = r"""
\begin{figure}[H]
\centering
\includegraphics[width=0.82\textwidth]{figures/full_ordinance_reaudit_decade.png}
\caption{Original versus re-audited full-ordinance counts by decade.}
\end{figure}
""" if fig_reaudit_ok else r"""
\noindent\textit{Figure omitted in this build environment (plotting dependency unavailable).}
"""

    signal_figure_block = r"""
\begin{figure}[H]
\centering
\includegraphics[width=0.62\textwidth]{figures/first_signal_type_counts.png}
\caption{First observed zoning signal type across cities.}
\end{figure}
""" if fig_signal_ok else r"""
\noindent\textit{Figure omitted in this build environment (plotting dependency unavailable).}
"""

    shift_bucket_figure_block = r"""
\begin{figure}[H]
\centering
\includegraphics[width=0.84\textwidth]{figures/full_shift_bucket_counts.png}
\caption{How rows originally labeled full ordinance were reclassified after audit.}
\end{figure}
""" if fig_shift_bucket_ok else r"""
\noindent\textit{Figure omitted in this build environment (plotting dependency unavailable).}
"""

    shift_city_figure_block = r"""
\begin{figure}[H]
\centering
\includegraphics[width=0.98\textwidth]{figures/full_shift_city_stacked.png}
\caption{City-level breakdown of reclassification outcomes for original full-ordinance rows.}
\end{figure}
""" if fig_shift_city_ok else r"""
\noindent\textit{Figure omitted in this build environment (plotting dependency unavailable).}
"""

    image_blocks = []
    for i, ex in enumerate(example_images, start=1):
        lbl = _latex_escape(ex["manual_label"])
        dtxt = _latex_escape(ex["issue_date"])
        iid = _latex_escape(ex["issue_id"])
        ex_txt = _latex_escape(ex["evidence_excerpt"])
        image_blocks.append(
            rf"""\begin{{figure}}[H]
\centering
\includegraphics[width=0.92\textwidth]{{{ex["file"]}}}
\caption{{Example {i}: original full-ordinance row reclassified as {lbl} ({dtxt}; {iid}).}}
\end{{figure}}
\noindent\textit{{Excerpt used in re-audit:}} {ex_txt}
"""
        )
    image_examples_block = (
        "\n".join(image_blocks)
        if image_blocks
        else r"\noindent\textit{No page-image examples were available in this build.}"
    )

    tex = rf"""
\documentclass[11pt]{{article}}
\usepackage[margin=1in]{{geometry}}
\usepackage{{booktabs}}
\usepackage{{graphicx}}
\usepackage{{float}}
\usepackage{{hyperref}}
\title{{Zoning Pipeline Assessment and Recommendation}}
\author{{Automated pipeline iteration report}}
\date{{{dt.date.today().isoformat()}}}
\begin{{document}}
\maketitle
\tableofcontents
\clearpage

\section{{Executive recommendation}}
Move forward with a four-part production pipeline:
\begin{{enumerate}}
\item Ontology v2 issue typing (two-stage: zoning presence + legal subtype).
\item LLM-only zoning text extraction for each issue.
\item One LLM request per city using the full chronological newspaper panel.
\item Lean longitudinal schema (panel\_lean\_numeric\_quote\_v1) with evidence-linked outputs.
\end{{enumerate}}

\begin{{table}}[H]
\centering
\caption{{Recommended production pipeline and rationale}}
\resizebox{{\textwidth}}{{!}}{{\input{{tables/recommended_pipeline.tex}}}}
\end{{table}}

\section{{What changed and why}}
Three major changes came out of iteration:
\begin{{enumerate}}
\item \textbf{{Full-panel context became the default unit.}} The extraction now reads each city as one longitudinal sequence instead of issue-isolated interpretation.
\item \textbf{{Issue typing was tightened.}} Ontology v2 reduced false full-ordinance calls by separating full code adoption from amendments and notices.
\item \textbf{{Long-panel stability was hardened.}} A lean output schema improved reliability on very large city panels, with clipping fallback only when one panel exceeded context limits.
\end{{enumerate}}

\section{{Data coverage for this run}}
This run includes {n_city} cities and {n_issues} issues ({min_issues} to {max_issues} issues per city; median {median_issues:.1f}).

{issue_figure_block}

\begin{{table}}[H]
\centering
\caption{{City panel coverage summary}}
\resizebox{{\textwidth}}{{!}}{{\input{{tables/coverage_by_city.tex}}}}
\end{{table}}

\section{{Evidence for classification changes}}
The table below summarizes ontology A/B outcomes from the larger 243-issue diagnostic run used for prompt selection.

\begin{{table}}[H]
\centering
\caption{{Ontology A/B summary (v2 full diagnostic sample)}}
\resizebox{{0.78\textwidth}}{{!}}{{\input{{tables/ontology_ab_summary.tex}}}}
\end{{table}}

Manual review showed that many rows previously called full ordinances were actually amendments, notices, or non-zoning legal text.

{reaudit_figure_block}

\begin{{table}}[H]
\centering
\caption{{City-level re-audit of rows originally labeled full ordinance}}
\resizebox{{\textwidth}}{{!}}{{\input{{tables/full_ordinance_reaudit_city.tex}}}}
\end{{table}}

\section{{Examples of classification shifts (focus: full ordinance vs amendment)}}
This section shows concrete shift patterns among rows that were originally in the full-ordinance bucket.
The largest shift is from original full-ordinance calls into amendment buckets, followed by notice/hearing buckets.

{shift_bucket_figure_block}

\begin{{table}}[H]
\centering
\caption{{Re-audit bucket shares for rows originally labeled full ordinance}}
\resizebox{{0.84\textwidth}}{{!}}{{\input{{tables/full_shift_bucket_summary.tex}}}}
\end{{table}}

{shift_city_figure_block}

\begin{{table}}[H]
\centering
\caption{{Illustrative rows reclassified from original full ordinance to zoning amendments}}
\resizebox{{\textwidth}}{{!}}{{\input{{tables/full_to_amend_examples.tex}}}}
\end{{table}}

\begin{{table}}[H]
\centering
\caption{{Illustrative rows reclassified from original full ordinance to notice/non-zoning buckets}}
\resizebox{{\textwidth}}{{!}}{{\input{{tables/full_to_notice_examples.tex}}}}
\end{{table}}

\subsection*{{Page-image examples of these shifts}}
Each image below is an original newspaper page tied to an issue that was initially bucketed as full ordinance and then reclassified.
{image_examples_block}

\section{{Longitudinal outputs from the refined 20-city run}}
Final refined outputs include:
\begin{{itemize}}
\item first-event timing per city,
\item stage-level category shares (early/mid/late),
\item phase segments with canonical archetype labels,
\item amendment wave typing,
\item numeric-constraint trajectories.
\end{{itemize}}

In this run, overflow handling was used in {overflow_cities} city panel(s), with {clipped_issues} clipped issue blocks total.

{signal_figure_block}

\begin{{table}}[H]
\centering
\caption{{City-level longitudinal summary (refined run)}}
\resizebox{{\textwidth}}{{!}}{{\input{{tables/longitudinal_city_summary.tex}}}}
\end{{table}}

\begin{{table}}[H]
\centering
\caption{{Mean category shares by stage (percent)}}
\resizebox{{0.86\textwidth}}{{!}}{{\input{{tables/category_stage_means.tex}}}}
\end{{table}}

\begin{{table}}[H]
\centering
\caption{{Phase archetype counts (unit = phase segment; total segments = {phase_rows})}}
\resizebox{{0.62\textwidth}}{{!}}{{\input{{tables/phase_counts.tex}}}}
\end{{table}}

\begin{{table}}[H]
\centering
\caption{{Amendment wave type counts (total waves = {wave_rows})}}
\resizebox{{0.58\textwidth}}{{!}}{{\input{{tables/wave_counts.tex}}}}
\end{{table}}

\section{{Extraction method recommendation}}
Given the current constraint to avoid deterministic extraction logic, LLM-only extraction is the default recommendation.
The Levittown end-to-end smoke run is included below as an operational check.

\begin{{table}}[H]
\centering
\caption{{Single-city extraction comparison (Levittown smoke run)}}
\resizebox{{0.86\textwidth}}{{!}}{{\input{{tables/single_city_extraction_compare.tex}}}}
\end{{table}}

\section{{What to run next (production path)}}
\begin{{enumerate}}
\item Run ontology v2 + LLM-only extraction + panel\_lean\_numeric\_quote\_v1 longitudinal extraction for all 20 cities in one clean production run.
\item Regenerate the standalone descriptive report from that production output directory.
\item Keep focused manual audit packets for 3--5 cities with the largest measured changes to verify transcript-to-claim alignment.
\end{{enumerate}}

\end{{document}}
"""
    _write_text(output_dir / "report.tex", tex)

    compiled = False
    if bool(args.compile_pdf):
        compiled = _compile_pdf(output_dir)

    provenance = {
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "compiled_pdf": bool(compiled),
        "image_examples_count": len(example_images),
        "inputs": {
            "selected_panel_issues": str(run_dir / "panels" / "selected_panel_issues.csv"),
            "city_summary_refined": str(
                run_dir
                / "longitudinal_iter_v13_full20_refined"
                / "final"
                / "panel_lean_numeric_quote_v1"
                / "city_summary.csv"
            ),
            "ontology_ab_summary": str(run_dir / "ontology_ab" / "ab_runs" / "v2_full243" / "ab_variant_summary.csv"),
            "full_ordinance_reaudit_city": str(
                run_dir / "longitudinal_iter_v13_full20_refined" / "full_ordinance_reaudit_city_summary_v2.csv"
            ),
            "full_ordinance_reaudit_all": str(
                run_dir / "longitudinal_iter_v13_full20_refined" / "full_ordinance_reaudit_all_cities_v2.csv"
            ),
            "levittown_manual_with_pages": str(
                run_dir / "longitudinal_iter_v13_full20_refined" / "levittown_full_ordinance_manual_review_with_pages.csv"
            ),
        },
    }
    _write_text(output_dir / "provenance.json", json.dumps(provenance, indent=2))

    print(json.dumps({"output_dir": str(output_dir), "compiled_pdf": bool(compiled)}, indent=2))


if __name__ == "__main__":
    main()
