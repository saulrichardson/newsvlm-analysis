#!/usr/bin/env python3
"""
Build a reader-friendly LaTeX writeup that consolidates:
  1) Topic discovery artifacts (UMAP + cluster frequency) with the latest cluster labels
  2) City×time RHS measurement artifacts (dashboards, event candidates, regimes, outcomes overlays)

This report is designed to be readable by a broad research audience:
  - avoids “computer labels” where feasible (uses friendly titles/legends)
  - starts with topic discovery (UMAP/cluster frequency)
  - then walks city-by-city through the constructed RHS plots

It does NOT choose a causal design or a treatment definition; it documents the data
products that can be used downstream (DiD/event studies/etc.).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import shutil
from pathlib import Path
from typing import Any

import pandas as pd


def _safe_slug(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9_]+", "", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s[:120] or "x"


def _tex_escape(s: str) -> str:
    return (
        s.replace("\\", "\\textbackslash{}")
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


def _copy(src: Path, dst: Path) -> None:
    if not src.is_file():
        raise SystemExit(f"Missing file to copy: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _find_latest(dir_path: Path, pattern: str) -> Path | None:
    paths = sorted(dir_path.glob(pattern))
    if not paths:
        return None
    # Prefer newest by mtime, fall back to lexicographic order.
    paths.sort(key=lambda p: (p.stat().st_mtime, str(p)))
    return paths[-1]


def _render_top32_table(top32: pd.DataFrame) -> str:
    cols = ["city_name", "state_abbr", "n_issues_sum", "n_months_obs"]
    missing = [c for c in cols if c not in top32.columns]
    if missing:
        raise SystemExit(f"top32 CSV missing required columns: {missing}")
    df = top32[cols].copy()
    df["city_name"] = df["city_name"].astype(str)
    df["state_abbr"] = df["state_abbr"].astype(str)
    df["n_issues_sum"] = pd.to_numeric(df["n_issues_sum"], errors="coerce")
    df["n_months_obs"] = pd.to_numeric(df["n_months_obs"], errors="coerce")
    df = df.sort_values(["n_issues_sum", "city_name"], ascending=[False, True]).reset_index(drop=True)

    lines = []
    lines.append("\\begin{longtable}{llrr}")
    lines.append("\\caption{Sample A city-by-state units (top 32 by issue coverage)}\\\\")
    lines.append("\\toprule")
    lines.append("City & State & Total issues & Months observed\\\\")
    lines.append("\\midrule")
    lines.append("\\endfirsthead")
    lines.append("\\toprule")
    lines.append("City & State & Total issues & Months observed\\\\")
    lines.append("\\midrule")
    lines.append("\\endhead")
    for r in df.itertuples(index=False):
        lines.append(
            f"{_tex_escape(str(r.city_name))} & "
            f"{_tex_escape(str(r.state_abbr).upper())} & "
            f"{int(r.n_issues_sum):,d} & "
            f"{int(r.n_months_obs):,d}\\\\"
        )
    lines.append("\\bottomrule")
    lines.append("\\end{longtable}")
    return "\n".join(lines) + "\n"


def _render_top_clusters_table(all_clusters_csv: Path, *, label: str, top_k: int = 10) -> str:
    df = pd.read_csv(all_clusters_csv)
    req = ["topic_name", "cluster_weight_total"]
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing columns in {all_clusters_csv}: {missing}")

    df = df.copy()
    df["topic_name"] = df["topic_name"].astype(str)
    df["cluster_weight_total"] = pd.to_numeric(df["cluster_weight_total"], errors="coerce")
    df = df.dropna(subset=["cluster_weight_total"])

    total = float(df["cluster_weight_total"].sum())
    if total <= 0:
        total = 1.0

    df = df.sort_values("cluster_weight_total", ascending=False).head(int(top_k)).reset_index(drop=True)
    df["share_pct"] = 100.0 * df["cluster_weight_total"] / total

    lines = []
    lines.append("\\begin{longtable}{lr}")
    label_title = {
        "full_ordinance": "Full ordinances",
        "amendment_substantial": "Substantial amendments",
        "amendment_targeted": "Targeted amendments",
    }.get(str(label), str(label).replace("_", " ").title())
    lines.append(f"\\caption{{Top {int(top_k)} cluster labels by total weight ({_tex_escape(label_title)})}}\\\\")
    lines.append("\\toprule")
    lines.append("Cluster label & Weight share (\\%)\\\\")
    lines.append("\\midrule")
    lines.append("\\endfirsthead")
    lines.append("\\toprule")
    lines.append("Cluster label & Weight share (\\%)\\\\")
    lines.append("\\midrule")
    lines.append("\\endhead")
    for r in df.itertuples(index=False):
        name = _tex_escape(str(r.topic_name))
        pct = float(r.share_pct)
        lines.append(f"{name} & {pct:.2f}\\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{longtable}")
    return "\n".join(lines) + "\n"


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build a consolidated LaTeX report for topic discovery + RHS measures.")
    ap.add_argument("--output-dir", required=True, help="Directory to write report.tex/figures/tables into.")
    ap.add_argument(
        "--issue-topics-report-dir",
        default="reports/issue_topics_report",
        help="Directory containing the updated topic-discovery report assets (figures + tables).",
    )
    ap.add_argument(
        "--top32-csv",
        required=True,
        help="Path to top32_city_state.csv (Sample A city list + coverage stats).",
    )

    ap.add_argument("--dashboards-dir", required=True, help="Directory containing dashboards (32 city pngs).")
    ap.add_argument("--event-emb-dir", required=True, help="Directory containing embedding-jump event plots (32 city pngs).")
    ap.add_argument("--event-mech-dir", required=True, help="Directory containing mechanics-jump event plots (32 city pngs).")
    ap.add_argument("--regimes-dir", required=True, help="Directory containing regime plots (32 city pngs).")

    ap.add_argument("--overlay-emb-dir", default="", help="Optional: directory containing embedding-jump outcome overlays.")
    ap.add_argument("--overlay-mech-dir", default="", help="Optional: directory containing mechanics-jump outcome overlays.")

    ap.add_argument("--title", default="Zoning Text Topics and City-by-Time RHS Measures", help="Report title.")
    ap.add_argument("--top-clusters", type=int, default=10, help="How many top clusters to show per label.")
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")

    out_dir = Path(args.output_dir).expanduser().resolve()
    figs_dir = out_dir / "figures"
    tables_dir = out_dir / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    figs_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    issue_topics_dir = Path(args.issue_topics_report_dir).expanduser().resolve()
    topics_fig_dir = issue_topics_dir / "figures"
    topics_tbl_dir = issue_topics_dir / "tables"
    if not topics_fig_dir.is_dir() or not topics_tbl_dir.is_dir():
        raise SystemExit(f"--issue-topics-report-dir does not look valid: {issue_topics_dir}")

    top32_csv = Path(args.top32_csv).expanduser().resolve()
    top32 = pd.read_csv(top32_csv)

    dashboards_dir = Path(args.dashboards_dir).expanduser().resolve()
    event_emb_dir = Path(args.event_emb_dir).expanduser().resolve()
    event_mech_dir = Path(args.event_mech_dir).expanduser().resolve()
    regimes_dir = Path(args.regimes_dir).expanduser().resolve()

    overlay_emb_dir = Path(str(args.overlay_emb_dir)).expanduser().resolve() if str(args.overlay_emb_dir).strip() else None
    overlay_mech_dir = Path(str(args.overlay_mech_dir)).expanduser().resolve() if str(args.overlay_mech_dir).strip() else None

    # Copy topic-discovery figures (already updated to the latest cluster labels).
    topic_labels = ["full_ordinance", "amendment_substantial", "amendment_targeted"]
    for lab in topic_labels:
        _copy(topics_fig_dir / f"{lab}__cluster_frequency_year.png", figs_dir / "topics" / f"{lab}__cluster_frequency_year.png")
        _copy(topics_fig_dir / f"{lab}__umap_scatter.png", figs_dir / "topics" / f"{lab}__umap_scatter.png")

        all_clusters_csv = topics_tbl_dir / f"{lab}__all_clusters.csv"
        tex = _render_top_clusters_table(all_clusters_csv, label=lab, top_k=int(args.top_clusters))
        (tables_dir / f"{lab}__top_clusters_by_weight_share.tex").write_text(tex, encoding="utf-8")

    # Sample A table.
    (tables_dir / "sample_a_top32.tex").write_text(_render_top32_table(top32), encoding="utf-8")

    # Copy city-level figures into a stable layout to make LaTeX simple.
    city_rows = top32[["city_name", "state_abbr", "n_issues_sum", "n_months_obs"]].to_dict("records")
    city_manifest: list[dict[str, Any]] = []

    for row in city_rows:
        city = str(row["city_name"])
        st = str(row["state_abbr"]).lower()
        key = f"{_safe_slug(city)}__{_safe_slug(st)}"

        # The plotting scripts use a consistent slug scheme: city_{slug}_{st}_<stamp>.png
        dashboard = _find_latest(dashboards_dir, f"city_{_safe_slug(city)}_{_safe_slug(st)}_*.png")
        ev_emb = _find_latest(event_emb_dir, f"city_{_safe_slug(city)}_{_safe_slug(st)}_*.png")
        ev_mech = _find_latest(event_mech_dir, f"city_{_safe_slug(city)}_{_safe_slug(st)}_*.png")
        regime = _find_latest(regimes_dir, f"regimes_k4_city_{_safe_slug(city)}_{_safe_slug(st)}_*.png")

        if dashboard is None:
            raise SystemExit(f"Missing dashboard plot for {city}, {st} in {dashboards_dir}")
        if ev_emb is None:
            raise SystemExit(f"Missing embedding event plot for {city}, {st} in {event_emb_dir}")
        if ev_mech is None:
            raise SystemExit(f"Missing mechanics event plot for {city}, {st} in {event_mech_dir}")
        if regime is None:
            raise SystemExit(f"Missing regime plot for {city}, {st} in {regimes_dir}")

        dst_base = figs_dir / "cities" / key
        _copy(dashboard, dst_base / "dashboard.png")
        _copy(ev_emb, dst_base / "event_embedding_jump.png")
        _copy(ev_mech, dst_base / "event_mechanics_jump.png")
        _copy(regime, dst_base / "regime_k4.png")

        overlay_emb = None
        overlay_mech = None
        if overlay_emb_dir is not None and overlay_emb_dir.is_dir():
            overlay_emb = _find_latest(overlay_emb_dir, f"city_{_safe_slug(city)}_{_safe_slug(st)}_*overlay_*.png")
            if overlay_emb is not None:
                _copy(overlay_emb, dst_base / "outcome_overlay_embedding.png")
        if overlay_mech_dir is not None and overlay_mech_dir.is_dir():
            overlay_mech = _find_latest(overlay_mech_dir, f"city_{_safe_slug(city)}_{_safe_slug(st)}_*overlay_*.png")
            if overlay_mech is not None:
                _copy(overlay_mech, dst_base / "outcome_overlay_mechanics.png")

        city_manifest.append(
            {
                "city_name": city,
                "state_abbr": st,
                "city_key": key,
                "n_issues_sum": int(row["n_issues_sum"]),
                "n_months_obs": int(row["n_months_obs"]),
                "has_outcome_overlay": bool(overlay_emb is not None or overlay_mech is not None),
            }
        )

    (out_dir / "city_manifest.json").write_text(json.dumps(city_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # Write LaTeX.
    title = str(args.title).strip()
    # Keep the title page stable (no build-time timestamp).
    # If you want a fixed date, set it explicitly here.
    title_date = ""
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    doc: list[str] = []
    doc.append("\\documentclass[11pt]{article}")
    doc.append("\\usepackage[margin=1in]{geometry}")
    doc.append("\\usepackage{graphicx}")
    doc.append("\\usepackage{booktabs}")
    doc.append("\\usepackage{longtable}")
    doc.append("\\usepackage{float}")
    doc.append("\\usepackage{hyperref}")
    doc.append("\\usepackage{caption}")
    doc.append("")
    doc.append("\\title{%s}" % _tex_escape(title))
    doc.append("\\date{%s}" % _tex_escape(title_date))
    doc.append("\\begin{document}")
    doc.append("\\maketitle")

    doc.append("\\tableofcontents")
    doc.append("\\clearpage")

    doc.append("\\section{Overview}")
    doc.append(
        "This document summarizes the topic-discovery and measurement pipeline we applied to zoning-related "
        "newspaper text. We updated the prompt used to label discovered text clusters with mechanism-specific, "
        "human-readable descriptions, and then assembled a city-by-month panel of right-hand-side (RHS) measures "
        "from the labeled text."
    )
    doc.append("\\begin{itemize}")
    doc.append("\\item \\textbf{Topic discovery:} cluster frequency over time and UMAP scatter plots.")
    doc.append("\\item \\textbf{City-by-time measures:} dashboards, event-candidate plots, and regime timelines (Sample A).")
    doc.append("\\end{itemize}")

    doc.append("\\section{Pipeline overview (high level)}")
    doc.append("\\begin{itemize}")
    doc.append("\\item Each newspaper issue is split into chunks (\\~3,000 characters).")
    doc.append("\\item Chunks are embedded (OpenAI text embeddings) and clustered (UMAP $\\rightarrow$ HDBSCAN) within three corpora: full ordinances, substantial amendments, targeted amendments.")
    doc.append("\\item Each discovered cluster is labeled by a language model using a prompt designed to produce mechanism-specific, human-readable labels.")
    doc.append("\\item Cluster labels are also mapped into a structured \\emph{mechanics schema}: action tags (e.g., rezoning vs text amendment), dimensions (e.g., setbacks, parking), instruments, decision bodies.")
    doc.append("\\item We aggregate mechanics tags into a city-by-month panel and construct multiple RHS summaries: level (step-function) and change (jump intensity, regime switches).")
    doc.append("\\end{itemize}")

    doc.append("\\section{Topic discovery (updated cluster labels)}")
    doc.append(
        "The next pages show updated topic-discovery figures using the updated cluster labels: "
        "(i) cluster frequency-over-time (top 50 clusters; legend outside the plot), and "
        "(ii) UMAP scatter plots for each corpus. "
        "These figures help answer: \\emph{what kinds of zoning mechanisms appear in the newspaper text, and when?}"
    )

    for lab in topic_labels:
        lab_title = lab.replace("_", " ").title()
        doc.append("\\subsection{%s}" % _tex_escape(lab_title))
        doc.append("\\paragraph{Top clusters by weight (summary)}")
        doc.append("\\input{tables/%s__top_clusters_by_weight_share.tex}" % _tex_escape(lab))
        doc.append("")
        doc.append("\\paragraph{Cluster frequency over time (top 50)}")
        doc.append("\\clearpage")
        doc.append("\\newgeometry{margin=0.25in}")
        doc.append("\\thispagestyle{empty}")
        doc.append("\\begin{center}")
        doc.append(
            "\\includegraphics[width=\\linewidth,height=0.98\\textheight,keepaspectratio]{%s}"
            % _tex_escape(f"figures/topics/{lab}__cluster_frequency_year.png")
        )
        doc.append("\\end{center}")
        doc.append("\\restoregeometry")
        doc.append("\\clearpage")
        doc.append("\\paragraph{UMAP scatter (clusters labeled)}")
        doc.append("\\newgeometry{margin=0.25in}")
        doc.append("\\thispagestyle{empty}")
        doc.append("\\begin{center}")
        doc.append(
            "\\includegraphics[width=\\linewidth,height=0.98\\textheight,keepaspectratio]{%s}"
            % _tex_escape(f"figures/topics/{lab}__umap_scatter.png")
        )
        doc.append("\\end{center}")
        doc.append("\\restoregeometry")
        doc.append("\\clearpage")

    doc.append("\\section{City-by-month RHS measures (what we built)}")
    doc.append(
        "We construct a canonical city-by-month panel (Sample A shown below) with multiple interpretable RHS views:"
    )
    doc.append("\\begin{itemize}")
    doc.append("\\item \\textbf{Mechanics shares (level):} city-month shares of action tags and dimension tags, aggregated from labeled clusters.")
    doc.append("\\item \\textbf{Step-function state:} forward-filled between observed months (motivated by policy being piecewise-constant between updates).")
    doc.append("\\item \\textbf{Semantic change intensity:} embedding-based jump measure that spikes when the text content shifts sharply.")
    doc.append("\\item \\textbf{Mechanics change intensity:} PCA-based jump on the mechanics-tag vectors (spikes when the mechanism mix shifts).")
    doc.append("\\item \\textbf{Policy regimes:} discrete types (KMeans on PCs) that produce regime-switch event candidates.")
    doc.append("\\end{itemize}")

    doc.append("\\section{Sample A (top 32 city-state units)}")
    doc.append(
        "To keep plots readable while still being data-rich, Sample A is defined as the top 32 city-state units "
        "by total issue coverage (exactly those with at least 100 issues)."
    )
    doc.append("\\input{tables/sample_a_top32.tex}")

    doc.append("\\section{City-by-city plots (Sample A)}")
    doc.append(
        "For each city, we show: (1) a dashboard of RHS time-series (mechanics shares + change intensity), "
        "(2) event-candidate plots for semantic vs mechanics change, and (3) the inferred policy regime timeline. "
        "Outcome overlays are included where available (currently only a small 3-city FHFA HPI mapping)."
    )

    for row in city_manifest:
        city = str(row["city_name"])
        st = str(row["state_abbr"]).upper()
        key = str(row["city_key"])
        issues_sum = int(row["n_issues_sum"])
        months_obs = int(row["n_months_obs"])
        has_overlay = bool(row["has_outcome_overlay"])

        doc.append("\\subsection{%s, %s}" % (_tex_escape(city), _tex_escape(st)))
        doc.append("\\begin{itemize}")
        doc.append(f"\\item Coverage: {issues_sum:,d} issues across {months_obs:,d} observed months.")
        doc.append("\\item The dashboard uses human-readable labels for zoning mechanisms (e.g., ``Setbacks'', ``Parking'', ``Amend zoning map / rezone'').")
        doc.append("\\end{itemize}")

        # Dashboard (full page)
        doc.append("\\clearpage")
        doc.append("\\newgeometry{margin=0.25in}")
        doc.append("\\thispagestyle{empty}")
        doc.append("\\begin{center}")
        doc.append(
            "\\includegraphics[width=\\linewidth,height=0.98\\textheight,keepaspectratio]{%s}"
            % _tex_escape(f"figures/cities/{key}/dashboard.png")
        )
        doc.append("\\end{center}")
        doc.append("\\restoregeometry")

        # Events + regime (stacked on one page)
        doc.append("\\clearpage")
        doc.append("\\newgeometry{margin=0.4in}")
        doc.append("\\thispagestyle{empty}")
        doc.append("\\begin{center}")
        doc.append("\\textbf{Event candidates and policy regime timeline}\\\\")
        doc.append("\\vspace{0.2em}")
        doc.append(
            "\\includegraphics[width=\\linewidth,height=0.34\\textheight,keepaspectratio]{%s}"
            % _tex_escape(f"figures/cities/{key}/event_embedding_jump.png")
        )
        doc.append("\\vspace{0.2em}")
        doc.append(
            "\\includegraphics[width=\\linewidth,height=0.34\\textheight,keepaspectratio]{%s}"
            % _tex_escape(f"figures/cities/{key}/event_mechanics_jump.png")
        )
        doc.append("\\vspace{0.2em}")
        doc.append(
            "\\includegraphics[width=\\linewidth,height=0.24\\textheight,keepaspectratio]{%s}"
            % _tex_escape(f"figures/cities/{key}/regime_k4.png")
        )
        doc.append("\\end{center}")
        doc.append("\\restoregeometry")

        # Optional outcomes overlays
        if has_overlay:
            doc.append("\\clearpage")
            doc.append("\\newgeometry{margin=0.4in}")
            doc.append("\\thispagestyle{empty}")
            doc.append("\\begin{center}")
            doc.append("\\textbf{Outcome overlays (FHFA HPI from FRED; exploratory)}\\\\")
            emb_path = figs_dir / "cities" / key / "outcome_overlay_embedding.png"
            mech_path = figs_dir / "cities" / key / "outcome_overlay_mechanics.png"
            if emb_path.is_file():
                doc.append(
                    "\\includegraphics[width=\\linewidth,height=0.45\\textheight,keepaspectratio]{%s}"
                    % _tex_escape(f"figures/cities/{key}/outcome_overlay_embedding.png")
                )
                doc.append("\\vspace{0.2em}")
            if mech_path.is_file():
                doc.append(
                    "\\includegraphics[width=\\linewidth,height=0.45\\textheight,keepaspectratio]{%s}"
                    % _tex_escape(f"figures/cities/{key}/outcome_overlay_mechanics.png")
                )
            doc.append("\\end{center}")
            doc.append("\\restoregeometry")

    doc.append("\\end{document}")
    (out_dir / "report.tex").write_text("\n".join(doc) + "\n", encoding="utf-8")

    # Minimal Makefile (latexmk).
    (out_dir / "Makefile").write_text(
        "\n".join(
            [
                "all: report.pdf",
                "",
                "report.pdf: report.tex",
                "\tlatexmk -pdf -interaction=nonstopmode report.tex",
                "",
                "clean:",
                "\tlatexmk -C",
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (out_dir / ".gitignore").write_text("\n".join(["report.pdf", "*.aux", "*.log", "*.out", "*.toc", "*.lof", "*.lot", "report.fls", "report.fdb_latexmk"]) + "\n", encoding="utf-8")

    provenance = {
        "created_at": now,
        "stamp": stamp,
        "issue_topics_report_dir": str(issue_topics_dir),
        "top32_csv": str(top32_csv),
        "dashboards_dir": str(dashboards_dir),
        "event_emb_dir": str(event_emb_dir),
        "event_mech_dir": str(event_mech_dir),
        "regimes_dir": str(regimes_dir),
        "overlay_emb_dir": str(overlay_emb_dir) if overlay_emb_dir is not None else "",
        "overlay_mech_dir": str(overlay_mech_dir) if overlay_mech_dir is not None else "",
        "n_cities": int(len(city_manifest)),
    }
    (out_dir / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"Done. wrote={out_dir}")


if __name__ == "__main__":
    main()
