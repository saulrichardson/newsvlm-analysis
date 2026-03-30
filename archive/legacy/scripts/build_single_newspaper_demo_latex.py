#!/usr/bin/env python3
"""
Build a compact single-newspaper demo report from a completed fullnewspaper_v3 run.

Purpose:
  Show the pipeline artifact chain end-to-end for one newspaper with strong coverage.
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
import pandas as pd


def _safe_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _safe_jsonl(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    try:
        return pd.read_json(path, lines=True)
    except Exception:
        return pd.DataFrame()


def _safe_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _ensure(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _write(path: Path, text: str) -> None:
    _ensure(path.parent)
    path.write_text(text, encoding="utf-8")


def _tex_escape(x: Any) -> str:
    s = str(x or "")
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


def _clip(x: Any, n: int = 280) -> str:
    s = str(x or "").replace("\n", " ").strip()
    if len(s) <= n:
        return s
    return s[: n - 3] + "..."


def _to_latex_table(df: pd.DataFrame, out_path: Path, *, index: bool = False, float_fmt: str = "{:.3f}") -> None:
    _ensure(out_path.parent)
    if df.empty:
        out_path.write_text("\\begin{tabular}{l}No data\\\\\\end{tabular}\n", encoding="utf-8")
        return

    def _fmt(v: Any) -> str:
        if isinstance(v, float):
            return float_fmt.format(v)
        return str(v)

    out_path.write_text(
        df.to_latex(index=index, escape=True, na_rep="", float_format=lambda x: _fmt(x)),
        encoding="utf-8",
    )


def _compile_pdf(output_dir: Path) -> bool:
    tex = output_dir / "report.tex"
    if not tex.is_file():
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


def _pick_png_for_issue(png_dir: Path, issue_id: str) -> Path | None:
    if "__" not in issue_id:
        return None
    slug, date_s = issue_id.split("__", 1)
    try:
        d = dt.date.fromisoformat(date_s)
    except Exception:
        return None
    mon = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"][d.month - 1]
    pattern = f"{slug}-{mon}-{d.day:02d}-{d.year}-p-*.png"
    hits = sorted(png_dir.glob(pattern))
    return hits[0] if hits else None


def build(run_root: Path, newspaper_slug: str, output_dir: Path, compile_pdf: bool = True) -> dict[str, Any]:
    class_csv = run_root / "classification" / "issue_labels_v3.csv"
    clean_csv = run_root / "cleaning" / "clean_zoning_text_flat.csv"
    assign_csv = run_root / "events" / "issue_event_assignments_v3.csv"
    clusters_csv = run_root / "events" / "ordinance_event_clusters_v3.csv"
    long_csv = run_root / "longitudinal" / "city_longitudinal_summary.csv"
    panel_jsonl = run_root / "longitudinal" / "panel_outputs_v3.jsonl"
    inventory_json = run_root / "organized_corpus_v3" / "summary" / "export_inventory.json"

    cls = _safe_csv(class_csv)
    clean = _safe_csv(clean_csv)
    assign = _safe_csv(assign_csv)
    clusters = _safe_csv(clusters_csv)
    long_df = _safe_csv(long_csv)
    panel = _safe_jsonl(panel_jsonl)
    inv = _safe_json(inventory_json)

    _ensure(output_dir)
    fig_dir = output_dir / "figures"
    tbl_dir = output_dir / "tables"
    _ensure(fig_dir)
    _ensure(tbl_dir)

    cls_n = cls[cls["newspaper_slug"] == newspaper_slug].copy() if not cls.empty else pd.DataFrame()
    clean_n = clean[clean["newspaper_slug"] == newspaper_slug].copy() if not clean.empty else pd.DataFrame()
    assign_n = assign[assign["newspaper_slug"] == newspaper_slug].copy() if not assign.empty else pd.DataFrame()
    clusters_n = clusters[clusters["newspaper_slug"] == newspaper_slug].copy() if not clusters.empty else pd.DataFrame()
    long_n = long_df[long_df["newspaper_slug"] == newspaper_slug].copy() if not long_df.empty else pd.DataFrame()
    panel_n = panel[panel["newspaper_slug"] == newspaper_slug].copy() if not panel.empty else pd.DataFrame()

    # Core demo metrics.
    issue_count = int(cls_n["issue_id"].nunique()) if not cls_n.empty else 0
    clean_nonempty = int((clean_n["clean_chars"] > 0).sum()) if not clean_n.empty else 0
    full_rows = int(cls_n["issue_class"].isin(["full_code_comprehensive", "full_code_recodification"]).sum()) if not cls_n.empty else 0
    amend_rows = int(cls_n["issue_class"].isin(["amendment_substantial", "amendment_targeted"]).sum()) if not cls_n.empty else 0
    dedup_full = int(clusters_n["event_type"].eq("full_ordinance").sum()) if not clusters_n.empty else 0
    dedup_amend = int(clusters_n["event_type"].eq("amendment").sum()) if not clusters_n.empty else 0

    metrics = pd.DataFrame(
        [
            ["Issue rows in timeline", issue_count],
            ["Issue rows with non-empty clean zoning text", clean_nonempty],
            ["Issue rows classified full-code", full_rows],
            ["Issue rows classified amendment", amend_rows],
            ["Deduped full-ordinance events", dedup_full],
            ["Deduped amendment events", dedup_amend],
        ],
        columns=["Metric", "Value"],
    )
    _to_latex_table(metrics, tbl_dir / "metrics.tex", index=False, float_fmt="{:.0f}")

    # Class count table + bar plot.
    class_counts = (
        cls_n.groupby("issue_class", dropna=False)["issue_id"].count().reset_index(name="issue_rows").sort_values("issue_rows", ascending=False)
        if not cls_n.empty
        else pd.DataFrame(columns=["issue_class", "issue_rows"])
    )
    _to_latex_table(class_counts, tbl_dir / "class_counts.tex", index=False, float_fmt="{:.0f}")
    if not class_counts.empty:
        fig, ax = plt.subplots(figsize=(8.6, 4.4))
        ax.barh(class_counts["issue_class"], class_counts["issue_rows"], color="#4c78a8")
        ax.invert_yaxis()
        ax.set_xlabel("Issue rows")
        ax.set_title(f"Issue class counts: {newspaper_slug}")
        fig.tight_layout()
        fig.savefig(fig_dir / "class_counts.png", dpi=220)
        plt.close(fig)

    # Dedup example cluster.
    dedup_example = pd.DataFrame(columns=["issue_id", "issue_date", "issue_class", "publication_role"])
    dedup_header = {"event_id": "", "event_type": "", "issue_count": 0}
    if not clusters_n.empty and not assign_n.empty:
        top = clusters_n[clusters_n["issue_count"] > 1].sort_values(["issue_count", "first_issue_date"], ascending=[False, True])
        if not top.empty:
            e = top.iloc[0]
            eid = e.get("ordinance_event_id", "")
            dedup_header = {"event_id": str(eid), "event_type": str(e.get("event_type", "")), "issue_count": int(e.get("issue_count", 0) or 0)}
            dedup_example = assign_n[assign_n["ordinance_event_id"] == eid][["issue_id", "issue_date", "issue_class", "publication_role"]].sort_values("issue_date")
    _to_latex_table(dedup_example, tbl_dir / "dedup_example.tex", index=False, float_fmt="{:.0f}")

    # Choose one full + one amendment issue example by clean_chars.
    full_example = pd.DataFrame()
    amend_example = pd.DataFrame()
    if not cls_n.empty:
        full_example = cls_n[cls_n["issue_class"].isin(["full_code_comprehensive", "full_code_recodification"])].sort_values(
            ["clean_chars", "issue_date"], ascending=[False, True]
        )
        amend_example = cls_n[cls_n["issue_class"].isin(["amendment_substantial", "amendment_targeted"])].sort_values(
            ["clean_chars", "issue_date"], ascending=[False, True]
        )

    png_search = run_root / "reports" / "fullnewspaper_v3" / "figures" / "appendix_pages"
    if not png_search.is_dir():
        png_search = run_root / "reports" / "fullnewspaper_v3" / "figures" / "evidence_pages"

    full_img = None
    amend_img = None
    full_issue_id = ""
    amend_issue_id = ""

    if not full_example.empty:
        for _, r in full_example.iterrows():
            iid = str(r["issue_id"])
            src = _pick_png_for_issue(png_search, iid)
            if src and src.is_file():
                full_issue_id = iid
                full_img = fig_dir / src.name
                full_img.write_bytes(src.read_bytes())
                full_example = pd.DataFrame([r])
                break
        if not full_issue_id:
            full_issue_id = str(full_example.iloc[0]["issue_id"])
            full_example = full_example.head(1)

    if not amend_example.empty:
        for _, r in amend_example.iterrows():
            iid = str(r["issue_id"])
            src = _pick_png_for_issue(png_search, iid)
            if src and src.is_file():
                amend_issue_id = iid
                amend_img = fig_dir / src.name
                amend_img.write_bytes(src.read_bytes())
                amend_example = pd.DataFrame([r])
                break
        if not amend_issue_id:
            amend_issue_id = str(amend_example.iloc[0]["issue_id"])
            amend_example = amend_example.head(1)

    # Pull evidence quote + clean snippet.
    def _extract_quotes(row_df: pd.DataFrame) -> tuple[str, str]:
        if row_df.empty:
            return "", ""
        r = row_df.iloc[0]
        ev_raw = r.get("evidence_quotes", "")
        quote = ""
        try:
            ev = ast.literal_eval(ev_raw) if isinstance(ev_raw, str) else ev_raw
            if isinstance(ev, list) and ev:
                quote = str(ev[0])
            else:
                quote = str(ev_raw)
        except Exception:
            quote = str(ev_raw)
        c = clean_n[clean_n["issue_id"] == r.get("issue_id", "")]
        snippet = str(c.iloc[0]["clean_zoning_text"]) if not c.empty else ""
        return _clip(quote, 420), _clip(snippet, 900)

    full_quote, full_snippet = _extract_quotes(full_example)
    amend_quote, amend_snippet = _extract_quotes(amend_example)

    # Longitudinal extract fields.
    timing = pd.DataFrame(columns=["first_signal_date", "first_signal_type", "first_full_date", "first_full_status"])
    if not long_n.empty:
        timing = long_n[["first_signal_date", "first_signal_type", "first_full_date", "first_full_status"]].copy()
    _to_latex_table(timing, tbl_dir / "timing.tex", index=False, float_fmt="{:.0f}")

    narrative = ""
    if not panel_n.empty:
        parsed = panel_n.iloc[0].get("parsed_json", {})
        if isinstance(parsed, dict):
            narrative = _clip(parsed.get("narrative_summary", ""), 1200)

    # Simple year plot, raw rows vs dedup events for full+amend.
    yearly_raw = pd.DataFrame(columns=["year", "full", "amend"])
    yearly_dedup = pd.DataFrame(columns=["year", "full", "amend"])
    if not assign_n.empty:
        d = assign_n[assign_n["event_type"].isin(["full_ordinance", "amendment"])].copy()
        if not d.empty:
            d["year"] = pd.to_datetime(d["issue_date"], errors="coerce").dt.year
            d = d[d["year"].notna()].copy()
            d["year"] = d["year"].astype(int)
            raw = d.groupby(["year", "event_type"])["issue_id"].count().reset_index(name="n")
            yearly_raw = raw.pivot(index="year", columns="event_type", values="n").fillna(0).reset_index()
            yearly_raw["full"] = yearly_raw.get("full_ordinance", 0).astype(int)
            yearly_raw["amend"] = yearly_raw.get("amendment", 0).astype(int)
            yearly_raw = yearly_raw[["year", "full", "amend"]]

            ded = d.copy()
            ded["event_key"] = ded["newspaper_slug"].astype(str) + "::" + ded["ordinance_event_id"].astype(str)
            ded = ded[ded["ordinance_event_id"].astype(str).ne("")]
            by = ded.groupby(["year", "event_type"])["event_key"].nunique().reset_index(name="n")
            yearly_dedup = by.pivot(index="year", columns="event_type", values="n").fillna(0).reset_index()
            yearly_dedup["full"] = yearly_dedup.get("full_ordinance", 0).astype(int)
            yearly_dedup["amend"] = yearly_dedup.get("amendment", 0).astype(int)
            yearly_dedup = yearly_dedup[["year", "full", "amend"]]

    if not yearly_raw.empty or not yearly_dedup.empty:
        fig, ax = plt.subplots(figsize=(9.2, 4.8))
        if not yearly_raw.empty:
            ax.plot(yearly_raw["year"], yearly_raw["full"], color="#e45756", marker="o", label="Full rows (pre-dedup)")
            ax.plot(yearly_raw["year"], yearly_raw["amend"], color="#4c78a8", marker="o", label="Amend rows (pre-dedup)")
        if not yearly_dedup.empty:
            ax.plot(yearly_dedup["year"], yearly_dedup["full"], color="#e45756", linestyle="--", marker="x", label="Full events (dedup)")
            ax.plot(yearly_dedup["year"], yearly_dedup["amend"], color="#4c78a8", linestyle="--", marker="x", label="Amend events (dedup)")
        ax.set_xlabel("Year")
        ax.set_ylabel("Count")
        ax.set_title(f"Full vs amendment counts over time: {newspaper_slug}")
        ax.legend(ncol=2, fontsize=8)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(fig_dir / "yearly_raw_vs_dedup.png", dpi=220)
        plt.close(fig)

    tex = f"""
\\documentclass[11pt]{{article}}
\\usepackage[margin=1in]{{geometry}}
\\usepackage{{booktabs}}
\\usepackage{{graphicx}}
\\usepackage{{float}}
\\usepackage{{hyperref}}
\\usepackage{{xurl}}
\\setlength{{\\parskip}}{{0.45em}}
\\setlength{{\\parindent}}{{0pt}}
\\title{{Single-Newspaper Demo: {_tex_escape(newspaper_slug)}}}
\\author{{Pipeline demo from completed production run}}
\\date{{{_tex_escape(str(dt.date.today()))}}}
\\begin{{document}}
\\maketitle
\\tableofcontents
\\clearpage

\\section{{Demo scope}}
This document demonstrates the full pipeline on one newspaper with strong coverage.
The demo newspaper is \\texttt{{{_tex_escape(newspaper_slug)}}}.
All results are generated from one completed production run and traced through issue-level extraction, classification, event deduplication, and timeline outputs.

\\begin{{table}}[H]
\\centering
\\caption{{Core demo metrics}}
\\resizebox{{0.72\\textwidth}}{{!}}{{\\input{{tables/metrics.tex}}}}
\\end{{table}}

\\section{{Issue-level extraction and classification}}
The pipeline first extracts clean zoning-law text from mixed newspaper content, then classifies each issue into the zoning ontology.

\\begin{{table}}[H]
\\centering
\\caption{{Issue class counts}}
\\resizebox{{0.60\\textwidth}}{{!}}{{\\input{{tables/class_counts.tex}}}}
\\end{{table}}

\\begin{{figure}}[H]
\\centering
\\includegraphics[width=0.82\\textwidth]{{figures/class_counts.png}}
\\caption{{Issue class distribution for this newspaper.}}
\\end{{figure}}

\\subsection{{Full-ordinance example}}
\\begin{{figure}}[H]
\\centering
{f"\\includegraphics[width=0.78\\textwidth]{{figures/{_tex_escape(full_img.name)}}}" if full_img else "\\fbox{No full-ordinance example page image available}"}
\\caption{{Example page from the full-ordinance stream.}}
\\end{{figure}}
\\textbf{{Issue id:}} \\texttt{{{_tex_escape(full_issue_id)}}}

\\textbf{{Evidence quote:}} {_tex_escape(full_quote or "N/A")}

\\textbf{{Extracted clean legal text snippet:}} {_tex_escape(full_snippet or "N/A")}

\\subsection{{Amendment example}}
\\begin{{figure}}[H]
\\centering
{f"\\includegraphics[width=0.78\\textwidth]{{figures/{_tex_escape(amend_img.name)}}}" if amend_img else "\\fbox{No amendment example page image available}"}
\\caption{{Example page from the amendment stream.}}
\\end{{figure}}
\\textbf{{Issue id:}} \\texttt{{{_tex_escape(amend_issue_id)}}}

\\textbf{{Evidence quote:}} {_tex_escape(amend_quote or "N/A")}

\\textbf{{Extracted clean legal text snippet:}} {_tex_escape(amend_snippet or "N/A")}

\\section{{Event deduplication}}
Repeated newspaper publications are collapsed into one legal event id.

\\begin{{table}}[H]
\\centering
\\caption{{Dedup cluster example (multi-issue event)}}
\\resizebox{{\\textwidth}}{{!}}{{\\input{{tables/dedup_example.tex}}}}
\\end{{table}}

Event id \\texttt{{{_tex_escape(dedup_header.get("event_id", ""))}}} is typed as {_tex_escape(dedup_header.get("event_type", ""))} and contains {_tex_escape(dedup_header.get("issue_count", 0))} publication rows.

\\begin{{figure}}[H]
\\centering
\\includegraphics[width=0.88\\textwidth]{{figures/yearly_raw_vs_dedup.png}}
\\caption{{Raw publication rows vs deduplicated event counts over time.}}
\\end{{figure}}

\\section{{Timeline extraction outputs}}
The timeline extractor reads the full issue sequence and returns first-signal timing, first-full timing, and longitudinal narrative.

\\begin{{table}}[H]
\\centering
\\caption{{First-signal and first-full timing}}
\\resizebox{{0.70\\textwidth}}{{!}}{{\\input{{tables/timing.tex}}}}
\\end{{table}}

\\textbf{{Timeline narrative (excerpt):}} {_tex_escape(narrative or "N/A")}

\\section{{Run-level context}}
This demo is sourced from a completed production run with full inventory:
\\begin{{itemize}}
\\item full-ordinance events (run-wide): {_tex_escape(inv.get("full_ordinance_event_count", inv.get("event_rows_full_ordinance", "")))}
\\item amendment events (run-wide): {_tex_escape(inv.get("amendment_event_count", inv.get("event_rows_amendment", "")))}
\\item law-only issue rows (run-wide): {_tex_escape(inv.get("law_issue_row_count", inv.get("issue_rows_law_only", "")))}
\\item total issue rows (run-wide): {_tex_escape(inv.get("issue_row_count", inv.get("issue_rows_total", "")))}
\\end{{itemize}}

\\end{{document}}
"""

    _write(output_dir / "report.tex", tex.strip() + "\n")

    prov = {
        "created_at": dt.datetime.now().isoformat(),
        "run_root": str(run_root),
        "newspaper_slug": newspaper_slug,
        "output_dir": str(output_dir),
        "full_issue_example": full_issue_id,
        "amend_issue_example": amend_issue_id,
        "dedup_example": dedup_header,
    }
    _write(output_dir / "provenance.json", json.dumps(prov, indent=2))

    compiled = _compile_pdf(output_dir) if compile_pdf else False
    return {
        "compiled_pdf": compiled,
        "report_tex": str(output_dir / "report.tex"),
        "report_pdf": str(output_dir / "report.pdf"),
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build single-newspaper demo latex report.")
    ap.add_argument("--run-root", type=Path, required=True)
    ap.add_argument("--newspaper-slug", type=str, required=True)
    ap.add_argument("--output-dir", type=Path, default=None)
    ap.add_argument("--no-compile-pdf", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    run_root = args.run_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve() if args.output_dir else (run_root / "reports" / f"demo_{args.newspaper_slug}")
    out = build(run_root, args.newspaper_slug.strip(), output_dir, compile_pdf=not args.no_compile_pdf)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
