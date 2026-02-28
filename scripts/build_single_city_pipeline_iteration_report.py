#!/usr/bin/env python3
"""
Build a concise iteration report for a single-city pipeline run.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any

import pandas as pd


def _norm(x: Any) -> str:
    return str(x or "").strip()


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _fmt_float(v: Any, nd: int = 3) -> str:
    try:
        x = float(v)
    except Exception:
        return "NA"
    if pd.isna(x):
        return "NA"
    return f"{x:.{nd}f}"


def _md_table(df: pd.DataFrame, cols: list[str] | None = None, max_rows: int = 20) -> str:
    if df.empty:
        return "_(no rows)_"
    d = df.copy()
    if cols:
        keep = [c for c in cols if c in d.columns]
        d = d[keep]
    d = d.head(max_rows).copy()
    columns = [str(c) for c in d.columns]
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows: list[str] = []
    for _, row in d.iterrows():
        cells: list[str] = []
        for c in columns:
            v = row.get(c, "")
            if pd.isna(v):
                s = ""
            else:
                s = str(v)
            s = s.replace("\n", " ").replace("|", "\\|").strip()
            cells.append(s)
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, sep, *rows])


def _best_variant(scorecard: pd.DataFrame) -> str:
    if scorecard.empty or "variant" not in scorecard.columns:
        return ""
    d = scorecard.sort_values(
        ["mean_overall_quality", "parse_valid_share", "mean_coverage_component", "variant"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    return _norm(d.iloc[0]["variant"]) if not d.empty else ""


def _load_first_row(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        s = raw.strip()
        if not s:
            continue
        try:
            obj = json.loads(s)
        except Exception:
            continue
        if isinstance(obj, dict):
            return obj
    return {}


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build single-city pipeline iteration report.")
    ap.add_argument(
        "--pipeline-dir",
        default="/Users/saulrichardson/projects/newspapers/newspaper-analysis/reports/runs/pi_v1_run60_fullnewspaper/single_city_pipeline/levittown_smoke_20260220",
    )
    ap.add_argument("--output-name", default="ITERATION_REPORT.md")
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    base = Path(args.pipeline_dir).expanduser().resolve()
    if not base.is_dir():
        raise SystemExit(f"Missing pipeline dir: {base}")

    ontology_summary = _safe_read_csv(base / "ontology_ab" / "ab_variant_summary.csv")
    ontology_docclass = _safe_read_csv(base / "ontology_ab" / "ab_variant_b_doc_class_distribution.csv")
    ontology_merged = _safe_read_csv(base / "ontology_ab" / "ab_outputs_merged.csv")

    extraction_comp = _safe_read_csv(base / "extraction" / "experiment" / "method_comparison.csv")
    city_llm_only = _safe_read_csv(base / "extraction" / "llm_only" / "city_metrics.csv")
    city_llm_hybrid = _safe_read_csv(base / "extraction" / "llm_hybrid" / "city_metrics.csv")

    sc_llm_only = _safe_read_csv(base / "longitudinal" / "llm_only" / "variant_scorecard.csv")
    sc_llm_hybrid = _safe_read_csv(base / "longitudinal" / "llm_hybrid" / "variant_scorecard.csv")
    sc_llm_hybrid_retry = _safe_read_csv(base / "longitudinal" / "llm_hybrid_retry_openai" / "variant_scorecard.csv")

    best_only = _best_variant(sc_llm_only)
    best_hybrid = _best_variant(sc_llm_hybrid)
    best_hybrid_retry = _best_variant(sc_llm_hybrid_retry)

    psummary = _safe_read_csv(base / "pipeline_summary.csv")
    prov_path = base / "pipeline_provenance.json"
    prov = json.loads(prov_path.read_text(encoding="utf-8")) if prov_path.is_file() else {}

    city_key = _norm(prov.get("city_key") or "")
    created = dt.datetime.now().isoformat(timespec="seconds")

    b_projection_counts = (
        ontology_merged["b_legacy_projection_label"].astype(str).value_counts().rename_axis("b_legacy_projection_label").reset_index(name="count")
        if (not ontology_merged.empty and "b_legacy_projection_label" in ontology_merged.columns)
        else pd.DataFrame()
    )

    lines: list[str] = []
    lines.append(f"# Single-City Iteration Report: {city_key or 'unknown_city'}")
    lines.append("")
    lines.append(f"Generated: `{created}`")
    lines.append("")
    lines.append("## 1) What was tested")
    lines.append("- Issue-type discovery ontology: baseline `A` vs stricter ontology `B`.")
    lines.append("- Zoning text extraction: `llm_only` (two-pass LLM) vs `llm_hybrid` (rules-recall candidates + LLM cleaning).")
    lines.append("- Longitudinal panel extraction variants: `compact_closed_v2`, `compact_numeric_quote_v1`, `panel_lean_numeric_quote_v1`.")
    lines.append("- Provider robustness check: retried failed `llm_hybrid` variants with `openai:gpt-5-mini`.")
    lines.append("")

    lines.append("## 2) Ontology discovery (issue classification)")
    lines.append(_md_table(ontology_summary))
    lines.append("")
    lines.append("Top ontology-B document classes:")
    lines.append(_md_table(ontology_docclass))
    lines.append("")
    lines.append("Ontology-B projected legacy labels:")
    lines.append(_md_table(b_projection_counts))
    lines.append("")

    lines.append("## 3) Zoning text extraction comparison")
    lines.append(_md_table(extraction_comp))
    lines.append("")
    if not extraction_comp.empty:
        d = extraction_comp.sort_values(["score", "rank"], ascending=[False, True]).reset_index(drop=True)
        best_method = _norm(d.iloc[0]["method"])
        lines.append(f"Best extraction method by composite score: `{best_method}`")
        lines.append("")
    lines.append("City metrics (`llm_only`):")
    lines.append(_md_table(city_llm_only))
    lines.append("")
    lines.append("City metrics (`llm_hybrid`):")
    lines.append(_md_table(city_llm_hybrid))
    lines.append("")

    lines.append("## 4) Longitudinal variant outcomes")
    lines.append("`llm_only` scorecard:")
    lines.append(_md_table(sc_llm_only))
    lines.append("")
    lines.append("`llm_hybrid` scorecard (Gemini):")
    lines.append(_md_table(sc_llm_hybrid))
    lines.append("")
    lines.append("`llm_hybrid` retry scorecard (OpenAI):")
    lines.append(_md_table(sc_llm_hybrid_retry))
    lines.append("")
    lines.append(f"Best `llm_only` variant: `{best_only or 'NA'}`")
    lines.append(f"Best `llm_hybrid` variant under Gemini: `{best_hybrid or 'NA'}`")
    lines.append(f"Best `llm_hybrid` variant under OpenAI retry: `{best_hybrid_retry or 'NA'}`")
    lines.append("")

    lines.append("## 5) Provider/runtime failure diagnosis")
    err_rows: list[dict[str, Any]] = []
    for v in ["compact_numeric_quote_v1", "panel_lean_numeric_quote_v1"]:
        p = base / "longitudinal" / "llm_hybrid" / "experiments" / v / "sample" / "results" / "openai_errors_shard000.jsonl"
        rec = _load_first_row(p)
        if rec:
            body = ((rec.get("response") or {}).get("body") or {}) if isinstance(rec, dict) else {}
            err = (body.get("error") or {}) if isinstance(body, dict) else {}
            err_rows.append(
                {
                    "variant": v,
                    "error_type": _norm(err.get("type")),
                    "error_message": _norm(err.get("message"))[:220],
                }
            )
    lines.append(_md_table(pd.DataFrame.from_records(err_rows)))
    lines.append("")
    lines.append("Interpretation: numeric-heavy variants for `llm_hybrid` hit gateway 500s under Gemini in this run, but succeeded under OpenAI retry. This points to provider/runtime fragility rather than guaranteed prompt failure.")
    lines.append("")

    lines.append("## 6) Current recommendation (single-city pilot)")
    lines.append("1. Keep ontology `B` for issue discovery/classification.")
    lines.append("2. Keep both extraction outputs for now:")
    lines.append("   - `llm_hybrid` as high-precision zoning corpus.")
    lines.append("   - `llm_only` as high-recall comparison corpus.")
    lines.append("3. For longitudinal extraction, use `panel_lean_numeric_quote_v1` as primary; keep `compact_closed_v2` as fallback.")
    lines.append("4. For production scaling, use provider fallback (Gemini -> OpenAI) for numeric-heavy variants.")
    lines.append("")

    lines.append("## 7) Key artifacts")
    lines.append(f"- Pipeline summary: `{base / 'pipeline_summary.csv'}`")
    lines.append(f"- Ontology outputs: `{base / 'ontology_ab'}`")
    lines.append(f"- Extraction outputs: `{base / 'extraction'}`")
    lines.append(f"- Longitudinal outputs (`llm_only`): `{base / 'longitudinal' / 'llm_only'}`")
    lines.append(f"- Longitudinal outputs (`llm_hybrid`): `{base / 'longitudinal' / 'llm_hybrid'}`")
    lines.append(f"- Longitudinal retry outputs (`llm_hybrid_retry_openai`): `{base / 'longitudinal' / 'llm_hybrid_retry_openai'}`")
    lines.append(f"- Manual review packets: `{base / 'manual_review'}`")
    lines.append("")

    out_path = base / _norm(args.output_name)
    out_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
