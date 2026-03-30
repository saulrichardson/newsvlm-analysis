#!/usr/bin/env python3
"""
Describe an RHS regression panel (city×time, etc.) and emit a lightweight feature catalog.

This is intentionally "artifact-first": it inspects an already-built panel (Parquet/CSV)
and produces machine-readable + human-readable summaries so we can decide later:
  - which mechanisms look like plausible "treatments" or exposures
  - which tags are too sparse/noisy
  - what time coverage and unit coverage look like
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
from pathlib import Path

import pandas as pd


def _load(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise SystemExit(f"Missing input: {path}")
    suf = path.suffix.lower()
    if suf == ".parquet":
        return pd.read_parquet(path)
    if suf == ".csv":
        return pd.read_csv(path)
    raise SystemExit(f"Unsupported input format: {path} (expected .parquet or .csv)")


def _group_for_column(col: str) -> str:
    if col.startswith("d_action__"):
        return "delta_action"
    if col.startswith("d_dim__"):
        return "delta_dimension"
    if col.startswith("d_body__"):
        return "delta_body"
    if col.startswith("d_instr__"):
        return "delta_instrument"
    if col.startswith("d_zr__"):
        return "delta_zoning_relevance"
    if col.startswith("action__"):
        return "action"
    if col.startswith("dim__"):
        return "dimension"
    if col.startswith("body__"):
        return "decision_body"
    if col.startswith("instr__"):
        return "instrument"
    if col.startswith("zr__"):
        return "zoning_relevance"
    if col.startswith("doc_weight_"):
        return "quality_doc_weight"
    if col in {"n_issues", "chunks_total", "chunk_chars_total"}:
        return "coverage"
    return "other"


def _is_numeric(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s.dtype)


def _describe_numeric(s: pd.Series) -> dict[str, float | int | None]:
    s = pd.to_numeric(s, errors="coerce")
    if s.notna().sum() == 0:
        return {
            "mean": None,
            "std": None,
            "min": None,
            "p25": None,
            "p50": None,
            "p75": None,
            "max": None,
            "pct_nonzero": None,
        }
    q = s.quantile([0.25, 0.50, 0.75], interpolation="linear")
    return {
        "mean": float(s.mean()),
        "std": float(s.std(ddof=0)),
        "min": float(s.min()),
        "p25": float(q.loc[0.25]),
        "p50": float(q.loc[0.50]),
        "p75": float(q.loc[0.75]),
        "max": float(s.max()),
        "pct_nonzero": float((s.fillna(0.0) != 0.0).mean()),
    }


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Describe a RHS panel and emit feature catalog artifacts.")
    ap.add_argument("--input", required=True, help="Path to RHS panel (.parquet or .csv).")
    ap.add_argument("--output-dir", required=True, help="Directory to write outputs.")
    ap.add_argument(
        "--unit-cols",
        default="city_name,state_abbr",
        help="Comma-separated columns defining the unit (for coverage stats only).",
    )
    ap.add_argument(
        "--time-col",
        default="bucket_month",
        help="Time bucket column name (for coverage stats only).",
    )
    ap.add_argument("--top-k", type=int, default=25, help="How many top features to show in the markdown summary.")
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    input_path = Path(args.input).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    df = _load(input_path)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")

    unit_cols = [c.strip() for c in str(args.unit_cols).split(",") if c.strip()]
    time_col = str(args.time_col).strip()

    for c in unit_cols + ([time_col] if time_col else []):
        if c and c not in df.columns:
            raise SystemExit(f"--unit-cols/--time-col references missing column: {c}")

    # Dataset-level stats.
    stats: dict[str, object] = {
        "created": stamp,
        "input": str(input_path),
        "rows": int(len(df)),
        "cols": int(len(df.columns)),
        "time_col": time_col,
        "unit_cols": unit_cols,
    }
    if unit_cols:
        stats["n_units"] = int(df[unit_cols].dropna().drop_duplicates().shape[0])
    if time_col:
        stats["n_time_buckets"] = int(df[time_col].dropna().nunique())
        stats["time_min"] = None if df[time_col].dropna().empty else str(df[time_col].dropna().min())
        stats["time_max"] = None if df[time_col].dropna().empty else str(df[time_col].dropna().max())

    # Per-column catalog.
    rows: list[dict[str, object]] = []
    for col in df.columns:
        s = df[col]
        row: dict[str, object] = {
            "column": col,
            "group": _group_for_column(col),
            "dtype": str(s.dtype),
            "pct_nonnull": float(s.notna().mean()),
        }
        if _is_numeric(s):
            row.update(_describe_numeric(s))
        rows.append(row)

    catalog = pd.DataFrame.from_records(rows).sort_values(["group", "column"]).reset_index(drop=True)

    # Write machine-readable.
    (out_dir / f"rhs_panel_stats_{stamp}.json").write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    catalog.to_csv(out_dir / f"rhs_feature_catalog_{stamp}.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    catalog.to_parquet(out_dir / f"rhs_feature_catalog_{stamp}.parquet", index=False)

    # Write markdown summary.
    num = catalog[catalog["group"].isin(["action", "dimension", "decision_body", "instrument", "zoning_relevance"])].copy()
    if not num.empty and "std" in num.columns:
        num = num.sort_values("std", ascending=False)

    top_k = int(args.top_k)
    top = num.head(top_k) if top_k > 0 else num.head(0)

    def _fmt(x: object, *, digits: int = 4) -> str:
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return "NA"
        try:
            return f"{float(x):.{digits}f}"
        except Exception:
            return "NA"

    lines: list[str] = []
    lines.append("# RHS panel summary")
    lines.append("")
    lines.append(f"Created: {stamp}")
    lines.append(f"Input: `{input_path}`")
    lines.append("")
    lines.append("## Coverage")
    lines.append(f"- rows: {stats.get('rows')}")
    lines.append(f"- cols: {stats.get('cols')}")
    if "n_units" in stats:
        lines.append(f"- units ({','.join(unit_cols)}): {stats.get('n_units')}")
    if "n_time_buckets" in stats:
        lines.append(f"- time buckets ({time_col}): {stats.get('n_time_buckets')} ({stats.get('time_min')} → {stats.get('time_max')})")
    lines.append("")
    lines.append("## Top features by time-series variability (std)")
    if top.empty:
        lines.append("- (none)")
    else:
        for _, r in top.iterrows():
            lines.append(
                f"- `{r['column']}` ({r['group']}): mean={_fmt(r.get('mean'))} std={_fmt(r.get('std'))} pct_nonzero={_fmt(r.get('pct_nonzero'), digits=3)}"
            )
    lines.append("")
    (out_dir / f"rhs_panel_summary_{stamp}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Done. wrote={out_dir} stamp={stamp}")


if __name__ == "__main__":
    main()
