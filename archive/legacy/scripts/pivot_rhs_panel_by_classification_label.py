#!/usr/bin/env python3
"""
Pivot a stratified panel (city×time×classification_label) into a wide city×time panel
with classification_label-specific feature columns.

Motivation:
  - You often want city-by-time as the canonical unit.
  - But ordinance types (full ordinance vs targeted/substantial amendments) can have
    distinct mechanics, and it can be useful to keep them separated as exposures.

This script is deterministic: it never calls an LLM or invents labels; it only reshapes.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
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


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Pivot RHS panel by classification_label into wide city×time.")
    ap.add_argument("--input", required=True, help="Input stratified panel (.parquet or .csv). Must include classification_label.")
    ap.add_argument("--output-dir", required=True, help="Directory to write outputs.")
    ap.add_argument(
        "--index-cols",
        default="city_name,state_abbr,bucket_month",
        help="Comma-separated index columns for the wide output (default: city_name,state_abbr,bucket_month).",
    )
    ap.add_argument(
        "--label-col",
        default="classification_label",
        help="Column name containing the classification label to pivot on.",
    )
    ap.add_argument(
        "--include-cols",
        default="",
        help=(
            "Optional comma-separated list of non-numeric columns to carry through using 'first' per index. "
            "Useful for stable IDs like state_id,city_id."
        ),
    )
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    input_path = Path(args.input).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    df = _load(input_path)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")

    label_col = str(args.label_col).strip()
    if label_col not in df.columns:
        raise SystemExit(f"Missing label column: {label_col}")

    index_cols = [c.strip() for c in str(args.index_cols).split(",") if c.strip()]
    if not index_cols:
        raise SystemExit("--index-cols cannot be empty")
    missing_idx = [c for c in index_cols if c not in df.columns]
    if missing_idx:
        raise SystemExit(f"Missing index columns: {missing_idx}")

    include_cols = [c.strip() for c in str(args.include_cols).split(",") if c.strip()]
    missing_inc = [c for c in include_cols if c not in df.columns]
    if missing_inc:
        raise SystemExit(f"Missing --include-cols columns: {missing_inc}")

    # Identify numeric feature columns to pivot.
    ignore = set(index_cols + [label_col])
    feat_cols = [c for c in df.columns if c not in ignore and pd.api.types.is_numeric_dtype(df[c].dtype)]
    if not feat_cols:
        raise SystemExit("No numeric feature columns found to pivot.")

    # Pivot to MultiIndex columns: (feature, classification_label) then flatten to {label}__{feature}
    wide = (
        df[index_cols + [label_col] + feat_cols]
        .pivot_table(index=index_cols, columns=label_col, values=feat_cols, aggfunc="first")
    )
    wide.columns = [f"{str(lbl)}__{str(feat)}" for feat, lbl in wide.columns.to_list()]
    wide = wide.reset_index()

    # Carry through stable non-numeric IDs if requested.
    if include_cols:
        extra = df[index_cols + include_cols].drop_duplicates(subset=index_cols, keep="first")
        wide = wide.merge(extra, on=index_cols, how="left")

    out_parquet = out_dir / f"rhs_city_time_wide_by_label_{stamp}.parquet"
    out_csv = out_dir / f"rhs_city_time_wide_by_label_{stamp}.csv"
    wide.to_parquet(out_parquet, index=False)
    wide.to_csv(out_csv, index=False, quoting=csv.QUOTE_MINIMAL)

    print(f"Done. wrote={out_dir} rows={len(wide)} cols={len(wide.columns)}")


if __name__ == "__main__":
    main()

