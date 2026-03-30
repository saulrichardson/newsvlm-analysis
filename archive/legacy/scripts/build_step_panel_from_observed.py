#!/usr/bin/env python3
"""
Convert an *observed* unit×time panel into a step-function panel by forward-filling.

This is useful when the underlying process is "policy updates happen on discrete event dates"
and you want a DiD-friendly panel where the RHS variables are piecewise-constant between events.

Example use:
  - mechanics-tag shares per city-month (only months with newspaper issues)
  -> step-function city-month panel where missing months are filled forward.

This script is deterministic and does not call external APIs.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
from pathlib import Path
from typing import Literal

import pandas as pd


Bucket = Literal["month", "year"]


def _month_index(bucket_ym: str) -> int:
    y, m = bucket_ym.split("-", 1)
    return int(y) * 12 + (int(m) - 1)


def _year_index(bucket_y: str) -> int:
    return int(bucket_y)


def _load_table(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise SystemExit(f"Missing input: {path}")
    suf = path.suffix.lower()
    if suf == ".parquet":
        return pd.read_parquet(path)
    if suf == ".csv":
        return pd.read_csv(path)
    raise SystemExit(f"Unsupported input format: {path} (expected .parquet or .csv)")


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build a step-function panel from an observed unit×time panel.")
    ap.add_argument("--input", required=True, help="Observed panel (.parquet or .csv).")
    ap.add_argument("--output-dir", required=True, help="Directory to write outputs.")
    ap.add_argument(
        "--unit-cols",
        default="city_name,state_abbr",
        help="Comma-separated unit columns (default: city_name,state_abbr).",
    )
    ap.add_argument(
        "--bucket-col",
        default="bucket_month",
        help="Time bucket column name (default: bucket_month).",
    )
    ap.add_argument("--bucket", choices=["month", "year"], default="month", help="Bucket type for indexing.")
    ap.add_argument(
        "--fill-prefix",
        action="append",
        default=["action__", "dim__", "body__", "instr__", "zr__"],
        help=(
            "Column prefix to forward-fill (repeatable). "
            "Default fills action__/dim__/body__/instr__/zr__ shares."
        ),
    )
    ap.add_argument(
        "--count-col",
        default="n_issues",
        help="Name of observation-count column; will be set to 0 when missing (default: n_issues).",
    )
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    input_path = Path(args.input).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    df = _load_table(input_path)

    unit_cols = [c.strip() for c in str(args.unit_cols).split(",") if c.strip()]
    bucket_col = str(args.bucket_col).strip()
    count_col = str(args.count_col).strip()
    bucket: Bucket = str(args.bucket)  # type: ignore[assignment]

    for c in unit_cols + [bucket_col]:
        if c not in df.columns:
            raise SystemExit(f"Missing required column: {c}")

    fill_prefixes = [str(p) for p in args.fill_prefix]
    fill_cols = [c for c in df.columns if any(c.startswith(p) for p in fill_prefixes)]
    if not fill_cols:
        raise SystemExit(f"No columns match --fill-prefix={fill_prefixes}")

    idx_fn = _month_index if bucket == "month" else _year_index

    # Build a bucket grid per unit between its min and max observed buckets.
    rows = []
    df = df.copy()
    df["_t_idx"] = df[bucket_col].astype(str).apply(idx_fn)

    for key, g in df.groupby(unit_cols, sort=False):
        g = g.sort_values("_t_idx")
        imin = int(g["_t_idx"].min())
        imax = int(g["_t_idx"].max())
        if imax < imin:
            continue

        if bucket == "year":
            buckets = [str(y) for y in range(imin, imax + 1)]
        else:
            buckets = []
            for mi in range(imin, imax + 1):
                y = mi // 12
                m = (mi % 12) + 1
                buckets.append(f"{y:04d}-{m:02d}")

        key_dict = {k: v for k, v in zip(unit_cols, key if isinstance(key, tuple) else (key,))}
        base = pd.DataFrame({**key_dict, bucket_col: buckets})
        merged = base.merge(g.drop(columns=["_t_idx"]), on=unit_cols + [bucket_col], how="left")

        merged["observed"] = 0
        if count_col in merged.columns:
            merged["observed"] = merged[count_col].fillna(0).astype(float).gt(0).astype(int)
            merged[count_col] = merged[count_col].fillna(0).astype(int)
        else:
            # If there is no count column, treat any non-null in fill_cols as "observed".
            merged["observed"] = merged[fill_cols].notna().any(axis=1).astype(int)

        # Forward fill policy share columns.
        merged[fill_cols] = merged[fill_cols].ffill()
        rows.append(merged)

    if not rows:
        raise SystemExit("No rows produced (unexpected).")

    out = pd.concat(rows, ignore_index=True)
    out["_t_idx"] = out[bucket_col].astype(str).apply(idx_fn)
    out = out.sort_values(unit_cols + ["_t_idx"]).drop(columns=["_t_idx"]).reset_index(drop=True)

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"step_panel_{stamp}"
    out_csv = out_dir / f"{stem}.csv"
    out.to_csv(out_csv, index=False, quoting=csv.QUOTE_MINIMAL)
    try:
        out.to_parquet(out_dir / f"{stem}.parquet", index=False)
    except Exception:
        pass

    print(f"Done. wrote={out_dir} rows={len(out)} cols={len(out.columns)}")


if __name__ == "__main__":
    main()

