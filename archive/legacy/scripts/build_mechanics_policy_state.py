#!/usr/bin/env python3
"""
Build low-dimensional "policy state" indices from mechanics-tag share panels.

This mirrors scripts/build_city_policy_state_from_embeddings.py, but operates on the
interpretable mechanics-tag panel produced by scripts/build_regression_rhs_from_topics.py.

Use cases:
  - Create PCA indices (mechanics_pc1..k) that summarize zoning mechanism content over time.
  - Compute jump sizes (mechanics_jump_cosine_pca / mechanics_jump_l2_pca) for event-style designs.
  - Compute "complexity" measures like entropy/effective-number across dimensions/actions.

This script is deterministic and does not call external APIs.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def _load_table(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise SystemExit(f"Missing input: {path}")
    suf = path.suffix.lower()
    if suf == ".parquet":
        return pd.read_parquet(path)
    if suf == ".csv":
        return pd.read_csv(path)
    raise SystemExit(f"Unsupported input format: {path} (expected .parquet or .csv)")


def _month_index(bucket_ym: str) -> int:
    y, m = str(bucket_ym).split("-", 1)
    return int(y) * 12 + (int(m) - 1)


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    an = float(np.linalg.norm(a))
    bn = float(np.linalg.norm(b))
    if an == 0.0 or bn == 0.0:
        return float("nan")
    return float(1.0 - float(np.dot(a, b) / (an * bn)))


def _entropy_from_nonneg(v: np.ndarray) -> float:
    """
    Shannon entropy of a non-negative vector after normalization.
    Returns NaN if sum is zero.
    """
    v = np.asarray(v, dtype=np.float64)
    v = np.clip(v, 0.0, np.inf)
    s = float(v.sum())
    if s <= 0.0:
        return float("nan")
    p = v / s
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build mechanics-based policy state indices and jump measures.")
    ap.add_argument("--input-panel", required=True, help="Mechanics city×time panel (.parquet or .csv).")
    ap.add_argument("--output-dir", required=True, help="Directory to write outputs.")

    ap.add_argument("--unit-cols", default="city_name,state_abbr", help="Comma-separated unit columns.")
    ap.add_argument("--bucket-col", default="bucket_month", help="Time bucket column name.")
    ap.add_argument(
        "--feature-prefix",
        action="append",
        default=["action__", "dim__"],
        help="Prefix of feature columns to include in PCA (repeatable). Default: action__ + dim__.",
    )
    ap.add_argument(
        "--fit-on-observed-only",
        action="store_true",
        help="Fit PCA only on rows where observed==1 (if observed column exists).",
    )
    ap.add_argument("--pca-components", type=int, default=8, help="How many PCA components to keep.")
    ap.add_argument(
        "--output-format",
        choices=["csv", "parquet", "both"],
        default="both",
        help="Output file format. Parquet requires a parquet engine.",
    )
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    in_path = Path(args.input_panel).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    df = _load_table(in_path)

    unit_cols = [c.strip() for c in str(args.unit_cols).split(",") if c.strip()]
    bucket_col = str(args.bucket_col).strip()
    for c in unit_cols + [bucket_col]:
        if c not in df.columns:
            raise SystemExit(f"Missing required column: {c}")

    prefixes = [str(p) for p in args.feature_prefix]
    feat_cols = [c for c in df.columns if any(c.startswith(p) for p in prefixes)]
    if not feat_cols:
        raise SystemExit(f"No feature columns found for --feature-prefix={prefixes}")

    # Prepare design matrix.
    X = df[feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    if X.ndim != 2 or X.shape[0] != len(df):
        raise SystemExit("Internal error: feature matrix shape mismatch")

    n_components = int(args.pca_components)
    if n_components <= 0:
        raise SystemExit("--pca-components must be > 0")
    if n_components > X.shape[1]:
        raise SystemExit(f"--pca-components too large: {n_components} > n_features={X.shape[1]}")

    fit_mask = np.ones((len(df),), dtype=bool)
    if bool(args.fit_on_observed_only) and "observed" in df.columns:
        fit_mask = pd.to_numeric(df["observed"], errors="coerce").fillna(0.0).to_numpy(dtype=float) > 0.0
        if int(fit_mask.sum()) < max(10, n_components * 3):
            raise SystemExit(
                f"Too few observed rows to fit PCA: observed_rows={int(fit_mask.sum())}. "
                "Rerun without --fit-on-observed-only or check the input panel."
            )

    # PCA over mechanics shares.
    #
    # Our mechanics feature matrices are typically modest width (dozens to low hundreds),
    # so using full SVD is stable and avoids rare randomized-SVD numerical warnings.
    # Keep deterministic behavior regardless.
    svd_solver = "full" if X.shape[1] <= 250 else "randomized"
    pca = PCA(n_components=n_components, svd_solver=svd_solver, random_state=0)
    pca.fit(X[fit_mask].astype(np.float64, copy=False))
    # Some BLAS/numpy builds can emit spurious RuntimeWarnings during large matrix multiplies
    # even when the final values are finite. Suppress them to keep runs clean.
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        scores = pca.transform(X.astype(np.float64, copy=False)).astype(np.float32)

    # Add PCA scores.
    out = df[unit_cols + [bucket_col]].copy()
    for i in range(n_components):
        out[f"mechanics_pc{i+1}"] = scores[:, i]

    # Add entropy / effective-number measures for interpretability.
    # NOTE: mechanics shares are not guaranteed to sum to 1 because tags can co-occur; we normalize within prefix.
    action_cols = [c for c in df.columns if c.startswith("action__")]
    dim_cols = [c for c in df.columns if c.startswith("dim__")]
    if action_cols:
        A = df[action_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        ent = np.array([_entropy_from_nonneg(r) for r in A], dtype=np.float32)
        out["action_entropy"] = ent
        out["action_effective_n"] = np.exp(ent.astype(np.float64)).astype(np.float32)
        out["action_mass"] = A.sum(axis=1).astype(np.float32)
    if dim_cols:
        D = df[dim_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        ent = np.array([_entropy_from_nonneg(r) for r in D], dtype=np.float32)
        out["dim_entropy"] = ent
        out["dim_effective_n"] = np.exp(ent.astype(np.float64)).astype(np.float32)
        out["dim_mass"] = D.sum(axis=1).astype(np.float32)

    # Carry observation count columns if present.
    for c in ["observed", "n_issues"]:
        if c in df.columns:
            out[c] = pd.to_numeric(df[c], errors="coerce")

    # Compute within-unit jump sizes on mechanics PC vectors.
    out["_t_idx"] = out[bucket_col].astype(str).apply(_month_index)
    out = out.sort_values(unit_cols + ["_t_idx"]).reset_index(drop=True)

    pc_cols = [f"mechanics_pc{i+1}" for i in range(n_components)]

    def _jump_metrics(g: pd.DataFrame) -> pd.DataFrame:
        g = g.copy()
        prev = g[pc_cols].shift(1)
        cur = g[pc_cols]
        cd = []
        l2 = []
        for a, b in zip(cur.to_numpy(), prev.to_numpy()):
            if np.any(np.isnan(b)) or np.any(np.isnan(a)):
                cd.append(np.nan)
                l2.append(np.nan)
                continue
            cd.append(_cosine_distance(a.astype(np.float32), b.astype(np.float32)))
            l2.append(float(np.linalg.norm((a - b).astype(np.float32))))
        g["mechanics_jump_cosine_pca"] = np.array(cd, dtype=np.float32)
        g["mechanics_jump_l2_pca"] = np.array(l2, dtype=np.float32)
        return g

    # Pandas 2.3+ deprecates including grouping columns inside apply(). We don't
    # need them inside _jump_metrics, so exclude to keep runs clean.
    out = out.groupby(unit_cols, group_keys=False, sort=False).apply(_jump_metrics, include_groups=False)
    out = out.drop(columns=["_t_idx"])

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"mechanics_policy_state_{stamp}"
    out_csv = out_dir / f"{stem}.csv"
    out.to_csv(out_csv, index=False, quoting=csv.QUOTE_MINIMAL)
    if str(args.output_format) in {"parquet", "both"}:
        try:
            out.to_parquet(out_dir / f"{stem}.parquet", index=False)
        except Exception as e:  # noqa: BLE001
            raise SystemExit(
                f"Failed to write parquet output (missing engine?): {out_dir / f'{stem}.parquet'}\n"
                f"Error: {e}\n"
                "Fix: install pyarrow or rerun with --output-format csv."
            ) from e

    pca_info = {
        "created_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "input": str(in_path),
        "unit_cols": unit_cols,
        "bucket_col": bucket_col,
        "feature_prefix": prefixes,
        "n_rows": int(len(df)),
        "n_features": int(X.shape[1]),
        "fit_on_observed_only": bool(args.fit_on_observed_only) and ("observed" in df.columns),
        "observed_rows_fit": int(fit_mask.sum()),
        "n_components": int(n_components),
        "explained_variance_ratio": [float(x) for x in pca.explained_variance_ratio_],
    }
    (out_dir / f"pca_info_{stamp}.json").write_text(json.dumps(pca_info, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    np.savez_compressed(
        out_dir / f"pca_model_{stamp}.npz",
        components_=pca.components_.astype(np.float32),
        mean_=pca.mean_.astype(np.float32),
        explained_variance_ratio_=pca.explained_variance_ratio_.astype(np.float32),
        feature_cols=np.array(feat_cols, dtype=object),
    )

    print(f"Done. wrote={out_dir} rows={len(out)} cols={len(out.columns)} features={len(feat_cols)}")


if __name__ == "__main__":
    main()
