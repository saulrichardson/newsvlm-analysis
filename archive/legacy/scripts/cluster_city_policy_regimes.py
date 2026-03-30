#!/usr/bin/env python3
"""
Cluster city×time "policy states" into discrete regimes (types) for DiD/event-study style designs.

This implements RHS option (A): regime/type indicators:
  - Cluster city×time vectors (e.g., embedding PCs, mechanics PCs) into K regimes.
  - Define treatment events as "entering regime r" or "switching regimes".

Inputs:
  - A city×time panel (.parquet or .csv) with columns:
      city_name, state_abbr, bucket_month (or chosen bucket col)
    plus numeric features to cluster on (e.g., pc1..pcN, mechanics_pc1..).

Outputs:
  - regime_assignments_<stamp>.parquet/.csv (city×time with regime_k{K} columns)
  - regime_summaries_k{K}_<stamp>.csv (per-regime feature means + diagnostics)
  - regime_events_k{K}_<stamp>.csv (per-city regime switch events)
  - figures/*.png (optional per-city regime timelines)

Deterministic:
  - Uses KMeans(random_state=0, n_init=20).
  - Uses StandardScaler on clustering features.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


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


def _safe_slug(s: str) -> str:
    s = str(s).strip().lower()
    out = []
    for ch in s:
        if ch.isalnum():
            out.append(ch)
        elif ch in {" ", "-", "_"}:
            out.append("_")
    t = "".join(out).strip("_")
    while "__" in t:
        t = t.replace("__", "_")
    return t[:120] or "x"


def _parse_k_list(s: str) -> list[int]:
    out = []
    for part in str(s).split(","):
        p = part.strip()
        if not p:
            continue
        out.append(int(p))
    if not out:
        raise SystemExit("--k-list must contain at least one integer")
    if any(k <= 1 for k in out):
        raise SystemExit("--k-list values must be >= 2")
    return out


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Cluster city×time policy states into discrete regimes.")
    ap.add_argument("--input-panel", required=True, help="City×time panel (.parquet or .csv).")
    ap.add_argument("--output-dir", required=True, help="Directory to write outputs.")
    ap.add_argument("--unit-cols", default="city_name,state_abbr", help="Comma-separated unit columns.")
    ap.add_argument("--bucket-col", default="bucket_month", help="Time bucket column name.")
    ap.add_argument("--k-list", default="4,6,8", help="Comma-separated K values to try (default: 4,6,8).")
    ap.add_argument(
        "--cluster-cols",
        default="pc1,pc2,pc3,mechanics_pc1,mechanics_pc2,mechanics_pc3",
        help=(
            "Comma-separated feature columns used for clustering. "
            "Columns not present are ignored; must leave at least 2 columns."
        ),
    )
    ap.add_argument(
        "--fit-on-observed-only",
        action="store_true",
        help="Fit KMeans on observed==1 rows (if observed column exists), then predict for all rows.",
    )
    ap.add_argument(
        "--plot-top-cities",
        type=int,
        default=0,
        help="If >0, render per-city regime timeline plots for this many cities (ranked by n_issues sum).",
    )
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    in_path = Path(args.input_panel).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    df = _load_table(in_path)

    unit_cols = [c.strip() for c in str(args.unit_cols).split(",") if c.strip()]
    bucket_col = str(args.bucket_col).strip()
    for c in unit_cols + [bucket_col]:
        if c not in df.columns:
            raise SystemExit(f"Missing required column: {c}")

    df = df.copy()
    df["_t_idx"] = df[bucket_col].astype(str).apply(_month_index)
    df = df.sort_values(unit_cols + ["_t_idx"]).reset_index(drop=True)

    # Determine clustering feature columns.
    wanted = [c.strip() for c in str(args.cluster_cols).split(",") if c.strip()]
    cluster_cols = [c for c in wanted if c in df.columns]
    if len(cluster_cols) < 2:
        raise SystemExit(
            f"Need at least 2 clustering columns, but only found {len(cluster_cols)} from {wanted}. "
            "Pass --cluster-cols with columns that exist in the input panel."
        )

    X = df[cluster_cols].apply(pd.to_numeric, errors="coerce")
    # Drop rows with any NaN in cluster features.
    ok = X.notna().all(axis=1)
    if int(ok.sum()) < 100:
        raise SystemExit(f"Too few rows with complete clustering features: {int(ok.sum())}")

    fit_mask = ok.to_numpy()
    if bool(args.fit_on_observed_only) and "observed" in df.columns:
        obs = pd.to_numeric(df["observed"], errors="coerce").fillna(0.0).to_numpy(dtype=float) > 0.0
        fit_mask = fit_mask & obs
        if int(fit_mask.sum()) < 100:
            raise SystemExit(f"Too few observed rows to fit regimes: {int(fit_mask.sum())}")

    X_fit = X[fit_mask].to_numpy(dtype=np.float64, copy=False)
    X_all = X.to_numpy(dtype=np.float64, copy=False)

    scaler = StandardScaler()
    scaler.fit(X_fit)
    X_fit_z = scaler.transform(X_fit)
    X_all_z = scaler.transform(X_all)

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")

    k_list = _parse_k_list(str(args.k_list))
    assignments = df[unit_cols + [bucket_col]].copy()
    if "observed" in df.columns:
        assignments["observed"] = pd.to_numeric(df["observed"], errors="coerce").fillna(0).astype(int)
    if "n_issues" in df.columns:
        assignments["n_issues"] = pd.to_numeric(df["n_issues"], errors="coerce").fillna(0).astype(int)

    # For summaries, include a small interpretable feature set if present.
    interpret_cols = []
    for c in ["jump_cosine_pca", "mechanics_jump_cosine_pca", "action_effective_n", "dim_effective_n", "action_mass", "dim_mass"]:
        if c in df.columns:
            interpret_cols.append(c)
    # Include a few action__/dim__ columns if present (top by overall std).
    for prefix in ("action__", "dim__"):
        cols = [c for c in df.columns if c.startswith(prefix)]
        if cols:
            s = df[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
            top = s.std(ddof=0).sort_values(ascending=False).head(8).index.tolist()
            interpret_cols.extend(top)

    interpret_cols = list(dict.fromkeys(interpret_cols))

    events_rows: list[dict[str, Any]] = []

    for k in k_list:
        km = KMeans(n_clusters=int(k), random_state=0, n_init=20)
        # Suppress rare BLAS/numpy RuntimeWarnings during large matmuls.
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            km.fit(X_fit_z)
            labs = km.predict(X_all_z)
        col = f"regime_k{k}"
        assignments[col] = labs.astype(int)

        # Per-regime summaries.
        summ = df.copy()
        summ[col] = labs.astype(int)
        rows = []
        for rid, g in summ.groupby(col, sort=True):
            row: dict[str, Any] = {"k": int(k), "regime": int(rid), "rows": int(len(g))}
            # Feature means for interpret cols.
            for c in interpret_cols:
                if c not in g.columns:
                    continue
                s = pd.to_numeric(g[c], errors="coerce")
                row[f"mean__{c}"] = float(s.mean()) if s.notna().any() else np.nan
            # Cluster feature means (in original units).
            for c, mu in zip(cluster_cols, g[cluster_cols].apply(pd.to_numeric, errors="coerce").mean().tolist()):
                row[f"mean__{c}"] = float(mu) if mu is not None and math.isfinite(float(mu)) else np.nan
            rows.append(row)
        pd.DataFrame.from_records(rows).to_csv(
            out_dir / f"regime_summaries_k{k}_{stamp}.csv",
            index=False,
            quoting=csv.QUOTE_MINIMAL,
        )

        # Regime switch events (within unit).
        tmp = assignments[unit_cols + [bucket_col, col]].copy()
        tmp["_t_idx"] = tmp[bucket_col].astype(str).apply(_month_index)
        tmp = tmp.sort_values(unit_cols + ["_t_idx"]).reset_index(drop=True)
        tmp["prev_regime"] = tmp.groupby(unit_cols, sort=False)[col].shift(1)
        tmp["switch"] = (tmp[col] != tmp["prev_regime"]).astype(int)
        # Exclude the first observation per unit (prev_regime NaN).
        tmp = tmp[tmp["prev_regime"].notna()].copy()
        switches = tmp[tmp["switch"] == 1].copy()
        for r in switches.itertuples(index=False):
            ev = {c: getattr(r, c) for c in unit_cols}
            ev.update({"k": int(k), "bucket": getattr(r, bucket_col), "regime_from": int(r.prev_regime), "regime_to": int(getattr(r, col))})
            events_rows.append(ev)

    # Write assignments.
    try:
        assignments.to_parquet(out_dir / f"regime_assignments_{stamp}.parquet", index=False)
    except Exception:
        pass
    assignments.to_csv(out_dir / f"regime_assignments_{stamp}.csv", index=False, quoting=csv.QUOTE_MINIMAL)

    pd.DataFrame.from_records(events_rows).to_csv(
        out_dir / f"regime_events_{stamp}.csv", index=False, quoting=csv.QUOTE_MINIMAL
    )

    # Optional: per-city regime plots.
    plot_n = int(args.plot_top_cities)
    if plot_n > 0:
        if "n_issues" in assignments.columns:
            city_rank = (
                assignments.groupby(unit_cols, as_index=False)["n_issues"]
                .sum()
                .rename(columns={"n_issues": "issues_sum"})
                .sort_values("issues_sum", ascending=False)
            )
        else:
            city_rank = assignments.groupby(unit_cols, as_index=False).size().rename(columns={"size": "issues_sum"})
        cities = city_rank.head(plot_n)[unit_cols].to_dict("records")

        # Use the first K for plotting.
        k0 = k_list[0]
        col0 = f"regime_k{k0}"
        if col0 not in assignments.columns:
            plot_n = 0
        cmap = plt.get_cmap("tab10")

        for row in cities:
            mask = np.ones((len(assignments),), dtype=bool)
            for c in unit_cols:
                mask &= assignments[c].astype(str).eq(str(row[c])).to_numpy()
            g = assignments[mask].copy()
            if g.empty:
                continue
            g["_t_idx"] = g[bucket_col].astype(str).apply(_month_index)
            g = g.sort_values("_t_idx")
            x_labels = g[bucket_col].astype(str).tolist()
            xs = np.arange(len(x_labels), dtype=int)
            regimes = g[col0].astype(int).to_numpy()

            fig, ax = plt.subplots(figsize=(14, 2.8))
            # Regime band.
            for i, r_id in enumerate(regimes):
                ax.axvspan(i - 0.5, i + 0.5, color=cmap(int(r_id) % 10), alpha=0.22, linewidth=0)
            ax.set_yticks([])
            ax.set_title(f"{row.get('city_name')}, {row.get('state_abbr')} — policy regime (k={k0})")
            tick_step = max(1, len(xs) // 12)
            tick_idx = list(range(0, len(xs), tick_step))
            ax.set_xticks(tick_idx)
            ax.set_xticklabels([x_labels[i] for i in tick_idx], rotation=45, ha="right")
            fig.tight_layout()
            fig.savefig(
                fig_dir
                / f"regimes_k{k0}_city_{_safe_slug(row.get('city_name'))}_{_safe_slug(row.get('state_abbr'))}_{stamp}.png",
                dpi=160,
            )
            plt.close(fig)

    meta = {
        "created_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "input": str(in_path),
        "unit_cols": unit_cols,
        "bucket_col": bucket_col,
        "cluster_cols": cluster_cols,
        "k_list": k_list,
        "fit_on_observed_only": bool(args.fit_on_observed_only) and ("observed" in df.columns),
        "rows_fit": int(fit_mask.sum()),
    }
    (out_dir / f"README_{stamp}.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"Done. wrote={out_dir} rows={len(df)} ks={k_list} cluster_cols={len(cluster_cols)}")


if __name__ == "__main__":
    main()
