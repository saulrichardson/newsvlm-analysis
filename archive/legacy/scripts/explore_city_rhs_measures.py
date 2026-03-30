#!/usr/bin/env python3
"""
Explore city×time RHS measures for downstream causal designs.

Inputs:
  - Mechanics-tag panel (from scripts/build_regression_rhs_from_topics.py)
  - Embedding-derived policy-state panel (from scripts/build_city_policy_state_from_embeddings.py)

Outputs:
  - merged panel (Parquet + CSV)
  - data-driven feature ranking tables
  - per-city time-series “dashboards” for selected measures

This script is deterministic and intended to “let the data speak”:
  - it ranks candidate measures by within-city variation + non-sparsity
  - it does NOT pick treatments or outcomes
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


_ACTION_FRIENDLY = {
    "adopt_ordinance": "Adopt ordinance",
    "amend_text": "Amend ordinance text",
    "amend_map_rezone": "Amend zoning map / rezone",
    "annexation": "Annexation",
    "create_district": "Create / revise district",
    "create_overlay": "Create / revise overlay district",
    "definitions": "Definitions / terms",
    "nonconforming": "Nonconforming uses / structures",
    "administration": "Administration / governance",
    "public_hearing_notice": "Public hearing notices",
    "appeals": "Appeals",
    "variance": "Variances",
    "conditional_use": "Conditional use permits",
    "special_exception": "Special exceptions",
    "site_plan": "Site plan review",
    "subdivision_plat": "Subdivision / plat approval",
    "pud": "Planned Unit Development (PUD)",
    "enforcement_penalties": "Enforcement / penalties",
    "permit_licensing": "Permits / licensing",
    "fees": "Fees",
}

_DIM_FRIENDLY = {
    # Use regulation
    "permitted_uses": "Permitted uses",
    "conditional_uses": "Conditional uses",
    "prohibited_uses": "Prohibited uses",
    # Dimensional / bulk standards
    "setbacks": "Setbacks",
    "yards": "Yards",
    "lot_area": "Minimum lot area",
    "lot_width": "Minimum lot width",
    "building_height": "Height limits",
    "floor_area_ratio": "Floor-area ratio (FAR)",
    "lot_coverage": "Lot coverage",
    "density": "Density limits",
    "dwelling_units": "Dwelling unit limits",
    # Development standards
    "parking": "Parking",
    "loading": "Loading",
    "signs": "Signs",
    "billboards": "Billboards",
    "landscaping_buffers": "Landscaping / buffers",
    "fences_walls": "Fences / walls",
    "accessory_structures": "Accessory structures",
    "home_occupations": "Home occupations",
    # Housing types / districts
    "mobile_homes": "Mobile homes",
    "trailer_parks": "Trailer parks",
    "manufactured_housing": "Manufactured housing",
    "districts": "Zoning districts (structure)",
    "definitions_terms": "Definitions / terms (dimension)",
    "nonconforming_uses": "Nonconforming uses (dimension)",
    # Processes often present in zoning corpora
    "subdivision_standards": "Subdivision standards",
    "site_plan_requirements": "Site plan requirements",
    "pud_standards": "PUD standards",
    # Common overlays / special topics
    "floodplain": "Floodplain",
    "airport": "Airport overlay / zoning",
    "adult_business": "Adult business regulation",
    "alcohol": "Alcohol regulation",
    "historic_district": "Historic district",
    "utilities_sewer_water": "Utilities / sewer / water",
}


def _pretty_feature_label(col: str) -> str:
    c = str(col)
    if c.startswith("action__"):
        tag = c.split("action__", 1)[1]
        return _ACTION_FRIENDLY.get(tag, tag.replace("_", " ").title())
    if c.startswith("dim__"):
        tag = c.split("dim__", 1)[1]
        return _DIM_FRIENDLY.get(tag, tag.replace("_", " ").title())
    if c.startswith(("body__", "instr__", "zr__")):
        return c.split("__", 1)[1].replace("_", " ").title()
    return c


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
    y, m = bucket_ym.split("-", 1)
    return int(y) * 12 + (int(m) - 1)


def _rank_features(
    df: pd.DataFrame,
    *,
    group_cols: list[str],
    feature_cols: list[str],
    min_nonzero: float,
) -> pd.DataFrame:
    """
    Rank features by mean within-group std (so we pick variables that move over time within cities).
    """
    rows: list[dict[str, Any]] = []
    for c in feature_cols:
        if c not in df.columns:
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        pct_nonnull = float(s.notna().mean())
        pct_nonzero = float((s.fillna(0.0) != 0.0).mean())
        if pct_nonzero < float(min_nonzero):
            continue
        # within-group std (drop groups with <2 obs)
        stds = df.assign(_v=s).dropna(subset=group_cols + ["_v"]).groupby(group_cols)["_v"].std(ddof=0)
        stds = stds[stds.notna()]
        within_std = float(stds.mean()) if not stds.empty else 0.0
        rows.append(
            {
                "column": c,
                "within_group_std_mean": within_std,
                "overall_std": float(s.std(ddof=0)),
                "mean": float(s.mean()),
                "pct_nonnull": pct_nonnull,
                "pct_nonzero": pct_nonzero,
                "min": float(s.min()) if s.notna().any() else np.nan,
                "max": float(s.max()) if s.notna().any() else np.nan,
            }
        )
    out = pd.DataFrame.from_records(rows)
    if out.empty:
        return out
    return out.sort_values(["within_group_std_mean", "overall_std"], ascending=False).reset_index(drop=True)


def _choose_top(df_rank: pd.DataFrame, *, prefix: str, k: int) -> list[str]:
    if df_rank.empty:
        return []
    cols = [c for c in df_rank["column"].tolist() if str(c).startswith(prefix)]
    return cols[:k]


def _safe_fname(s: str) -> str:
    s = s.strip().lower()
    out = []
    for ch in s:
        if ch.isalnum():
            out.append(ch)
        elif ch in {" ", "-", "_"}:
            out.append("_")
    return "".join(out).strip("_")[:120] or "city"


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Explore RHS measures and render city time-series dashboards.")
    ap.add_argument("--mechanics-panel", required=True, help="Path to mechanics city×month panel (.parquet or .csv).")
    ap.add_argument(
        "--embedding-panel",
        required=True,
        help="Path to embedding-derived city×month panel (.csv from Greene is OK).",
    )
    ap.add_argument("--output-dir", required=True, help="Directory to write outputs.")
    ap.add_argument("--bucket-col", default="bucket_month", help="Time bucket column name (default: bucket_month).")
    ap.add_argument("--top-cities", type=int, default=12, help="How many cities to render dashboards for.")
    ap.add_argument(
        "--min-nonzero",
        type=float,
        default=0.05,
        help="Drop candidate features with pct_nonzero below this threshold.",
    )
    ap.add_argument("--top-action", type=int, default=3, help="How many action__ features to plot per city.")
    ap.add_argument("--top-dim", type=int, default=3, help="How many dim__ features to plot per city.")
    ap.add_argument(
        "--friendly-labels",
        action="store_true",
        help="Use reader-friendly legend labels (recommended for LaTeX reports).",
    )
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    mech = _load_table(Path(args.mechanics_panel).expanduser().resolve())
    emb = _load_table(Path(args.embedding_panel).expanduser().resolve())

    bucket_col = str(args.bucket_col)
    for c in ["city_name", "state_abbr", bucket_col]:
        if c not in mech.columns:
            raise SystemExit(f"Mechanics panel missing required column: {c}")
        if c not in emb.columns:
            raise SystemExit(f"Embedding panel missing required column: {c}")

    merged = mech.merge(emb, on=["city_name", "state_abbr", bucket_col], how="inner", suffixes=("", "__emb"))
    if merged.empty:
        raise SystemExit("No rows after merging mechanics and embedding panels (check key columns).")

    # Sort time for plotting and for within-city std calcs.
    merged["_t_idx"] = merged[bucket_col].astype(str).apply(_month_index)
    merged = merged.sort_values(["city_name", "state_abbr", "_t_idx"]).reset_index(drop=True)

    # Candidate feature sets.
    mech_candidates = [c for c in merged.columns if c.startswith(("action__", "dim__", "body__", "instr__"))]
    # Also consider deltas (first differences) if present.
    mech_candidates += [c for c in merged.columns if c.startswith(("d_action__", "d_dim__", "d_body__", "d_instr__"))]
    emb_candidates = [c for c in merged.columns if c.startswith(("pc", "jump_"))]

    group_cols = ["city_name", "state_abbr"]
    mech_rank = _rank_features(
        merged,
        group_cols=group_cols,
        feature_cols=sorted(set(mech_candidates)),
        min_nonzero=float(args.min_nonzero),
    )
    emb_rank = _rank_features(
        merged,
        group_cols=group_cols,
        feature_cols=sorted(set(emb_candidates)),
        min_nonzero=0.01,  # PCA/jumps are continuous; don’t threshold hard
    )

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    mech_rank_path = out_dir / f"feature_ranking_mechanics_{stamp}.csv"
    emb_rank_path = out_dir / f"feature_ranking_embeddings_{stamp}.csv"
    mech_rank.to_csv(mech_rank_path, index=False, quoting=csv.QUOTE_MINIMAL)
    emb_rank.to_csv(emb_rank_path, index=False, quoting=csv.QUOTE_MINIMAL)

    # Choose a small set of “best” measures to visualize.
    top_action = _choose_top(mech_rank, prefix="action__", k=int(args.top_action))
    top_dim = _choose_top(mech_rank, prefix="dim__", k=int(args.top_dim))
    # Always include jump_cosine_pca if present.
    emb_measures = []
    if "jump_cosine_pca" in merged.columns:
        emb_measures.append("jump_cosine_pca")
    # Add a couple PCs for interpretability.
    for c in ["pc1", "pc2", "pc3"]:
        if c in merged.columns:
            emb_measures.append(c)
    emb_measures = list(dict.fromkeys(emb_measures))  # stable dedupe

    selected = {
        "top_action": top_action,
        "top_dim": top_dim,
        "embedding_measures": emb_measures,
        "notes": {
            "min_nonzero_mechanics": float(args.min_nonzero),
            "ranking_metric": "mean within-city std",
        },
    }
    (out_dir / f"selected_features_{stamp}.json").write_text(
        json.dumps(selected, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    # Save merged panel for downstream work.
    merged_out_csv = out_dir / f"merged_city_month_panel_{stamp}.csv"
    merged.drop(columns=["_t_idx"]).to_csv(merged_out_csv, index=False, quoting=csv.QUOTE_MINIMAL)
    try:
        merged_out_pq = out_dir / f"merged_city_month_panel_{stamp}.parquet"
        merged.drop(columns=["_t_idx"]).to_parquet(merged_out_pq, index=False)
    except Exception:
        # Parquet optional; CSV is the source-of-truth fallback for portability.
        pass

    # Pick top cities by observation density.
    if "n_issues" in merged.columns:
        city_counts = (
            merged.groupby(["city_name", "state_abbr"], as_index=False)["n_issues"].sum().rename(columns={"n_issues": "issues_sum"})
        )
    else:
        city_counts = merged.groupby(["city_name", "state_abbr"], as_index=False).size().rename(columns={"size": "issues_sum"})
    city_counts = city_counts.sort_values("issues_sum", ascending=False).reset_index(drop=True)

    n_cities = int(args.top_cities)
    cities = city_counts.head(n_cities)[["city_name", "state_abbr"]].to_dict("records")

    # Render per-city dashboards.
    for row in cities:
        city = str(row["city_name"])
        st = str(row["state_abbr"])
        g = merged[(merged["city_name"] == city) & (merged["state_abbr"] == st)].copy()
        if g.empty:
            continue

        # Panel layout: (jump + n_issues), (actions), (dimensions)
        fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(14, 9), sharex=True)

        x = g[bucket_col].astype(str).tolist()

        # 1) Jump intensity + n_issues
        ax = axes[0]
        if "jump_cosine_pca" in g.columns:
            lab = "Semantic change intensity (embedding jump)" if bool(args.friendly_labels) else "jump_cosine_pca"
            ax.plot(x, g["jump_cosine_pca"].astype(float), color="black", linewidth=1.2, label=lab)
        if "mechanics_jump_cosine_pca" in g.columns:
            lab = "Mechanics change intensity (tag PCA jump)" if bool(args.friendly_labels) else "mechanics_jump_cosine_pca"
            ax.plot(x, g["mechanics_jump_cosine_pca"].astype(float), color="#B22222", linewidth=1.1, alpha=0.9, label=lab)
        ax.set_ylabel("Change intensity")
        ax.grid(True, alpha=0.25)
        ax2 = ax.twinx()
        if "n_issues" in g.columns:
            ax2.bar(x, g["n_issues"].astype(float), color="#4C78A8", alpha=0.25, width=1.0, label="n_issues")
            ax2.set_ylabel("n_issues")
        # Keep legends compact
        ax.legend(loc="upper left", fontsize=9, frameon=False)

        # 2) Actions
        ax = axes[1]
        for c in top_action:
            if c in g.columns:
                lab = _pretty_feature_label(c) if bool(args.friendly_labels) else c
                ax.plot(x, g[c].astype(float), linewidth=1.1, label=lab)
        ax.set_ylabel("Action share")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper left", fontsize=8, frameon=False, ncol=1)

        # 3) Dimensions
        ax = axes[2]
        for c in top_dim:
            if c in g.columns:
                lab = _pretty_feature_label(c) if bool(args.friendly_labels) else c
                ax.plot(x, g[c].astype(float), linewidth=1.1, label=lab)
        ax.set_ylabel("Dimension share")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper left", fontsize=8, frameon=False, ncol=1)

        # X axis: reduce ticks to ~12
        tick_step = max(1, len(x) // 12)
        idx = list(range(0, len(x), tick_step))
        axes[2].set_xticks(idx)
        axes[2].set_xticklabels([x[i] for i in idx], rotation=45, ha="right")

        fig.suptitle(f"{city}, {st} — zoning text RHS measures ({bucket_col})", fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.96])

        out_path = fig_dir / f"city_{_safe_fname(city)}_{_safe_fname(st)}_{stamp}.png"
        fig.savefig(out_path, dpi=200)
        plt.close(fig)

    print(f"Done. wrote={out_dir} merged_rows={len(merged)} cities_plotted={len(cities)}")


if __name__ == "__main__":
    main()
