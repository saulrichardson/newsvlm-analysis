#!/usr/bin/env python3
"""
Overlay text-derived RHS measures with lightweight outcome series for visual exploration.

This is explicitly *not* a causal design implementation. It exists to help decide:
  - which RHS construction is stable + interpretable (jumps, regimes, tag shares, PCs)
  - where candidate “events” might plausibly line up with price dynamics

The default outcome overlay is a quarterly FHFA HPI series fetched from FRED via
scripts/fetch_fred_series.py (no API key required).
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Plot RHS measures with FRED outcome overlays (quarterly).")
    ap.add_argument("--rhs-panel", required=True, help="Merged STEP panel (.parquet or .csv).")
    ap.add_argument("--fred-panel", required=True, help="Tidy FRED panel (.parquet or .csv) from fetch_fred_series.py.")
    ap.add_argument("--output-dir", required=True, help="Directory to write figures + merged quarterly panel.")
    ap.add_argument("--score-col", default="jump_cosine_pca", help="RHS score column to plot on the left axis.")
    ap.add_argument(
        "--score-label",
        default="",
        help="Optional human-friendly label for the RHS score in plots (does not change the column used).",
    )
    ap.add_argument(
        "--series-id",
        default="",
        help="Optional: force a specific FRED series_id (otherwise the first series per city is used).",
    )
    ap.add_argument(
        "--events-by-city",
        default="",
        help="Optional: events_by_city CSV from build_rhs_event_candidates.py (adds vertical markers).",
    )
    ap.add_argument("--top-cities", type=int, default=6, help="How many cities to plot (ranked by RHS coverage).")
    return ap.parse_args()


def _load(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise SystemExit(f"Missing input: {path}")
    suf = path.suffix.lower()
    if suf == ".parquet":
        return pd.read_parquet(path)
    if suf == ".csv":
        return pd.read_csv(path)
    raise SystemExit(f"Unsupported input format: {path} (expected .parquet or .csv)")


def _safe_slug(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9_]+", "", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s[:120] or "x"


def main() -> None:
    args = _parse_args()
    rhs_path = Path(args.rhs_panel).expanduser().resolve()
    fred_path = Path(args.fred_panel).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    score_col = str(args.score_col).strip()
    if not score_col:
        raise SystemExit("--score-col must be non-empty")
    score_label = str(args.score_label or "").strip() or score_col

    rhs = _load(rhs_path)
    fred = _load(fred_path)

    for c in ["city_name", "state_abbr", "bucket_month", score_col]:
        if c not in rhs.columns:
            raise SystemExit(f"RHS panel missing required column: {c}")
    for c in ["city_name", "state_abbr", "series_id", "date", "value"]:
        if c not in fred.columns:
            raise SystemExit(f"FRED panel missing required column: {c}")

    # RHS -> quarterly
    rhs = rhs.copy()
    rhs["date"] = pd.to_datetime(rhs["bucket_month"].astype(str) + "-01", errors="coerce")
    rhs = rhs.dropna(subset=["date"])
    rhs["quarter"] = rhs["date"].dt.to_period("Q").astype(str)
    rhs[score_col] = pd.to_numeric(rhs[score_col], errors="coerce")
    rhs["n_issues"] = pd.to_numeric(rhs.get("n_issues", 0), errors="coerce").fillna(0).astype(int)

    rhs_q = (
        rhs.groupby(["city_name", "state_abbr", "quarter"], as_index=False)
        .agg({score_col: "max", "n_issues": "sum"})
        .rename(columns={score_col: "rhs_score_q_max", "n_issues": "n_issues_q_sum"})
    )

    # FRED -> quarterly
    fred = fred.copy()
    fred["date"] = pd.to_datetime(fred["date"], errors="coerce")
    fred["value"] = pd.to_numeric(fred["value"], errors="coerce")
    fred = fred.dropna(subset=["date"]).reset_index(drop=True)
    fred["quarter"] = fred["date"].dt.to_period("Q").astype(str)

    series_id_override = str(args.series_id or "").strip()
    if series_id_override:
        fred = fred[fred["series_id"].astype(str) == series_id_override].copy()
        if fred.empty:
            raise SystemExit(f"--series-id={series_id_override} filtered FRED panel down to 0 rows")

    # Optional: events -> quarterly markers
    ev_q = None
    events_by_city = str(args.events_by_city or "").strip()
    if events_by_city:
        ev = _load(Path(events_by_city).expanduser().resolve())
        for c in ["city_name", "state_abbr", "bucket_month"]:
            if c not in ev.columns:
                raise SystemExit(f"--events-by-city missing required column: {c}")
        ev = ev.copy()
        ev["date"] = pd.to_datetime(ev["bucket_month"].astype(str) + "-01", errors="coerce")
        ev = ev.dropna(subset=["date"])
        ev["quarter"] = ev["date"].dt.to_period("Q").astype(str)
        ev_q = ev[["city_name", "state_abbr", "quarter"]].drop_duplicates()

    # Restrict to cities we have outcomes for.
    cities_with_outcomes = fred[["city_name", "state_abbr"]].dropna().drop_duplicates()
    rhs_q = rhs_q.merge(cities_with_outcomes, on=["city_name", "state_abbr"], how="inner")
    if rhs_q.empty:
        raise SystemExit("No RHS rows remain after restricting to cities present in FRED panel")

    # Pick top cities by RHS coverage (within the set that has outcomes).
    city_rank = (
        rhs_q.groupby(["city_name", "state_abbr"], as_index=False)["n_issues_q_sum"]
        .sum()
        .rename(columns={"n_issues_q_sum": "issues_sum"})
        .sort_values("issues_sum", ascending=False)
        .reset_index(drop=True)
    )
    top_n = int(args.top_cities)
    if top_n <= 0:
        raise SystemExit("--top-cities must be > 0")
    cities = city_rank.head(top_n)[["city_name", "state_abbr"]].to_dict("records")

    merged_rows = []
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")

    for row in cities:
        city = str(row["city_name"])
        st = str(row["state_abbr"])

        g_rhs = rhs_q[(rhs_q["city_name"] == city) & (rhs_q["state_abbr"] == st)].copy()
        g_fred = fred[(fred["city_name"] == city) & (fred["state_abbr"] == st)].copy()
        if g_rhs.empty or g_fred.empty:
            continue

        # If multiple series_ids remain for a city (rare), choose one deterministically.
        if not series_id_override:
            sid = sorted(g_fred["series_id"].dropna().astype(str).unique().tolist())[0]
            g_fred = g_fred[g_fred["series_id"].astype(str) == sid].copy()

        g_fred_q = (
            g_fred[["quarter", "series_id", "value"]]
            .groupby(["quarter", "series_id"], as_index=False)["value"]
            .last()
            .rename(columns={"value": "outcome_value"})
        )

        g = g_rhs.merge(g_fred_q, on="quarter", how="left")
        g["series_id"] = g["series_id"].astype(str)
        g = g.sort_values("quarter").reset_index(drop=True)
        g["rhs_score_q_max"] = pd.to_numeric(g["rhs_score_q_max"], errors="coerce")
        g["outcome_value"] = pd.to_numeric(g["outcome_value"], errors="coerce")

        g["city_name"] = city
        g["state_abbr"] = st
        merged_rows.append(g)

        x = g["quarter"].astype(str).tolist()
        y_score = g["rhs_score_q_max"]
        y_out = g["outcome_value"]
        sid = str(g["series_id"].dropna().unique().tolist()[0]) if g["series_id"].notna().any() else "fred"

        fig, ax = plt.subplots(figsize=(14, 4.8))
        ax.plot(x, y_score, color="black", linewidth=1.2, label=f"{score_label} (q max)")
        ax.set_ylabel(score_label)
        ax.grid(True, alpha=0.25)

        ax2 = ax.twinx()
        ax2.plot(x, y_out, color="#4C78A8", linewidth=1.4, alpha=0.85, label=sid)
        ax2.set_ylabel(sid)

        if ev_q is not None:
            g_ev = ev_q[(ev_q["city_name"] == city) & (ev_q["state_abbr"] == st)]
            for q in sorted(g_ev["quarter"].dropna().astype(str).unique().tolist()):
                if q in set(x):
                    ax.axvline(q, color="red", alpha=0.12, linewidth=1.0)

        tick_step = max(1, len(x) // 12)
        ax.set_xticks(list(range(0, len(x), tick_step)))
        ax.set_xticklabels([x[i] for i in range(0, len(x), tick_step)], rotation=45, ha="right")
        ax.set_title(f"{city}, {st} — RHS vs outcome overlay")

        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=9, frameon=False)

        fname = f"city_{_safe_slug(city)}_{_safe_slug(st)}_{_safe_slug(score_col)}_overlay_{stamp}.png"
        fig.savefig(fig_dir / fname, dpi=200, bbox_inches="tight")
        plt.close(fig)

    if not merged_rows:
        raise SystemExit("No city plots produced (likely no overlap between RHS and FRED panel cities).")

    merged = pd.concat(merged_rows, ignore_index=True)
    merged = merged[["city_name", "state_abbr", "quarter", "rhs_score_q_max", "n_issues_q_sum", "series_id", "outcome_value"]]
    merged.to_csv(out_dir / f"rhs_outcome_quarter_panel_{stamp}.csv", index=False, quoting=csv.QUOTE_MINIMAL)

    print(f"Done. wrote={out_dir} cities_plotted={merged[['city_name','state_abbr']].drop_duplicates().shape[0]}")


if __name__ == "__main__":
    main()
