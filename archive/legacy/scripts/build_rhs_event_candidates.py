#!/usr/bin/env python3
"""
Build candidate "event" objects from text-derived RHS measures for downstream causal designs.

This script intentionally does NOT choose a treatment definition. Instead it produces:
  - a ranked list of candidate event months per city (e.g., biggest embedding jump)
  - per-event summaries of what appears to have changed (top delta mechanics tags)
  - per-event issue lists (issue_id + classification_label) for manual inspection
  - optional per-city plots with event markers

Inputs:
  1) merged city×month panel (mechanics + embeddings), typically from:
       scripts/explore_city_rhs_measures.py  (STEP run recommended)
     Required columns:
       - city_name, state_abbr, bucket_month
       - jump_cosine_pca (embedding change intensity) (or jump_l2_pca)
       - n_issues (count of issues in that city×month)
       - observed (0/1) (from step mechanics panel)
     Optional columns:
       - d_* delta columns (from mechanics observed panel; in step panels these are populated on observed months)

  2) issue-level mechanics panel, typically from:
       scripts/build_regression_rhs_from_topics.py
     Required columns:
       - issue_id, issue_date, classification_label, city_name, state_abbr

Outputs:
  - events_by_city_topk_<stamp>.csv
  - events_global_topn_<stamp>.csv
  - event_issues_<stamp>.csv
  - event_top_deltas_<stamp>.csv
  - event_packets/*.md (human-readable per-event summaries)
  - figures/*.png (optional per-city plots)

This script is deterministic and intended to "let the data speak" while producing
artifact-first audit trails for manual reading.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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
    # bucket_ym: YYYY-MM
    y, m = str(bucket_ym).split("-", 1)
    return int(y) * 12 + (int(m) - 1)


def _safe_slug(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9_]+", "", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s[:120] or "x"


def _jsonable(x: Any) -> Any:
    if x is None:
        return None
    if isinstance(x, (np.floating, float)):
        xf = float(x)
        if not math.isfinite(xf):
            return None
        return xf
    if isinstance(x, (np.integer, int)):
        return int(x)
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    # pandas scalar types
    if isinstance(x, (pd.Timestamp, pd.Timedelta)):
        return str(x)
    return x


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build candidate event months from RHS panels.")
    ap.add_argument("--merged-step-panel", required=True, help="Merged city×month panel (.parquet or .csv).")
    ap.add_argument("--issue-panel", required=True, help="Issue-level mechanics panel (.parquet or .csv).")
    ap.add_argument("--output-dir", required=True, help="Directory to write outputs.")

    ap.add_argument("--bucket-col", default="bucket_month", help="Time bucket column (default: bucket_month).")
    ap.add_argument(
        "--event-score-col",
        default="jump_cosine_pca",
        help=(
            "Column to rank events by (higher = more event-like). "
            "Default: jump_cosine_pca. "
            "You can also pass derived columns like mechanics_jump_cosine_pca, etc."
        ),
    )
    ap.add_argument(
        "--event-score-label",
        default="",
        help=(
            "Optional human-friendly label for plots/captions (does not change the column used). "
            "If omitted, uses --event-score-col."
        ),
    )
    ap.add_argument(
        "--require-observed",
        action="store_true",
        help="Only keep event months where observed==1 (recommended for STEP panels).",
    )
    ap.add_argument("--top-k-per-city", type=int, default=5, help="How many events to keep per city.")
    ap.add_argument("--top-n-global", type=int, default=250, help="How many events to keep globally.")
    ap.add_argument("--top-deltas", type=int, default=12, help="How many delta features to surface per event.")
    ap.add_argument(
        "--plot-top-cities",
        type=int,
        default=25,
        help="Plot jump time series for this many most-observed cities (0 disables).",
    )
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    packets_dir = out_dir / "event_packets"
    packets_dir.mkdir(parents=True, exist_ok=True)
    figs_dir = out_dir / "figures"
    figs_dir.mkdir(parents=True, exist_ok=True)

    merged = _load_table(Path(args.merged_step_panel).expanduser().resolve())
    issues = _load_table(Path(args.issue_panel).expanduser().resolve())

    bucket_col = str(args.bucket_col)
    score_col = str(args.event_score_col).strip()
    if not score_col:
        raise SystemExit("--event-score-col must be non-empty")
    score_label = str(args.event_score_label or "").strip() or score_col

    required_merged = ["city_name", "state_abbr", bucket_col, "n_issues", score_col]
    for c in required_merged:
        if c not in merged.columns:
            raise SystemExit(f"Merged panel missing required column: {c}")

    required_issue = ["issue_id", "issue_date", "classification_label", "city_name", "state_abbr"]
    for c in required_issue:
        if c not in issues.columns:
            raise SystemExit(f"Issue panel missing required column: {c}")

    # Normalize issue bucket column.
    issues = issues.copy()
    issues[bucket_col] = issues["issue_date"].astype(str).str.slice(0, 7)

    # Candidate event rows: keep months with at least one issue (and optionally observed flag).
    ev = merged.copy()
    ev["_t_idx"] = ev[bucket_col].astype(str).apply(_month_index)

    if bool(args.require_observed):
        if "observed" not in ev.columns:
            raise SystemExit("--require-observed was set but merged panel has no 'observed' column")
        ev = ev[ev["observed"].astype(float) > 0.0]
    else:
        ev = ev[ev["n_issues"].astype(float) > 0.0]

    ev[score_col] = pd.to_numeric(ev[score_col], errors="coerce")
    ev = ev[ev[score_col].notna()]

    if ev.empty:
        raise SystemExit("No candidate event rows after filtering (check n_issues/observed and score column).")

    # Delta feature set (optional).
    d_cols = [c for c in ev.columns if c.startswith("d_")]
    if d_cols:
        d_mat = ev[d_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        ev["mechanics_delta_l1"] = d_mat.abs().sum(axis=1).astype(np.float32)
        ev["mechanics_delta_linf"] = d_mat.abs().max(axis=1).astype(np.float32)
        ev["mechanics_delta_nnz"] = (d_mat.abs() > 0).sum(axis=1).astype(np.int32)
    else:
        ev["mechanics_delta_l1"] = np.nan
        ev["mechanics_delta_linf"] = np.nan
        ev["mechanics_delta_nnz"] = np.nan

    # Stable event_id.
    ev["event_id"] = (
        ev["city_name"].astype(str).apply(_safe_slug)
        + "__"
        + ev["state_abbr"].astype(str).apply(_safe_slug)
        + "__"
        + ev[bucket_col].astype(str).apply(_safe_slug)
    )

    top_k = int(args.top_k_per_city)
    if top_k <= 0:
        raise SystemExit("--top-k-per-city must be > 0")
    top_n = int(args.top_n_global)
    if top_n <= 0:
        raise SystemExit("--top-n-global must be > 0")
    top_deltas_n = int(args.top_deltas)
    if top_deltas_n <= 0:
        raise SystemExit("--top-deltas must be > 0")

    # Rank: within-city top-k.
    ev_sorted = ev.sort_values(["city_name", "state_abbr", score_col], ascending=[True, True, False])
    ev_city = ev_sorted.groupby(["city_name", "state_abbr"], sort=False).head(top_k).copy()

    # Global top-n.
    ev_global = ev.sort_values(score_col, ascending=False).head(top_n).copy()

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    events_city_path = out_dir / f"events_by_city_top{top_k}_{stamp}.csv"
    events_global_path = out_dir / f"events_global_top{top_n}_{stamp}.csv"

    keep_cols = [
        "event_id",
        "city_name",
        "state_abbr",
        bucket_col,
        "n_issues",
        score_col,
        "jump_l2_pca" if "jump_l2_pca" in ev.columns else None,
        "mechanics_delta_l1",
        "mechanics_delta_linf",
        "mechanics_delta_nnz",
        "pc1" if "pc1" in ev.columns else None,
        "pc2" if "pc2" in ev.columns else None,
        "pc3" if "pc3" in ev.columns else None,
    ]
    keep_cols = [c for c in keep_cols if c is not None and c in ev.columns]

    ev_city[keep_cols].to_csv(events_city_path, index=False, quoting=csv.QUOTE_MINIMAL)
    ev_global[keep_cols].to_csv(events_global_path, index=False, quoting=csv.QUOTE_MINIMAL)

    # Event -> issue list (join to issue panel).
    # Emit one row per issue for easy filtering.
    ev_keys = ev_city[["event_id", "city_name", "state_abbr", bucket_col]].drop_duplicates()
    ev_issue = ev_keys.merge(issues, on=["city_name", "state_abbr", bucket_col], how="left")
    ev_issue["missing_issue_join"] = ev_issue["issue_id"].isna().astype(int)
    ev_issue = ev_issue.sort_values(["city_name", "state_abbr", bucket_col, "issue_date", "issue_id"]).reset_index(drop=True)
    ev_issue_path = out_dir / f"event_issues_top{top_k}_{stamp}.csv"
    ev_issue.to_csv(ev_issue_path, index=False, quoting=csv.QUOTE_MINIMAL)

    # Per-event top delta features (if deltas exist).
    deltas_rows: list[dict[str, Any]] = []
    if d_cols:
        ev_city_idx = ev_city.set_index("event_id", drop=False)
        for eid in ev_city_idx.index.tolist():
            row = ev_city_idx.loc[eid]
            dv = row[d_cols].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
            abs_sorted = dv.abs().sort_values(ascending=False)
            for rank, (col, abs_val) in enumerate(abs_sorted.head(top_deltas_n).items(), start=1):
                deltas_rows.append(
                    {
                        "event_id": str(eid),
                        "city_name": str(row["city_name"]),
                        "state_abbr": str(row["state_abbr"]),
                        bucket_col: str(row[bucket_col]),
                        "delta_column": str(col),
                        "delta_value": float(dv[col]),
                        "abs_delta": float(abs_val),
                        "rank": int(rank),
                    }
                )
    deltas_path = out_dir / f"event_top_deltas_top{top_k}_{stamp}.csv"
    pd.DataFrame.from_records(deltas_rows).to_csv(deltas_path, index=False, quoting=csv.QUOTE_MINIMAL)

    # Write per-event markdown packets.
    issue_share_cols_action = [c for c in issues.columns if c.startswith("action__")]
    issue_share_cols_dim = [c for c in issues.columns if c.startswith("dim__")]
    issues_by_id = issues.set_index("issue_id", drop=False)

    def _top_features(row: pd.Series, cols: list[str], k: int = 4) -> list[tuple[str, float]]:
        vals: list[tuple[str, float]] = []
        for c in cols:
            v = row.get(c)
            try:
                fv = float(v)
            except Exception:
                continue
            if not math.isfinite(fv) or fv == 0.0:
                continue
            vals.append((c, fv))
        vals.sort(key=lambda t: (-abs(t[1]), t[0]))
        return vals[:k]

    # Pre-group joined issues by event_id for fast packet generation.
    ev_issue_groups = {k: g.copy() for k, g in ev_issue.groupby("event_id", sort=False)}

    for _, e in ev_city[keep_cols].iterrows():
        eid = str(e["event_id"])
        city = str(e["city_name"])
        st = str(e["state_abbr"])
        bm = str(e[bucket_col])

        # Top deltas for this event.
        top_delta_lines: list[str] = []
        if deltas_rows:
            sub = [r for r in deltas_rows if r["event_id"] == eid]
            for r in sub[:top_deltas_n]:
                top_delta_lines.append(f"- `{r['delta_column']}` = {r['delta_value']:+.3f}")

        # Issue list for this event.
        g_issues = ev_issue_groups.get(eid)
        issue_lines: list[str] = []
        if g_issues is not None:
            g_issues = g_issues[g_issues["issue_id"].notna()].copy()
            for _, ir in g_issues.iterrows():
                iid = str(ir["issue_id"])
                idate = str(ir.get("issue_date") or "")
                cl = str(ir.get("classification_label") or "")
                if iid in issues_by_id.index:
                    row = issues_by_id.loc[iid]
                    top_a = _top_features(row, issue_share_cols_action, k=4)
                    top_d = _top_features(row, issue_share_cols_dim, k=4)
                    a_txt = ", ".join([f"`{c}`={v:.2f}" for c, v in top_a]) if top_a else "(none)"
                    d_txt = ", ".join([f"`{c}`={v:.2f}" for c, v in top_d]) if top_d else "(none)"
                else:
                    a_txt = "(issue row missing)"
                    d_txt = "(issue row missing)"
                issue_lines.append(
                    f"- `{iid}` ({idate}) [{cl}]\n"
                    f"  - actions: {a_txt}\n"
                    f"  - dims: {d_txt}"
                )

        summary = {k: _jsonable(e.get(k)) for k in keep_cols}

        md = []
        md.append(f"# Event packet: {city}, {st} — {bm}")
        md.append("")
        md.append("## Summary")
        md.append("")
        md.append("```json")
        md.append(json.dumps(summary, indent=2, default=str))
        md.append("```")
        md.append("")
        md.append("## Top mechanics deltas (diagnostic)")
        md.append("")
        if top_delta_lines:
            md.extend(top_delta_lines)
        else:
            md.append("(No d_* delta columns found in merged panel.)")
        md.append("")
        md.append("## Issues in this city×month")
        md.append("")
        if issue_lines:
            md.extend(issue_lines)
        else:
            md.append("(No issues joined for this event; see event_issues CSV for missing join flags.)")
        md.append("")
        (packets_dir / f"event_{_safe_slug(eid)}.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    # Optional plots: jump over time with event markers.
    plot_n = int(args.plot_top_cities)
    if plot_n > 0:
        city_rank = (
            merged.groupby(["city_name", "state_abbr"], as_index=False)["n_issues"]
            .sum()
            .rename(columns={"n_issues": "issues_sum"})
            .sort_values("issues_sum", ascending=False)
            .reset_index(drop=True)
        )
        cities = city_rank.head(plot_n)[["city_name", "state_abbr"]].to_dict("records")

        city_events = (
            ev_city[["city_name", "state_abbr", bucket_col, score_col]]
            .copy()
            .sort_values(["city_name", "state_abbr", score_col], ascending=[True, True, False])
        )

        for r in cities:
            city = str(r["city_name"])
            st = str(r["state_abbr"])
            g = merged[(merged["city_name"] == city) & (merged["state_abbr"] == st)].copy()
            if g.empty:
                continue
            g["_t_idx"] = g[bucket_col].astype(str).apply(_month_index)
            g = g.sort_values("_t_idx")

            x_labels = g[bucket_col].astype(str).tolist()
            xs = np.arange(len(x_labels), dtype=int)
            y = pd.to_numeric(g[score_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            n = pd.to_numeric(g["n_issues"], errors="coerce").fillna(0.0).to_numpy(dtype=float)

            fig, ax = plt.subplots(figsize=(14, 4))
            ax.plot(xs, y, color="black", linewidth=1.2, label=score_label)
            ax2 = ax.twinx()
            ax2.bar(xs, n, color="#8fb3d9", alpha=0.35, label="n_issues")

            # Mark selected events.
            evm = city_events[(city_events["city_name"] == city) & (city_events["state_abbr"] == st)][bucket_col].astype(str).tolist()
            if evm:
                ev_idx = {m: i for i, m in enumerate(x_labels)}
                for m in evm:
                    if m in ev_idx:
                        ax.axvline(ev_idx[m], color="#cc0000", alpha=0.12)

            tick_step = max(1, len(xs) // 12)
            tick_idx = list(range(0, len(xs), tick_step))
            ax.set_xticks(tick_idx)
            ax.set_xticklabels([x_labels[i] for i in tick_idx], rotation=45, ha="right")
            ax.set_title(f"{city}, {st} — {score_label} (event candidates highlighted)")
            ax.set_ylabel(score_label)
            ax2.set_ylabel("n_issues")
            fig.tight_layout()
            fig.savefig(figs_dir / f"city_{_safe_slug(city)}_{_safe_slug(st)}_{stamp}.png", dpi=160)
            plt.close(fig)

    meta = {
        "created_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "inputs": {
            "merged_step_panel": str(Path(args.merged_step_panel).expanduser().resolve()),
            "issue_panel": str(Path(args.issue_panel).expanduser().resolve()),
        },
        "params": {
            "bucket_col": bucket_col,
            "event_score_col": score_col,
            "event_score_label": score_label,
            "require_observed": bool(args.require_observed),
            "top_k_per_city": top_k,
            "top_n_global": top_n,
            "top_deltas": top_deltas_n,
            "plot_top_cities": plot_n,
        },
        "outputs": {
            "events_by_city": str(events_city_path),
            "events_global": str(events_global_path),
            "event_issues": str(ev_issue_path),
            "event_top_deltas": str(deltas_path),
            "event_packets_dir": str(packets_dir),
            "figures_dir": str(figs_dir),
        },
    }
    (out_dir / f"README_{stamp}.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(
        "Done. "
        f"wrote={out_dir} "
        f"candidate_rows={len(ev)} "
        f"events_city={len(ev_city)} "
        f"events_global={len(ev_global)} "
        f"d_cols={len(d_cols)}"
    )


if __name__ == "__main__":
    main()
