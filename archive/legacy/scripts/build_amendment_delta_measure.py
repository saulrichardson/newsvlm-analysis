#!/usr/bin/env python3
"""
Build amendment topical summaries and prior-doc delta measures.

Inputs:
  - <run_dir>/panels/selected_panel_issues.csv
  - <run_dir>/llm_panel_vs_issue/parsed_issue_outputs.jsonl

Outputs:
  - <output_dir>/amendment_topic_shares.csv
  - <output_dir>/amendment_topic_shares_by_region_urbanicity_year.csv
  - <output_dir>/amendment_delta_events.csv
  - <output_dir>/amendment_delta_category_summary.csv
  - <output_dir>/amendment_delta_matching_diagnostics.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


_FIXED_CATEGORY_KEYS = [
    "land_use_restrictions",
    "bulk_dimensional_standards",
    "procedural_governance",
    "nonconforming_use",
    "definitions",
    "parking_transport",
    "environmental_protection",
    "historic_preservation",
    "flexible_zoning",
    "inclusionary_affordable_housing",
    "enforcement_penalties",
    "miscellaneous_site_rules",
]


def _iter_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                out.append(obj)
    return out


def _parse_issue_id_from_custom_id(custom_id: str) -> str:
    parts = str(custom_id or "").split("::", 2)
    return parts[2] if len(parts) >= 3 else ""


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build amendment delta measures.")
    ap.add_argument(
        "--run-dir",
        default="/Users/saulrichardson/projects/newspapers/newspaper-analysis/reports/runs/prototype_zoning_panel_analysis/prototype_zoning_panel_analysis_iter8_scale30",
        help="Prototype run directory.",
    )
    ap.add_argument(
        "--output-dir",
        default="/Users/saulrichardson/projects/newspapers/newspaper-analysis/reports/runs/prototype_zoning_panel_analysis/prototype_zoning_panel_analysis_iter8_scale30/amendments",
        help="Output directory for amendment tables.",
    )
    ap.add_argument(
        "--max-back-years",
        type=int,
        default=10,
        help="Max lookback window for prior-doc matching.",
    )
    return ap.parse_args()


def _extract_issue_measure_maps(parsed_issue_path: Path) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, Any]]]:
    cat_map: dict[str, dict[str, float]] = {}
    num_map: dict[str, dict[str, Any]] = {}
    for row in _iter_jsonl(parsed_issue_path):
        issue_id = _parse_issue_id_from_custom_id(str(row.get("custom_id") or ""))
        parsed = row.get("parsed_json")
        if not issue_id or not isinstance(parsed, dict):
            continue
        cs = parsed.get("category_shares")
        if isinstance(cs, dict):
            cat_map[issue_id] = {k: float(cs.get(k) or 0.0) for k in _FIXED_CATEGORY_KEYS}
        nc = parsed.get("numeric_constraints")
        if isinstance(nc, dict):
            num_map[issue_id] = dict(nc)
    return cat_map, num_map


def _load_selected_issues(run_dir: Path) -> pd.DataFrame:
    selected_issues_path = run_dir / "panels" / "selected_panel_issues.csv"
    if selected_issues_path.is_file():
        return pd.read_csv(selected_issues_path)

    rows: list[dict[str, Any]] = []
    panels_dir = run_dir / "panels"
    for city_dir in sorted(p for p in panels_dir.iterdir() if p.is_dir()):
        issue_path = city_dir / "issue_texts.jsonl"
        if not issue_path.is_file():
            continue
        for obj in _iter_jsonl(issue_path):
            rows.append(
                {
                    "city_key": str(obj.get("city_key") or city_dir.name),
                    "city_name": str(obj.get("city_name") or ""),
                    "state_abbr": str(obj.get("state_abbr") or "").lower(),
                    "region": str(obj.get("region") or ""),
                    "urbanicity_proxy": str(obj.get("urbanicity_proxy") or ""),
                    "issue_id": str(obj.get("issue_id") or ""),
                    "issue_date": str(obj.get("issue_date") or ""),
                    "classification_label": str(obj.get("classification_label") or ""),
                    "text_chars": int(obj.get("text_chars") or 0),
                    "page_count": int(obj.get("page_count") or 0),
                }
            )
    if not rows:
        raise SystemExit(f"Missing selected issues and could not reconstruct from {panels_dir}")
    return pd.DataFrame.from_records(rows)


def main() -> None:
    args = _parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    parsed_issue_path = run_dir / "llm_panel_vs_issue" / "parsed_issue_outputs.jsonl"
    if not parsed_issue_path.is_file():
        raise SystemExit(f"Missing parsed issue outputs JSONL: {parsed_issue_path}")

    issue_df = _load_selected_issues(run_dir)
    issue_df["issue_date"] = pd.to_datetime(issue_df["issue_date"], errors="coerce")
    issue_df = issue_df.dropna(subset=["issue_date"]).copy()
    issue_df["issue_year"] = issue_df["issue_date"].dt.year
    issue_df["state_abbr"] = issue_df["state_abbr"].astype(str).str.lower()

    cat_map, num_map = _extract_issue_measure_maps(parsed_issue_path)

    # 1) Within-amendment topical summary.
    amend_labels = {"amendment_substantial", "amendment_targeted"}
    amend = issue_df[issue_df["classification_label"].isin(amend_labels)].copy()
    share_rows: list[dict[str, Any]] = []
    for r in amend.itertuples(index=False):
        cs = cat_map.get(str(r.issue_id), {k: 0.0 for k in _FIXED_CATEGORY_KEYS})
        for k in _FIXED_CATEGORY_KEYS:
            share_rows.append(
                {
                    "city_key": str(r.city_key),
                    "city_name": str(r.city_name),
                    "state_abbr": str(r.state_abbr),
                    "region": str(r.region),
                    "urbanicity_proxy": str(r.urbanicity_proxy),
                    "issue_id": str(r.issue_id),
                    "issue_date": str(pd.Timestamp(r.issue_date).date()),
                    "issue_year": int(r.issue_year),
                    "classification_label": str(r.classification_label),
                    "category": k,
                    "share": float(cs.get(k) or 0.0),
                }
            )
    amend_topic = pd.DataFrame.from_records(share_rows)
    amend_topic = amend_topic.sort_values(["city_key", "issue_date", "category"]).reset_index(drop=True)
    amend_topic_path = out_dir / "amendment_topic_shares.csv"
    amend_topic.to_csv(amend_topic_path, index=False, quoting=csv.QUOTE_MINIMAL)

    amend_topic_grp = (
        amend_topic.groupby(["region", "urbanicity_proxy", "issue_year", "classification_label", "category"], dropna=False)
        .agg(
            n_issues=("issue_id", "nunique"),
            mean_share=("share", "mean"),
            p50_share=("share", "median"),
            p90_share=("share", lambda s: float(pd.Series(s).quantile(0.9))),
        )
        .reset_index()
        .sort_values(["region", "urbanicity_proxy", "issue_year", "classification_label", "category"])
    )
    amend_topic_grp_path = out_dir / "amendment_topic_shares_by_region_urbanicity_year.csv"
    amend_topic_grp.to_csv(amend_topic_grp_path, index=False, quoting=csv.QUOTE_MINIMAL)

    # 2) Prior-doc deterministic delta coding.
    events: list[dict[str, Any]] = []
    max_back_days = int(args.max_back_years) * 365.25
    for city_key, g in issue_df.sort_values(["issue_date"]).groupby("city_key", dropna=False):
        g = g.sort_values(["issue_date", "issue_id"]).reset_index(drop=True)
        for row in g.itertuples(index=False):
            label = str(row.classification_label)
            if label not in amend_labels:
                continue
            cands = g[g["issue_date"] < row.issue_date].copy()
            if cands.empty:
                events.append(
                    {
                        "city_key": city_key,
                        "issue_id": str(row.issue_id),
                        "issue_date": str(pd.Timestamp(row.issue_date).date()),
                        "classification_label": label,
                        "matched_prior_issue_id": "",
                        "matched_prior_label": "",
                        "lag_days": math.nan,
                        "match_quality": "unmatched_no_prior",
                        "added_shares": "{}",
                        "removed_shares": "{}",
                        "clarified_shares": "{}",
                        "numeric_changes": "{}",
                    }
                )
                continue

            cands["lag_days"] = (pd.Timestamp(row.issue_date) - cands["issue_date"]).dt.days
            cands = cands[cands["lag_days"] <= max_back_days].copy()
            if cands.empty:
                events.append(
                    {
                        "city_key": city_key,
                        "issue_id": str(row.issue_id),
                        "issue_date": str(pd.Timestamp(row.issue_date).date()),
                        "classification_label": label,
                        "matched_prior_issue_id": "",
                        "matched_prior_label": "",
                        "lag_days": math.nan,
                        "match_quality": "unmatched_window",
                        "added_shares": "{}",
                        "removed_shares": "{}",
                        "clarified_shares": "{}",
                        "numeric_changes": "{}",
                    }
                )
                continue

            cands["prior_rank"] = cands["classification_label"].map({"full_ordinance": 0, "amendment_substantial": 1, "amendment_targeted": 2}).fillna(9)
            cands = cands.sort_values(["prior_rank", "lag_days", "issue_date"], ascending=[True, True, False]).reset_index(drop=True)
            prior = cands.iloc[0]

            cs_cur = cat_map.get(str(row.issue_id), {k: 0.0 for k in _FIXED_CATEGORY_KEYS})
            cs_prev = cat_map.get(str(prior.issue_id), {k: 0.0 for k in _FIXED_CATEGORY_KEYS})
            added = {k: max(0.0, float(cs_cur.get(k, 0.0)) - float(cs_prev.get(k, 0.0))) for k in _FIXED_CATEGORY_KEYS}
            removed = {k: max(0.0, float(cs_prev.get(k, 0.0)) - float(cs_cur.get(k, 0.0))) for k in _FIXED_CATEGORY_KEYS}
            clarified = {k: min(float(cs_cur.get(k, 0.0)), float(cs_prev.get(k, 0.0))) for k in _FIXED_CATEGORY_KEYS}

            nc_cur = num_map.get(str(row.issue_id), {})
            nc_prev = num_map.get(str(prior.issue_id), {})
            numeric_keys = ["max_min_lot_size_sqft", "max_height_ft", "max_front_setback_ft", "max_parking_spaces_per_unit"]
            num_delta: dict[str, Any] = {}
            for nk in numeric_keys:
                cv = nc_cur.get(nk)
                pv = nc_prev.get(nk)
                if isinstance(cv, (int, float)) and isinstance(pv, (int, float)):
                    num_delta[nk] = float(cv) - float(pv)
                else:
                    num_delta[nk] = None

            events.append(
                {
                    "city_key": city_key,
                    "issue_id": str(row.issue_id),
                    "issue_date": str(pd.Timestamp(row.issue_date).date()),
                    "classification_label": label,
                    "matched_prior_issue_id": str(prior.issue_id),
                    "matched_prior_label": str(prior.classification_label),
                    "lag_days": int(prior.lag_days),
                    "match_quality": "matched",
                    "added_shares": json.dumps(added, ensure_ascii=False),
                    "removed_shares": json.dumps(removed, ensure_ascii=False),
                    "clarified_shares": json.dumps(clarified, ensure_ascii=False),
                    "numeric_changes": json.dumps(num_delta, ensure_ascii=False),
                }
            )

    ev = pd.DataFrame.from_records(events).sort_values(["city_key", "issue_date", "issue_id"]).reset_index(drop=True)
    ev_path = out_dir / "amendment_delta_events.csv"
    ev.to_csv(ev_path, index=False, quoting=csv.QUOTE_MINIMAL)

    # Category summary across matched events.
    cat_sum_rows: list[dict[str, Any]] = []
    matched = ev[ev["match_quality"] == "matched"].copy()
    for r in matched.itertuples(index=False):
        added = json.loads(str(r.added_shares))
        removed = json.loads(str(r.removed_shares))
        clarified = json.loads(str(r.clarified_shares))
        for k in _FIXED_CATEGORY_KEYS:
            cat_sum_rows.append(
                {
                    "classification_label": str(r.classification_label),
                    "category": k,
                    "added_share": float(added.get(k) or 0.0),
                    "removed_share": float(removed.get(k) or 0.0),
                    "clarified_share": float(clarified.get(k) or 0.0),
                }
            )
    cat_sum = pd.DataFrame.from_records(cat_sum_rows)
    if cat_sum.empty:
        cat_agg = pd.DataFrame(
            columns=["classification_label", "category", "n_events", "mean_added_share", "mean_removed_share", "mean_clarified_share"]
        )
    else:
        cat_agg = (
            cat_sum.groupby(["classification_label", "category"], dropna=False)
            .agg(
                n_events=("category", "count"),
                mean_added_share=("added_share", "mean"),
                mean_removed_share=("removed_share", "mean"),
                mean_clarified_share=("clarified_share", "mean"),
            )
            .reset_index()
            .sort_values(["classification_label", "category"])
        )
    cat_agg_path = out_dir / "amendment_delta_category_summary.csv"
    cat_agg.to_csv(cat_agg_path, index=False, quoting=csv.QUOTE_MINIMAL)

    # Matching diagnostics.
    diag_rows = []
    for label, g in ev.groupby("classification_label", dropna=False):
        n_total = int(len(g))
        n_match = int((g["match_quality"] == "matched").sum())
        lag = pd.to_numeric(g.loc[g["match_quality"] == "matched", "lag_days"], errors="coerce").dropna()
        diag_rows.append(
            {
                "classification_label": label,
                "n_amendments": n_total,
                "n_matched": n_match,
                "matched_share": float(n_match / max(1, n_total)),
                "median_lag_days": float(lag.median()) if not lag.empty else math.nan,
                "p90_lag_days": float(lag.quantile(0.9)) if not lag.empty else math.nan,
            }
        )
    diag = pd.DataFrame.from_records(diag_rows).sort_values("classification_label")
    diag_path = out_dir / "amendment_delta_matching_diagnostics.csv"
    diag.to_csv(diag_path, index=False, quoting=csv.QUOTE_MINIMAL)

    print(
        "Done. "
        f"amendment_topic={amend_topic_path} "
        f"amendment_topic_grouped={amend_topic_grp_path} "
        f"delta_events={ev_path} "
        f"delta_summary={cat_agg_path} "
        f"matching_diag={diag_path}"
    )


if __name__ == "__main__":
    main()
