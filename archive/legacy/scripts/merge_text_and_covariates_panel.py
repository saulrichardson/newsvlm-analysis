#!/usr/bin/env python3
"""
Merge text-derived zoning measures with external covariates into analysis panels.

Outputs:
  - analysis_panel/city_year_adoption_panel.parquet
  - analysis_panel/city_year_content_panel.parquet
  - analysis_panel/merge_manifest.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
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


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Merge text measures and covariates into analysis-ready panels.")
    ap.add_argument(
        "--run-dir",
        default="/Users/saulrichardson/projects/newspapers/newspaper-analysis/reports/runs/prototype_zoning_panel_analysis/prototype_zoning_panel_analysis_iter8_scale30",
        help="Prototype run directory.",
    )
    ap.add_argument(
        "--covariates-csv",
        default="",
        help="Optional explicit covariates CSV path (default: <run-dir>/covariates/city_year_external_covariates.csv).",
    )
    ap.add_argument(
        "--output-dir",
        default="",
        help="Output directory (default: <run-dir>/analysis_panel).",
    )
    return ap.parse_args()


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


def _parse_issue_id(custom_id: str) -> tuple[str, str]:
    parts = str(custom_id or "").split("::", 2)
    if len(parts) >= 3 and parts[0] == "issue":
        return str(parts[1]), str(parts[2])
    return "", ""


def _load_selected_issues(run_dir: Path) -> pd.DataFrame:
    p = run_dir / "panels" / "selected_panel_issues.csv"
    if p.is_file():
        return pd.read_csv(p)
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
        raise SystemExit(f"Missing selected issues CSV and could not reconstruct from {panels_dir}")
    return pd.DataFrame.from_records(rows)


def _load_issue_llm_measures(parsed_issue_path: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for obj in _iter_jsonl(parsed_issue_path):
        city_key, issue_id = _parse_issue_id(str(obj.get("custom_id") or ""))
        parsed = obj.get("parsed_json")
        if not issue_id or not isinstance(parsed, dict):
            continue
        cs = parsed.get("category_shares") if isinstance(parsed.get("category_shares"), dict) else {}
        measures = parsed.get("measures") if isinstance(parsed.get("measures"), dict) else {}
        numeric = parsed.get("numeric_constraints") if isinstance(parsed.get("numeric_constraints"), dict) else {}
        row = {
            "city_key": city_key,
            "issue_id": issue_id,
            "llm_issue_complexity_0_100": pd.to_numeric(measures.get("regulatory_complexity_0_100"), errors="coerce"),
            "llm_issue_procedural_intensity_0_100": pd.to_numeric(measures.get("procedural_intensity_0_100"), errors="coerce"),
            "llm_issue_dimensional_specificity_0_100": pd.to_numeric(measures.get("dimensional_specificity_0_100"), errors="coerce"),
            "llm_max_min_lot_size_sqft": pd.to_numeric(numeric.get("max_min_lot_size_sqft"), errors="coerce"),
            "llm_max_height_ft": pd.to_numeric(numeric.get("max_height_ft"), errors="coerce"),
            "llm_max_front_setback_ft": pd.to_numeric(numeric.get("max_front_setback_ft"), errors="coerce"),
            "llm_max_parking_spaces_per_unit": pd.to_numeric(numeric.get("max_parking_spaces_per_unit"), errors="coerce"),
        }
        for k in _FIXED_CATEGORY_KEYS:
            row[f"share__{k}"] = pd.to_numeric(cs.get(k), errors="coerce")
        rows.append(row)
    return pd.DataFrame.from_records(rows)


def _build_adoption_panel(
    *,
    issues: pd.DataFrame,
    cov: pd.DataFrame,
    first_full: pd.DataFrame,
    city_meta: pd.DataFrame,
) -> pd.DataFrame:
    city_bounds = (
        issues.groupby("city_key", dropna=False)
        .agg(
            city_name=("city_name", "first"),
            state_abbr=("state_abbr", "first"),
            region=("region", "first"),
            urbanicity_proxy=("urbanicity_proxy", "first"),
            min_issue_year=("year", "min"),
            max_issue_year=("year", "max"),
        )
        .reset_index()
    )
    rows: list[dict[str, Any]] = []
    ff_map = {
        str(r.city_key): (int(r.first_full_ordinance_year) if pd.notna(r.first_full_ordinance_year) else None)
        for r in first_full.itertuples(index=False)
    }
    for r in city_bounds.itertuples(index=False):
        lo = int(r.min_issue_year)
        hi = int(r.max_issue_year)
        first_year = ff_map.get(str(r.city_key))
        for y in range(lo, hi + 1):
            adopted = int(first_year is not None and y >= int(first_year))
            at_risk = int(first_year is None or y <= int(first_year))
            adopt_event = int(first_year is not None and y == int(first_year))
            rows.append(
                {
                    "city_key": str(r.city_key),
                    "city_name": str(r.city_name),
                    "state_abbr": str(r.state_abbr),
                    "region": str(r.region),
                    "urbanicity_proxy": str(r.urbanicity_proxy),
                    "year": int(y),
                    "first_full_ordinance_year": first_year,
                    "adopted_by_year": adopted,
                    "at_risk": at_risk,
                    "adopt_event": adopt_event,
                }
            )
    panel = pd.DataFrame.from_records(rows)

    issue_ag = (
        issues.groupby(["city_key", "year"], dropna=False)
        .agg(
            n_issues=("issue_id", "nunique"),
            n_full_ordinance_issues=("classification_label", lambda s: int((s == "full_ordinance").sum())),
            llm_issue_complexity_mean=("llm_issue_complexity_0_100", "mean"),
        )
        .reset_index()
    )
    issue_ag["full_ordinance_issue_share"] = (
        issue_ag["n_full_ordinance_issues"] / issue_ag["n_issues"].replace(0, math.nan)
    )

    panel = panel.merge(issue_ag, on=["city_key", "year"], how="left")
    panel = panel.merge(cov, on=["city_key", "year"], how="left")
    panel = panel.merge(
        city_meta[
            [
                "city_key",
                "county_fips",
                "county_name",
                "place_geoid",
                "state_fips",
                "place_fips",
            ]
        ].drop_duplicates(),
        on="city_key",
        how="left",
    )
    return panel.sort_values(["city_key", "year"]).reset_index(drop=True)


def _build_content_panel(
    *,
    issues: pd.DataFrame,
    cov: pd.DataFrame,
    zone_metrics: pd.DataFrame,
) -> pd.DataFrame:
    share_cols = [f"share__{k}" for k in _FIXED_CATEGORY_KEYS]

    def _agg(g: pd.DataFrame) -> pd.Series:
        out: dict[str, Any] = {
            "n_issues": int(g["issue_id"].nunique()),
            "llm_issue_complexity_mean": pd.to_numeric(g["llm_issue_complexity_0_100"], errors="coerce").mean(),
            "llm_issue_procedural_intensity_mean": pd.to_numeric(g["llm_issue_procedural_intensity_0_100"], errors="coerce").mean(),
            "llm_issue_dimensional_specificity_mean": pd.to_numeric(g["llm_issue_dimensional_specificity_0_100"], errors="coerce").mean(),
        }
        for c in share_cols:
            out[f"{c}_mean"] = pd.to_numeric(g[c], errors="coerce").mean()
        return pd.Series(out)

    by_label = (
        issues.groupby(["city_key", "city_name", "state_abbr", "region", "urbanicity_proxy", "year", "classification_label"], dropna=False)
        .apply(_agg)
        .reset_index()
    )
    pooled = (
        issues.groupby(["city_key", "city_name", "state_abbr", "region", "urbanicity_proxy", "year"], dropna=False)
        .apply(_agg)
        .reset_index()
    )
    pooled["classification_label"] = "pooled_all"
    content = pd.concat([by_label, pooled], ignore_index=True)

    zm = zone_metrics.copy()
    if not zm.empty:
        zm["issue_year"] = pd.to_numeric(zm["issue_year"], errors="coerce")
        zm = zm.rename(columns={"issue_year": "year"})
        zm = (
            zm.groupby(["city_key", "year"], dropna=False)
            .agg(
                distinct_zone_code_count_max=("distinct_zone_code_count_max", "max"),
                max_min_lot_size_sqft=("max_min_lot_size_sqft", "max"),
                max_height_ft_mention=("max_height_ft_mention", "max"),
                max_front_setback_ft=("max_front_setback_ft", "max"),
                max_parking_spaces_per_unit=("max_parking_spaces_per_unit", "max"),
            )
            .reset_index()
        )
        content = content.merge(zm, on=["city_key", "year"], how="left")

    content = content.merge(cov, on=["city_key", "year"], how="left")
    return content.sort_values(["city_key", "year", "classification_label"]).reset_index(drop=True)


def main() -> None:
    args = _parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve() if str(args.output_dir).strip() else (run_dir / "analysis_panel")
    out_dir.mkdir(parents=True, exist_ok=True)

    parsed_issue_path = run_dir / "llm_panel_vs_issue" / "parsed_issue_outputs.jsonl"
    first_full_path = run_dir / "tables" / "initial_full_ordinance_by_city.csv"
    cov_path = Path(args.covariates_csv).expanduser().resolve() if str(args.covariates_csv).strip() else (
        run_dir / "covariates" / "city_year_external_covariates.csv"
    )
    place_xw_path = run_dir / "covariates" / "city_place_crosswalk.csv"
    county_xw_path = run_dir / "covariates" / "city_county_crosswalk.csv"
    zone_metrics_path = run_dir / "tables" / "distinct_zone_metrics_by_city_year.csv"

    for p in (parsed_issue_path, first_full_path, cov_path):
        if not p.is_file():
            raise SystemExit(f"Missing required input: {p}")

    issues = _load_selected_issues(run_dir)
    issues["state_abbr"] = issues["state_abbr"].astype(str).str.lower()
    issues["year"] = pd.to_datetime(issues["issue_date"], errors="coerce").dt.year
    issues = issues.dropna(subset=["year"]).copy()
    issues["year"] = issues["year"].astype(int)

    llm_issue = _load_issue_llm_measures(parsed_issue_path)
    issues = issues.merge(llm_issue, on=["city_key", "issue_id"], how="left")

    cov = pd.read_csv(cov_path)
    cov["year"] = pd.to_numeric(cov["year"], errors="coerce").astype("Int64")
    cov = cov.dropna(subset=["year"]).copy()
    cov["year"] = cov["year"].astype(int)

    first_full = pd.read_csv(first_full_path)
    first_full["state_abbr"] = first_full["state_abbr"].astype(str).str.lower()
    first_full["city_key"] = (
        first_full["city_name"].astype(str).str.lower().str.replace(r"[^a-z0-9]+", "_", regex=True).str.strip("_")
        + "__"
        + first_full["state_abbr"].astype(str).str.lower().str.replace(r"[^a-z0-9]+", "_", regex=True).str.strip("_")
    )

    place_xw = pd.read_csv(place_xw_path) if place_xw_path.is_file() else pd.DataFrame(columns=["city_key"])
    county_xw = pd.read_csv(county_xw_path) if county_xw_path.is_file() else pd.DataFrame(columns=["city_key"])
    city_meta = place_xw.merge(county_xw, on=["city_key", "city_name", "state_abbr", "state_fips", "place_geoid"], how="outer")

    zone_metrics = pd.read_csv(zone_metrics_path) if zone_metrics_path.is_file() else pd.DataFrame()

    adoption_panel = _build_adoption_panel(
        issues=issues,
        cov=cov,
        first_full=first_full,
        city_meta=city_meta,
    )
    content_panel = _build_content_panel(
        issues=issues,
        cov=cov,
        zone_metrics=zone_metrics,
    )

    ad_path = out_dir / "city_year_adoption_panel.parquet"
    ct_path = out_dir / "city_year_content_panel.parquet"
    adoption_panel.to_parquet(ad_path, index=False)
    content_panel.to_parquet(ct_path, index=False)

    manifest = {
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "run_dir": str(run_dir),
        "output_dir": str(out_dir),
        "inputs": {
            "selected_panel_issues": str(run_dir / "panels" / "selected_panel_issues.csv"),
            "parsed_issue_outputs": str(parsed_issue_path),
            "initial_full_ordinance_by_city": str(first_full_path),
            "covariates": str(cov_path),
            "city_place_crosswalk": str(place_xw_path),
            "city_county_crosswalk": str(county_xw_path),
            "distinct_zone_metrics_by_city_year": str(zone_metrics_path),
        },
        "outputs": {
            "city_year_adoption_panel_parquet": str(ad_path),
            "city_year_content_panel_parquet": str(ct_path),
        },
        "n_issue_rows": int(len(issues)),
        "n_adoption_rows": int(len(adoption_panel)),
        "n_content_rows": int(len(content_panel)),
        "n_city_keys": int(issues["city_key"].nunique()),
    }
    (out_dir / "merge_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(
        "Done. "
        f"adoption_rows={len(adoption_panel)} "
        f"content_rows={len(content_panel)} "
        f"cities={issues['city_key'].nunique()}"
    )


if __name__ == "__main__":
    main()
