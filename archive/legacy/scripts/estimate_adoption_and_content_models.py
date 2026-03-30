#!/usr/bin/env python3
"""
Estimate adoption hazard and content/complexity models from merged panels.

Outputs:
  - models/adoption_model_table.csv
  - models/content_model_table.csv
  - models/model_spec_manifest.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


_BASE_COVARIATE_COLUMNS = [
    "population_place",
    "median_household_income_place",
    "median_home_value_place",
    "vacancy_rate_place",
    "permits_per_1000_pop",
    "unemployment_rate_county_pct",
    "per_capita_income_county",
]


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Estimate adoption and content models.")
    ap.add_argument(
        "--run-dir",
        default="/Users/saulrichardson/projects/newspapers/newspaper-analysis/reports/runs/prototype_zoning_panel_analysis/prototype_zoning_panel_analysis_iter8_scale30",
        help="Prototype run directory.",
    )
    ap.add_argument(
        "--adoption-panel",
        default="",
        help="Path to city_year_adoption_panel.parquet (default: <run-dir>/analysis_panel/city_year_adoption_panel.parquet).",
    )
    ap.add_argument(
        "--content-panel",
        default="",
        help="Path to city_year_content_panel.parquet (default: <run-dir>/analysis_panel/city_year_content_panel.parquet).",
    )
    ap.add_argument(
        "--output-dir",
        default="",
        help="Output directory (default: <run-dir>/models).",
    )
    ap.add_argument("--year-min", type=int, default=0, help="Optional year minimum filter.")
    ap.add_argument("--year-max", type=int, default=0, help="Optional year maximum filter.")
    return ap.parse_args()


def _log1p(series: pd.Series) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce")
    x = x.where(x >= 0)
    return np.log1p(x)


def _prep_covariates(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["log_population_place"] = _log1p(out.get("population_place"))
    out["log_median_household_income_place"] = _log1p(out.get("median_household_income_place"))
    out["log_median_home_value_place"] = _log1p(out.get("median_home_value_place"))
    out["log_per_capita_income_county"] = _log1p(out.get("per_capita_income_county"))
    out["vacancy_rate_place"] = pd.to_numeric(out.get("vacancy_rate_place"), errors="coerce")
    out["permits_per_1000_pop"] = pd.to_numeric(out.get("permits_per_1000_pop"), errors="coerce")
    out["unemployment_rate_county_pct"] = pd.to_numeric(out.get("unemployment_rate_county_pct"), errors="coerce")

    # Mean imputation + missing indicators keeps panels with partial coverage.
    for c in (
        "log_population_place",
        "log_median_household_income_place",
        "log_median_home_value_place",
        "log_per_capita_income_county",
        "vacancy_rate_place",
        "permits_per_1000_pop",
        "unemployment_rate_county_pct",
    ):
        miss = out[c].isna().astype(int)
        out[f"{c}__missing"] = miss
        mu = pd.to_numeric(out[c], errors="coerce").mean()
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(mu if pd.notna(mu) else 0.0)
    return out


def _coef_table(result: Any, *, model_name: str, subset: str, depvar: str) -> pd.DataFrame:
    params = result.params
    bse = result.bse
    pvals = result.pvalues
    conf = result.conf_int()
    rows: list[dict[str, Any]] = []
    for k in params.index.tolist():
        coef = float(params[k])
        se = float(bse[k]) if k in bse else math.nan
        p = float(pvals[k]) if k in pvals else math.nan
        ci_l = float(conf.loc[k, 0]) if k in conf.index else math.nan
        ci_u = float(conf.loc[k, 1]) if k in conf.index else math.nan
        rows.append(
            {
                "model_name": model_name,
                "subset": subset,
                "dependent_variable": depvar,
                "term": str(k),
                "coef": coef,
                "std_err": se,
                "p_value": p,
                "ci_95_low": ci_l,
                "ci_95_high": ci_u,
                "odds_ratio": (float(math.exp(coef)) if depvar == "adopt_event" else math.nan),
            }
        )
    return pd.DataFrame.from_records(rows)


def _fit_adoption_model(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    dat = df.copy()
    dat = dat[(dat["at_risk"] == 1) & dat["adopt_event"].notna()].copy()
    dat["year"] = pd.to_numeric(dat["year"], errors="coerce").astype("Int64")
    dat = dat.dropna(subset=["year"]).copy()
    dat["year"] = dat["year"].astype(int)
    dat["decade"] = (dat["year"] // 10) * 10

    region_col = next((c for c in ("region", "region_x", "region_y") if c in dat.columns), "")
    if not region_col:
        dat["region_std"] = "unknown"
    else:
        dat["region_std"] = (
            dat[region_col]
            .astype(str)
            .str.strip()
            .replace({"": "unknown", "nan": "unknown", "None": "unknown"})
            .fillna("unknown")
        )

    # Prepare optional text signals with explicit missingness indicators.
    text_cols_raw = ["n_issues", "full_ordinance_issue_share", "llm_issue_complexity_mean"]
    text_rows_mask = np.zeros(len(dat), dtype=bool)
    for c in text_cols_raw:
        v = pd.to_numeric(dat.get(c), errors="coerce")
        dat[c] = v
        text_rows_mask = text_rows_mask | v.notna().to_numpy()
        dat[f"{c}__missing"] = v.isna().astype(int)
        if v.notna().any():
            dat[c] = v.fillna(float(v.median()))
        else:
            dat[c] = 0.0
    dat["text_signal_available"] = text_rows_mask.astype(int)

    # External covariates may be structurally unavailable in historical at-risk years.
    cov_nonmissing_rows = {
        c: int(pd.to_numeric(dat.get(c), errors="coerce").notna().sum()) for c in _BASE_COVARIATE_COLUMNS
    }
    cov_terms: list[str] = []
    cov_miss_terms: list[str] = []
    if any(v > 0 for v in cov_nonmissing_rows.values()):
        dat = _prep_covariates(dat)
        cov_terms = [
            c
            for c in (
                "log_population_place",
                "log_median_household_income_place",
                "log_median_home_value_place",
                "vacancy_rate_place",
                "permits_per_1000_pop",
                "unemployment_rate_county_pct",
                "log_per_capita_income_county",
            )
            if c in dat.columns and pd.to_numeric(dat[c], errors="coerce").nunique(dropna=False) > 1
        ]
        cov_miss_terms = [
            f"{c}__missing"
            for c in cov_terms
            if f"{c}__missing" in dat.columns and pd.to_numeric(dat[f"{c}__missing"], errors="coerce").nunique(dropna=False) > 1
        ]

    text_terms = [
        c for c in text_cols_raw if c in dat.columns and pd.to_numeric(dat[c], errors="coerce").nunique(dropna=False) > 1
    ]
    text_miss_terms = [
        f"{c}__missing"
        for c in text_cols_raw
        if f"{c}__missing" in dat.columns and pd.to_numeric(dat[f"{c}__missing"], errors="coerce").nunique(dropna=False) > 1
    ]

    def _fit_one(*, model_name: str, subset: str, formula: str, frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
        errs: list[str] = []
        # Prefer GLM Binomial + robust covariance for better numerical stability.
        try:
            glm = smf.glm(formula=formula, data=frame, family=sm.families.Binomial())
            res = glm.fit(cov_type="HC1")
            tab = _coef_table(res, model_name=model_name, subset=subset, depvar="adopt_event")
            return (
                tab,
                {
                    "model_name": model_name,
                    "subset": subset,
                    "status": "ok",
                    "fit_engine": "glm_binomial_hc1",
                    "formula": formula,
                    "n_rows": int(len(frame)),
                    "n_events": int(pd.to_numeric(frame["adopt_event"], errors="coerce").fillna(0).sum()),
                    "aic": float(res.aic) if hasattr(res, "aic") else math.nan,
                    "bic": float(res.bic_llf) if hasattr(res, "bic_llf") else math.nan,
                },
            )
        except Exception as e:
            errs.append(str(e))
        # Secondary fallback.
        try:
            logit = smf.logit(formula=formula, data=frame)
            res2 = logit.fit(disp=False, maxiter=300)
            tab2 = _coef_table(res2, model_name=model_name, subset=subset, depvar="adopt_event")
            return (
                tab2,
                {
                    "model_name": model_name,
                    "subset": subset,
                    "status": "ok",
                    "fit_engine": "logit_mle",
                    "formula": formula,
                    "n_rows": int(len(frame)),
                    "n_events": int(pd.to_numeric(frame["adopt_event"], errors="coerce").fillna(0).sum()),
                    "aic": float(res2.aic) if hasattr(res2, "aic") else math.nan,
                    "bic": float(res2.bic) if hasattr(res2, "bic") else math.nan,
                    "converged": bool(getattr(res2, "mle_retvals", {}).get("converged", True)),
                },
            )
        except Exception as e:
            errs.append(str(e))
        return (
            pd.DataFrame(),
            {
                "model_name": model_name,
                "subset": subset,
                "status": "failed",
                "formula": formula,
                "n_rows": int(len(frame)),
                "n_events": int(pd.to_numeric(frame["adopt_event"], errors="coerce").fillna(0).sum()),
                "errors": errs,
            },
        )

    rhs_base = []
    if dat["region_std"].nunique(dropna=False) > 1:
        rhs_base.append("C(region_std)")
    if dat["decade"].nunique(dropna=False) > 1:
        rhs_base.append("C(decade)")
    if not rhs_base:
        rhs_base = ["1"]

    specs: list[dict[str, Any]] = [
        {
            "model_name": "adoption_hazard_logit_text_region_restricted",
            "subset": "text_signal_rows",
            "frame": dat[dat["text_signal_available"] == 1].copy(),
            "rhs": ["C(region_std)"] + text_terms + text_miss_terms,
        },
        {
            "model_name": "adoption_hazard_logit_text_region_decade_restricted",
            "subset": "text_signal_rows",
            "frame": dat[dat["text_signal_available"] == 1].copy(),
            "rhs": rhs_base + text_terms + text_miss_terms,
        },
        {
            "model_name": "adoption_hazard_logit_baseline_region_decade",
            "subset": "at_risk_all_rows",
            "frame": dat.copy(),
            "rhs": rhs_base,
        },
    ]
    if cov_terms:
        specs.append(
            {
                "model_name": "adoption_hazard_logit_covariates_region_decade",
                "subset": "at_risk_all_rows",
                "frame": dat.copy(),
                "rhs": rhs_base + cov_terms + cov_miss_terms,
            }
        )

    tables: list[pd.DataFrame] = []
    spec_meta: list[dict[str, Any]] = []
    for spec in specs:
        frame = spec["frame"]
        rhs_terms: list[str] = []
        for t in spec["rhs"]:
            if not t:
                continue
            if t == "1":
                rhs_terms.append(t)
                continue
            if t.startswith("C(") and t.endswith(")"):
                c = t[2:-1]
                if c in frame.columns and frame[c].astype(str).nunique(dropna=False) > 1:
                    rhs_terms.append(t)
                continue
            if t in frame.columns and pd.to_numeric(frame[t], errors="coerce").nunique(dropna=False) > 1:
                rhs_terms.append(t)
        if frame.empty:
            spec_meta.append(
                {
                    "model_name": spec["model_name"],
                    "subset": spec["subset"],
                    "status": "skipped_empty_frame",
                    "n_rows": 0,
                    "n_events": 0,
                }
            )
            continue
        if not rhs_terms:
            rhs_terms = ["1"]
        formula = "adopt_event ~ " + " + ".join(rhs_terms)
        tab, meta = _fit_one(
            model_name=spec["model_name"],
            subset=spec["subset"],
            formula=formula,
            frame=frame,
        )
        if not tab.empty:
            tables.append(tab)
        spec_meta.append(meta)

    out = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame(
        columns=[
            "model_name",
            "subset",
            "dependent_variable",
            "term",
            "coef",
            "std_err",
            "p_value",
            "ci_95_low",
            "ci_95_high",
            "odds_ratio",
        ]
    )
    meta = {
        "status": ("ok" if tables else "failed"),
        "n_rows_at_risk": int(len(dat)),
        "n_events_at_risk": int(pd.to_numeric(dat["adopt_event"], errors="coerce").fillna(0).sum()),
        "text_signal_rows": int(dat["text_signal_available"].sum()),
        "text_terms_used": text_terms,
        "text_missing_terms_used": text_miss_terms,
        "covariate_nonmissing_rows": cov_nonmissing_rows,
        "covariate_terms_used": cov_terms,
        "models": spec_meta,
    }
    return out, meta


def _fit_content_models(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    content = df.copy()
    content["year"] = pd.to_numeric(content["year"], errors="coerce").astype("Int64")
    content = content.dropna(subset=["year"]).copy()
    content["year"] = content["year"].astype(int)
    content = _prep_covariates(content)

    depvars = [
        "llm_issue_complexity_mean",
        "distinct_zone_code_count_max",
        "share__land_use_restrictions_mean",
        "share__bulk_dimensional_standards_mean",
        "share__procedural_governance_mean",
    ]
    subsets: dict[str, pd.DataFrame] = {
        "pooled_all": content[content["classification_label"] == "pooled_all"].copy(),
        "full_ordinance": content[content["classification_label"] == "full_ordinance"].copy(),
        "amendments_only": content[content["classification_label"].isin(["amendment_substantial", "amendment_targeted"])].copy(),
    }

    base_terms = [
        "log_population_place",
        "log_median_household_income_place",
        "log_median_home_value_place",
        "vacancy_rate_place",
        "permits_per_1000_pop",
        "unemployment_rate_county_pct",
        "log_per_capita_income_county",
    ]
    miss_terms = [f"{c}__missing" for c in base_terms]
    out_tables: list[pd.DataFrame] = []
    specs: list[dict[str, Any]] = []

    for subset_name, d in subsets.items():
        for dep in depvars:
            dd = d.copy()
            dd[dep] = pd.to_numeric(dd.get(dep), errors="coerce")
            dd = dd.dropna(subset=[dep]).copy()
            if dd.empty or dd["city_key"].nunique() < 4:
                specs.append(
                    {
                        "model_name": "content_fe_ols",
                        "subset": subset_name,
                        "dependent_variable": dep,
                        "status": "skipped_insufficient_data",
                        "n_rows": int(len(dd)),
                        "n_city_keys": int(dd["city_key"].nunique()) if not dd.empty else 0,
                    }
                )
                continue

            formulas = [
                dep + " ~ " + " + ".join(base_terms + miss_terms) + " + C(city_key) + C(year)",
                dep + " ~ " + " + ".join(base_terms + miss_terms) + " + C(year)",
                dep + " ~ " + " + ".join(base_terms + miss_terms),
            ]
            chosen = ""
            res = None
            errs: list[str] = []
            for f in formulas:
                try:
                    model = smf.ols(formula=f, data=dd)
                    # HC1 for robust inference in heteroskedastic panel settings.
                    res = model.fit(cov_type="HC1")
                    chosen = f
                    break
                except Exception as e:
                    errs.append(str(e))
                    continue
            if res is None:
                specs.append(
                    {
                        "model_name": "content_fe_ols",
                        "subset": subset_name,
                        "dependent_variable": dep,
                        "status": "failed",
                        "n_rows": int(len(dd)),
                        "n_city_keys": int(dd["city_key"].nunique()),
                        "spec_candidates": formulas,
                        "errors": errs,
                    }
                )
                continue

            tab = _coef_table(res, model_name="content_fe_ols", subset=subset_name, depvar=dep)
            out_tables.append(tab)
            specs.append(
                {
                    "model_name": "content_fe_ols",
                    "subset": subset_name,
                    "dependent_variable": dep,
                    "status": "ok",
                    "n_rows": int(len(dd)),
                    "n_city_keys": int(dd["city_key"].nunique()),
                    "formula": chosen,
                    "r2": float(res.rsquared) if hasattr(res, "rsquared") else math.nan,
                    "adj_r2": float(res.rsquared_adj) if hasattr(res, "rsquared_adj") else math.nan,
                    "aic": float(res.aic) if hasattr(res, "aic") else math.nan,
                    "bic": float(res.bic) if hasattr(res, "bic") else math.nan,
                }
            )

    out = pd.concat(out_tables, ignore_index=True) if out_tables else pd.DataFrame(
        columns=[
            "model_name",
            "subset",
            "dependent_variable",
            "term",
            "coef",
            "std_err",
            "p_value",
            "ci_95_low",
            "ci_95_high",
            "odds_ratio",
        ]
    )
    return out, specs


def main() -> None:
    args = _parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve() if str(args.output_dir).strip() else (run_dir / "models")
    out_dir.mkdir(parents=True, exist_ok=True)

    adoption_path = (
        Path(args.adoption_panel).expanduser().resolve()
        if str(args.adoption_panel).strip()
        else run_dir / "analysis_panel" / "city_year_adoption_panel.parquet"
    )
    content_path = (
        Path(args.content_panel).expanduser().resolve()
        if str(args.content_panel).strip()
        else run_dir / "analysis_panel" / "city_year_content_panel.parquet"
    )
    if not adoption_path.is_file():
        raise SystemExit(f"Missing adoption panel: {adoption_path}")
    if not content_path.is_file():
        raise SystemExit(f"Missing content panel: {content_path}")

    adoption = pd.read_parquet(adoption_path)
    content = pd.read_parquet(content_path)
    if int(args.year_min) > 0:
        adoption = adoption[pd.to_numeric(adoption["year"], errors="coerce") >= int(args.year_min)].copy()
        content = content[pd.to_numeric(content["year"], errors="coerce") >= int(args.year_min)].copy()
    if int(args.year_max) > 0:
        adoption = adoption[pd.to_numeric(adoption["year"], errors="coerce") <= int(args.year_max)].copy()
        content = content[pd.to_numeric(content["year"], errors="coerce") <= int(args.year_max)].copy()

    adoption_table, adoption_meta = _fit_adoption_model(adoption)
    content_table, content_specs = _fit_content_models(content)

    ad_path = out_dir / "adoption_model_table.csv"
    ct_path = out_dir / "content_model_table.csv"
    ad_opt_path = out_dir / "adoption_model_option_diagnostics.csv"
    adoption_table.to_csv(ad_path, index=False)
    content_table.to_csv(ct_path, index=False)
    ad_models = adoption_meta.get("models") if isinstance(adoption_meta, dict) else []
    if isinstance(ad_models, list) and ad_models:
        pd.DataFrame.from_records(ad_models).to_csv(ad_opt_path, index=False)
    else:
        pd.DataFrame().to_csv(ad_opt_path, index=False)

    manifest = {
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "run_dir": str(run_dir),
        "inputs": {
            "adoption_panel": str(adoption_path),
            "content_panel": str(content_path),
        },
        "filters": {"year_min": int(args.year_min), "year_max": int(args.year_max)},
        "adoption_model": adoption_meta,
        "content_models": content_specs,
        "n_adoption_rows_input": int(len(adoption)),
        "n_content_rows_input": int(len(content)),
        "outputs": {
            "adoption_model_table": str(ad_path),
            "content_model_table": str(ct_path),
            "adoption_model_option_diagnostics": str(ad_opt_path),
        },
    }
    (out_dir / "model_spec_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(
        "Done. "
        f"adoption_terms={len(adoption_table)} "
        f"content_terms={len(content_table)} "
        f"content_specs={len(content_specs)}"
    )


if __name__ == "__main__":
    main()
