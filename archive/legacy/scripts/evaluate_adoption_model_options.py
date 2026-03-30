#!/usr/bin/env python3
"""
Evaluate estimable adoption-model feature sets with grouped cross-validation.

This script is intentionally predictive-focused. It scores model options using
GroupKFold split by city_key to reduce panel leakage.

Output:
  - models/adoption_option_grid_ml_groupcv_estimable.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.metrics import log_loss
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Evaluate adoption option grid with grouped CV.")
    ap.add_argument(
        "--run-dir",
        default="/Users/saulrichardson/projects/newspapers/newspaper-analysis/reports/runs/prototype_zoning_panel_analysis/prototype_zoning_panel_analysis_iter8_scale30",
        help="Prototype run directory.",
    )
    ap.add_argument(
        "--adoption-panel",
        default="",
        help="Optional explicit adoption panel parquet (default: <run-dir>/analysis_panel/city_year_adoption_panel.parquet).",
    )
    ap.add_argument(
        "--output-csv",
        default="",
        help="Optional output CSV path (default: <run-dir>/models/adoption_option_grid_ml_groupcv_estimable.csv).",
    )
    return ap.parse_args()


def _resolve_region(df: pd.DataFrame) -> pd.Series:
    for c in ("region", "region_x", "region_y"):
        if c in df.columns:
            return (
                df[c]
                .astype(str)
                .str.strip()
                .replace({"": "unknown", "nan": "unknown", "None": "unknown"})
                .fillna("unknown")
            )
    return pd.Series("unknown", index=df.index)


def _eval_option(
    *,
    name: str,
    frame: pd.DataFrame,
    num_cols: list[str],
    cat_cols: list[str],
    require_any_num: bool = False,
) -> dict[str, Any]:
    dd = frame.copy()
    for c in (num_cols + cat_cols):
        if c not in dd.columns:
            dd[c] = np.nan

    if require_any_num and num_cols:
        mask = np.zeros(len(dd), dtype=bool)
        for c in num_cols:
            mask = mask | pd.to_numeric(dd[c], errors="coerce").notna().to_numpy()
        dd = dd.loc[mask].copy()

    cols = num_cols + cat_cols + ["adopt_event", "city_key"]
    dd = dd[cols].copy()
    if dd.empty:
        return {
            "option": name,
            "status": "empty_after_filter",
            "n_rows": 0,
            "n_events": 0,
            "n_features_num": 0,
            "n_features_cat": 0,
            "group_cv_auc_mean": np.nan,
            "group_cv_logloss_mean": np.nan,
            "group_cv_brier_mean": np.nan,
            "n_folds_scored": 0,
        }

    y = pd.to_numeric(dd.pop("adopt_event"), errors="coerce").fillna(0).astype(int).to_numpy()
    groups = dd.pop("city_key").astype(str).to_numpy()

    num_keep = [c for c in num_cols if pd.to_numeric(dd[c], errors="coerce").nunique(dropna=True) > 1]
    cat_keep = [c for c in cat_cols if dd[c].astype(str).nunique(dropna=True) > 1]
    if (not num_keep) and (not cat_keep):
        return {
            "option": name,
            "status": "not_estimable",
            "n_rows": int(len(dd)),
            "n_events": int(y.sum()),
            "n_features_num": 0,
            "n_features_cat": 0,
            "group_cv_auc_mean": np.nan,
            "group_cv_logloss_mean": np.nan,
            "group_cv_brier_mean": np.nan,
            "n_folds_scored": 0,
        }

    transformers: list[tuple[str, Any, list[str]]] = []
    if num_keep:
        transformers.append(
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                num_keep,
            )
        )
    if cat_keep:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("ohe", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                cat_keep,
            )
        )

    pre = ColumnTransformer(transformers, remainder="drop")
    clf = LogisticRegression(max_iter=4000, solver="lbfgs", class_weight="balanced", C=1.0)
    pipe = Pipeline([("pre", pre), ("clf", clf)])

    aucs: list[float] = []
    lls: list[float] = []
    briers: list[float] = []
    gkf = GroupKFold(n_splits=5)
    for tr, te in gkf.split(dd, y, groups):
        x_tr = dd.iloc[tr]
        x_te = dd.iloc[te]
        y_tr = y[tr]
        y_te = y[te]
        if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                pipe.fit(x_tr, y_tr)
                p = pipe.predict_proba(x_te)[:, 1]
            aucs.append(float(roc_auc_score(y_te, p)))
            lls.append(float(log_loss(y_te, p, labels=[0, 1])))
            briers.append(float(brier_score_loss(y_te, p)))
        except Exception:
            continue

    status = "ok" if aucs else "degenerate_folds"
    return {
        "option": name,
        "status": status,
        "n_rows": int(len(dd)),
        "n_events": int(y.sum()),
        "n_features_num": int(len(num_keep)),
        "n_features_cat": int(len(cat_keep)),
        "group_cv_auc_mean": float(np.mean(aucs)) if aucs else np.nan,
        "group_cv_logloss_mean": float(np.mean(lls)) if lls else np.nan,
        "group_cv_brier_mean": float(np.mean(briers)) if briers else np.nan,
        "n_folds_scored": int(len(aucs)),
    }


def main() -> None:
    args = _parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    adoption_panel = (
        Path(args.adoption_panel).expanduser().resolve()
        if str(args.adoption_panel).strip()
        else run_dir / "analysis_panel" / "city_year_adoption_panel.parquet"
    )
    out_csv = (
        Path(args.output_csv).expanduser().resolve()
        if str(args.output_csv).strip()
        else run_dir / "models" / "adoption_option_grid_ml_groupcv_estimable.csv"
    )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if not adoption_panel.is_file():
        raise SystemExit(f"Missing adoption panel: {adoption_panel}")

    df = pd.read_parquet(adoption_panel)
    df = df[(df["at_risk"] == 1) & df["adopt_event"].notna()].copy()
    df["adopt_event"] = pd.to_numeric(df["adopt_event"], errors="coerce").fillna(0).astype(int)
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["year"]).copy()
    df["year"] = df["year"].astype(int)
    df["decade"] = (df["year"] // 10) * 10
    df["region"] = _resolve_region(df)

    # Constrain numeric text-derived features to known ranges where possible.
    df["n_issues"] = pd.to_numeric(df.get("n_issues"), errors="coerce").clip(lower=0, upper=50)
    df["full_ordinance_issue_share"] = pd.to_numeric(df.get("full_ordinance_issue_share"), errors="coerce").clip(lower=0, upper=1)
    df["llm_issue_complexity_mean"] = pd.to_numeric(df.get("llm_issue_complexity_mean"), errors="coerce").clip(lower=0, upper=100)

    options: list[dict[str, Any]] = []
    options.append(
        _eval_option(
            name="H1_region_decade_baseline",
            frame=df,
            num_cols=[],
            cat_cols=["region", "decade"],
            require_any_num=False,
        )
    )
    options.append(
        _eval_option(
            name="H2_text_region_decade",
            frame=df,
            num_cols=["n_issues", "full_ordinance_issue_share", "llm_issue_complexity_mean"],
            cat_cols=["region", "decade"],
            require_any_num=True,
        )
    )
    options.append(
        _eval_option(
            name="H3_text_region",
            frame=df,
            num_cols=["n_issues", "full_ordinance_issue_share", "llm_issue_complexity_mean"],
            cat_cols=["region"],
            require_any_num=True,
        )
    )
    options.append(
        _eval_option(
            name="H4_text_decade",
            frame=df,
            num_cols=["n_issues", "full_ordinance_issue_share", "llm_issue_complexity_mean"],
            cat_cols=["decade"],
            require_any_num=True,
        )
    )
    options.append(
        _eval_option(
            name="H5_text_only",
            frame=df,
            num_cols=["n_issues", "full_ordinance_issue_share", "llm_issue_complexity_mean"],
            cat_cols=[],
            require_any_num=True,
        )
    )
    options.append(
        _eval_option(
            name="H6_region_only",
            frame=df,
            num_cols=[],
            cat_cols=["region"],
            require_any_num=False,
        )
    )

    out = pd.DataFrame.from_records(options)
    out = out.sort_values(["group_cv_auc_mean", "group_cv_logloss_mean"], ascending=[False, True], na_position="last")
    out.to_csv(out_csv, index=False)
    print(f"Done. options={len(out)} output={out_csv}")


if __name__ == "__main__":
    main()
