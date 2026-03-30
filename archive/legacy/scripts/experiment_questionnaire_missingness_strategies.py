#!/usr/bin/env python3
"""
Experiment: missingness-handling strategies for questionnaire PCA.

Goal
----
Quantify the tradeoffs of a few concrete approaches to "missingness dominates PCA swings"
using *actual artifacts* from the repo:
  - questionnaire normalized outputs (normalized.jsonl)
  - PCA workbook (scores + loadings) as the current baseline
  - Questions.xlsx for question types and orientation

Strategies evaluated (current default + 3 alternatives)
-------------------------------------------------------
1) baseline_current_pca:
   Uses the existing PCA workbook scores; serves as the reference.

2) baseline_residualized_vs_coverage (global):
   Keeps baseline PCs but subtracts a *global* linear coverage effect:
     residual(score) = score - (alpha + beta * coverage)
   This is analysis-only (doesn't change loadings/features).

2b) baseline_residualized_PC1_only (global):
   Only residualize Principal_Component_1 (leave others untouched).

2c) baseline_residualized_PC1_within_slug:
   Residualize PC1 within each slug separately (helps if coverage differs by slug).

3) obs_augmented_pca_mean_impute:
   Builds a new feature matrix that separates:
     - q_value (oriented numeric/binary), missing as NaN then mean-imputed
     - q_obs (0/1 indicator of whether an answer was observed)
   Runs PCA on [q_value_imputed, q_obs] and identifies a "coverage component"
   by correlation with coverage.

4) carry_forward_pca_mean_impute (last-any):
   Applies within-slug carry-forward on q_value across years before mean-imputation
   and PCA. This tests the "ordinance persistence" idea.

4b) carry_forward_observed_only_mean_impute:
   Carry-forward only from last *observed* (non-null) value (do not propagate carried values).

4c) carry_forward_max_gap_2_mean_impute:
   Carry-forward from last observed value only if it is within the last 2 years.

4d) carry_forward_gap_sweep:
   Repeat 4c for several max-gap settings to show the stability/assumption tradeoff curve.

Outputs
-------
Writes:
  - A markdown summary with key diagnostics (correlations, volatility, etc.)
  - An Excel workbook with per-strategy scores (optional)

This is intentionally an experiment script, not a production pipeline change.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from os.path import expanduser
from pathlib import Path
from typing import Any


def _eprint(msg: str) -> None:
    print(msg, flush=True)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise SystemExit(f"Missing file: {path}")
    out: list[dict[str, Any]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:  # noqa: BLE001
            raise SystemExit(f"Invalid JSON in {path}: {exc}") from exc
        if not isinstance(obj, dict):
            raise SystemExit(f"Expected JSON object in {path}, got {type(obj).__name__}")
        out.append(obj)
    return out


def _safe_float(v: Any) -> float | None:
    if v is None:
        return None
    if isinstance(v, bool):
        return 1.0 if v else 0.0
    if isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v)):
        return float(v)
    if isinstance(v, str) and v.strip():
        try:
            return float(v)
        except Exception:
            return None
    return None


def _corr(a, b) -> float | None:
    import numpy as np  # type: ignore

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    if int(mask.sum()) < 3:
        return None
    return float(np.corrcoef(a[mask], b[mask])[0, 1])


def _ols_residuals(y, x):
    """Return residuals of y ~ 1 + x."""
    import numpy as np  # type: ignore

    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    mask = np.isfinite(y) & np.isfinite(x)
    if int(mask.sum()) < 3:
        return y - np.nanmean(y)
    xm = float(np.mean(x[mask]))
    ym = float(np.mean(y[mask]))
    x0 = x[mask] - xm
    denom = float(np.sum(x0 * x0))
    if denom == 0:
        beta = 0.0
    else:
        beta = float(np.sum(x0 * (y[mask] - ym)) / denom)
    alpha = ym - beta * xm
    yhat = alpha + beta * x
    return y - yhat


def _ols_residuals_by_group(df, *, y_col: str, x_col: str, group_col: str) -> list[float]:
    """Return residuals for y ~ 1+x, fit separately within each group."""
    import numpy as np  # type: ignore

    out = np.full(shape=(df.shape[0],), fill_value=np.nan, dtype=float)
    for gval, g in df[[group_col, y_col, x_col]].groupby(group_col):
        idx = g.index.to_numpy()
        y = g[y_col].to_numpy(dtype=float)
        x = g[x_col].to_numpy(dtype=float)
        out[idx] = _ols_residuals(y, x)
    return out.tolist()


@dataclass(frozen=True)
class QInfo:
    qid: str
    qtype: str
    positive_means_stricter: bool
    short: str


def _load_questions_meta(*, questions_xlsx: Path, processed_sheet: str) -> dict[str, QInfo]:
    import pandas as pd  # type: ignore

    if not questions_xlsx.is_file():
        raise SystemExit(f"Questions workbook not found: {questions_xlsx}")
    df = pd.read_excel(questions_xlsx, sheet_name=processed_sheet)
    if "Include" not in df.columns or "ID" not in df.columns:
        raise SystemExit(f"Questions sheet missing required columns (Include, ID). Have: {list(df.columns)}")
    df = df[df["Include"].astype(str).str.strip().str.lower() == "yes"].copy()
    df["ID"] = df["ID"].astype(str)

    out: dict[str, QInfo] = {}
    for _, r in df.iterrows():
        qid = str(r.get("ID") or "").strip()
        if not qid:
            continue
        qt = str(r.get("Question Type") or "").strip() or "Unknown"
        pms_raw = r.get("Positive Means Stricter")
        if isinstance(pms_raw, bool):
            pms = pms_raw
        elif isinstance(pms_raw, (int, float)) and not (isinstance(pms_raw, float) and math.isnan(pms_raw)):
            pms = bool(int(pms_raw))
        elif isinstance(pms_raw, str) and pms_raw.strip():
            pms = pms_raw.strip().lower() in {"true", "t", "yes", "y", "1"}
        else:
            # Fail loudly: orientation is required for comparability with baseline PCA.
            raise SystemExit(f"Questions.xlsx missing Positive Means Stricter for ID={qid}")
        out[qid] = QInfo(
            qid=qid,
            qtype=qt,
            positive_means_stricter=pms,
            short=str(r.get("Short Question") or "").strip(),
        )
    return out


def _parse_custom_id_to_slug_year(custom_id: str) -> tuple[str, int] | None:
    parts = custom_id.split("__")
    if len(parts) != 2:
        return None
    slug = parts[0].strip()
    if not slug:
        return None
    try:
        year = int(parts[1])
    except Exception:
        return None
    return slug, year


def _metric_sort_key(metric: str) -> tuple[int, str]:
    m = re.match(r"^Principal_Component_(\d+)$", metric)
    if m:
        return (int(m.group(1)), metric)
    if metric == "Overall_Index":
        return (10_000, metric)
    return (99_999, metric)


def main() -> None:
    ap = argparse.ArgumentParser(description="Test missingness-handling strategies on questionnaire PCA.")
    ap.add_argument("--pca-xlsx", required=True, help="Baseline PCA workbook (scores/imputed/loadings).")
    ap.add_argument("--questionnaire-results-dir", required=True, help="Questionnaire results dir containing normalized.jsonl.")
    ap.add_argument("--questions-xlsx", required=True, help="Questions workbook (for orientation + types).")
    ap.add_argument("--questions-processed-sheet", default="Processed Info")
    ap.add_argument("--n-components", type=int, default=5)
    ap.add_argument("--out-md", default="", help="Optional: write markdown summary to this path.")
    ap.add_argument("--out-xlsx", default="", help="Optional: write an xlsx with per-strategy scores to this path.")
    args = ap.parse_args()

    import numpy as np  # type: ignore
    import pandas as pd  # type: ignore
    from sklearn.decomposition import PCA  # type: ignore
    from sklearn.preprocessing import StandardScaler  # type: ignore

    pca_xlsx = Path(expanduser(str(args.pca_xlsx))).resolve()
    results_dir = Path(expanduser(str(args.questionnaire_results_dir))).resolve()
    questions_xlsx = Path(expanduser(str(args.questions_xlsx))).resolve()

    if not pca_xlsx.is_file():
        raise SystemExit(f"--pca-xlsx not found: {pca_xlsx}")
    if not results_dir.is_dir():
        raise SystemExit(f"--questionnaire-results-dir not found: {results_dir}")
    norm_path = results_dir / "normalized.jsonl"
    if not norm_path.is_file():
        raise SystemExit(f"Expected normalized.jsonl in results dir: {norm_path}")

    # Baseline workbook
    scores = pd.read_excel(pca_xlsx, sheet_name="scores")
    imputed = pd.read_excel(pca_xlsx, sheet_name="imputed")
    loadings = pd.read_excel(pca_xlsx, sheet_name="loadings", index_col=0)
    metrics = sorted([c for c in scores.columns if c.startswith("Principal_Component_")], key=_metric_sort_key)
    if "Overall_Index" in scores.columns:
        metrics.append("Overall_Index")

    if "slug" not in scores.columns or "page_year" not in scores.columns:
        raise SystemExit("Baseline PCA scores must contain (slug, page_year).")

    # Feature columns (same set as baseline PCA used).
    feature_cols = [c for c in imputed.columns if c not in {"slug", "page_year"}]
    if not feature_cols:
        raise SystemExit("No feature columns found in baseline imputed sheet.")

    qmeta = _load_questions_meta(questions_xlsx=questions_xlsx, processed_sheet=str(args.questions_processed_sheet))
    missing_qmeta = [c for c in feature_cols if c not in qmeta]
    if missing_qmeta:
        raise SystemExit(f"Feature columns missing from Questions.xlsx meta: {missing_qmeta[:10]}{'...' if len(missing_qmeta)>10 else ''}")

    # Load normalized answers (slug-year group outputs).
    norm_rows = _read_jsonl(norm_path)
    ans_by_key: dict[tuple[str, int], dict[str, dict[str, Any]]] = {}
    for row in norm_rows:
        cid = row.get("custom_id")
        if not isinstance(cid, str) or not cid.strip():
            continue
        ky = _parse_custom_id_to_slug_year(cid)
        if ky is None:
            continue
        answers_by_id = ((row.get("normalized") or {}).get("answers_by_id") or {})
        if isinstance(answers_by_id, dict):
            ans_by_key[ky] = answers_by_id

    # Restrict to the rows used by baseline PCA.
    base_keys = [(str(r.get("slug")), int(r.get("page_year"))) for _, r in scores.iterrows() if pd.notna(r.get("slug")) and pd.notna(r.get("page_year"))]
    base_keys = list(dict.fromkeys(base_keys))  # preserve order, unique

    # Build value and obs matrices from normalized answers.
    V = pd.DataFrame(index=pd.MultiIndex.from_tuples(base_keys, names=["slug", "page_year"]), columns=feature_cols, dtype=float)
    O = pd.DataFrame(index=V.index, columns=feature_cols, dtype=float)

    parse_fail_counts: dict[str, int] = {qid: 0 for qid in feature_cols}
    for slug, year in base_keys:
        answers = ans_by_key.get((slug, year), {})
        for qid in feature_cols:
            a = answers.get(qid)
            if not isinstance(a, dict):
                V.loc[(slug, year), qid] = np.nan
                O.loc[(slug, year), qid] = 0.0
                continue
            raw = a.get("answer")
            obs = 0.0 if raw is None else 1.0
            O.loc[(slug, year), qid] = obs

            qi = qmeta[qid]
            val: float | None
            if raw is None:
                val = None
            elif qi.qtype == "Binary":
                if raw is True:
                    val = 1.0
                elif raw is False:
                    val = 0.0
                else:
                    # schema says bool, but fail gracefully for experiment and count it.
                    val = _safe_float(raw)
                    if val is None:
                        parse_fail_counts[qid] += 1
            else:
                val = _safe_float(raw)
                if val is None:
                    parse_fail_counts[qid] += 1

            if val is None:
                V.loc[(slug, year), qid] = np.nan
            else:
                # Orientation: make higher = stricter.
                if not qi.positive_means_stricter:
                    if qi.qtype == "Binary":
                        val = 1.0 - val
                    else:
                        val = -1.0 * val
                V.loc[(slug, year), qid] = float(val)

    # Coverage from O
    answered = O.sum(axis=1)
    coverage = answered / float(len(feature_cols))
    cov_df = pd.DataFrame({"answered": answered, "coverage": coverage}, index=V.index).reset_index()

    # Baseline scores merged with coverage.
    base_scores = scores.copy()
    base_scores["slug"] = base_scores["slug"].astype(str)
    base_scores["page_year"] = base_scores["page_year"].astype(int)
    base_scores = base_scores.merge(cov_df, on=["slug", "page_year"], how="left")

    # Diagnostics: correlations vs coverage
    def corr_table(df, label: str) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for m in metrics:
            c = df[m].corr(df["coverage"]) if m in df.columns else None
            rows.append({"strategy": label, "metric": m, "corr_with_coverage": c})
        return pd.DataFrame(rows)

    corr_baseline = corr_table(base_scores, "baseline_current_pca")

    # Strategy 2: residualize baseline scores vs coverage.
    resid_scores = base_scores[["slug", "page_year", "coverage", "answered"]].copy()
    for m in metrics:
        resid_scores[m] = _ols_residuals(base_scores[m].to_numpy(dtype=float), base_scores["coverage"].to_numpy(dtype=float))
    corr_resid = corr_table(resid_scores, "baseline_residualized_vs_coverage")

    # Strategy 2b: residualize only PC1 (global), leave others unchanged.
    resid_pc1_only = base_scores.copy()
    if "Principal_Component_1" in resid_pc1_only.columns:
        resid_pc1_only["Principal_Component_1"] = _ols_residuals(
            resid_pc1_only["Principal_Component_1"].to_numpy(dtype=float),
            resid_pc1_only["coverage"].to_numpy(dtype=float),
        )
    corr_resid_pc1_only = corr_table(resid_pc1_only, "baseline_residualized_PC1_only")

    # Strategy 2c: residualize PC1 within each slug.
    resid_pc1_within = base_scores.copy()
    if "Principal_Component_1" in resid_pc1_within.columns:
        resid_pc1_within["Principal_Component_1"] = _ols_residuals_by_group(
            resid_pc1_within, y_col="Principal_Component_1", x_col="coverage", group_col="slug"
        )
    corr_resid_pc1_within = corr_table(resid_pc1_within, "baseline_residualized_PC1_within_slug")

    # Strategy 3: obs-augmented PCA.
    # Mean-impute values (so missingness doesn't change the value column; the obs column carries missingness).
    V_mean = V.copy()
    for qid in feature_cols:
        col = V_mean[qid].astype(float)
        mu = float(np.nanmean(col.to_numpy(dtype=float))) if np.isfinite(col.to_numpy(dtype=float)).any() else 0.0
        V_mean[qid] = col.fillna(mu)
    X_obsaug = pd.concat([V_mean.add_suffix("__val"), O.add_suffix("__obs")], axis=1)
    X_obsaug_arr = X_obsaug.to_numpy(dtype=float)
    scaler = StandardScaler()
    Xn = scaler.fit_transform(X_obsaug_arr)
    n_comp = int(args.n_components)
    n_comp = max(1, min(n_comp, Xn.shape[0], Xn.shape[1]))
    pca = PCA(n_components=n_comp)
    pcs = pca.fit_transform(Xn)
    obsaug_scores = pd.DataFrame(pcs, columns=[f"PC{i}" for i in range(1, n_comp + 1)])
    obsaug_scores[["slug", "page_year"]] = cov_df[["slug", "page_year"]]
    obsaug_scores = obsaug_scores.merge(cov_df, on=["slug", "page_year"], how="left")

    # Identify coverage component: max |corr(PCk, coverage)|.
    cov_corrs = {c: obsaug_scores[c].corr(obsaug_scores["coverage"]) for c in obsaug_scores.columns if c.startswith("PC")}
    cov_comp = max(cov_corrs, key=lambda k: abs(float(cov_corrs[k]) if cov_corrs[k] is not None else -1.0))

    corr_obsaug = pd.DataFrame(
        [{"strategy": "obs_augmented_pca_mean_impute", "metric": k, "corr_with_coverage": v} for k, v in sorted(cov_corrs.items())]
    )
    corr_obsaug["coverage_component"] = corr_obsaug["metric"].apply(lambda m: m == cov_comp)

    # Strategy 4: carry-forward values within slug before mean-impute + PCA.
    def carry_forward_matrix(*, V_in, mode: str, max_gap_years: int | None) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return (V_filled, carried_mask) where carried_mask==1 for carried-forward cells.

        mode:
          - "last_any": carry-forward from last non-missing (including previously carried)
          - "observed_only": carry-forward only from last *observed* value in the original V_in (no propagation)
        """
        V_out = V_in.copy()
        carried_mask = pd.DataFrame(0.0, index=V_out.index, columns=feature_cols, dtype=float)
        # Determine which cells are originally observed (non-missing).
        observed_mask = ~V_in.isna()

        for slug in sorted({s for s, _y in V_out.index.tolist()}):
            sub_years = sorted({y for s, y in V_out.index.tolist() if s == slug})
            last_val: dict[str, float] = {}
            last_year: dict[str, int] = {}
            for year in sub_years:
                for qid in feature_cols:
                    v = V_out.loc[(slug, year), qid]
                    is_obs = bool(observed_mask.loc[(slug, year), qid])
                    if is_obs:
                        # Observed values always update the carry state.
                        last_val[qid] = float(v)
                        last_year[qid] = int(year)
                        continue

                    if mode == "observed_only":
                        # We do NOT update last_val from carried values; state already reflects last observed.
                        pass
                    elif mode == "last_any":
                        # State may have been updated by carried values (because we write into V_out).
                        # But we only update last_val when we hit a non-missing value (handled above).
                        pass
                    else:
                        raise SystemExit(f"Unknown carry_forward mode: {mode}")

                    if qid not in last_val:
                        continue
                    if max_gap_years is not None:
                        gap = int(year) - int(last_year.get(qid, year))
                        if gap > int(max_gap_years):
                            continue
                    V_out.loc[(slug, year), qid] = last_val[qid]
                    carried_mask.loc[(slug, year), qid] = 1.0

        return V_out, carried_mask

    V_cf, carried = carry_forward_matrix(V_in=V, mode="last_any", max_gap_years=None)

    V_cf_mean = V_cf.copy()
    for qid in feature_cols:
        col = V_cf_mean[qid].astype(float)
        mu = float(np.nanmean(col.to_numpy(dtype=float))) if np.isfinite(col.to_numpy(dtype=float)).any() else 0.0
        V_cf_mean[qid] = col.fillna(mu)

    X_cf_arr = V_cf_mean.to_numpy(dtype=float)
    Xn_cf = StandardScaler().fit_transform(X_cf_arr)
    pca_cf = PCA(n_components=min(n_comp, Xn_cf.shape[0], Xn_cf.shape[1]))
    pcs_cf = pca_cf.fit_transform(Xn_cf)
    cf_scores = pd.DataFrame(pcs_cf, columns=[f"PC{i}" for i in range(1, pcs_cf.shape[1] + 1)])
    cf_scores[["slug", "page_year"]] = cov_df[["slug", "page_year"]]
    cf_scores = cf_scores.merge(cov_df, on=["slug", "page_year"], how="left")

    corr_cf = pd.DataFrame(
        [{"strategy": "carry_forward_pca_mean_impute", "metric": c, "corr_with_coverage": cf_scores[c].corr(cf_scores["coverage"])} for c in cf_scores.columns if c.startswith("PC")]
    )
    carried_rate = float(np.mean(carried.to_numpy(dtype=float)))

    # Strategy 4b: carry-forward observed-only (no propagation).
    V_cf_obs, carried_obs = carry_forward_matrix(V_in=V, mode="observed_only", max_gap_years=None)
    V_cf_obs_mean = V_cf_obs.copy()
    for qid in feature_cols:
        col = V_cf_obs_mean[qid].astype(float)
        mu = float(np.nanmean(col.to_numpy(dtype=float))) if np.isfinite(col.to_numpy(dtype=float)).any() else 0.0
        V_cf_obs_mean[qid] = col.fillna(mu)
    Xn_cf_obs = StandardScaler().fit_transform(V_cf_obs_mean.to_numpy(dtype=float))
    pca_cf_obs = PCA(n_components=min(n_comp, Xn_cf_obs.shape[0], Xn_cf_obs.shape[1]))
    pcs_cf_obs = pca_cf_obs.fit_transform(Xn_cf_obs)
    cf_obs_scores = pd.DataFrame(pcs_cf_obs, columns=[f"PC{i}" for i in range(1, pcs_cf_obs.shape[1] + 1)])
    cf_obs_scores[["slug", "page_year"]] = cov_df[["slug", "page_year"]]
    cf_obs_scores = cf_obs_scores.merge(cov_df, on=["slug", "page_year"], how="left")
    corr_cf_obs = pd.DataFrame(
        [{"strategy": "carry_forward_observed_only_mean_impute", "metric": c, "corr_with_coverage": cf_obs_scores[c].corr(cf_obs_scores["coverage"])} for c in cf_obs_scores.columns if c.startswith("PC")]
    )
    carried_obs_rate = float(np.mean(carried_obs.to_numpy(dtype=float)))

    # Strategy 4c: carry-forward observed-only with max gap (2 years).
    V_cf_gap2, carried_gap2 = carry_forward_matrix(V_in=V, mode="observed_only", max_gap_years=2)
    V_cf_gap2_mean = V_cf_gap2.copy()
    for qid in feature_cols:
        col = V_cf_gap2_mean[qid].astype(float)
        mu = float(np.nanmean(col.to_numpy(dtype=float))) if np.isfinite(col.to_numpy(dtype=float)).any() else 0.0
        V_cf_gap2_mean[qid] = col.fillna(mu)
    Xn_cf_gap2 = StandardScaler().fit_transform(V_cf_gap2_mean.to_numpy(dtype=float))
    pca_cf_gap2 = PCA(n_components=min(n_comp, Xn_cf_gap2.shape[0], Xn_cf_gap2.shape[1]))
    pcs_cf_gap2 = pca_cf_gap2.fit_transform(Xn_cf_gap2)
    cf_gap2_scores = pd.DataFrame(pcs_cf_gap2, columns=[f"PC{i}" for i in range(1, pcs_cf_gap2.shape[1] + 1)])
    cf_gap2_scores[["slug", "page_year"]] = cov_df[["slug", "page_year"]]
    cf_gap2_scores = cf_gap2_scores.merge(cov_df, on=["slug", "page_year"], how="left")
    corr_cf_gap2 = pd.DataFrame(
        [{"strategy": "carry_forward_max_gap_2_mean_impute", "metric": c, "corr_with_coverage": cf_gap2_scores[c].corr(cf_gap2_scores["coverage"])} for c in cf_gap2_scores.columns if c.startswith("PC")]
    )
    carried_gap2_rate = float(np.mean(carried_gap2.to_numpy(dtype=float)))

    def carry_stats(carried_mask: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return (carry_rate_by_qid, max_consecutive_run_by_qid).

        - carry_rate_by_qid: mean carried_mask across rows per qid
        - max_consecutive_run_by_qid: longest streak of carried years within any slug for that qid
        """

        rates = carried_mask.mean(axis=0).reset_index()
        rates.columns = ["qid", "carried_rate"]

        streak_rows: list[dict[str, Any]] = []
        for qid in feature_cols:
            best = 0
            for slug in sorted({s for s, _y in carried_mask.index.tolist()}):
                sub = carried_mask.loc[slug][qid]
                years = sub.index.tolist()
                # years are ints; ensure sorted
                years = sorted(years)
                cur = 0
                last_year = None
                for y in years:
                    v = float(sub.loc[y])
                    if v >= 0.5:
                        # consecutive years only
                        if last_year is not None and int(y) == int(last_year) + 1:
                            cur += 1
                        else:
                            cur = 1
                        best = max(best, cur)
                    else:
                        cur = 0
                    last_year = y
            streak_rows.append({"qid": qid, "max_consecutive_carried_years": int(best)})

        streak = pd.DataFrame(streak_rows)
        return rates.sort_values("carried_rate", ascending=False), streak.sort_values("max_consecutive_carried_years", ascending=False)

    carry_rate_by_qid, carry_streak_by_qid = carry_stats(carried)
    carry_gap2_rate_by_qid, carry_gap2_streak_by_qid = carry_stats(carried_gap2)

    # Volatility comparison: max abs YoY per slug.
    def max_abs_yoy(df, metric_cols: list[str]) -> pd.DataFrame:
        out_rows: list[dict[str, Any]] = []
        for slug, g in df.dropna(subset=["slug", "page_year"]).groupby("slug"):
            g = g.sort_values("page_year")
            for m in metric_cols:
                if m not in g.columns:
                    continue
                y = g[m].to_numpy(dtype=float)
                diffs = np.diff(y)
                v = float(np.max(np.abs(diffs))) if len(diffs) else 0.0
                out_rows.append({"slug": slug, "metric": m, "max_abs_yoy": v})
        return pd.DataFrame(out_rows)

    # Gap sweep for carry-forward: see tradeoff curve.
    sweep_gaps: list[int | None] = [None, 1, 2, 3, 5]
    sweep_rows: list[dict[str, Any]] = []
    for gap in sweep_gaps:
        V_gap, carried_gap = carry_forward_matrix(V_in=V, mode="observed_only", max_gap_years=gap)
        V_gap_mean = V_gap.copy()
        for qid in feature_cols:
            col = V_gap_mean[qid].astype(float)
            mu = float(np.nanmean(col.to_numpy(dtype=float))) if np.isfinite(col.to_numpy(dtype=float)).any() else 0.0
            V_gap_mean[qid] = col.fillna(mu)
        Xn_gap = StandardScaler().fit_transform(V_gap_mean.to_numpy(dtype=float))
        pca_gap = PCA(n_components=min(n_comp, Xn_gap.shape[0], Xn_gap.shape[1]))
        pcs_gap = pca_gap.fit_transform(Xn_gap)
        gap_scores = pd.DataFrame(pcs_gap, columns=[f"PC{i}" for i in range(1, pcs_gap.shape[1] + 1)])
        gap_scores[["slug", "page_year"]] = cov_df[["slug", "page_year"]]
        gap_scores = gap_scores.merge(cov_df, on=["slug", "page_year"], how="left")
        pc_cols = [c for c in gap_scores.columns if c.startswith("PC")]
        corr_map = {pc: gap_scores[pc].corr(gap_scores["coverage"]) for pc in pc_cols}
        cov_pc = max(corr_map, key=lambda k: abs(float(corr_map[k]) if corr_map[k] is not None else -1.0))
        # volatility of the coverage component
        vol_covpc = max_abs_yoy(gap_scores, [cov_pc]).rename(columns={"metric": "pc"}).assign(pc=cov_pc)
        sweep_rows.append(
            {
                "max_gap_years": None if gap is None else int(gap),
                "max_gap_years_label": "None" if gap is None else str(int(gap)),
                "carry_rate": float(np.mean(carried_gap.to_numpy(dtype=float))),
                "coverage_component": cov_pc,
                "corr_with_coverage": float(corr_map[cov_pc]) if corr_map.get(cov_pc) is not None else None,
                "max_abs_yoy_mean": float(vol_covpc["max_abs_yoy"].mean()) if not vol_covpc.empty else None,
                "max_abs_yoy_max": float(vol_covpc["max_abs_yoy"].max()) if not vol_covpc.empty else None,
                "variance_explained_covpc": float(pca_gap.explained_variance_ratio_[int(cov_pc.replace("PC", "")) - 1]) if cov_pc.startswith("PC") else None,
            }
        )
    sweep_df = pd.DataFrame(sweep_rows)
    if not sweep_df.empty:
        def _gap_sort(v: Any) -> int:
            if v is None or (isinstance(v, float) and math.isnan(v)):
                return -1
            try:
                return int(v)
            except Exception:
                return 99_999

        sweep_df["max_gap_sort"] = sweep_df["max_gap_years"].apply(_gap_sort)

    vol_base = max_abs_yoy(base_scores, metrics).assign(strategy="baseline_current_pca")
    vol_resid = max_abs_yoy(resid_scores, metrics).assign(strategy="baseline_residualized_vs_coverage")
    vol_resid_pc1_only = max_abs_yoy(resid_pc1_only, metrics).assign(strategy="baseline_residualized_PC1_only")
    vol_resid_pc1_within = max_abs_yoy(resid_pc1_within, metrics).assign(strategy="baseline_residualized_PC1_within_slug")
    vol_obsaug = max_abs_yoy(obsaug_scores, [c for c in obsaug_scores.columns if c.startswith("PC")]).assign(strategy="obs_augmented_pca_mean_impute")
    vol_cf = max_abs_yoy(cf_scores, [c for c in cf_scores.columns if c.startswith("PC")]).assign(strategy="carry_forward_pca_mean_impute")
    vol_cf_obs = max_abs_yoy(cf_obs_scores, [c for c in cf_obs_scores.columns if c.startswith("PC")]).assign(strategy="carry_forward_observed_only_mean_impute")
    vol_cf_gap2 = max_abs_yoy(cf_gap2_scores, [c for c in cf_gap2_scores.columns if c.startswith("PC")]).assign(strategy="carry_forward_max_gap_2_mean_impute")
    vol_all = pd.concat(
        [vol_base, vol_resid, vol_resid_pc1_only, vol_resid_pc1_within, vol_obsaug, vol_cf, vol_cf_obs, vol_cf_gap2],
        ignore_index=True,
    )

    # Build markdown summary.
    md_lines: list[str] = []
    md_lines.append("# Missingness strategies experiment (questionnaire PCA)")
    md_lines.append("")
    md_lines.append("This report is generated from local artifacts; no simulation.")
    md_lines.append("")
    md_lines.append(f"- Baseline PCA workbook: `{pca_xlsx}`")
    md_lines.append(f"- Questionnaire results: `{norm_path}`")
    md_lines.append(f"- Questions workbook: `{questions_xlsx}` (sheet={args.questions_processed_sheet})")
    md_lines.append("")
    md_lines.append("## Data sanity checks")
    md_lines.append(f"- Rows in baseline PCA scores: {int(scores.shape[0])}")
    md_lines.append(f"- Rows with normalized answers matched: {int(len(ans_by_key))}")
    md_lines.append(f"- Feature columns (baseline): {int(len(feature_cols))}")
    md_lines.append(f"- Parse failures while converting normalized answers: {sum(parse_fail_counts.values())} (counts by qid are available in code)")
    md_lines.append("")
    md_lines.append("## Strategy comparison: correlation with coverage")
    md_lines.append("Coverage = answered_questions / total_questions (on baseline feature set).")
    md_lines.append("")

    corr_all = pd.concat([corr_baseline, corr_resid, corr_resid_pc1_only, corr_resid_pc1_within], ignore_index=True)
    # Add obsaug and carry-forward correlations (they use PC1..)
    corr_all = pd.concat(
        [
            corr_all,
            corr_obsaug.rename(columns={"metric": "metric"}).assign(strategy="obs_augmented_pca_mean_impute"),
            corr_cf.rename(columns={"metric": "metric"}).assign(strategy="carry_forward_pca_mean_impute"),
        ],
        ignore_index=True,
    )

    def md_table(df: pd.DataFrame, cols: list[str]) -> list[str]:
        out = []
        out.append("| " + " | ".join(cols) + " |")
        out.append("|" + "|".join(["---"] * len(cols)) + "|")
        for _, r in df.iterrows():
            row = []
            for c in cols:
                v = r.get(c)
                if isinstance(v, float) and not math.isnan(v):
                    row.append(f"{v:+.3f}" if "corr" in c else f"{v:.3f}")
                else:
                    row.append("" if v is None else str(v))
            out.append("| " + " | ".join(row) + " |")
        return out

    # Baseline + residual correlations
    md_lines.extend(md_table(corr_baseline.sort_values("metric"), ["metric", "corr_with_coverage"]))
    md_lines.append("")
    md_lines.append("### Residualized baseline (score ~ coverage)")
    md_lines.extend(md_table(corr_resid.sort_values("metric"), ["metric", "corr_with_coverage"]))
    md_lines.append("")

    md_lines.append("### Obs-augmented PCA (mean-imputed values + explicit obs flags)")
    md_lines.append(f"- Identified coverage component (max |corr| with coverage): **{cov_comp}**")
    md_lines.extend(md_table(corr_obsaug.sort_values("metric"), ["metric", "corr_with_coverage", "coverage_component"]))
    md_lines.append("")

    md_lines.append("### Carry-forward PCA (within-slug carry-forward, then mean-impute)")
    md_lines.append(f"- Mean carry-forward rate across all (row,feature) cells: {carried_rate:.3f}")
    md_lines.extend(md_table(corr_cf.sort_values("metric"), ["metric", "corr_with_coverage"]))
    md_lines.append("")

    md_lines.append("### Carry-forward PCA variants")
    md_lines.append(f"- Observed-only carry-forward rate: {carried_obs_rate:.3f}")
    md_lines.extend(md_table(corr_cf_obs.sort_values("metric"), ["metric", "corr_with_coverage"]))
    md_lines.append("")
    md_lines.append(f"- Observed-only with max gap=2 years carry-forward rate: {carried_gap2_rate:.3f}")
    md_lines.extend(md_table(corr_cf_gap2.sort_values("metric"), ["metric", "corr_with_coverage"]))
    md_lines.append("")

    md_lines.append("### Carry-forward: which questions are being inferred the most?")
    md_lines.append("These are computed on the baseline PCA feature set (25 questions), using carry-forward on oriented values.")
    md_lines.append("")
    md_lines.append("#### Carry-forward (no max gap): top questions by carried rate")
    md_lines.extend(md_table(carry_rate_by_qid.head(12), ["qid", "carried_rate"]))
    md_lines.append("")
    md_lines.append("#### Carry-forward (no max gap): longest consecutive carried streak (years)")
    md_lines.extend(md_table(carry_streak_by_qid.head(12), ["qid", "max_consecutive_carried_years"]))
    md_lines.append("")
    md_lines.append("#### Carry-forward (max gap=2): top questions by carried rate")
    md_lines.extend(md_table(carry_gap2_rate_by_qid.head(12), ["qid", "carried_rate"]))
    md_lines.append("")
    md_lines.append("#### Carry-forward (max gap=2): longest consecutive carried streak (years)")
    md_lines.extend(md_table(carry_gap2_streak_by_qid.head(12), ["qid", "max_consecutive_carried_years"]))
    md_lines.append("")

    md_lines.append("### Carry-forward gap sweep (coverage component only)")
    md_lines.append("For each max-gap setting we recompute PCA and identify the component most correlated with coverage.")
    md_lines.append("")
    md_lines.extend(
        md_table(
            sweep_df.sort_values("max_gap_sort"),
            ["max_gap_years_label", "carry_rate", "coverage_component", "corr_with_coverage", "variance_explained_covpc", "max_abs_yoy_mean", "max_abs_yoy_max"],
        )
    )
    md_lines.append("")

    md_lines.append("## Volatility proxy: max absolute year-to-year change")
    md_lines.append("This is not directly comparable across different PCA bases, but highlights how much each score swings over time.")
    md_lines.append("")
    # Summarize volatility by strategy+metric mean across slugs.
    vol_summary = (
        vol_all.groupby(["strategy", "metric"], as_index=False)
        .agg(max_abs_yoy_mean=("max_abs_yoy", "mean"), max_abs_yoy_max=("max_abs_yoy", "max"))
        .sort_values(["strategy", "metric"])
    )
    md_lines.extend(md_table(vol_summary, ["strategy", "metric", "max_abs_yoy_mean", "max_abs_yoy_max"]))
    md_lines.append("")

    md_lines.append("## Interpreting tradeoffs (what these tests can and cannot tell you)")
    md_lines.append("- `baseline_residualized_vs_coverage` removes linear coverage association without changing PCA features/loadings.")
    md_lines.append("- `baseline_residualized_PC1_only` targets the coverage-dominated component with minimal disruption to other PCs.")
    md_lines.append("- `baseline_residualized_PC1_within_slug` avoids using cross-slug coverage differences to fit the adjustment.")
    md_lines.append("- `obs_augmented_pca_mean_impute` forces missingness to express itself via explicit `__obs` features; you can ignore the coverage component.")
    md_lines.append("- `carry_forward_pca_mean_impute` encodes ordinance persistence; it reduces missingness effects but may smooth away true changes.")
    md_lines.append("- Carry-forward variants trade stability vs assumption strength: observed-only avoids propagating imputed values; max-gap limits long-range carry.")
    md_lines.append("")

    # Write outputs
    out_md_s = str(args.out_md).strip()
    if out_md_s:
        out_md = Path(expanduser(out_md_s)).resolve()
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
        _eprint(f"out_md\t{out_md}")
    else:
        # Print a short console summary.
        _eprint(f"baseline_PC1_corr_with_coverage\t{float(corr_baseline[corr_baseline['metric']=='Principal_Component_1']['corr_with_coverage'].iloc[0]):+.3f}")
        _eprint(f"obsaug_coverage_component\t{cov_comp}\t corr={cov_corrs.get(cov_comp)}")
        _eprint(f"carry_forward_rate\t{carried_rate:.3f}")

    out_xlsx_s = str(args.out_xlsx).strip()
    if out_xlsx_s:
        out_xlsx = Path(expanduser(out_xlsx_s)).resolve()
        out_xlsx.parent.mkdir(parents=True, exist_ok=True)
        with pd.ExcelWriter(out_xlsx, engine="openpyxl") as xw:
            base_scores.to_excel(xw, sheet_name="baseline_scores", index=False)
            resid_scores.to_excel(xw, sheet_name="baseline_residualized", index=False)
            resid_pc1_only.to_excel(xw, sheet_name="baseline_resid_pc1_only", index=False)
            resid_pc1_within.to_excel(xw, sheet_name="baseline_resid_pc1_within", index=False)
            obsaug_scores.to_excel(xw, sheet_name="obsaug_scores", index=False)
            cf_scores.to_excel(xw, sheet_name="carry_forward_scores", index=False)
            cf_obs_scores.to_excel(xw, sheet_name="carry_forward_obs_only", index=False)
            cf_gap2_scores.to_excel(xw, sheet_name="carry_forward_gap2", index=False)
            carry_rate_by_qid.to_excel(xw, sheet_name="carry_rate_by_qid", index=False)
            carry_streak_by_qid.to_excel(xw, sheet_name="carry_streak_by_qid", index=False)
            carry_gap2_rate_by_qid.to_excel(xw, sheet_name="carry_gap2_rate_by_qid", index=False)
            carry_gap2_streak_by_qid.to_excel(xw, sheet_name="carry_gap2_streak_by_qid", index=False)
            sweep_df.to_excel(xw, sheet_name="carry_gap_sweep", index=False)
            corr_all.to_excel(xw, sheet_name="corr_with_coverage", index=False)
            vol_all.to_excel(xw, sheet_name="max_abs_yoy", index=False)
            vol_summary.to_excel(xw, sheet_name="max_abs_yoy_summary", index=False)
        _eprint(f"out_xlsx\t{out_xlsx}")


if __name__ == "__main__":
    main()
