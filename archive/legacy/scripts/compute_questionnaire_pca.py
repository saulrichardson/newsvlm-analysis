#!/usr/bin/env python3
"""
Compute PCA / principal components over the ordinance questionnaire outputs.

This script is intended to replace the PCA-prep + PCA step currently done in:
  /Users/saulrichardson/Dropbox/Inclusionary Zoning/Github/ai-zoning

Key behaviors mirrored from ai-zoning (with adaptations for this repo):
  - Missing answers are treated as NaN
  - Binary question columns are filled with 0.5 before imputation
  - Remaining missing values are KNN-imputed
  - PCA is run on standardized features (StandardScaler)

Important differences vs ai-zoning:
  - Our Q/A data is box-level; we support grouping (e.g., by slug or page_id)
  - Our question set includes categorical questions; by default we drop them for PCA
  - Our continuous questions have units; we normalize a few common units (e.g., acres->sqft)

Input:
  - A merged questionnaire workbook produced by scripts/export_ordinance_questionnaire_answers_to_excel.py
    (sheet: answers_wide)
  - Questions.xlsx (Processed Info + Input Info)

Output:
  - A single XLSX with:
      * pivot_numeric (wide numeric matrix before imputation)
      * imputed (after fill + KNN + orientation transforms)
      * loadings (PCA loadings + variance explained + scaling info)
      * scores (PC scores + overall_index)
      * unit_summary (per-question unit stats used in conversions)
      * params (run parameters)
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from os.path import expanduser
from pathlib import Path
from typing import Any


def _eprint(msg: str) -> None:
    print(msg, flush=True)


def _json_dumps(v: Any) -> str:
    return json.dumps(v, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _read_answers_xlsx(*, xlsx_path: Path, sheet: str) -> "pd.DataFrame":
    try:
        import pandas as pd  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise SystemExit("Missing dependency: pandas (pip install pandas)") from exc

    try:
        return pd.read_excel(xlsx_path, sheet_name=sheet)
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"Failed to read {xlsx_path} sheet={sheet!r}: {exc}") from exc


def _read_questions_xlsx(*, xlsx_path: Path, processed_sheet: str, input_info_sheet: str) -> "pd.DataFrame":
    try:
        import pandas as pd  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise SystemExit("Missing dependency: pandas (pip install pandas)") from exc

    proc = pd.read_excel(xlsx_path, sheet_name=processed_sheet)
    if "Include" not in proc.columns or "ID" not in proc.columns:
        raise SystemExit(f"Questions processed sheet missing required columns (Include, ID). Have: {list(proc.columns)}")

    proc = proc[proc["Include"].astype(str).str.strip().str.lower() == "yes"].copy()
    proc["ID"] = proc["ID"].astype(str)

    required = {"ID", "Question Type", "Positive Means Stricter"}
    missing = sorted(required - set(proc.columns))
    if missing:
        raise SystemExit(f"Questions processed sheet missing columns: {missing}")

    info = pd.read_excel(xlsx_path, sheet_name=input_info_sheet)
    if "ID" in info.columns:
        info = info.copy()
        info["ID"] = info["ID"].astype(str)
    else:
        info = info.iloc[0:0].copy()

    # Merge possible answers onto proc if available.
    if "Possible Answers" in info.columns:
        keep = info[["ID", "Possible Answers"]].dropna(subset=["ID"]).copy()
        keep["Possible Answers"] = keep["Possible Answers"].astype(str)
        proc = proc.merge(keep, how="left", on="ID")
    else:
        proc["Possible Answers"] = None

    # Ensure uniqueness by ID
    dupes = proc["ID"][proc["ID"].duplicated()].unique().tolist()
    if dupes:
        raise SystemExit(f"Questions.xlsx has duplicate IDs in Processed Info: {dupes}")

    return proc


@dataclass(frozen=True)
class _QuestionMeta:
    qid: str
    qtype: str
    positive_means_stricter: bool
    possible_answers: list[str] | None


def _parse_question_meta(proc_df: "pd.DataFrame") -> dict[str, _QuestionMeta]:
    out: dict[str, _QuestionMeta] = {}
    for _, r in proc_df.iterrows():
        qid = str(r["ID"]).strip()
        if not qid:
            continue
        qtype = str(r["Question Type"]).strip()
        pos = bool(r["Positive Means Stricter"])
        poss_raw = r.get("Possible Answers")
        poss: list[str] | None = None
        if isinstance(poss_raw, str) and poss_raw.strip():
            poss = [p.strip() for p in poss_raw.split(";") if p.strip()]
        out[qid] = _QuestionMeta(qid=qid, qtype=qtype, positive_means_stricter=pos, possible_answers=poss)
    return out


def _normalize_unit(s: Any) -> str | None:
    if not isinstance(s, str):
        return None
    u = s.strip().lower()
    if not u:
        return None
    return u


_YEAR_RE = re.compile(r"-(\d{4})-p-\d+\s*$")


def _parse_year_from_page_id(page_id: Any) -> int | None:
    if not isinstance(page_id, str):
        return None
    m = _YEAR_RE.search(page_id.strip())
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _normalize_numeric_value(*, qid: str, value: Any, unit: Any, issues: list[str]) -> float | None:
    """Normalize a numeric answer to a canonical unit when we can.

    This only applies lightweight conversions grounded in observed units in the data:
      - 28Min/28Max: acres -> square feet (43560 sqft per acre)
      - Z6 (lot coverage): percent -> fraction (0..1)

    If units are unrecognized for these questions, we drop the value to NaN and record an issue.
    """

    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, (int, float)):
        v = float(value)
    else:
        try:
            v = float(str(value).strip())
        except Exception:  # noqa: BLE001
            issues.append(f"{qid}:non_numeric_value")
            return None

    u = _normalize_unit(unit)

    if qid in {"28Min", "28Max"}:
        if u is None:
            return v
        if u in {"acre", "acres"}:
            return v * 43560.0
        if u in {"sq ft", "sqft", "square feet", "square foot"}:
            return v
        # Some messy unit strings still clearly indicate sqft
        if "square" in u and "feet" in u:
            return v
        issues.append(f"{qid}:unexpected_unit:{u}")
        return None

    if qid == "Z6":
        if u is None:
            return v
        if u in {"%", "percent", "percentage"}:
            # Heuristic grounded in observed values: most are in 10..60,
            # but a few look like fractions (e.g., 0.1666667).
            return (v / 100.0) if v > 1.0 else v
        if "fraction" in u:
            return v
        issues.append(f"{qid}:unexpected_unit:{u}")
        return None

    # Default: keep numeric as-is.
    return v


def _binary_to_float(v: Any) -> float | None:
    if v is None:
        return None
    if v is True:
        return 1.0
    if v is False:
        return 0.0
    if isinstance(v, (int, float)):
        x = float(v)
        if x in {0.0, 1.0}:
            return x
        return None
    if isinstance(v, str):
        s = v.strip().lower()
        if s in {"yes", "true", "y"}:
            return 1.0
        if s in {"no", "false", "n"}:
            return 0.0
    return None


def _aggregate_best_confidence(
    gdf: "pd.DataFrame", *, qid: str, ans_col: str, conf_col: str, unit_col: str
) -> tuple[Any, Any]:
    # If confidence column missing, fall back to first non-null answer.
    if conf_col not in gdf.columns:
        sub = gdf[gdf[ans_col].notna()] if ans_col in gdf.columns else gdf.iloc[0:0]
        if sub.empty:
            return None, None
        row = sub.iloc[0]
        return row.get(ans_col), row.get(unit_col) if unit_col in gdf.columns else None

    sub = gdf[gdf[ans_col].notna()] if ans_col in gdf.columns else gdf.iloc[0:0]
    if sub.empty:
        return None, None
    # max confidence; break ties by first row
    sub2 = sub.copy()
    sub2[conf_col] = sub2[conf_col].fillna(-1.0)
    idx = sub2[conf_col].astype(float).idxmax()
    row = gdf.loc[idx]
    return row.get(ans_col), row.get(unit_col) if unit_col in gdf.columns else None


def _aggregate_majority_vote(gdf: "pd.DataFrame", *, ans_col: str) -> Any:
    if ans_col not in gdf.columns:
        return None
    vals = [v for v in gdf[ans_col].tolist() if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not vals:
        return None
    counts = Counter(vals)
    top = counts.most_common(2)
    if len(top) == 1:
        return top[0][0]
    if top[0][1] == top[1][1]:
        return None
    return top[0][0]


def _aggregate_numeric_median(gdf: "pd.DataFrame", *, qid: str, ans_col: str, unit_col: str) -> tuple[float | None, Any, list[str]]:
    issues: list[str] = []
    if ans_col not in gdf.columns:
        return None, None, issues
    vals: list[float] = []
    units: list[Any] = []
    for v, u in zip(gdf[ans_col].tolist(), gdf[unit_col].tolist() if unit_col in gdf.columns else [None] * len(gdf)):
        nv = _normalize_numeric_value(qid=qid, value=v, unit=u, issues=issues)
        if nv is None:
            continue
        vals.append(float(nv))
        units.append(u)
    if not vals:
        return None, None, issues
    vals_sorted = sorted(vals)
    mid = vals_sorted[len(vals_sorted) // 2]
    # Unit is informational only here; keep the most common unit string.
    unit_counts = Counter([_normalize_unit(u) for u in units if _normalize_unit(u) is not None])
    best_unit = unit_counts.most_common(1)[0][0] if unit_counts else None
    return float(mid), best_unit, issues


def _safe_quantile(series: "pd.Series", q: float) -> float | None:
    try:
        s = series.dropna()
        if s.empty:
            return None
        return float(s.quantile(q))
    except Exception:
        return None


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Compute PCA over questionnaire outputs (with ai-zoning-style imputation).")
    ap.add_argument(
        "--answers-xlsx",
        required=True,
        help="Path to questionnaire answers workbook (answers_wide sheet).",
    )
    ap.add_argument("--answers-sheet", default="answers_wide", help="Worksheet name in answers workbook.")
    ap.add_argument("--questions-xlsx", default="~/Downloads/Questions.xlsx", help="Path to Questions.xlsx")
    ap.add_argument("--questions-processed-sheet", default="Processed Info")
    ap.add_argument("--questions-input-info-sheet", default="Input Info")

    ap.add_argument(
        "--locations-parquet",
        default="",
        help=(
            "Optional: path to locations_headful.parquet (pub_slug -> city/state metadata). "
            "If provided, we join it onto the answers via answers.slug == locations.pub_slug and derive city_state."
        ),
    )
    ap.add_argument(
        "--locations-join-left",
        default="slug",
        help="Column name in answers_wide to join on (default: slug).",
    )
    ap.add_argument(
        "--locations-join-right",
        default="pub_slug",
        help="Column name in locations parquet to join on (default: pub_slug).",
    )

    ap.add_argument(
        "--group-key",
        default="slug",
        help="Column to group by before PCA (e.g., slug or page_id).",
    )
    ap.add_argument(
        "--group-keys",
        default="",
        help=(
            "Comma-separated list of columns to group by (overrides --group-key). "
            "Example: --group-keys city_state,page_year"
        ),
    )
    ap.add_argument(
        "--aggregation",
        choices=["best_confidence", "majority_vote"],
        default="best_confidence",
        help="How to aggregate box-level answers into one value per group.",
    )
    ap.add_argument(
        "--categorical-mode",
        choices=["drop", "onehot"],
        default="drop",
        help="How to handle categorical questions for PCA (default: drop).",
    )
    ap.add_argument(
        "--imputer-neighbors",
        type=int,
        default=50,
        help="KNNImputer n_neighbors (ai-zoning uses 50).",
    )
    ap.add_argument(
        "--binary-fill",
        type=float,
        default=0.5,
        help="Fill value for missing binary questions before KNN (ai-zoning uses 0.5).",
    )
    ap.add_argument(
        "--winsorize-quantile",
        type=float,
        default=0.99,
        help="If >0, replace values above this quantile with NaN for numeric cols prior to KNN (ai-zoning often used 0.99).",
    )
    ap.add_argument(
        "--binarize-28max-one-acre",
        action="store_true",
        help="After imputation, convert 28Max to binary (1 if > 43560 sqft else 0), matching ai-zoning behavior.",
    )
    ap.add_argument("--n-components", type=int, default=5, help="Number of principal components to compute.")
    ap.add_argument(
        "--out-xlsx",
        required=True,
        help="Path to write output workbook.",
    )
    return ap.parse_args()


def main() -> None:
    args = _parse_args()

    answers_xlsx = Path(expanduser(args.answers_xlsx)).resolve()
    if not answers_xlsx.is_file():
        raise SystemExit(f"--answers-xlsx not found: {answers_xlsx}")

    questions_xlsx = Path(expanduser(args.questions_xlsx)).resolve()
    if not questions_xlsx.is_file():
        raise SystemExit(f"--questions-xlsx not found: {questions_xlsx}")

    out_xlsx = Path(expanduser(args.out_xlsx)).resolve()
    out_xlsx.parent.mkdir(parents=True, exist_ok=True)

    try:
        import numpy as np  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise SystemExit("Missing dependency: numpy (pip install numpy)") from exc

    try:
        import pandas as pd  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise SystemExit("Missing dependency: pandas (pip install pandas)") from exc

    try:
        from sklearn.decomposition import PCA  # type: ignore
        from sklearn.impute import KNNImputer  # type: ignore
        from sklearn.preprocessing import StandardScaler  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise SystemExit("Missing dependency: scikit-learn (pip install scikit-learn)") from exc

    answers_df = _read_answers_xlsx(xlsx_path=answers_xlsx, sheet=str(args.answers_sheet))
    # Optional locations join
    locations_parquet_s = str(args.locations_parquet).strip()
    if locations_parquet_s:
        locations_parquet = Path(expanduser(locations_parquet_s)).resolve()
        if not locations_parquet.is_file():
            raise SystemExit(f"--locations-parquet not found: {locations_parquet}")

        left_key = str(args.locations_join_left)
        right_key = str(args.locations_join_right)
        if left_key not in answers_df.columns:
            raise SystemExit(
                f"--locations-join-left {left_key!r} not found in answers sheet. Have: {list(answers_df.columns)}"
            )

        try:
            loc_df = pd.read_parquet(locations_parquet)
        except Exception as exc:  # noqa: BLE001
            raise SystemExit(f"Failed to read locations parquet {locations_parquet}: {exc}") from exc

        if right_key not in loc_df.columns:
            raise SystemExit(
                f"--locations-join-right {right_key!r} not found in locations parquet. Have: {list(loc_df.columns)}"
            )

        # Keep only columns we actually use; this also reduces the chance of name collisions.
        keep_cols = [
            right_key,
            "country_id",
            "country_abbr",
            "state_id",
            "state_abbr",
            "state_name",
            "city_id",
            "city_name",
            "pub_id",
            "pub_title",
            "pub_min_year",
            "pub_max_year",
        ]
        keep_cols = [c for c in keep_cols if c in loc_df.columns]
        loc_df = loc_df[keep_cols].copy()

        # Ensure right_key is unique.
        dupe_mask = loc_df[right_key].duplicated(keep="first")
        if bool(dupe_mask.any()):
            dupes = loc_df.loc[dupe_mask, right_key].astype(str).unique().tolist()
            raise SystemExit(f"locations parquet has duplicate {right_key!r} values (expected unique). Examples: {dupes[:10]}")

        answers_df = answers_df.merge(loc_df, how="left", left_on=left_key, right_on=right_key)

        missing = answers_df[answers_df[right_key].isna()][left_key].dropna().astype(str).unique().tolist()
        if missing:
            raise SystemExit(
                "locations join produced missing metadata for some slugs.\n"
                f"  left_key={left_key!r} right_key={right_key!r}\n"
                f"  missing_count={len(missing)}\n"
                f"  missing_examples={missing[:20]}"
            )

        # Derived helper: city_state key.
        if "city_name" in answers_df.columns and "state_abbr" in answers_df.columns:
            def _city_state(row) -> str | None:
                city = row.get("city_name")
                st = row.get("state_abbr")
                if not isinstance(city, str) or not city.strip():
                    return None
                if not isinstance(st, str) or not st.strip():
                    return None
                return f"{city.strip()}, {st.strip().upper()}"

            answers_df["city_state"] = answers_df.apply(_city_state, axis=1)

    # Derived helper: year from page_id (if present).
    if "page_id" in answers_df.columns and "page_year" not in answers_df.columns:
        answers_df["page_year"] = answers_df["page_id"].apply(_parse_year_from_page_id)

    if args.group_key not in answers_df.columns:
        raise SystemExit(
            f"--group-key {args.group_key!r} not found in answers sheet after optional joins/derivations. "
            f"Have: {list(answers_df.columns)}"
        )

    proc_df = _read_questions_xlsx(
        xlsx_path=questions_xlsx,
        processed_sheet=str(args.questions_processed_sheet),
        input_info_sheet=str(args.questions_input_info_sheet),
    )
    qmeta = _parse_question_meta(proc_df)
    if not qmeta:
        raise SystemExit("No included questions found in Questions.xlsx (Include == Yes).")

    # Build per-group aggregated answers.
    group_keys_raw = str(args.group_keys).strip()
    if group_keys_raw:
        group_keys = [p.strip() for p in group_keys_raw.split(",") if p.strip()]
    else:
        group_keys = [str(args.group_key).strip()]

    if not group_keys:
        raise SystemExit("No group keys provided (empty --group-key/--group-keys).")

    missing_group_cols = [c for c in group_keys if c not in answers_df.columns]
    if missing_group_cols:
        raise SystemExit(
            f"Group key column(s) missing from answers sheet after optional joins/derivations: {missing_group_cols}. "
            f"Have: {list(answers_df.columns)}"
        )

    group = answers_df.groupby(group_keys, dropna=False)

    rows: list[dict[str, Any]] = []
    group_meta_rows: list[dict[str, Any]] = []
    unit_issue_counts: Counter[str] = Counter()

    for group_value, gdf in group:
        if isinstance(group_value, tuple):
            if len(group_value) != len(group_keys):
                raise SystemExit(f"Internal error: group_value tuple len mismatch: {group_value}")
            group_kv = dict(zip(group_keys, group_value, strict=True))
        else:
            group_kv = {group_keys[0]: group_value}

        row: dict[str, Any] = dict(group_kv)
        row_issues: list[str] = []

        meta: dict[str, Any] = {
            **group_kv,
            "n_boxes": int(len(gdf)),
        }
        if "page_id" in gdf.columns:
            meta["n_pages"] = int(gdf["page_id"].nunique(dropna=True))
        if "slug" in gdf.columns:
            meta["n_slugs"] = int(gdf["slug"].nunique(dropna=True))
            # Keep a small sample to make debugging merges easier.
            uniq = [s for s in gdf["slug"].dropna().astype(str).unique().tolist() if s.strip()]
            meta["slug_examples_json"] = _json_dumps(sorted(uniq)[:25]) if uniq else None
        if "page_year" in gdf.columns:
            yrs = [y for y in gdf["page_year"].dropna().astype(int).tolist()]
            meta["min_page_year"] = min(yrs) if yrs else None
            meta["max_page_year"] = max(yrs) if yrs else None

        # If we joined location metadata, surface it in group_meta. Only include fields that are constant
        # within the group (otherwise we'd need arrays, which is awkward for Excel).
        for col in [
            "state_abbr",
            "state_name",
            "city_name",
            "city_id",
            "pub_id",
            "pub_title",
            "pub_min_year",
            "pub_max_year",
        ]:
            if col in gdf.columns:
                uniq_vals = [v for v in gdf[col].dropna().unique().tolist()]
                meta[col] = uniq_vals[0] if len(uniq_vals) == 1 else None

        for qid, qm in qmeta.items():
            ans_col = f"{qid}__answer"
            conf_col = f"{qid}__confidence"
            unit_col = f"{qid}__unit"

            # If the answer column is missing, we assume this question was not exported.
            if ans_col not in answers_df.columns:
                continue

            if qm.qtype in {"Continuous", "Numerical"}:
                if args.aggregation == "majority_vote":
                    v, unit_best, issues = _aggregate_numeric_median(gdf, qid=qid, ans_col=ans_col, unit_col=unit_col)
                    row[qid] = v
                    row[f"{qid}__unit"] = unit_best
                    row_issues.extend(issues)
                else:
                    raw_v, raw_u = _aggregate_best_confidence(gdf, qid=qid, ans_col=ans_col, conf_col=conf_col, unit_col=unit_col)
                    issues: list[str] = []
                    v = _normalize_numeric_value(qid=qid, value=raw_v, unit=raw_u, issues=issues)
                    row[qid] = v
                    row[f"{qid}__unit"] = _normalize_unit(raw_u)
                    row_issues.extend(issues)
                continue

            if qm.qtype == "Binary":
                if args.aggregation == "majority_vote":
                    raw = _aggregate_majority_vote(gdf, ans_col=ans_col)
                else:
                    raw, _ = _aggregate_best_confidence(gdf, qid=qid, ans_col=ans_col, conf_col=conf_col, unit_col=unit_col)
                row[qid] = _binary_to_float(raw)
                continue

            if qm.qtype == "Categorical":
                if args.categorical_mode == "drop":
                    continue
                # onehot mode: we keep the raw answer for later encoding
                if args.aggregation == "majority_vote":
                    raw = _aggregate_majority_vote(gdf, ans_col=ans_col)
                else:
                    raw, _ = _aggregate_best_confidence(gdf, qid=qid, ans_col=ans_col, conf_col=conf_col, unit_col=unit_col)
                row[qid] = raw
                continue

            raise SystemExit(f"Unsupported Question Type for PCA: {qm.qtype!r} (ID={qid})")

        if row_issues:
            for it in row_issues:
                unit_issue_counts[it] += 1
            row["issues"] = ";".join(sorted(set(row_issues)))
            meta["issues"] = row["issues"]
        rows.append(row)
        group_meta_rows.append(meta)

    if not rows:
        raise SystemExit("No groups produced (empty answers input?)")

    agg_df = pd.DataFrame(rows)
    agg_df = agg_df.set_index(group_keys)

    group_meta_df = pd.DataFrame(group_meta_rows)
    if not group_meta_df.empty and all(k in group_meta_df.columns for k in group_keys):
        group_meta_df = group_meta_df.set_index(group_keys)

    # Build numeric feature matrix X (dropping non-feature columns like units/issues).
    feature_cols: list[str] = []
    categorical_qids: list[str] = []
    binary_qids: list[str] = []
    numeric_qids: list[str] = []

    for qid, qm in qmeta.items():
        if qid not in agg_df.columns and qm.qtype != "Categorical":
            # Not exported or no data.
            continue
        if qm.qtype == "Binary":
            if qid in agg_df.columns:
                feature_cols.append(qid)
                binary_qids.append(qid)
        elif qm.qtype in {"Continuous", "Numerical"}:
            if qid in agg_df.columns:
                feature_cols.append(qid)
                numeric_qids.append(qid)
        elif qm.qtype == "Categorical":
            if args.categorical_mode != "drop" and qid in agg_df.columns:
                categorical_qids.append(qid)
        else:
            continue

    X = agg_df[feature_cols].copy() if feature_cols else agg_df.iloc[:, 0:0].copy()
    if X.shape[1] == 0:
        raise SystemExit("No numeric features available for PCA (after filtering).")

    # One-hot encode categoricals if requested.
    onehot_cols: list[str] = []
    if args.categorical_mode == "onehot" and categorical_qids:
        for qid in categorical_qids:
            qm = qmeta[qid]
            opts = qm.possible_answers or []
            if not opts:
                raise SystemExit(
                    f"Categorical onehot requested but Questions.xlsx lacks Possible Answers for ID={qid}. "
                    "Add options in Input Info sheet or use --categorical-mode drop."
                )
            # Create 1/0 columns; unknown stays NaN.
            for opt in opts:
                col = f"{qid}__{opt}"
                onehot_cols.append(col)
                X[col] = agg_df[qid].apply(lambda v: (1.0 if v == opt else (0.0 if v is not None else np.nan)))

    # Drop rows that are completely empty across features (no signals).
    all_nan_mask = X.isna().all(axis=1)
    dropped_all_nan = int(all_nan_mask.sum())
    if dropped_all_nan:
        X = X[~all_nan_mask].copy()

    # Drop columns that are completely NaN (KNNImputer cannot learn them).
    all_nan_cols = [c for c in X.columns if X[c].isna().all()]
    if all_nan_cols:
        X = X.drop(columns=all_nan_cols)

    if X.shape[0] < 2:
        raise SystemExit(f"Need at least 2 groups/rows for PCA; have {X.shape[0]}")

    # Winsorize numeric columns if configured (drop extreme outliers to NaN).
    wins_q = float(args.winsorize_quantile)
    winsorized_cols: list[str] = []
    if wins_q and 0.0 < wins_q < 1.0:
        for qid in numeric_qids:
            if qid not in X.columns:
                continue
            thr = _safe_quantile(X[qid], wins_q)
            if thr is None:
                continue
            X.loc[X[qid] > thr, qid] = np.nan
            winsorized_cols.append(qid)

    # ai-zoning-style: fill missing binary columns with 0.5
    for qid in binary_qids:
        if qid in X.columns:
            X[qid] = X[qid].fillna(float(args.binary_fill))

    # For one-hot columns, we intentionally do NOT fill with 0.5; keep NaN for KNN.

    # KNN Impute remaining NaNs
    n_neighbors = int(args.imputer_neighbors)
    if n_neighbors >= X.shape[0]:
        n_neighbors = max(1, X.shape[0] - 1)
    imputer = KNNImputer(n_neighbors=n_neighbors, weights="uniform")
    imputed_arr = imputer.fit_transform(X)
    imputed_df = pd.DataFrame(imputed_arr, columns=X.columns, index=X.index)

    # Orientation transform: make "higher = stricter" for Binary and numeric features.
    # For binary: 1-x (keeps 0.5 fixed).
    # For numeric: multiply by -1 when Positive Means Stricter == False.
    for qid, qm in qmeta.items():
        if qid not in imputed_df.columns:
            continue
        if qm.positive_means_stricter:
            continue
        if qm.qtype == "Binary":
            imputed_df[qid] = 1.0 - imputed_df[qid]
        elif qm.qtype in {"Continuous", "Numerical"}:
            imputed_df[qid] = -1.0 * imputed_df[qid]
        # Categorical one-hot columns are not oriented.

    if args.binarize_28max_one_acre and "28Max" in imputed_df.columns:
        imputed_df["28Max"] = (imputed_df["28Max"] > 43560.0).astype(float)

    # Standardize + PCA
    scaler = StandardScaler()
    normalized = scaler.fit_transform(imputed_df)

    n_components = int(args.n_components)
    if n_components < 1:
        raise SystemExit("--n-components must be >= 1")
    if n_components > imputed_df.shape[1]:
        n_components = imputed_df.shape[1]

    pca = PCA(n_components=n_components)
    pcs = pca.fit_transform(normalized)

    # Build loadings table (match ai-zoning style: PC rows + variance explained + scaling info)
    pc_names = [f"PC{i}" for i in range(1, n_components + 1)]
    loadings_df = pd.DataFrame(pca.components_[:n_components, :], index=pc_names, columns=imputed_df.columns)
    loadings_df["Variance Explained"] = pca.explained_variance_ratio_[:n_components]

    scaling_info = pd.DataFrame(
        {"Mean": scaler.mean_, "Standard_Deviation": scaler.scale_},
        index=imputed_df.columns,
    ).T
    loadings_df = pd.concat([loadings_df, scaling_info], axis=0)

    # Scores / indices
    scores_df = pd.DataFrame(pcs, index=imputed_df.index, columns=[f"Principal_Component_{i}" for i in range(1, n_components + 1)])
    scores_df["Overall_Index"] = normalized.sum(axis=1)
    scores_df = scores_df.sort_values(by="Principal_Component_1" if "Principal_Component_1" in scores_df.columns else scores_df.columns[0])

    # Summarize units used for numeric questions (from aggregated dataframe).
    unit_rows: list[dict[str, Any]] = []
    for qid in numeric_qids:
        ucol = f"{qid}__unit"
        if ucol not in agg_df.columns:
            continue
        units = [_normalize_unit(u) for u in agg_df[ucol].tolist()]
        counts = Counter([u for u in units if u is not None])
        unit_rows.append(
            {
                "question_id": qid,
                "non_null_values": int(agg_df[qid].notna().sum()) if qid in agg_df.columns else 0,
                "distinct_units": len(counts),
                "top_units_json": _json_dumps(dict(counts.most_common(10))),
            }
        )
    unit_summary_df = pd.DataFrame(unit_rows).sort_values(by=["distinct_units", "non_null_values"], ascending=[False, False])

    params = {
        "answers_xlsx": str(answers_xlsx),
        "answers_sheet": str(args.answers_sheet),
        "questions_xlsx": str(questions_xlsx),
        "group_keys": group_keys,
        "aggregation": str(args.aggregation),
        "categorical_mode": str(args.categorical_mode),
        "imputer_neighbors": int(args.imputer_neighbors),
        "binary_fill": float(args.binary_fill),
        "winsorize_quantile": float(args.winsorize_quantile),
        "binarize_28max_one_acre": bool(args.binarize_28max_one_acre),
        "n_components": int(args.n_components),
        "dropped_all_nan_rows": dropped_all_nan,
        "dropped_all_nan_cols": all_nan_cols,
        "winsorized_numeric_cols": winsorized_cols,
        "unit_issue_counts_top": dict(unit_issue_counts.most_common(50)),
    }

    # NOTE: merge_cells=False avoids writing MultiIndex outputs with merged cells + blank repeated values,
    # which makes downstream reading/analysis much less error-prone.
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as xw:
        X.to_excel(xw, sheet_name="pivot_numeric", merge_cells=False)
        imputed_df.to_excel(xw, sheet_name="imputed", merge_cells=False)
        loadings_df.to_excel(xw, sheet_name="loadings")
        scores_df.to_excel(xw, sheet_name="scores", merge_cells=False)
        if not group_meta_df.empty:
            group_meta_df.to_excel(xw, sheet_name="group_meta", merge_cells=False)
        unit_summary_df.to_excel(xw, sheet_name="unit_summary", index=False)
        pd.DataFrame([{"k": k, "v": _json_dumps(v) if isinstance(v, (dict, list)) else v} for k, v in params.items()]).to_excel(
            xw, sheet_name="params", index=False
        )

    _eprint(f"out_xlsx\t{out_xlsx}")
    _eprint(f"group_keys\t{','.join(group_keys)}")
    _eprint(f"groups_total\t{agg_df.shape[0]}")
    _eprint(f"groups_used\t{X.shape[0]}")
    _eprint(f"features\t{X.shape[1]}")
    _eprint(f"binary_features\t{len([c for c in binary_qids if c in X.columns])}")
    _eprint(f"numeric_features\t{len([c for c in numeric_qids if c in X.columns])}")
    _eprint(f"onehot_features\t{len(onehot_cols)}")
    _eprint(f"dropped_all_nan_rows\t{dropped_all_nan}")
    _eprint(f"dropped_all_nan_cols\t{len(all_nan_cols)}")
    _eprint(f"winsorized_numeric_cols\t{len(winsorized_cols)}")
    _eprint(f"n_neighbors_used\t{n_neighbors}")
    _eprint(f"n_components_used\t{n_components}")


if __name__ == "__main__":
    main()
