from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from .io_utils import clean_optional_str, write_json, write_parquet, write_text

EXCLUDED_MOTIVES = {"other", "unclear"}
DISCRETION_INSTRUMENTS = {"conditional_use", "special_exception", "variance", "appeal", "permit_administration"}
STRUCTURAL_INSTRUMENTS = {"district_framework", "zoning_map", "definition", "purpose"}
VINTAGE_BIN_ORDER = ["1920_1949", "1950s", "1960s", "1970_plus", "unknown"]
SECTION_CLUSTER_EXPERIMENT_NAME = "section_semantic_hdbscan"
SECTION_LOCAL_CLUSTER_EXPERIMENT_NAME = "section_semantic_hdbscan_local_baseline"


def _safe_share(numerator: float, denominator: float) -> float:
    denom = float(denominator or 0.0)
    if denom <= 0.0:
        return 0.0
    return float(numerator) / denom


def _sample_docs(run_root: Path, sample: str) -> pd.DataFrame:
    sample_path = run_root / "corpus" / f"{sample}_sample.parquet"
    if sample_path.is_file():
        return pd.read_parquet(sample_path)
    docs = pd.read_parquet(run_root / "corpus" / "docs.parquet")
    if "sample" in docs.columns:
        docs = docs[docs["sample"].astype(str) == str(sample)].copy()
    return docs.reset_index(drop=True)


def _normalized_position(section_number: int, section_count: int) -> float:
    if int(section_count) <= 1:
        return 0.0
    return (int(section_number) - 1) / float(max(1, int(section_count) - 1))


def _row_normalize(matrix: np.ndarray) -> np.ndarray:
    arr = np.asarray(matrix, dtype=np.float64)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.where(norms > 0.0, norms, 1.0)
    return np.nan_to_num(arr / norms, nan=0.0, posinf=0.0, neginf=0.0)


def _cosine_similarity_matrix(matrix: np.ndarray) -> np.ndarray:
    normalized = _row_normalize(matrix)
    sims = np.einsum("ij,kj->ik", normalized, normalized, dtype=np.float64)
    return np.clip(np.nan_to_num(sims, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0)


def _cosine_similarity_to_centroid(matrix: np.ndarray, centroid: np.ndarray) -> np.ndarray:
    normalized = _row_normalize(matrix)
    centroid_norm = _row_normalize(np.asarray(centroid, dtype=np.float64).reshape(1, -1))
    sims = np.einsum("ij,kj->ik", normalized, centroid_norm, dtype=np.float64).reshape(-1)
    return np.clip(np.nan_to_num(sims, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0)


def _architecture_anchor(title: str) -> str:
    text = clean_optional_str(title).lower()
    text = re.sub(r"\([^)]*\)", " ", text)
    text = re.split(r"[:;\u2014-]", text, maxsplit=1)[0]
    text = re.sub(r"[^a-z0-9\s]+", " ", text)
    text = re.sub(r"\b(article|section|chapter|part)\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = text.split()
    if not tokens:
        return "unknown"
    return " ".join(tokens[:5])


def _vintage_bin(year: Any) -> str:
    if pd.isna(year):
        return "unknown"
    y = int(year)
    if y <= 1949:
        return "1920_1949"
    if y <= 1959:
        return "1950s"
    if y <= 1969:
        return "1960s"
    return "1970_plus"


def _add_vintage_fields(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "issue_year" in out.columns:
        years = pd.to_numeric(out["issue_year"], errors="coerce")
    elif "issue_date" in out.columns:
        years = pd.to_datetime(out["issue_date"], errors="coerce").dt.year
    else:
        years = pd.Series(np.nan, index=out.index, dtype=float)
    out["issue_year"] = years.astype("Int64")
    out["vintage_bin"] = out["issue_year"].map(_vintage_bin).astype(str)
    return out


def _normalized_entropy(values: Any) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr) & (arr > 0.0)]
    if arr.size <= 1:
        return 0.0
    probs = arr / arr.sum()
    entropy = float(-(probs * np.log2(probs)).sum())
    denom = float(np.log2(arr.size))
    if denom <= 0.0:
        return 0.0
    return entropy / denom


def _fit_ols_models(
    df: pd.DataFrame,
    *,
    target_col: str,
    complexity_col: str,
    categorical_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = ["ordinance_doc_id", target_col, complexity_col, *categorical_cols]
    if df.empty or any(col not in df.columns for col in required):
        return pd.DataFrame(), pd.DataFrame()
    model_df = df[required].copy()
    for col in categorical_cols:
        model_df[col] = model_df[col].astype(str).replace({"": "unknown"}).fillna("unknown")
    model_df[target_col] = pd.to_numeric(model_df[target_col], errors="coerce")
    model_df[complexity_col] = pd.to_numeric(model_df[complexity_col], errors="coerce")
    model_df = model_df.dropna(subset=[target_col, complexity_col]).reset_index(drop=True)
    if len(model_df) < 12:
        return pd.DataFrame(), pd.DataFrame()

    formulas = [
        ("raw", f"{target_col} ~ {complexity_col}"),
        ("vintage_level", f"{target_col} ~ {complexity_col} + C(vintage_bin) + C(jurisdiction_level)"),
        (
            "vintage_level_region",
            f"{target_col} ~ {complexity_col} + C(vintage_bin) + C(jurisdiction_level) + C(jurisdiction_region)",
        ),
    ]
    rows: list[dict[str, Any]] = []
    residual_frames: list[pd.DataFrame] = []
    for model_name, formula in formulas:
        try:
            result = smf.ols(formula, data=model_df).fit(cov_type="HC1")
        except Exception as exc:  # noqa: BLE001
            rows.append({"model_name": model_name, "formula": formula, "error": str(exc)})
            continue
        coef = float(result.params.get(complexity_col, np.nan))
        se = float(result.bse.get(complexity_col, np.nan))
        pvalue = float(result.pvalues.get(complexity_col, np.nan))
        conf = result.conf_int().loc[complexity_col].tolist() if complexity_col in result.params.index else [np.nan, np.nan]
        rows.append(
            {
                "model_name": model_name,
                "formula": formula,
                "nobs": int(result.nobs),
                "r_squared": float(getattr(result, "rsquared", np.nan)),
                "adj_r_squared": float(getattr(result, "rsquared_adj", np.nan)),
                "complexity_coef": coef,
                "complexity_se": se,
                "complexity_pvalue": pvalue,
                "complexity_ci_low": float(conf[0]),
                "complexity_ci_high": float(conf[1]),
            }
        )
        resid_df = model_df[["ordinance_doc_id", target_col, complexity_col, *categorical_cols]].copy()
        resid_df["model_name"] = model_name
        resid_df["fitted_value"] = result.fittedvalues
        resid_df["residual"] = result.resid
        residual_frames.append(resid_df)
    model_table = pd.DataFrame.from_records(rows)
    residual_table = pd.concat(residual_frames, ignore_index=True) if residual_frames else pd.DataFrame()
    return model_table, residual_table


def _fit_group_effect_model(
    df: pd.DataFrame,
    *,
    target_col: str,
    categorical_cols: list[str],
    extra_id_cols: list[str] | None = None,
) -> pd.DataFrame:
    extra_id_cols = extra_id_cols or []
    required = [target_col, *categorical_cols, *extra_id_cols]
    if df.empty or any(col not in df.columns for col in required):
        return pd.DataFrame()
    model_df = df[required].copy()
    for col in categorical_cols:
        model_df[col] = model_df[col].astype(str).replace({"": "unknown"}).fillna("unknown")
    model_df[target_col] = pd.to_numeric(model_df[target_col], errors="coerce")
    model_df = model_df.dropna(subset=[target_col]).reset_index(drop=True)
    if len(model_df) < 16:
        return pd.DataFrame()
    formula = f"{target_col} ~ " + " + ".join(f"C({col})" for col in categorical_cols)
    try:
        result = smf.ols(formula, data=model_df).fit(cov_type="HC1")
    except Exception:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    conf_df = result.conf_int()
    for term, coef in result.params.items():
        if term == "Intercept":
            continue
        match = re.match(r"C\((?P<group>[^)]+)\)\[T\.(?P<value>.+)\]", str(term))
        if not match:
            continue
        group_name = clean_optional_str(match.group("group"))
        level_value = clean_optional_str(match.group("value"))
        rows.append(
            {
                **{col: clean_optional_str(model_df[col].iloc[0]) for col in extra_id_cols if col in model_df.columns},
                "target_col": target_col,
                "formula": formula,
                "term": str(term),
                "group_name": group_name,
                "group_value": level_value,
                "coefficient": float(coef),
                "std_error": float(result.bse.get(term, np.nan)),
                "pvalue": float(result.pvalues.get(term, np.nan)),
                "ci_low": float(conf_df.loc[term, 0]) if term in conf_df.index else np.nan,
                "ci_high": float(conf_df.loc[term, 1]) if term in conf_df.index else np.nan,
                "nobs": int(result.nobs),
                "r_squared": float(getattr(result, "rsquared", np.nan)),
            }
        )
    return pd.DataFrame.from_records(rows)


def _vintage_adjusted_mean_table(df: pd.DataFrame, *, group_col: str, vintage_col: str, value_cols: list[str]) -> pd.DataFrame:
    if df.empty or group_col not in df.columns or vintage_col not in df.columns:
        return pd.DataFrame(columns=[group_col, "ordinance_count", "vintage_bin_count", "vintage_weight_coverage"])
    overall_weights = (
        df[["ordinance_doc_id", vintage_col]]
        .drop_duplicates()
        [vintage_col]
        .astype(str)
        .value_counts(normalize=True)
    )
    raw = (
        df.groupby(group_col, dropna=False)
        .agg(
            ordinance_count=("ordinance_doc_id", "nunique"),
            **{f"{col}__raw_mean": (col, "mean") for col in value_cols},
        )
        .reset_index()
    )
    rows: list[dict[str, Any]] = []
    for group_value, group in df.groupby(group_col, dropna=False):
        vintages = group[vintage_col].astype(str).unique().tolist()
        present_weights = overall_weights[overall_weights.index.isin(vintages)]
        coverage = float(present_weights.sum()) if not present_weights.empty else 0.0
        norm_weights = present_weights / coverage if coverage > 0.0 else present_weights
        by_vintage = group.groupby(vintage_col, dropna=False)[value_cols].mean(numeric_only=True)
        row: dict[str, Any] = {
            group_col: clean_optional_str(group_value),
            "ordinance_count": int(group["ordinance_doc_id"].nunique()),
            "vintage_bin_count": int(group[vintage_col].astype(str).nunique()),
            "vintage_weight_coverage": coverage,
        }
        for col in value_cols:
            weighted = 0.0
            used = False
            for vintage_name, weight in norm_weights.items():
                if vintage_name in by_vintage.index and pd.notna(by_vintage.loc[vintage_name, col]):
                    weighted += float(by_vintage.loc[vintage_name, col]) * float(weight)
                    used = True
            row[f"{col}__vintage_adjusted"] = float(weighted) if used else np.nan
        rows.append(row)
    adjusted = pd.DataFrame.from_records(rows)
    return raw.merge(adjusted, on=[group_col, "ordinance_count"], how="left")


def _vintage_adjusted_presence_table(
    df: pd.DataFrame,
    *,
    group_col: str,
    vintage_col: str,
    category_cols: list[str],
    value_col: str = "present",
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=[group_col, *category_cols, "presence_rate_raw", "presence_rate_vintage_adjusted"])
    overall_weights = (
        df[["ordinance_doc_id", vintage_col]]
        .drop_duplicates()
        [vintage_col]
        .astype(str)
        .value_counts(normalize=True)
    )
    overall_presence = (
        df.groupby(category_cols, dropna=False)[value_col]
        .mean()
        .reset_index(name="overall_presence_rate")
    )
    raw = (
        df.groupby([group_col, *category_cols], dropna=False)[value_col]
        .mean()
        .reset_index(name="presence_rate_raw")
    )
    rows: list[dict[str, Any]] = []
    for keys, group in df.groupby([group_col, *category_cols], dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        group_value = clean_optional_str(keys[0])
        cats = list(keys[1:])
        vintages = group[vintage_col].astype(str).unique().tolist()
        present_weights = overall_weights[overall_weights.index.isin(vintages)]
        coverage = float(present_weights.sum()) if not present_weights.empty else 0.0
        norm_weights = present_weights / coverage if coverage > 0.0 else present_weights
        by_vintage = group.groupby(vintage_col, dropna=False)[value_col].mean()
        weighted = 0.0
        used = False
        for vintage_name, weight in norm_weights.items():
            if vintage_name in by_vintage.index and pd.notna(by_vintage.loc[vintage_name]):
                weighted += float(by_vintage.loc[vintage_name]) * float(weight)
                used = True
        row = {
            group_col: group_value,
            "vintage_weight_coverage": coverage,
            "ordinance_count": int(group["ordinance_doc_id"].nunique()),
            "vintage_bin_count": int(group[vintage_col].astype(str).nunique()),
            "presence_rate_vintage_adjusted": float(weighted) if used else np.nan,
        }
        for col, val in zip(category_cols, cats, strict=False):
            row[col] = val
        rows.append(row)
    adjusted = pd.DataFrame.from_records(rows)
    out = raw.merge(adjusted, on=[group_col, *category_cols], how="left").merge(overall_presence, on=category_cols, how="left")
    out["presence_lift_vs_overall_raw"] = out["presence_rate_raw"] / out["overall_presence_rate"].clip(lower=1e-9)
    out["presence_lift_vs_overall_vintage_adjusted"] = out["presence_rate_vintage_adjusted"] / out["overall_presence_rate"].clip(lower=1e-9)
    return out


def _vintage_adjusted_category_mean_table(
    df: pd.DataFrame,
    *,
    group_col: str,
    vintage_col: str,
    category_cols: list[str],
    value_col: str,
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=[group_col, *category_cols, f"{value_col}__raw_mean", f"{value_col}__vintage_adjusted"])
    overall_weights = (
        df[["ordinance_doc_id", vintage_col]]
        .drop_duplicates()
        [vintage_col]
        .astype(str)
        .value_counts(normalize=True)
    )
    overall_mean = (
        df.groupby(category_cols, dropna=False)[value_col]
        .mean()
        .reset_index(name="overall_mean")
    )
    raw = (
        df.groupby([group_col, *category_cols], dropna=False)
        .agg(
            ordinance_count=("ordinance_doc_id", "nunique"),
            **{f"{value_col}__raw_mean": (value_col, "mean")},
        )
        .reset_index()
    )
    rows: list[dict[str, Any]] = []
    for keys, group in df.groupby([group_col, *category_cols], dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        group_value = clean_optional_str(keys[0])
        cats = list(keys[1:])
        vintages = group[vintage_col].astype(str).unique().tolist()
        present_weights = overall_weights[overall_weights.index.isin(vintages)]
        coverage = float(present_weights.sum()) if not present_weights.empty else 0.0
        norm_weights = present_weights / coverage if coverage > 0.0 else present_weights
        by_vintage = group.groupby(vintage_col, dropna=False)[value_col].mean()
        weighted = 0.0
        used = False
        for vintage_name, weight in norm_weights.items():
            if vintage_name in by_vintage.index and pd.notna(by_vintage.loc[vintage_name]):
                weighted += float(by_vintage.loc[vintage_name]) * float(weight)
                used = True
        row = {
            group_col: group_value,
            "ordinance_count": int(group["ordinance_doc_id"].nunique()),
            "vintage_bin_count": int(group[vintage_col].astype(str).nunique()),
            "vintage_weight_coverage": coverage,
            f"{value_col}__vintage_adjusted": float(weighted) if used else np.nan,
        }
        for col, val in zip(category_cols, cats, strict=False):
            row[col] = val
        rows.append(row)
    adjusted = pd.DataFrame.from_records(rows)
    out = raw.merge(adjusted, on=[group_col, *category_cols], how="left").merge(overall_mean, on=category_cols, how="left")
    out[f"{value_col}_lift_vs_overall_raw"] = out[f"{value_col}__raw_mean"] / out["overall_mean"].clip(lower=1e-9)
    out[f"{value_col}_lift_vs_overall_vintage_adjusted"] = out[f"{value_col}__vintage_adjusted"] / out["overall_mean"].clip(lower=1e-9)
    return out


def _aggregate_ordinance_vectors(section_meta: pd.DataFrame, section_matrix: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
    rows: list[dict[str, Any]] = []
    vectors: list[np.ndarray] = []
    for ordinance_doc_id, group in section_meta.groupby("ordinance_doc_id", dropna=False):
        row_idx = group["embedding_row_index"].to_numpy(dtype=int)
        vectors.append(section_matrix[row_idx].mean(axis=0))
        first = group.sort_values(["section_number", "section_id"]).iloc[0]
        rows.append(
            {
                "ordinance_doc_id": clean_optional_str(ordinance_doc_id),
                "issue_id": clean_optional_str(first.get("issue_id")),
                "census_id_pid6": clean_optional_str(first.get("census_id_pid6")),
                "jurisdiction_region": clean_optional_str(first.get("jurisdiction_region")),
                "jurisdiction_level": clean_optional_str(first.get("jurisdiction_level")),
                "title_line": clean_optional_str(first.get("title", "")),
                "section_count": int(group["section_id"].nunique()),
                "embedding_row_index": len(rows),
            }
        )
    vector_matrix = np.asarray(vectors, dtype=np.float32) if vectors else np.zeros((0, 0), dtype=np.float32)
    return pd.DataFrame.from_records(rows), vector_matrix


def _top_neighbor_rows(meta: pd.DataFrame, matrix: np.ndarray, *, k: int = 5) -> pd.DataFrame:
    if meta.empty or matrix.size == 0:
        return pd.DataFrame(
            columns=[
                "ordinance_doc_id",
                "neighbor_rank",
                "neighbor_ordinance_doc_id",
                "cosine_similarity",
                "jurisdiction_region",
                "neighbor_region",
                "jurisdiction_level",
                "neighbor_level",
            ]
        )
    sims = _cosine_similarity_matrix(matrix)
    rows: list[dict[str, Any]] = []
    for i, row in meta.reset_index(drop=True).iterrows():
        order = np.argsort(-sims[i])
        rank = 0
        for j in order:
            if int(i) == int(j):
                continue
            neighbor = meta.iloc[int(j)]
            rank += 1
            rows.append(
                {
                    "ordinance_doc_id": clean_optional_str(row["ordinance_doc_id"]),
                    "neighbor_rank": int(rank),
                    "neighbor_ordinance_doc_id": clean_optional_str(neighbor["ordinance_doc_id"]),
                    "cosine_similarity": float(sims[i, j]),
                    "jurisdiction_region": clean_optional_str(row.get("jurisdiction_region")),
                    "neighbor_region": clean_optional_str(neighbor.get("jurisdiction_region")),
                    "jurisdiction_level": clean_optional_str(row.get("jurisdiction_level")),
                    "neighbor_level": clean_optional_str(neighbor.get("jurisdiction_level")),
                }
            )
            if rank >= int(k):
                break
    return pd.DataFrame.from_records(rows)


def _prototype_rows(meta: pd.DataFrame, matrix: np.ndarray, *, group_col: str, top_n: int = 5) -> pd.DataFrame:
    if meta.empty or matrix.size == 0 or group_col not in meta.columns:
        return pd.DataFrame(columns=[group_col, "prototype_rank", "ordinance_doc_id", "cosine_similarity"])
    rows: list[dict[str, Any]] = []
    for group_value, group in meta.groupby(group_col, dropna=False):
        idx = group["embedding_row_index"].to_numpy(dtype=int)
        if len(idx) == 0:
            continue
        centroid = np.asarray(matrix[idx], dtype=np.float64).mean(axis=0, keepdims=True)
        sims = _cosine_similarity_to_centroid(np.asarray(matrix[idx], dtype=np.float64), centroid)
        order = np.argsort(-sims)
        for rank, pos in enumerate(order[: int(top_n)], start=1):
            sample_row = group.iloc[int(pos)]
            rows.append(
                {
                    group_col: clean_optional_str(group_value),
                    "prototype_rank": int(rank),
                    "ordinance_doc_id": clean_optional_str(sample_row["ordinance_doc_id"]),
                    "cosine_similarity": float(sims[int(pos)]),
                    "jurisdiction_region": clean_optional_str(sample_row.get("jurisdiction_region")),
                    "jurisdiction_level": clean_optional_str(sample_row.get("jurisdiction_level")),
                    "title_line": clean_optional_str(sample_row.get("title_line")),
                }
            )
    return pd.DataFrame.from_records(rows)


def _build_architecture_family_outputs(
    *,
    docs: pd.DataFrame,
    sections: pd.DataFrame,
    section_assign: pd.DataFrame,
    doc_arch_df: pd.DataFrame,
    architecture_dir: Path,
    sample: str,
) -> dict[str, Any]:
    summary: dict[str, Any] = {"available": False, "family_count": 0, "top_k": 0}
    if section_assign.empty:
        return summary
    non_noise = section_assign[section_assign["cluster_id"].astype(int) >= 0].copy()
    if non_noise.empty:
        return summary
    if "normalized_position" not in non_noise.columns:
        non_noise = non_noise.merge(
            sections[["section_id", "normalized_position"]],
            on="section_id",
            how="left",
        )
    if "word_count" not in non_noise.columns:
        non_noise = non_noise.merge(
            sections[["section_id", "word_count"]],
            on="section_id",
            how="left",
        )
    non_noise["word_count"] = pd.to_numeric(non_noise["word_count"], errors="coerce").fillna(0.0)
    non_noise["normalized_position"] = pd.to_numeric(non_noise["normalized_position"], errors="coerce").fillna(0.0)
    non_noise["early_flag"] = (non_noise["normalized_position"] <= (1.0 / 3.0)).astype(int)
    non_noise["late_flag"] = (non_noise["normalized_position"] >= (2.0 / 3.0)).astype(int)
    non_noise["early_word_count"] = non_noise["word_count"] * non_noise["early_flag"]
    non_noise["late_word_count"] = non_noise["word_count"] * non_noise["late_flag"]

    label_map = (
        non_noise[["cluster_id", "cluster_label"]]
        .drop_duplicates()
        .assign(cluster_id=lambda d: d["cluster_id"].astype(int))
        .set_index("cluster_id")["cluster_label"]
        .to_dict()
    )
    total_words = (
        non_noise.groupby("ordinance_doc_id", dropna=False)["word_count"]
        .sum()
        .rename("doc_word_total")
        .reset_index()
    )
    by_cluster = (
        non_noise.groupby(["ordinance_doc_id", "cluster_id"], dropna=False)
        .agg(
            word_count=("word_count", "sum"),
            early_word_count=("early_word_count", "sum"),
            late_word_count=("late_word_count", "sum"),
        )
        .reset_index()
        .merge(total_words, on="ordinance_doc_id", how="left")
    )
    by_cluster["cluster_id"] = by_cluster["cluster_id"].astype(int)
    by_cluster["share"] = by_cluster["word_count"] / by_cluster["doc_word_total"].clip(lower=1.0)
    by_cluster["early_share"] = by_cluster["early_word_count"] / by_cluster["doc_word_total"].clip(lower=1.0)
    by_cluster["late_share"] = by_cluster["late_word_count"] / by_cluster["doc_word_total"].clip(lower=1.0)

    feature_specs = [("share", "share"), ("early_share", "early"), ("late_share", "late")]
    feature_frames: list[pd.DataFrame] = []
    for value_col, prefix in feature_specs:
        pivot = (
            by_cluster.pivot_table(
                index="ordinance_doc_id",
                columns="cluster_id",
                values=value_col,
                aggfunc="sum",
                fill_value=0.0,
            )
            .sort_index(axis=1)
            .rename(columns=lambda cid: f"{prefix}__c{int(cid)}")
        )
        feature_frames.append(pivot)
    architecture_matrix_df = pd.concat(feature_frames, axis=1).fillna(0.0)
    if architecture_matrix_df.empty or len(architecture_matrix_df) < 12:
        return summary
    x = architecture_matrix_df.to_numpy(dtype=np.float64)
    means = x.mean(axis=0, keepdims=True)
    stds = x.std(axis=0, keepdims=True)
    stds = np.where(stds > 0.0, stds, 1.0)
    x_scaled = np.nan_to_num((x - means) / stds, nan=0.0, posinf=0.0, neginf=0.0)

    sweep_rows: list[dict[str, Any]] = []
    best_score = -np.inf
    best_k = 0
    best_labels: np.ndarray | None = None
    max_k = min(8, max(3, len(architecture_matrix_df) - 1))
    for k in range(3, max_k + 1):
        try:
            model = KMeans(n_clusters=k, random_state=42, n_init=20)
            labels = model.fit_predict(x_scaled)
        except Exception:
            continue
        unique = np.unique(labels)
        if len(unique) <= 1:
            score = np.nan
        else:
            try:
                score = float(silhouette_score(x_scaled, labels, metric="euclidean"))
            except Exception:
                score = np.nan
        sweep_rows.append(
            {
                "cluster_count": int(k),
                "doc_count": int(len(architecture_matrix_df)),
                "silhouette_score": float(score) if np.isfinite(score) else np.nan,
            }
        )
        if np.isfinite(score) and float(score) > best_score:
            best_score = float(score)
            best_k = int(k)
            best_labels = labels
    sweep_df = pd.DataFrame.from_records(sweep_rows)
    write_parquet(sweep_df, architecture_dir / f"architecture_family_sweep__{sample}.parquet")
    if best_labels is None or best_k <= 0:
        return summary

    assignments = architecture_matrix_df.reset_index().rename(columns={"index": "ordinance_doc_id"}).copy()
    assignments["architecture_family_id"] = best_labels.astype(int)
    assignments = assignments.merge(
        docs[
            [
                "ordinance_doc_id",
                "issue_id",
                "jurisdiction_region",
                "jurisdiction_level",
                "vintage_bin",
                "census_id_pid6",
            ]
        ],
        on="ordinance_doc_id",
        how="left",
    ).merge(
        doc_arch_df[
            [
                "ordinance_doc_id",
                "opening_signature",
                "closing_signature",
                "dominant_section_kind_by_words",
                "first_substantive_section_kind",
                "first_substantive_section_motive",
            ]
        ],
        on="ordinance_doc_id",
        how="left",
    )

    feature_cols = [c for c in architecture_matrix_df.columns if c != "ordinance_doc_id"]
    family_rows: list[dict[str, Any]] = []
    prototype_rows: list[dict[str, Any]] = []
    x_df = pd.DataFrame(x_scaled, index=architecture_matrix_df.index, columns=feature_cols)
    for family_id, family in assignments.groupby("architecture_family_id", dropna=False):
        doc_ids = family["ordinance_doc_id"].astype(str).tolist()
        family_matrix = x_df.loc[doc_ids]
        centroid = family_matrix.mean(axis=0).to_numpy(dtype=np.float64)
        sims = _cosine_similarity_to_centroid(family_matrix.to_numpy(dtype=np.float64), centroid)
        order = np.argsort(-sims)
        family_means = family_matrix.mean(axis=0)
        top_share_cols = family_means[[c for c in feature_cols if c.startswith("share__")]].sort_values(ascending=False)
        top_early_cols = family_means[[c for c in feature_cols if c.startswith("early__")]].sort_values(ascending=False)
        top_late_cols = family_means[[c for c in feature_cols if c.startswith("late__")]].sort_values(ascending=False)

        def _col_to_label(col_name: str) -> str:
            try:
                cluster_id = int(str(col_name).split("__c", 1)[1])
            except Exception:
                return clean_optional_str(col_name)
            return clean_optional_str(label_map.get(cluster_id, f"cluster_{cluster_id}"))

        top_share_labels = [_col_to_label(col) for col in top_share_cols.head(2).index.tolist()]
        opening_label = _col_to_label(top_early_cols.head(1).index.tolist()[0]) if not top_early_cols.empty else "unknown"
        closing_label = _col_to_label(top_late_cols.head(1).index.tolist()[0]) if not top_late_cols.empty else "unknown"
        label_parts = [opening_label, *top_share_labels[:2]]
        cluster_label = " | ".join([part for part in label_parts if clean_optional_str(part)])
        cluster_description = (
            f"Opens with {opening_label.lower()}, emphasizes "
            f"{', '.join(label.lower() for label in top_share_labels[:2]) or 'mixed section families'}, "
            f"and tends to close with {closing_label.lower()}."
        )
        dominant_opening = (
            family["first_substantive_section_kind"].astype(str).replace("", np.nan).dropna().value_counts().index.tolist()
        )
        dominant_motive = (
            family["first_substantive_section_motive"].astype(str).replace("", np.nan).dropna().value_counts().index.tolist()
        )
        family_rows.append(
            {
                "architecture_family_id": int(family_id),
                "ordinance_count": int(family["ordinance_doc_id"].nunique()),
                "cluster_label": cluster_label,
                "cluster_description": cluster_description,
                "top_opening_section_cluster_label": opening_label,
                "top_closing_section_cluster_label": closing_label,
                "top_share_cluster_label_1": top_share_labels[0] if len(top_share_labels) > 0 else "",
                "top_share_cluster_label_2": top_share_labels[1] if len(top_share_labels) > 1 else "",
                "dominant_first_substantive_section_kind": clean_optional_str(dominant_opening[0]) if dominant_opening else "",
                "dominant_first_substantive_motive": clean_optional_str(dominant_motive[0]) if dominant_motive else "",
                "mean_section_cluster_entropy": float(
                    family.merge(
                        sections.groupby("ordinance_doc_id", dropna=False)["section_id"].nunique().rename("section_count"),
                        on="ordinance_doc_id",
                        how="left",
                    )["section_count"].mean()
                )
                if not family.empty
                else 0.0,
                "silhouette_score": float(best_score),
            }
        )
        for rank, pos in enumerate(order[:3], start=1):
            sample_row = family.iloc[int(pos)]
            prototype_rows.append(
                {
                    "architecture_family_id": int(family_id),
                    "prototype_rank": int(rank),
                    "ordinance_doc_id": clean_optional_str(sample_row["ordinance_doc_id"]),
                    "jurisdiction_region": clean_optional_str(sample_row.get("jurisdiction_region")),
                    "jurisdiction_level": clean_optional_str(sample_row.get("jurisdiction_level")),
                    "first_substantive_section_kind": clean_optional_str(sample_row.get("first_substantive_section_kind")),
                    "cosine_similarity": float(sims[int(pos)]),
                }
            )

    family_df = pd.DataFrame.from_records(family_rows).sort_values("architecture_family_id").reset_index(drop=True)
    assignments = assignments.merge(
        family_df[["architecture_family_id", "cluster_label", "cluster_description"]],
        on="architecture_family_id",
        how="left",
    )
    prototype_df = pd.DataFrame.from_records(prototype_rows).sort_values(
        ["architecture_family_id", "prototype_rank"]
    ).reset_index(drop=True)
    presence_df = assignments[["ordinance_doc_id", "architecture_family_id", "cluster_label", "jurisdiction_region", "jurisdiction_level", "vintage_bin"]].copy()
    presence_df["present"] = 1.0
    family_presence_region = _vintage_adjusted_presence_table(
        presence_df,
        group_col="jurisdiction_region",
        vintage_col="vintage_bin",
        category_cols=["architecture_family_id", "cluster_label"],
        value_col="present",
    )
    family_presence_level = _vintage_adjusted_presence_table(
        presence_df,
        group_col="jurisdiction_level",
        vintage_col="vintage_bin",
        category_cols=["architecture_family_id", "cluster_label"],
        value_col="present",
    )
    top_region_families = (
        family_presence_region.sort_values(
            ["jurisdiction_region", "presence_lift_vs_overall_vintage_adjusted", "presence_rate_vintage_adjusted"],
            ascending=[True, False, False],
        )
        .groupby("jurisdiction_region", dropna=False)
        .head(4)
        .reset_index(drop=True)
    )
    write_parquet(assignments.sort_values(["architecture_family_id", "ordinance_doc_id"]).reset_index(drop=True), architecture_dir / f"architecture_family_assignments__{sample}.parquet")
    write_parquet(family_df, architecture_dir / f"architecture_family_summary__{sample}.parquet")
    write_parquet(prototype_df, architecture_dir / f"architecture_family_prototypes__{sample}.parquet")
    write_parquet(family_presence_region, architecture_dir / f"architecture_family_by_region_vintage_adjusted__{sample}.parquet")
    write_parquet(family_presence_level, architecture_dir / f"architecture_family_by_level_vintage_adjusted__{sample}.parquet")
    write_parquet(top_region_families, architecture_dir / f"top_architecture_family_contrasts__{sample}.parquet")
    summary.update(
        {
            "available": True,
            "family_count": int(family_df["architecture_family_id"].nunique()),
            "top_k": int(best_k),
            "sweep_rows": int(len(sweep_df)),
            "region_contrast_rows": int(len(top_region_families)),
        }
    )
    return summary


def _build_regulatory_style_family_outputs(
    *,
    style_df: pd.DataFrame,
    style_dir: Path,
    sample: str,
) -> dict[str, Any]:
    summary: dict[str, Any] = {"available": False, "family_count": 0, "top_k": 0}
    if style_df.empty or len(style_df) < 12:
        return summary

    feature_cols = [
        "direct_rule_share_of_operative",
        "discretion_share_of_operative",
        "exception_or_condition_share_of_operative",
        "threshold_share_of_operative",
        "distinct_instrument_type_count",
        "instrument_entropy",
        "top_instrument_share",
        "operative_clause_count_log",
    ]
    if any(col not in style_df.columns for col in feature_cols):
        return summary

    feature_names = {
        "direct_rule_share_of_operative": "direct-rule-heavy",
        "discretion_share_of_operative": "discretion-heavy",
        "exception_or_condition_share_of_operative": "exception-and-condition-heavy",
        "threshold_share_of_operative": "threshold-heavy",
        "distinct_instrument_type_count": "high instrument variety",
        "instrument_entropy": "high instrument diversity",
        "top_instrument_share": "top-instrument concentration",
        "operative_clause_count_log": "large operative scale",
    }
    x = style_df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    means = x.mean(axis=0, keepdims=True)
    stds = x.std(axis=0, keepdims=True)
    stds = np.where(stds > 0.0, stds, 1.0)
    x_scaled = np.nan_to_num((x - means) / stds, nan=0.0, posinf=0.0, neginf=0.0)

    sweep_rows: list[dict[str, Any]] = []
    best_score = -np.inf
    best_k = 0
    best_labels: np.ndarray | None = None
    max_k = min(6, max(3, len(style_df) - 1))
    for k in range(3, max_k + 1):
        try:
            model = KMeans(n_clusters=k, random_state=42, n_init=20)
            labels = model.fit_predict(x_scaled)
        except Exception:
            continue
        unique = np.unique(labels)
        if len(unique) <= 1:
            score = np.nan
        else:
            try:
                score = float(silhouette_score(x_scaled, labels, metric="euclidean"))
            except Exception:
                score = np.nan
        sweep_rows.append(
            {
                "cluster_count": int(k),
                "doc_count": int(len(style_df)),
                "silhouette_score": float(score) if np.isfinite(score) else np.nan,
            }
        )
        if np.isfinite(score) and float(score) > best_score:
            best_score = float(score)
            best_k = int(k)
            best_labels = labels
    sweep_df = pd.DataFrame.from_records(sweep_rows)
    write_parquet(sweep_df, style_dir / f"regulatory_style_family_sweep__{sample}.parquet")
    if best_labels is None or best_k <= 0:
        return summary

    assignments = style_df.copy()
    assignments["regulatory_style_family_id"] = best_labels.astype(int)
    x_df = pd.DataFrame(x_scaled, index=assignments.index, columns=feature_cols)

    family_rows: list[dict[str, Any]] = []
    prototype_rows: list[dict[str, Any]] = []
    for family_id, family in assignments.groupby("regulatory_style_family_id", dropna=False):
        idx = family.index.to_numpy(dtype=int)
        family_matrix = x_df.loc[idx]
        centroid = family_matrix.mean(axis=0)
        top_features = centroid.sort_values(ascending=False).head(3)
        label_parts = [feature_names.get(col, clean_optional_str(col)) for col in top_features.index.tolist()]
        cluster_label = " | ".join(label_parts)
        dominant_region = (
            family["jurisdiction_region"].astype(str).replace("", np.nan).dropna().value_counts().index.tolist()
        )
        dominant_level = (
            family["jurisdiction_level"].astype(str).replace("", np.nan).dropna().value_counts().index.tolist()
        )
        dominant_motive = (
            family["doc_primary_motive"].astype(str).replace("", np.nan).dropna().value_counts().index.tolist()
        )
        family_rows.append(
            {
                "regulatory_style_family_id": int(family_id),
                "ordinance_count": int(family["ordinance_doc_id"].nunique()),
                "cluster_label": cluster_label,
                "dominant_region": clean_optional_str(dominant_region[0]) if dominant_region else "",
                "dominant_level": clean_optional_str(dominant_level[0]) if dominant_level else "",
                "dominant_doc_primary_motive": clean_optional_str(dominant_motive[0]) if dominant_motive else "",
                "mean_direct_rule_share": float(family["direct_rule_share_of_operative"].mean()),
                "mean_discretion_share": float(family["discretion_share_of_operative"].mean()),
                "mean_exception_or_condition_share": float(family["exception_or_condition_share_of_operative"].mean()),
                "mean_threshold_share": float(family["threshold_share_of_operative"].mean()),
                "mean_distinct_instrument_type_count": float(family["distinct_instrument_type_count"].mean()),
                "mean_instrument_entropy": float(family["instrument_entropy"].mean()),
                "mean_top_instrument_share": float(family["top_instrument_share"].mean()),
                "silhouette_score": float(best_score),
            }
        )

        sims = _cosine_similarity_to_centroid(family_matrix.to_numpy(dtype=np.float64), centroid.to_numpy(dtype=np.float64))
        order = np.argsort(-sims)
        for rank, pos in enumerate(order[:3], start=1):
            sample_row = family.iloc[int(pos)]
            prototype_rows.append(
                {
                    "regulatory_style_family_id": int(family_id),
                    "prototype_rank": int(rank),
                    "ordinance_doc_id": clean_optional_str(sample_row["ordinance_doc_id"]),
                    "jurisdiction_region": clean_optional_str(sample_row.get("jurisdiction_region")),
                    "jurisdiction_level": clean_optional_str(sample_row.get("jurisdiction_level")),
                    "doc_primary_motive": clean_optional_str(sample_row.get("doc_primary_motive")),
                    "cosine_similarity": float(sims[int(pos)]),
                }
            )

    family_df = pd.DataFrame.from_records(family_rows).sort_values("regulatory_style_family_id").reset_index(drop=True)
    assignments = assignments.merge(
        family_df[["regulatory_style_family_id", "cluster_label"]],
        on="regulatory_style_family_id",
        how="left",
    )
    prototype_df = pd.DataFrame.from_records(prototype_rows).sort_values(
        ["regulatory_style_family_id", "prototype_rank"]
    ).reset_index(drop=True)

    presence_df = assignments[
        [
            "ordinance_doc_id",
            "regulatory_style_family_id",
            "cluster_label",
            "jurisdiction_region",
            "jurisdiction_level",
            "vintage_bin",
        ]
    ].copy()
    presence_df["present"] = 1.0
    family_presence_region = _vintage_adjusted_presence_table(
        presence_df,
        group_col="jurisdiction_region",
        vintage_col="vintage_bin",
        category_cols=["regulatory_style_family_id", "cluster_label"],
        value_col="present",
    )
    family_presence_level = _vintage_adjusted_presence_table(
        presence_df,
        group_col="jurisdiction_level",
        vintage_col="vintage_bin",
        category_cols=["regulatory_style_family_id", "cluster_label"],
        value_col="present",
    )
    top_region_families = (
        family_presence_region.sort_values(
            ["jurisdiction_region", "presence_lift_vs_overall_vintage_adjusted", "presence_rate_vintage_adjusted"],
            ascending=[True, False, False],
        )
        .groupby("jurisdiction_region", dropna=False)
        .head(4)
        .reset_index(drop=True)
    )

    write_parquet(assignments.sort_values(["regulatory_style_family_id", "ordinance_doc_id"]).reset_index(drop=True), style_dir / f"regulatory_style_family_assignments__{sample}.parquet")
    write_parquet(family_df, style_dir / f"regulatory_style_family_summary__{sample}.parquet")
    write_parquet(prototype_df, style_dir / f"regulatory_style_family_prototypes__{sample}.parquet")
    write_parquet(family_presence_region, style_dir / f"regulatory_style_family_by_region_vintage_adjusted__{sample}.parquet")
    write_parquet(family_presence_level, style_dir / f"regulatory_style_family_by_level_vintage_adjusted__{sample}.parquet")
    write_parquet(top_region_families, style_dir / f"top_regulatory_style_family_contrasts__{sample}.parquet")
    write_text(
        style_dir / f"regulatory_style_family_notes__{sample}.md",
        "\n".join(
            [
                "# Regulatory style families",
                "",
                f"- documents covered: {int(len(assignments))}",
                f"- chosen k: {int(best_k)}",
                f"- silhouette: {float(best_score):.3f}",
                "",
                *[
                    (
                        f"- family {int(row['regulatory_style_family_id'])}: `{clean_optional_str(row['cluster_label'])}` "
                        f"| n={int(row['ordinance_count'])} | region=`{clean_optional_str(row['dominant_region'])}` "
                        f"| level=`{clean_optional_str(row['dominant_level'])}` | motive=`{clean_optional_str(row['dominant_doc_primary_motive'])}`"
                    )
                    for row in family_df.to_dict(orient="records")
                ],
            ]
        ),
    )
    summary.update(
        {
            "available": True,
            "family_count": int(family_df["regulatory_style_family_id"].nunique()),
            "top_k": int(best_k),
            "sweep_rows": int(len(sweep_df)),
            "region_contrast_rows": int(len(top_region_families)),
            "silhouette_score": float(best_score),
        }
    )
    return summary


def build_llm_v2_research_outputs(*, run_root: Path, sample: str) -> dict[str, Any]:
    research_dir = run_root / "research"
    coherence_dir = research_dir / "coherence"
    architecture_dir = research_dir / "architecture"
    retrieval_dir = research_dir / "retrieval"
    governance_dir = research_dir / "governance_style"
    style_dir = research_dir / "style_families"
    coverage_dir = research_dir / "coverage"
    instruments_dir = research_dir / "instruments"
    contrastive_dir = research_dir / "contrastive"
    vintage_dir = research_dir / "vintage"
    complexity_dir = research_dir / "complexity"
    embedding_cmp_dir = research_dir / "embedding_comparison"
    clause_clusters_dir = research_dir / "clause_clusters"
    purpose_dir = research_dir / "purposes"
    for path in (coherence_dir, architecture_dir, retrieval_dir, governance_dir, style_dir, coverage_dir, instruments_dir, contrastive_dir, vintage_dir, complexity_dir, embedding_cmp_dir, clause_clusters_dir, purpose_dir):
        path.mkdir(parents=True, exist_ok=True)

    docs = _add_vintage_fields(_sample_docs(run_root, sample).copy())
    sections = pd.read_parquet(run_root / "structure" / f"sections__{sample}.parquet").copy()
    purposes = pd.read_parquet(run_root / "semantics" / f"document_purposes__{sample}.parquet").copy()
    section_motives = pd.read_parquet(run_root / "semantics" / f"section_motives__{sample}.parquet").copy()
    clauses_path = run_root / "semantics" / f"clauses__{sample}.parquet"
    if clauses_path.is_file():
        clauses = pd.read_parquet(clauses_path).copy()
    else:
        clauses = pd.DataFrame(
            columns=[
                "clause_id",
                "section_id",
                "ordinance_doc_id",
                "modality",
                "motive",
                "instrument_hint",
            ]
        )
    section_meta = pd.read_parquet(run_root / "embeddings" / "section" / "metadata.parquet").copy()
    section_matrix = np.load(run_root / "embeddings" / "section" / "embedding_matrix.npy")

    sections = sections.merge(
        section_motives[["section_id", "primary_motive", "secondary_motives", "confidence", "evidence"]],
        on="section_id",
        how="left",
    )
    sections["primary_motive"] = sections["primary_motive"].astype(str).replace("nan", "").fillna("")
    sections["word_count"] = sections["word_count"].fillna(0).astype(int)

    # Vintage composition summaries.
    vintage_counts = (
        docs.groupby("vintage_bin", dropna=False)["ordinance_doc_id"]
        .nunique()
        .reset_index(name="ordinance_count")
        .sort_values("vintage_bin")
        .reset_index(drop=True)
    )
    vintage_by_region = (
        docs.groupby(["jurisdiction_region", "vintage_bin"], dropna=False)["ordinance_doc_id"]
        .nunique()
        .reset_index(name="ordinance_count")
    )
    if not vintage_by_region.empty:
        totals = vintage_by_region.groupby("jurisdiction_region", dropna=False)["ordinance_count"].sum().rename("region_total").reset_index()
        vintage_by_region = vintage_by_region.merge(totals, on="jurisdiction_region", how="left")
        vintage_by_region["share_within_region"] = vintage_by_region["ordinance_count"] / vintage_by_region["region_total"].clip(lower=1)
        vintage_by_region = vintage_by_region.sort_values(["jurisdiction_region", "vintage_bin"]).reset_index(drop=True)
    vintage_by_level = (
        docs.groupby(["jurisdiction_level", "vintage_bin"], dropna=False)["ordinance_doc_id"]
        .nunique()
        .reset_index(name="ordinance_count")
    )
    if not vintage_by_level.empty:
        totals = vintage_by_level.groupby("jurisdiction_level", dropna=False)["ordinance_count"].sum().rename("level_total").reset_index()
        vintage_by_level = vintage_by_level.merge(totals, on="jurisdiction_level", how="left")
        vintage_by_level["share_within_level"] = vintage_by_level["ordinance_count"] / vintage_by_level["level_total"].clip(lower=1)
        vintage_by_level = vintage_by_level.sort_values(["jurisdiction_level", "vintage_bin"]).reset_index(drop=True)
    write_parquet(vintage_counts, vintage_dir / f"vintage_counts__{sample}.parquet")
    write_parquet(vintage_by_region, vintage_dir / f"vintage_distribution_by_region__{sample}.parquet")
    write_parquet(vintage_by_level, vintage_dir / f"vintage_distribution_by_level__{sample}.parquet")

    # Document-purpose surfaces.
    purpose_rows = purposes[["ordinance_doc_id", "motive", "confidence"]].copy()
    purpose_rows["motive"] = purpose_rows["motive"].astype(str).replace("nan", "").fillna("")
    purpose_rows["confidence"] = pd.to_numeric(purpose_rows["confidence"], errors="coerce").fillna(0.0)
    purpose_rows = purpose_rows[purpose_rows["motive"].astype(str) != ""].reset_index(drop=True)
    purpose_primary = (
        purpose_rows.sort_values(["ordinance_doc_id", "confidence", "motive"], ascending=[True, False, True])
        .groupby("ordinance_doc_id", as_index=False)
        .first()
        .rename(columns={"motive": "primary_motive", "confidence": "primary_motive_confidence"})
    )
    purpose_weighted = purpose_rows.copy()
    total_conf = purpose_weighted.groupby("ordinance_doc_id", dropna=False)["confidence"].transform("sum")
    count_conf = purpose_weighted.groupby("ordinance_doc_id", dropna=False)["confidence"].transform("count").clip(lower=1)
    purpose_weighted["purpose_weight"] = np.where(
        total_conf.gt(0),
        purpose_weighted["confidence"] / total_conf.clip(lower=1e-9),
        1.0 / count_conf,
    )
    purpose_weighted = purpose_weighted.merge(
        docs[["ordinance_doc_id", "jurisdiction_region", "jurisdiction_level", "vintage_bin"]],
        on="ordinance_doc_id",
        how="left",
    )
    purpose_primary_presence = purpose_primary.merge(
        docs[["ordinance_doc_id", "jurisdiction_region", "jurisdiction_level", "vintage_bin"]],
        on="ordinance_doc_id",
        how="left",
    ).rename(columns={"primary_motive": "motive"})
    purpose_primary_presence["present"] = 1
    primary_region = _vintage_adjusted_presence_table(
        purpose_primary_presence,
        group_col="jurisdiction_region",
        vintage_col="vintage_bin",
        category_cols=["motive"],
        value_col="present",
    )
    primary_level = _vintage_adjusted_presence_table(
        purpose_primary_presence,
        group_col="jurisdiction_level",
        vintage_col="vintage_bin",
        category_cols=["motive"],
        value_col="present",
    )
    weighted_region = _vintage_adjusted_category_mean_table(
        purpose_weighted,
        group_col="jurisdiction_region",
        vintage_col="vintage_bin",
        category_cols=["motive"],
        value_col="purpose_weight",
    )
    weighted_level = _vintage_adjusted_category_mean_table(
        purpose_weighted,
        group_col="jurisdiction_level",
        vintage_col="vintage_bin",
        category_cols=["motive"],
        value_col="purpose_weight",
    )
    top_primary_region = (
        primary_region.sort_values(
            ["presence_lift_vs_overall_vintage_adjusted", "presence_rate_vintage_adjusted", "jurisdiction_region", "motive"],
            ascending=[False, False, True, True],
        )
        .groupby("jurisdiction_region", dropna=False)
        .head(8)
        .reset_index(drop=True)
    )
    top_weighted_region = (
        weighted_region.sort_values(
            ["purpose_weight_lift_vs_overall_vintage_adjusted", "purpose_weight__vintage_adjusted", "jurisdiction_region", "motive"],
            ascending=[False, False, True, True],
        )
        .groupby("jurisdiction_region", dropna=False)
        .head(8)
        .reset_index(drop=True)
    )
    top_weighted_level = (
        weighted_level.sort_values(
            ["purpose_weight_lift_vs_overall_vintage_adjusted", "purpose_weight__vintage_adjusted", "jurisdiction_level", "motive"],
            ascending=[False, False, True, True],
        )
        .groupby("jurisdiction_level", dropna=False)
        .head(8)
        .reset_index(drop=True)
    )
    write_parquet(purpose_primary.sort_values("ordinance_doc_id").reset_index(drop=True), purpose_dir / f"primary_document_motives__{sample}.parquet")
    write_parquet(primary_region, purpose_dir / f"primary_motive_by_region_vintage_adjusted__{sample}.parquet")
    write_parquet(primary_level, purpose_dir / f"primary_motive_by_level_vintage_adjusted__{sample}.parquet")
    write_parquet(weighted_region, purpose_dir / f"weighted_purpose_by_region_vintage_adjusted__{sample}.parquet")
    write_parquet(weighted_level, purpose_dir / f"weighted_purpose_by_level_vintage_adjusted__{sample}.parquet")
    write_parquet(top_primary_region, purpose_dir / f"top_primary_motive_region_lifts__{sample}.parquet")
    write_parquet(top_weighted_region, purpose_dir / f"top_weighted_purpose_region_lifts__{sample}.parquet")
    write_parquet(top_weighted_level, purpose_dir / f"top_weighted_purpose_level_lifts__{sample}.parquet")

    # Cross-level coherence diagnostics.
    purpose_grouped = purposes.groupby("ordinance_doc_id", dropna=False)
    section_grouped = sections.groupby("ordinance_doc_id", dropna=False)
    coherence_rows: list[dict[str, Any]] = []
    packet_lines = ["# Frontier LLM v2 Section-Side Research Memo", "", "## Cross-Level Coherence", ""]
    for row in docs.to_dict(orient="records"):
        doc_id = clean_optional_str(row["ordinance_doc_id"])
        doc_purposes = purpose_grouped.get_group(doc_id) if doc_id in purpose_grouped.groups else pd.DataFrame(columns=purposes.columns)
        doc_sections = section_grouped.get_group(doc_id) if doc_id in section_grouped.groups else pd.DataFrame(columns=sections.columns)
        if not doc_purposes.empty:
            top_purpose = doc_purposes.sort_values(["confidence", "motive"], ascending=[False, True]).iloc[0]
            doc_primary_motive = clean_optional_str(top_purpose.get("motive"))
            doc_primary_purpose = clean_optional_str(top_purpose.get("description"))
            doc_primary_confidence = float(top_purpose.get("confidence", 0.0))
        else:
            doc_primary_motive = "unclear"
            doc_primary_purpose = ""
            doc_primary_confidence = 0.0

        section_word_totals = (
            doc_sections.groupby("primary_motive", dropna=False)["word_count"].sum().sort_values(ascending=False)
            if not doc_sections.empty
            else pd.Series(dtype=float)
        )
        total_words = float(section_word_totals.sum()) if not section_word_totals.empty else 0.0
        filtered_word_totals = section_word_totals.loc[~section_word_totals.index.astype(str).isin(EXCLUDED_MOTIVES)] if not section_word_totals.empty else section_word_totals
        section_primary_motive = clean_optional_str(section_word_totals.index[0]) if not section_word_totals.empty else "unclear"
        filtered_section_primary_motive = clean_optional_str(filtered_word_totals.index[0]) if not filtered_word_totals.empty else section_primary_motive
        doc_alignment_words = float(section_word_totals.get(doc_primary_motive, 0.0)) if not section_word_totals.empty else 0.0
        filtered_total_words = float(filtered_word_totals.sum()) if not filtered_word_totals.empty else 0.0
        filtered_alignment_words = float(filtered_word_totals.get(doc_primary_motive, 0.0)) if not filtered_word_totals.empty else 0.0
        coherence_rows.append(
            {
                "ordinance_doc_id": doc_id,
                "issue_id": clean_optional_str(row.get("issue_id")),
                "census_id_pid6": clean_optional_str(row.get("census_id_pid6")),
                "issue_year": int(row.get("issue_year")) if pd.notna(row.get("issue_year")) else pd.NA,
                "vintage_bin": clean_optional_str(row.get("vintage_bin")),
                "jurisdiction_region": clean_optional_str(row.get("jurisdiction_region")),
                "jurisdiction_level": clean_optional_str(row.get("jurisdiction_level")),
                "doc_primary_purpose": doc_primary_purpose,
                "doc_primary_motive": doc_primary_motive,
                "doc_primary_confidence": doc_primary_confidence,
                "section_primary_motive_weighted": section_primary_motive,
                "section_primary_motive_weighted_excl_other_unclear": filtered_section_primary_motive,
                "section_count": int(doc_sections["section_id"].nunique()) if not doc_sections.empty else 0,
                "distinct_section_motive_count": int(doc_sections["primary_motive"].nunique()) if not doc_sections.empty else 0,
                "purpose_section_exact_match_flag": int(doc_primary_motive == section_primary_motive and doc_primary_motive != ""),
                "purpose_section_filtered_match_flag": int(doc_primary_motive == filtered_section_primary_motive and doc_primary_motive != ""),
                "purpose_alignment_word_share": _safe_share(doc_alignment_words, total_words),
                "purpose_alignment_non_other_word_share": _safe_share(filtered_alignment_words, filtered_total_words),
            }
        )
    coherence_df = pd.DataFrame.from_records(coherence_rows)
    write_parquet(coherence_df, coherence_dir / f"document_section_motive_alignment__{sample}.parquet")
    purpose_alignment_by_motive = (
        coherence_df.groupby("doc_primary_motive", dropna=False)
        .agg(
            ordinance_count=("ordinance_doc_id", "nunique"),
            mean_alignment_share=("purpose_alignment_word_share", "mean"),
            mean_non_other_alignment_share=("purpose_alignment_non_other_word_share", "mean"),
            exact_match_rate=("purpose_section_exact_match_flag", "mean"),
            filtered_match_rate=("purpose_section_filtered_match_flag", "mean"),
            mean_distinct_section_motives=("distinct_section_motive_count", "mean"),
        )
        .reset_index()
        .sort_values(["ordinance_count", "mean_non_other_alignment_share", "doc_primary_motive"], ascending=[False, False, True])
        .reset_index(drop=True)
    )
    write_parquet(purpose_alignment_by_motive, purpose_dir / f"purpose_alignment_by_primary_motive__{sample}.parquet")
    by_region = (
        coherence_df.groupby("jurisdiction_region", dropna=False)
        .agg(
            ordinance_count=("ordinance_doc_id", "nunique"),
            exact_match_rate=("purpose_section_exact_match_flag", "mean"),
            filtered_match_rate=("purpose_section_filtered_match_flag", "mean"),
            mean_alignment_share=("purpose_alignment_word_share", "mean"),
            mean_non_other_alignment_share=("purpose_alignment_non_other_word_share", "mean"),
            mean_distinct_section_motives=("distinct_section_motive_count", "mean"),
        )
        .reset_index()
        .sort_values("jurisdiction_region")
        .reset_index(drop=True)
    )
    by_level = (
        coherence_df.groupby("jurisdiction_level", dropna=False)
        .agg(
            ordinance_count=("ordinance_doc_id", "nunique"),
            exact_match_rate=("purpose_section_exact_match_flag", "mean"),
            filtered_match_rate=("purpose_section_filtered_match_flag", "mean"),
            mean_alignment_share=("purpose_alignment_word_share", "mean"),
            mean_non_other_alignment_share=("purpose_alignment_non_other_word_share", "mean"),
        )
        .reset_index()
        .sort_values("jurisdiction_level")
        .reset_index(drop=True)
    )
    by_vintage = (
        coherence_df.groupby("vintage_bin", dropna=False)
        .agg(
            ordinance_count=("ordinance_doc_id", "nunique"),
            exact_match_rate=("purpose_section_exact_match_flag", "mean"),
            filtered_match_rate=("purpose_section_filtered_match_flag", "mean"),
            mean_alignment_share=("purpose_alignment_word_share", "mean"),
            mean_non_other_alignment_share=("purpose_alignment_non_other_word_share", "mean"),
            mean_distinct_section_motives=("distinct_section_motive_count", "mean"),
        )
        .reset_index()
        .sort_values("vintage_bin")
        .reset_index(drop=True)
    )
    coherence_metrics_df = coherence_df.rename(
        columns={
            "purpose_section_exact_match_flag": "exact_match_rate",
            "purpose_section_filtered_match_flag": "filtered_match_rate",
            "purpose_alignment_word_share": "mean_alignment_share",
            "purpose_alignment_non_other_word_share": "mean_non_other_alignment_share",
            "distinct_section_motive_count": "mean_distinct_section_motives",
        }
    )
    coherence_metric_cols = [
        "exact_match_rate",
        "filtered_match_rate",
        "mean_alignment_share",
        "mean_non_other_alignment_share",
        "mean_distinct_section_motives",
    ]
    coherence_region_adjusted = _vintage_adjusted_mean_table(
        coherence_metrics_df,
        group_col="jurisdiction_region",
        vintage_col="vintage_bin",
        value_cols=coherence_metric_cols,
    )
    coherence_level_adjusted = _vintage_adjusted_mean_table(
        coherence_metrics_df,
        group_col="jurisdiction_level",
        vintage_col="vintage_bin",
        value_cols=coherence_metric_cols,
    )
    low_alignment = coherence_df.sort_values(
        ["purpose_alignment_non_other_word_share", "purpose_alignment_word_share", "ordinance_doc_id"],
        ascending=[True, True, True],
    ).reset_index(drop=True)
    write_parquet(by_region, coherence_dir / f"coherence_by_region__{sample}.parquet")
    write_parquet(by_level, coherence_dir / f"coherence_by_level__{sample}.parquet")
    write_parquet(by_vintage, coherence_dir / f"coherence_by_vintage__{sample}.parquet")
    write_parquet(coherence_region_adjusted, coherence_dir / f"coherence_by_region_vintage_adjusted__{sample}.parquet")
    write_parquet(coherence_level_adjusted, coherence_dir / f"coherence_by_level_vintage_adjusted__{sample}.parquet")
    write_parquet(low_alignment, coherence_dir / f"low_alignment_ordinances__{sample}.parquet")

    # Section architecture summaries.
    section_counts = sections.groupby("ordinance_doc_id", dropna=False)["section_id"].nunique().rename("section_count")
    sections = sections.merge(section_counts, on="ordinance_doc_id", how="left")
    sections["architecture_anchor"] = sections["title"].astype(str).map(_architecture_anchor)
    sections["normalized_position"] = [
        _normalized_position(num, total)
        for num, total in zip(sections["section_number"].astype(int), sections["section_count"].astype(int), strict=False)
    ]
    kind_region = (
        sections.groupby(["jurisdiction_region", "architecture_anchor"], dropna=False)["word_count"]
        .sum()
        .reset_index(name="word_count")
        .sort_values(["jurisdiction_region", "word_count"], ascending=[True, False])
        .reset_index(drop=True)
    )
    region_totals = kind_region.groupby("jurisdiction_region", dropna=False)["word_count"].sum().rename("region_word_total").reset_index()
    kind_region = kind_region.merge(region_totals, on="jurisdiction_region", how="left")
    kind_region["share_within_region"] = kind_region["word_count"] / kind_region["region_word_total"].clip(lower=1)
    kind_level = (
        sections.groupby(["jurisdiction_level", "architecture_anchor"], dropna=False)["word_count"]
        .sum()
        .reset_index(name="word_count")
        .sort_values(["jurisdiction_level", "word_count"], ascending=[True, False])
        .reset_index(drop=True)
    )
    level_totals = kind_level.groupby("jurisdiction_level", dropna=False)["word_count"].sum().rename("level_word_total").reset_index()
    kind_level = kind_level.merge(level_totals, on="jurisdiction_level", how="left")
    kind_level["share_within_level"] = kind_level["word_count"] / kind_level["level_word_total"].clip(lower=1)
    section_kind_position = (
        sections.groupby("architecture_anchor", dropna=False)
        .agg(
            section_count=("section_id", "count"),
            mean_normalized_position=("normalized_position", "mean"),
            median_word_count=("word_count", "median"),
        )
        .reset_index()
        .sort_values(["section_count", "architecture_anchor"], ascending=[False, True])
        .reset_index(drop=True)
    )
    doc_arch_rows: list[dict[str, Any]] = []
    for doc_id, group in sections.groupby("ordinance_doc_id", dropna=False):
        ordered = group.sort_values(["section_number", "section_id"]).reset_index(drop=True)
        first_three = " | ".join(ordered["architecture_anchor"].astype(str).head(3).tolist())
        last_three = " | ".join(ordered["architecture_anchor"].astype(str).tail(3).tolist())
        dominant_kind = clean_optional_str(
            ordered.groupby("architecture_anchor", dropna=False)["word_count"].sum().sort_values(ascending=False).index[0]
        ) if not ordered.empty else ""
        substantive = ordered.loc[~ordered["primary_motive"].astype(str).isin(list(EXCLUDED_MOTIVES))].copy()
        first_substantive = clean_optional_str(substantive.iloc[0]["architecture_anchor"]) if not substantive.empty else ""
        first_substantive_motive = clean_optional_str(substantive.iloc[0]["primary_motive"]) if not substantive.empty else ""
        meta = docs.loc[docs["ordinance_doc_id"].astype(str) == str(doc_id)].head(1)
        meta_row = meta.iloc[0].to_dict() if not meta.empty else {}
        doc_arch_rows.append(
            {
                "ordinance_doc_id": clean_optional_str(doc_id),
                "issue_id": clean_optional_str(meta_row.get("issue_id")),
                "jurisdiction_region": clean_optional_str(meta_row.get("jurisdiction_region")),
                "jurisdiction_level": clean_optional_str(meta_row.get("jurisdiction_level")),
                "section_count": int(ordered["section_id"].nunique()),
                "opening_signature": first_three,
                "closing_signature": last_three,
                "dominant_section_kind_by_words": dominant_kind,
                "first_substantive_section_kind": first_substantive,
                "first_substantive_section_motive": first_substantive_motive,
            }
        )
    doc_arch_df = pd.DataFrame.from_records(doc_arch_rows)
    opening_region = (
        doc_arch_df.groupby(["jurisdiction_region", "first_substantive_section_kind"], dropna=False)["ordinance_doc_id"]
        .nunique()
        .reset_index(name="ordinance_count")
        .sort_values(["jurisdiction_region", "ordinance_count"], ascending=[True, False])
        .reset_index(drop=True)
    )
    opening_motive_region = (
        doc_arch_df.groupby(["jurisdiction_region", "first_substantive_section_motive"], dropna=False)["ordinance_doc_id"]
        .nunique()
        .reset_index(name="ordinance_count")
        .sort_values(["jurisdiction_region", "ordinance_count"], ascending=[True, False])
        .reset_index(drop=True)
    )
    write_parquet(kind_region, architecture_dir / f"section_kind_shares_by_region__{sample}.parquet")
    write_parquet(kind_level, architecture_dir / f"section_kind_shares_by_level__{sample}.parquet")
    write_parquet(section_kind_position, architecture_dir / f"section_kind_position_summary__{sample}.parquet")
    write_parquet(doc_arch_df, architecture_dir / f"document_architecture_profiles__{sample}.parquet")
    write_parquet(opening_region, architecture_dir / f"first_substantive_section_kind_by_region__{sample}.parquet")
    write_parquet(opening_motive_region, architecture_dir / f"first_substantive_section_motive_by_region__{sample}.parquet")

    # Retrieval and prototype surfaces from OpenAI section embeddings.
    ordinance_meta, ordinance_matrix = _aggregate_ordinance_vectors(section_meta, section_matrix)
    ordinance_meta = ordinance_meta.merge(
        docs[
            [
                "ordinance_doc_id",
                "issue_date",
                "publication_key",
                "publication_city_name",
                "publication_state_abbr",
                "jurisdiction_key",
                "jurisdiction_city_name",
                "jurisdiction_state_abbr",
            ]
        ],
        on="ordinance_doc_id",
        how="left",
    )
    ordinance_neighbors = _top_neighbor_rows(ordinance_meta, ordinance_matrix, k=5)
    region_prototypes = _prototype_rows(ordinance_meta, ordinance_matrix, group_col="jurisdiction_region", top_n=5)
    level_prototypes = _prototype_rows(ordinance_meta, ordinance_matrix, group_col="jurisdiction_level", top_n=5)
    coherence_for_merge = coherence_df[["ordinance_doc_id", "doc_primary_motive"]].copy()
    ordinance_meta = ordinance_meta.merge(coherence_for_merge, on="ordinance_doc_id", how="left")
    motive_prototypes = _prototype_rows(
        ordinance_meta[ordinance_meta["doc_primary_motive"].astype(str).ne("")].copy(),
        ordinance_matrix,
        group_col="doc_primary_motive",
        top_n=5,
    )
    write_parquet(ordinance_meta, retrieval_dir / f"ordinance_embedding_metadata__{sample}.parquet")
    np.save(retrieval_dir / f"ordinance_embedding_matrix__{sample}.npy", ordinance_matrix)
    write_parquet(ordinance_neighbors, retrieval_dir / f"ordinance_neighbors__{sample}.parquet")
    write_parquet(region_prototypes, retrieval_dir / f"region_prototypes__{sample}.parquet")
    write_parquet(level_prototypes, retrieval_dir / f"level_prototypes__{sample}.parquet")
    write_parquet(motive_prototypes, retrieval_dir / f"motive_prototypes__{sample}.parquet")
    neighbor_diagnostics = ordinance_neighbors.copy()
    if not neighbor_diagnostics.empty:
        neighbor_diagnostics["same_region_flag"] = (
            neighbor_diagnostics["jurisdiction_region"].astype(str) == neighbor_diagnostics["neighbor_region"].astype(str)
        ).astype(int)
        neighbor_diagnostics["same_level_flag"] = (
            neighbor_diagnostics["jurisdiction_level"].astype(str) == neighbor_diagnostics["neighbor_level"].astype(str)
        ).astype(int)
        neighbor_by_doc = (
            neighbor_diagnostics.groupby("ordinance_doc_id", dropna=False)
            .agg(
                mean_neighbor_similarity=("cosine_similarity", "mean"),
                nearest_neighbor_similarity=("cosine_similarity", "max"),
                same_region_neighbor_share=("same_region_flag", "mean"),
                same_level_neighbor_share=("same_level_flag", "mean"),
                dominant_neighbor_region=("neighbor_region", lambda s: clean_optional_str(pd.Series(s).astype(str).value_counts().index[0]) if not pd.Series(s).empty else ""),
                dominant_neighbor_level=("neighbor_level", lambda s: clean_optional_str(pd.Series(s).astype(str).value_counts().index[0]) if not pd.Series(s).empty else ""),
            )
            .reset_index()
            .merge(
                ordinance_meta[
                    [
                        "ordinance_doc_id",
                        "jurisdiction_region",
                        "jurisdiction_level",
                        "doc_primary_motive",
                    ]
                ],
                on="ordinance_doc_id",
                how="left",
            )
        )
        neighbor_by_region = (
            neighbor_by_doc.groupby("jurisdiction_region", dropna=False)
            .agg(
                ordinance_count=("ordinance_doc_id", "nunique"),
                mean_same_region_neighbor_share=("same_region_neighbor_share", "mean"),
                mean_same_level_neighbor_share=("same_level_neighbor_share", "mean"),
                mean_neighbor_similarity=("mean_neighbor_similarity", "mean"),
            )
            .reset_index()
            .sort_values("jurisdiction_region")
            .reset_index(drop=True)
        )
        neighbor_by_level = (
            neighbor_by_doc.groupby("jurisdiction_level", dropna=False)
            .agg(
                ordinance_count=("ordinance_doc_id", "nunique"),
                mean_same_region_neighbor_share=("same_region_neighbor_share", "mean"),
                mean_same_level_neighbor_share=("same_level_neighbor_share", "mean"),
                mean_neighbor_similarity=("mean_neighbor_similarity", "mean"),
            )
            .reset_index()
            .sort_values("jurisdiction_level")
            .reset_index(drop=True)
        )
        regional_base_rate = (
            ordinance_meta["jurisdiction_region"].astype(str).value_counts(normalize=True).rename_axis("jurisdiction_region").reset_index(name="region_sample_share")
        )
        neighbor_by_region = neighbor_by_region.merge(regional_base_rate, on="jurisdiction_region", how="left")
        neighbor_by_region["same_region_neighbor_lift"] = (
            neighbor_by_region["mean_same_region_neighbor_share"] / neighbor_by_region["region_sample_share"].clip(lower=1e-9)
        )
        level_base_rate = (
            ordinance_meta["jurisdiction_level"].astype(str).value_counts(normalize=True).rename_axis("jurisdiction_level").reset_index(name="level_sample_share")
        )
        neighbor_by_level = neighbor_by_level.merge(level_base_rate, on="jurisdiction_level", how="left")
        neighbor_by_level["same_level_neighbor_lift"] = (
            neighbor_by_level["mean_same_level_neighbor_share"] / neighbor_by_level["level_sample_share"].clip(lower=1e-9)
        )
        cross_region_bridges = (
            neighbor_by_doc.sort_values(
                ["same_region_neighbor_share", "mean_neighbor_similarity", "ordinance_doc_id"],
                ascending=[True, False, True],
            )
            .head(20)
            .reset_index(drop=True)
        )
        strong_regional_exemplars = (
            neighbor_by_doc.sort_values(
                ["same_region_neighbor_share", "mean_neighbor_similarity", "ordinance_doc_id"],
                ascending=[False, False, True],
            )
            .head(20)
            .reset_index(drop=True)
        )
    else:
        neighbor_by_doc = pd.DataFrame()
        neighbor_by_region = pd.DataFrame()
        neighbor_by_level = pd.DataFrame()
        cross_region_bridges = pd.DataFrame()
        strong_regional_exemplars = pd.DataFrame()
    write_parquet(neighbor_by_doc, retrieval_dir / f"neighbor_similarity_by_doc__{sample}.parquet")
    write_parquet(neighbor_by_region, retrieval_dir / f"neighbor_similarity_by_region__{sample}.parquet")
    write_parquet(neighbor_by_level, retrieval_dir / f"neighbor_similarity_by_level__{sample}.parquet")
    write_parquet(cross_region_bridges, retrieval_dir / f"cross_region_bridge_ordinances__{sample}.parquet")
    write_parquet(strong_regional_exemplars, retrieval_dir / f"strong_regional_exemplars__{sample}.parquet")
    bridge_lines = ["# Cross-region bridge ordinances", ""]
    if cross_region_bridges.empty:
        bridge_lines.append("No bridge ordinances available.")
    else:
        for row in cross_region_bridges.head(20).to_dict(orient="records"):
            bridge_lines.append(
                f"- `{clean_optional_str(row['ordinance_doc_id'])}` | region=`{clean_optional_str(row['jurisdiction_region'])}` "
                f"| level=`{clean_optional_str(row['jurisdiction_level'])}` | dominant neighbor region=`{clean_optional_str(row['dominant_neighbor_region'])}` "
                f"| same-region share={float(row['same_region_neighbor_share']):.2f} | "
                f"mean similarity={float(row['mean_neighbor_similarity']):.3f} | motive=`{clean_optional_str(row['doc_primary_motive'])}`"
            )
    write_text(retrieval_dir / f"cross_region_bridge_cases__{sample}.md", "\n".join(bridge_lines))
    exemplar_lines = ["# Strong regional exemplars", ""]
    if strong_regional_exemplars.empty:
        exemplar_lines.append("No exemplar ordinances available.")
    else:
        for row in strong_regional_exemplars.head(20).to_dict(orient="records"):
            exemplar_lines.append(
                f"- `{clean_optional_str(row['ordinance_doc_id'])}` | region=`{clean_optional_str(row['jurisdiction_region'])}` "
                f"| level=`{clean_optional_str(row['jurisdiction_level'])}` | same-region share={float(row['same_region_neighbor_share']):.2f} "
                f"| mean similarity={float(row['mean_neighbor_similarity']):.3f} | motive=`{clean_optional_str(row['doc_primary_motive'])}`"
            )
    write_text(retrieval_dir / f"strong_regional_exemplars__{sample}.md", "\n".join(exemplar_lines))

    enriched_path = run_root / "semantics" / f"enriched_clauses__{sample}.parquet"
    enriched = pd.read_parquet(enriched_path).copy() if enriched_path.is_file() else pd.DataFrame()
    clause_surface_available = bool(enriched_path.is_file())
    coverage_summary: dict[str, Any] = {"available": False, "doc_count": 0, "row_count": 0, "mean_clause_share": 0.0}
    clause_doc_cov = (
        clauses.groupby("ordinance_doc_id", dropna=False)
        .agg(
            clause_count=("clause_id", "nunique"),
            extracted_section_count=("section_id", "nunique"),
        )
        .reset_index()
    )
    coverage_df = (
        docs[
            [
                "ordinance_doc_id",
                "issue_id",
                "issue_year",
                "vintage_bin",
                "jurisdiction_region",
                "jurisdiction_level",
                "census_id_pid6",
            ]
        ]
        .merge(
            sections.groupby("ordinance_doc_id", dropna=False)["section_id"].nunique().rename("section_count").reset_index(),
            on="ordinance_doc_id",
            how="left",
        )
        .merge(clause_doc_cov, on="ordinance_doc_id", how="left")
    )
    if not enriched.empty:
        enriched_cov = (
            enriched.groupby("ordinance_doc_id", dropna=False)
            .agg(
                enriched_clause_count=("clause_id", "nunique"),
                enriched_section_count=("section_id", "nunique"),
                operative_clause_count=("operative_flag", "sum"),
                mean_enrichment_confidence=("confidence", "mean"),
            )
            .reset_index()
        )
        coverage_df = coverage_df.merge(enriched_cov, on="ordinance_doc_id", how="left")
    for col in [
        "section_count",
        "clause_count",
        "extracted_section_count",
        "enriched_clause_count",
        "enriched_section_count",
        "operative_clause_count",
        "mean_enrichment_confidence",
    ]:
        if col not in coverage_df.columns:
            coverage_df[col] = 0.0
        coverage_df[col] = pd.to_numeric(coverage_df[col], errors="coerce").fillna(0.0)
    coverage_df["section_extraction_share"] = coverage_df["extracted_section_count"] / coverage_df["section_count"].clip(lower=1)
    coverage_df["section_enrichment_share"] = coverage_df["enriched_section_count"] / coverage_df["section_count"].clip(lower=1)
    coverage_df["clause_enrichment_share"] = coverage_df["enriched_clause_count"] / coverage_df["clause_count"].clip(lower=1)
    coverage_df["operative_clause_share_within_enriched"] = coverage_df["operative_clause_count"] / coverage_df["enriched_clause_count"].clip(lower=1)
    coverage_df["has_clause_extraction"] = (coverage_df["clause_count"] > 0).astype(float)
    coverage_df["has_clause_enrichment"] = (coverage_df["enriched_clause_count"] > 0).astype(float)
    coverage_df["coverage_gap_clause_count"] = coverage_df["clause_count"] - coverage_df["enriched_clause_count"]
    coverage_by_region = (
        coverage_df.groupby("jurisdiction_region", dropna=False)
        .agg(
            ordinance_count=("ordinance_doc_id", "nunique"),
            doc_enrichment_rate=("has_clause_enrichment", "mean"),
            mean_clause_enrichment_share=("clause_enrichment_share", "mean"),
            mean_section_enrichment_share=("section_enrichment_share", "mean"),
            mean_enriched_clause_count=("enriched_clause_count", "mean"),
            mean_clause_count=("clause_count", "mean"),
        )
        .reset_index()
        .sort_values("jurisdiction_region")
        .reset_index(drop=True)
    )
    coverage_by_level = (
        coverage_df.groupby("jurisdiction_level", dropna=False)
        .agg(
            ordinance_count=("ordinance_doc_id", "nunique"),
            doc_enrichment_rate=("has_clause_enrichment", "mean"),
            mean_clause_enrichment_share=("clause_enrichment_share", "mean"),
            mean_section_enrichment_share=("section_enrichment_share", "mean"),
            mean_enriched_clause_count=("enriched_clause_count", "mean"),
            mean_clause_count=("clause_count", "mean"),
        )
        .reset_index()
        .sort_values("jurisdiction_level")
        .reset_index(drop=True)
    )
    coverage_by_vintage = (
        coverage_df.groupby("vintage_bin", dropna=False)
        .agg(
            ordinance_count=("ordinance_doc_id", "nunique"),
            doc_enrichment_rate=("has_clause_enrichment", "mean"),
            mean_clause_enrichment_share=("clause_enrichment_share", "mean"),
            mean_section_enrichment_share=("section_enrichment_share", "mean"),
            mean_enriched_clause_count=("enriched_clause_count", "mean"),
            mean_clause_count=("clause_count", "mean"),
        )
        .reset_index()
        .sort_values("vintage_bin")
        .reset_index(drop=True)
    )
    coverage_region_adjusted = _vintage_adjusted_mean_table(
        coverage_df,
        group_col="jurisdiction_region",
        vintage_col="vintage_bin",
        value_cols=[
            "has_clause_enrichment",
            "clause_enrichment_share",
            "section_enrichment_share",
            "enriched_clause_count",
            "clause_count",
        ],
    )
    coverage_level_adjusted = _vintage_adjusted_mean_table(
        coverage_df,
        group_col="jurisdiction_level",
        vintage_col="vintage_bin",
        value_cols=[
            "has_clause_enrichment",
            "clause_enrichment_share",
            "section_enrichment_share",
            "enriched_clause_count",
            "clause_count",
        ],
    )
    write_parquet(coverage_df.sort_values("ordinance_doc_id").reset_index(drop=True), coverage_dir / f"clause_enrichment_coverage_by_doc__{sample}.parquet")
    write_parquet(coverage_by_region, coverage_dir / f"clause_enrichment_coverage_by_region__{sample}.parquet")
    write_parquet(coverage_by_level, coverage_dir / f"clause_enrichment_coverage_by_level__{sample}.parquet")
    write_parquet(coverage_by_vintage, coverage_dir / f"clause_enrichment_coverage_by_vintage__{sample}.parquet")
    write_parquet(coverage_region_adjusted, coverage_dir / f"clause_enrichment_coverage_by_region_vintage_adjusted__{sample}.parquet")
    write_parquet(coverage_level_adjusted, coverage_dir / f"clause_enrichment_coverage_by_level_vintage_adjusted__{sample}.parquet")
    most_undercovered = coverage_df.sort_values(
        ["clause_enrichment_share", "clause_count", "ordinance_doc_id"],
        ascending=[True, False, True],
    ).head(25).reset_index(drop=True)
    best_covered = coverage_df.sort_values(
        ["clause_enrichment_share", "enriched_clause_count", "ordinance_doc_id"],
        ascending=[False, False, True],
    ).head(25).reset_index(drop=True)
    write_parquet(most_undercovered, coverage_dir / f"most_undercovered_ordinances__{sample}.parquet")
    write_parquet(best_covered, coverage_dir / f"best_covered_ordinances__{sample}.parquet")
    coverage_lines = ["# Clause enrichment coverage notes", ""]
    if coverage_region_adjusted.empty:
        coverage_lines.append("Coverage summaries unavailable.")
    else:
        best_region = coverage_region_adjusted.sort_values("has_clause_enrichment__vintage_adjusted", ascending=False).iloc[0]
        weak_region = coverage_region_adjusted.sort_values("has_clause_enrichment__vintage_adjusted", ascending=True).iloc[0]
        coverage_lines.extend(
            [
                "## Vintage-adjusted regional coverage",
                "",
                f"- strongest current region: `{clean_optional_str(best_region['jurisdiction_region'])}` "
                f"(doc coverage={float(best_region['has_clause_enrichment__vintage_adjusted']):.3f}, "
                f"mean clause share={float(best_region['clause_enrichment_share__vintage_adjusted']):.3f})",
                f"- weakest current region: `{clean_optional_str(weak_region['jurisdiction_region'])}` "
                f"(doc coverage={float(weak_region['has_clause_enrichment__vintage_adjusted']):.3f}, "
                f"mean clause share={float(weak_region['clause_enrichment_share__vintage_adjusted']):.3f})",
                "",
                "## Most undercovered ordinances",
                "",
            ]
        )
        for row in most_undercovered.head(12).to_dict(orient="records"):
            coverage_lines.append(
                f"- `{clean_optional_str(row['ordinance_doc_id'])}` | region=`{clean_optional_str(row['jurisdiction_region'])}` "
                f"| level=`{clean_optional_str(row['jurisdiction_level'])}` | clause_share={float(row['clause_enrichment_share']):.3f} "
                f"| enriched={int(row['enriched_clause_count'])}/{int(row['clause_count'])}"
            )
        coverage_lines.extend(["", "## Best covered ordinances", ""])
        for row in best_covered.head(12).to_dict(orient="records"):
            coverage_lines.append(
                f"- `{clean_optional_str(row['ordinance_doc_id'])}` | region=`{clean_optional_str(row['jurisdiction_region'])}` "
                f"| level=`{clean_optional_str(row['jurisdiction_level'])}` | clause_share={float(row['clause_enrichment_share']):.3f} "
                f"| enriched={int(row['enriched_clause_count'])}/{int(row['clause_count'])}"
            )
    write_text(coverage_dir / f"clause_enrichment_coverage_notes__{sample}.md", "\n".join(coverage_lines))
    coverage_summary = {
        "available": True,
        "doc_count": int((coverage_df["has_clause_enrichment"] > 0).sum()),
        "row_count": int(coverage_df["enriched_clause_count"].sum()),
        "mean_clause_share": float(coverage_df["clause_enrichment_share"].mean()),
        "mean_section_share": float(coverage_df["section_enrichment_share"].mean()),
    }

    section_cluster_summary: dict[str, Any] = {"available": False, "cluster_count": 0}
    top_region_contrasts = pd.DataFrame()
    top_region_contrasts_adjusted = pd.DataFrame()
    top_joint_effect_clusters = pd.DataFrame()
    cluster_presence_effects = pd.DataFrame()
    section_assign = pd.DataFrame()
    embedding_surface_summary: dict[str, Any] = {"available": False, "winner": "", "surface_count": 0}
    section_assign_path = run_root / "clusters" / "section_semantic_hdbscan" / "assignments.parquet"
    if section_assign_path.is_file():
        cluster_dir = research_dir / "section_clusters"
        cluster_dir.mkdir(parents=True, exist_ok=True)
        section_assign = pd.read_parquet(section_assign_path).copy()
        if "word_count" not in section_assign.columns:
            section_assign = section_assign.merge(
                sections[["section_id", "word_count", "jurisdiction_region", "jurisdiction_level"]],
                on="section_id",
                how="left",
            )
        section_assign = section_assign.merge(
            docs[["ordinance_doc_id", "vintage_bin"]],
            on="ordinance_doc_id",
            how="left",
        )
        non_noise = section_assign[section_assign["cluster_id"].astype(int) >= 0].copy()
        cluster_region = (
            non_noise.groupby(["jurisdiction_region", "cluster_id", "cluster_label"], dropna=False)["word_count"]
            .sum()
            .reset_index(name="word_count")
        )
        cluster_region_totals = cluster_region.groupby("jurisdiction_region", dropna=False)["word_count"].sum().rename("region_word_total").reset_index()
        cluster_region = cluster_region.merge(cluster_region_totals, on="jurisdiction_region", how="left")
        cluster_region["share_within_region"] = cluster_region["word_count"] / cluster_region["region_word_total"].clip(lower=1)
        overall_cluster = (
            non_noise.groupby(["cluster_id", "cluster_label"], dropna=False)["word_count"]
            .sum()
            .reset_index(name="overall_word_count")
        )
        overall_cluster["overall_share"] = overall_cluster["overall_word_count"] / overall_cluster["overall_word_count"].sum()
        cluster_region = cluster_region.merge(overall_cluster[["cluster_id", "cluster_label", "overall_share"]], on=["cluster_id", "cluster_label"], how="left")
        cluster_region["share_lift_vs_overall"] = cluster_region["share_within_region"] / cluster_region["overall_share"].clip(lower=1e-9)
        cluster_region["share_diff_vs_overall"] = cluster_region["share_within_region"] - cluster_region["overall_share"]
        cluster_level = (
            non_noise.groupby(["jurisdiction_level", "cluster_id", "cluster_label"], dropna=False)["word_count"]
            .sum()
            .reset_index(name="word_count")
        )
        cluster_level_totals = cluster_level.groupby("jurisdiction_level", dropna=False)["word_count"].sum().rename("level_word_total").reset_index()
        cluster_level = cluster_level.merge(cluster_level_totals, on="jurisdiction_level", how="left")
        cluster_level["share_within_level"] = cluster_level["word_count"] / cluster_level["level_word_total"].clip(lower=1)
        cluster_level = cluster_level.merge(
            overall_cluster[["cluster_id", "cluster_label", "overall_share"]],
            on=["cluster_id", "cluster_label"],
            how="left",
        )
        cluster_level["share_lift_vs_overall"] = cluster_level["share_within_level"] / cluster_level["overall_share"].clip(lower=1e-9)
        cluster_level["share_diff_vs_overall"] = cluster_level["share_within_level"] - cluster_level["overall_share"]
        top_region_contrasts = (
            cluster_region.sort_values(["jurisdiction_region", "share_lift_vs_overall", "word_count"], ascending=[True, False, False])
            .groupby("jurisdiction_region", dropna=False)
            .head(8)
            .reset_index(drop=True)
        )
        top_level_contrasts = (
            cluster_level.sort_values(["jurisdiction_level", "share_lift_vs_overall", "word_count"], ascending=[True, False, False])
            .groupby("jurisdiction_level", dropna=False)
            .head(8)
            .reset_index(drop=True)
        )
        cluster_doc_presence = (
            non_noise.groupby(["ordinance_doc_id", "cluster_id", "cluster_label"], dropna=False)
            .size()
            .reset_index(name="section_count")
        )
        cluster_doc_presence["present"] = 1.0
        cluster_catalog = cluster_doc_presence[["cluster_id", "cluster_label"]].drop_duplicates().reset_index(drop=True)
        doc_cluster_base = docs[
            ["ordinance_doc_id", "jurisdiction_region", "jurisdiction_level", "vintage_bin", "issue_year"]
        ].drop_duplicates().reset_index(drop=True)
        doc_cluster_base["_join_key"] = 1
        cluster_catalog["_join_key"] = 1
        doc_cluster_matrix = (
            doc_cluster_base.merge(cluster_catalog, on="_join_key", how="inner")
            .drop(columns="_join_key")
            .merge(
                cluster_doc_presence[["ordinance_doc_id", "cluster_id", "section_count", "present"]],
                on=["ordinance_doc_id", "cluster_id"],
                how="left",
            )
        )
        doc_cluster_matrix["section_count"] = doc_cluster_matrix["section_count"].fillna(0).astype(int)
        doc_cluster_matrix["present"] = doc_cluster_matrix["present"].fillna(0.0).astype(float)
        cluster_presence_region = _vintage_adjusted_presence_table(
            doc_cluster_matrix,
            group_col="jurisdiction_region",
            vintage_col="vintage_bin",
            category_cols=["cluster_id", "cluster_label"],
            value_col="present",
        )
        cluster_presence_level = _vintage_adjusted_presence_table(
            doc_cluster_matrix,
            group_col="jurisdiction_level",
            vintage_col="vintage_bin",
            category_cols=["cluster_id", "cluster_label"],
            value_col="present",
        )
        top_region_contrasts_adjusted = (
            cluster_presence_region.sort_values(
                ["jurisdiction_region", "presence_lift_vs_overall_vintage_adjusted", "presence_rate_vintage_adjusted"],
                ascending=[True, False, False],
            )
            .groupby("jurisdiction_region", dropna=False)
            .head(8)
            .reset_index(drop=True)
        )
        top_level_contrasts_adjusted = (
            cluster_presence_level.sort_values(
                ["jurisdiction_level", "presence_lift_vs_overall_vintage_adjusted", "presence_rate_vintage_adjusted"],
                ascending=[True, False, False],
            )
            .groupby("jurisdiction_level", dropna=False)
            .head(8)
            .reset_index(drop=True)
        )
        write_parquet(cluster_region, cluster_dir / f"section_cluster_shares_by_region__{sample}.parquet")
        write_parquet(cluster_level, cluster_dir / f"section_cluster_shares_by_level__{sample}.parquet")
        write_parquet(top_region_contrasts, cluster_dir / f"top_region_contrasts__{sample}.parquet")
        write_parquet(top_level_contrasts, cluster_dir / f"top_level_contrasts__{sample}.parquet")
        write_parquet(cluster_doc_presence, cluster_dir / f"section_cluster_doc_presence__{sample}.parquet")
        write_parquet(cluster_presence_region, cluster_dir / f"section_cluster_presence_by_region_vintage_adjusted__{sample}.parquet")
        write_parquet(cluster_presence_level, cluster_dir / f"section_cluster_presence_by_level_vintage_adjusted__{sample}.parquet")
        write_parquet(top_region_contrasts_adjusted, cluster_dir / f"top_region_contrasts_vintage_adjusted__{sample}.parquet")
        write_parquet(top_level_contrasts_adjusted, cluster_dir / f"top_level_contrasts_vintage_adjusted__{sample}.parquet")
        eligible_cluster_ids = (
            cluster_doc_presence.groupby(["cluster_id", "cluster_label"], dropna=False)["ordinance_doc_id"]
            .nunique()
            .reset_index(name="doc_presence_count")
        )
        eligible_cluster_ids = eligible_cluster_ids[
            eligible_cluster_ids["doc_presence_count"] >= max(6, int(round(len(doc_cluster_base) * 0.08)))
        ].sort_values(["doc_presence_count", "cluster_id"], ascending=[False, True])
        effect_frames: list[pd.DataFrame] = []
        for row in eligible_cluster_ids.to_dict(orient="records"):
            cluster_slice = (
                doc_cluster_matrix[doc_cluster_matrix["cluster_id"].astype(int) == int(row["cluster_id"])]
                .copy()
            )
            if cluster_slice["present"].nunique() < 2:
                continue
            effect_df = _fit_group_effect_model(
                cluster_slice,
                target_col="present",
                categorical_cols=["vintage_bin", "jurisdiction_level", "jurisdiction_region"],
                extra_id_cols=["cluster_id", "cluster_label"],
            )
            if effect_df.empty:
                continue
            effect_df["doc_presence_count"] = int(row["doc_presence_count"])
            effect_frames.append(effect_df)
        if effect_frames:
            cluster_presence_effects = pd.concat(effect_frames, ignore_index=True)
            cluster_presence_effects["abs_coefficient"] = cluster_presence_effects["coefficient"].abs()
            top_joint_effect_clusters = (
                cluster_presence_effects[
                    cluster_presence_effects["group_name"].isin(["jurisdiction_region", "jurisdiction_level"])
                ]
                .sort_values(["group_name", "abs_coefficient", "pvalue"], ascending=[True, False, True])
                .groupby("group_name", dropna=False)
                .head(15)
                .reset_index(drop=True)
            )
            write_parquet(cluster_presence_effects, cluster_dir / f"cluster_presence_region_level_effects__{sample}.parquet")
            write_parquet(top_joint_effect_clusters, cluster_dir / f"top_cluster_presence_region_level_effects__{sample}.parquet")
        section_cluster_summary = {
            "available": True,
            "cluster_count": int(non_noise["cluster_id"].astype(int).nunique()),
            "region_contrast_rows": int(len(top_region_contrasts)),
            "region_contrast_rows_vintage_adjusted": int(len(top_region_contrasts_adjusted)),
        }
        comparison_rows: list[dict[str, Any]] = []
        cluster_quality_rows: list[dict[str, Any]] = []
        surface_specs = [
            ("openai", run_root / "clusters" / SECTION_CLUSTER_EXPERIMENT_NAME / "assignments.parquet"),
            ("local_baseline", run_root / "clusters" / SECTION_LOCAL_CLUSTER_EXPERIMENT_NAME / "assignments.parquet"),
        ]
        for surface_name, assign_path in surface_specs:
            if not assign_path.is_file():
                continue
            surf = pd.read_parquet(assign_path).copy()
            surf = surf.merge(
                sections[["section_id", "word_count", "architecture_anchor", "primary_motive"]],
                on="section_id",
                how="left",
                suffixes=("", "_section"),
            )
            surf["word_count"] = pd.to_numeric(surf["word_count"], errors="coerce").fillna(0.0)
            cluster_count = int(surf.loc[surf["cluster_id"].astype(int) >= 0, "cluster_id"].astype(int).nunique())
            noise_rate = float((surf["cluster_id"].astype(int) < 0).mean())
            for cluster_id, group in surf.groupby("cluster_id", dropna=False):
                if int(cluster_id) < 0:
                    continue
                motive_counts = group.groupby("primary_motive", dropna=False)["word_count"].sum().sort_values(ascending=False)
                anchor_counts = group.groupby("architecture_anchor", dropna=False)["word_count"].sum().sort_values(ascending=False)
                motive_total = float(motive_counts.sum()) if not motive_counts.empty else 0.0
                anchor_total = float(anchor_counts.sum()) if not anchor_counts.empty else 0.0
                cluster_quality_rows.append(
                    {
                        "surface_name": surface_name,
                        "cluster_id": int(cluster_id),
                        "cluster_label": clean_optional_str(group["cluster_label"].iloc[0]),
                        "section_count": int(group["section_id"].nunique()),
                        "top_motive": clean_optional_str(motive_counts.index[0]) if not motive_counts.empty else "",
                        "top_motive_share": float(motive_counts.iloc[0] / motive_total) if motive_total > 0 else 0.0,
                        "motive_entropy": _normalized_entropy(motive_counts.to_numpy()) if not motive_counts.empty else 0.0,
                        "top_architecture_anchor": clean_optional_str(anchor_counts.index[0]) if not anchor_counts.empty else "",
                        "top_architecture_share": float(anchor_counts.iloc[0] / anchor_total) if anchor_total > 0 else 0.0,
                        "architecture_entropy": _normalized_entropy(anchor_counts.to_numpy()) if not anchor_counts.empty else 0.0,
                    }
                )
            quality_df = pd.DataFrame.from_records([row for row in cluster_quality_rows if row["surface_name"] == surface_name])
            comparison_rows.append(
                {
                    "surface_name": surface_name,
                    "cluster_count": int(cluster_count),
                    "noise_rate": float(noise_rate),
                    "mean_top_motive_share": float(quality_df["top_motive_share"].mean()) if not quality_df.empty else 0.0,
                    "mean_motive_entropy": float(quality_df["motive_entropy"].mean()) if not quality_df.empty else 0.0,
                    "mean_top_architecture_share": float(quality_df["top_architecture_share"].mean()) if not quality_df.empty else 0.0,
                    "mean_architecture_entropy": float(quality_df["architecture_entropy"].mean()) if not quality_df.empty else 0.0,
                }
            )
        comparison_df = pd.DataFrame.from_records(comparison_rows)
        cluster_quality_df = pd.DataFrame.from_records(cluster_quality_rows)
        write_parquet(comparison_df, embedding_cmp_dir / f"surface_comparison__{sample}.parquet")
        write_parquet(cluster_quality_df, embedding_cmp_dir / f"surface_cluster_quality__{sample}.parquet")
        if not comparison_df.empty:
            scored = comparison_df.copy()
            scored["surface_score"] = (
                scored["mean_top_motive_share"]
                + scored["mean_top_architecture_share"]
                - scored["mean_motive_entropy"]
                - scored["mean_architecture_entropy"]
                - scored["noise_rate"]
            )
            winner_row = scored.sort_values(["surface_score", "mean_top_motive_share"], ascending=[False, False]).iloc[0]
            embedding_surface_summary = {
                "available": True,
                "surface_count": int(len(comparison_df)),
                "winner": clean_optional_str(winner_row["surface_name"]),
                "winner_score": float(winner_row["surface_score"]),
            }

    architecture_family_summary = _build_architecture_family_outputs(
        docs=docs,
        sections=sections,
        section_assign=section_assign,
        doc_arch_df=doc_arch_df,
        architecture_dir=architecture_dir,
        sample=sample,
    )

    # Complexity summaries from sections, clauses, and available cluster structure.
    section_embedding_stats = (
        section_meta.groupby("ordinance_doc_id", dropna=False)
        .agg(
            section_count_from_embedding=("section_id", "nunique"),
            multi_segment_section_count=("embedding_segment_count", lambda s: int((pd.Series(s).fillna(0).astype(int) > 1).sum())),
            max_embedding_segment_count=("embedding_segment_count", "max"),
        )
        .reset_index()
    )
    section_motive_rows: list[dict[str, Any]] = []
    for ordinance_doc_id, group in sections.groupby("ordinance_doc_id", dropna=False):
        motive_counts = group["primary_motive"].astype(str).value_counts()
        section_motive_rows.append(
            {
                "ordinance_doc_id": clean_optional_str(ordinance_doc_id),
                "section_count": int(group["section_id"].nunique()),
                "mean_section_word_count": float(group["word_count"].mean()) if not group.empty else 0.0,
                "max_section_word_count": int(group["word_count"].max()) if not group.empty else 0,
                "distinct_section_motive_count": int(motive_counts.index.difference([""]).nunique()),
                "section_motive_entropy": _normalized_entropy(motive_counts.to_numpy()),
            }
        )
    section_motive_df = pd.DataFrame.from_records(section_motive_rows)
    clause_modality = clauses.pivot_table(
        index="ordinance_doc_id",
        columns="modality",
        values="clause_id",
        aggfunc="count",
        fill_value=0,
    )
    clause_modality = clause_modality.rename(columns={c: f"{str(c)}_clause_count" for c in clause_modality.columns})
    clause_doc = (
        clauses.groupby("ordinance_doc_id", dropna=False)
        .agg(
            clause_count=("clause_id", "nunique"),
            distinct_clause_motive_count=("motive", lambda s: int(pd.Series(s).astype(str).replace("", np.nan).dropna().nunique())),
        )
        .reset_index()
        .merge(clause_modality.reset_index(), on="ordinance_doc_id", how="left")
    )
    for modality in ("must", "may", "must_not", "definition", "other"):
        col = f"{modality}_clause_count"
        if col not in clause_doc.columns:
            clause_doc[col] = 0
    clause_doc["clause_modality_entropy"] = (
        clause_doc[[f"{m}_clause_count" for m in ("must", "may", "must_not", "definition", "other")]]
        .apply(lambda row: _normalized_entropy(row.to_numpy()), axis=1)
    )
    for modality in ("must", "may", "must_not", "definition"):
        clause_doc[f"{modality}_share"] = clause_doc[f"{modality}_clause_count"] / clause_doc["clause_count"].clip(lower=1)

    cluster_complexity = pd.DataFrame(columns=["ordinance_doc_id", "distinct_section_cluster_count", "section_cluster_entropy"])
    if not section_assign.empty:
        cluster_complexity_rows: list[dict[str, Any]] = []
        for ordinance_doc_id, group in section_assign.groupby("ordinance_doc_id", dropna=False):
            non_noise_group = group[group["cluster_id"].astype(int) >= 0].copy()
            cluster_counts = non_noise_group["cluster_id"].astype(int).value_counts()
            cluster_complexity_rows.append(
                {
                    "ordinance_doc_id": clean_optional_str(ordinance_doc_id),
                    "distinct_section_cluster_count": int(len(cluster_counts)),
                    "section_cluster_entropy": _normalized_entropy(cluster_counts.to_numpy()),
                }
            )
        cluster_complexity = pd.DataFrame.from_records(cluster_complexity_rows)

    complexity_df = (
        docs[
            [
                "ordinance_doc_id",
                "issue_id",
                "issue_date",
                "issue_year",
                "vintage_bin",
                "census_id_pid6",
                "jurisdiction_region",
                "jurisdiction_level",
                "word_count",
                "char_count",
                "length_quartile",
            ]
        ]
        .merge(section_motive_df, on="ordinance_doc_id", how="left")
        .merge(section_embedding_stats, on="ordinance_doc_id", how="left")
        .merge(clause_doc, on="ordinance_doc_id", how="left")
        .merge(cluster_complexity, on="ordinance_doc_id", how="left")
    )
    complexity_df["multi_segment_section_share"] = complexity_df["multi_segment_section_count"].fillna(0) / complexity_df["section_count"].clip(lower=1)
    complexity_df["clauses_per_1000_words"] = complexity_df["clause_count"].fillna(0) / (complexity_df["word_count"].clip(lower=1) / 1000.0)
    complexity_df["avg_clauses_per_section"] = complexity_df["clause_count"].fillna(0) / complexity_df["section_count"].clip(lower=1)
    complexity_metric_cols = [
        "word_count",
        "section_count",
        "clause_count",
        "clauses_per_1000_words",
        "distinct_section_motive_count",
        "section_motive_entropy",
        "clause_modality_entropy",
        "multi_segment_section_share",
    ]
    if "distinct_section_cluster_count" in complexity_df.columns:
        complexity_metric_cols.extend(["distinct_section_cluster_count", "section_cluster_entropy"])
    for col in complexity_metric_cols:
        if col not in complexity_df.columns:
            complexity_df[col] = 0.0
        complexity_df[col] = pd.to_numeric(complexity_df[col], errors="coerce").fillna(0.0)
    z_parts: list[pd.Series] = []
    for col in complexity_metric_cols:
        series = complexity_df[col].astype(float)
        std = float(series.std(ddof=0))
        z_parts.append((series - float(series.mean())) / std if std > 0.0 else pd.Series(0.0, index=series.index))
    z_frame = pd.concat(z_parts, axis=1)
    complexity_df["complexity_index_exploratory"] = z_frame.mean(axis=1)
    complexity_by_vintage = (
        complexity_df.groupby("vintage_bin", dropna=False)
        .agg(
            ordinance_count=("ordinance_doc_id", "nunique"),
            mean_complexity_index=("complexity_index_exploratory", "mean"),
            mean_clause_count=("clause_count", "mean"),
            mean_section_count=("section_count", "mean"),
            mean_clauses_per_1000_words=("clauses_per_1000_words", "mean"),
        )
        .reset_index()
        .sort_values("vintage_bin")
        .reset_index(drop=True)
    )
    complexity_region_adjusted = _vintage_adjusted_mean_table(
        complexity_df,
        group_col="jurisdiction_region",
        vintage_col="vintage_bin",
        value_cols=[
            "complexity_index_exploratory",
            "section_count",
            "clause_count",
            "clauses_per_1000_words",
            "section_motive_entropy",
            "clause_modality_entropy",
            "multi_segment_section_share",
        ],
    )
    complexity_level_adjusted = _vintage_adjusted_mean_table(
        complexity_df,
        group_col="jurisdiction_level",
        vintage_col="vintage_bin",
        value_cols=[
            "complexity_index_exploratory",
            "section_count",
            "clause_count",
            "clauses_per_1000_words",
            "section_motive_entropy",
            "clause_modality_entropy",
            "multi_segment_section_share",
        ],
    )
    write_parquet(complexity_df.sort_values("ordinance_doc_id").reset_index(drop=True), complexity_dir / f"complexity_by_doc__{sample}.parquet")
    write_parquet(complexity_by_vintage, complexity_dir / f"complexity_by_vintage__{sample}.parquet")
    write_parquet(complexity_region_adjusted, complexity_dir / f"complexity_by_region_vintage_adjusted__{sample}.parquet")
    write_parquet(complexity_level_adjusted, complexity_dir / f"complexity_by_level_vintage_adjusted__{sample}.parquet")
    complexity_effects = _fit_group_effect_model(
        complexity_df,
        target_col="complexity_index_exploratory",
        categorical_cols=["vintage_bin", "jurisdiction_level", "jurisdiction_region"],
    )
    if not complexity_effects.empty:
        complexity_effects["abs_coefficient"] = complexity_effects["coefficient"].abs()
        top_complexity_effects = (
            complexity_effects[complexity_effects["group_name"].isin(["jurisdiction_region", "jurisdiction_level"])]
            .sort_values(["group_name", "abs_coefficient", "pvalue"], ascending=[True, False, True])
            .groupby("group_name", dropna=False)
            .head(10)
            .reset_index(drop=True)
        )
    else:
        top_complexity_effects = pd.DataFrame()
    write_parquet(complexity_effects, complexity_dir / f"complexity_region_level_effects__{sample}.parquet")
    write_parquet(top_complexity_effects, complexity_dir / f"top_complexity_region_level_effects__{sample}.parquet")
    write_parquet(
        complexity_df.sort_values(["complexity_index_exploratory", "ordinance_doc_id"], ascending=[False, True]).head(25).reset_index(drop=True),
        complexity_dir / f"top_complexity_ordinances__{sample}.parquet",
    )
    coherence_complexity = coherence_df.merge(
        complexity_df[
            [
                "ordinance_doc_id",
                "complexity_index_exploratory",
                "clause_count",
                "section_count",
                "section_motive_entropy",
                "clause_modality_entropy",
            ]
        ].rename(columns={"section_count": "complexity_section_count"}),
        on="ordinance_doc_id",
        how="inner",
    )
    if not coherence_complexity.empty:
        coherence_complexity["complexity_quartile"] = pd.qcut(
            coherence_complexity["complexity_index_exploratory"],
            4,
            labels=["Q1", "Q2", "Q3", "Q4"],
            duplicates="drop",
        )
        coherence_complexity_quartiles = (
            coherence_complexity.groupby("complexity_quartile", observed=False)
            .agg(
                ordinance_count=("ordinance_doc_id", "count"),
                mean_alignment_share=("purpose_alignment_non_other_word_share", "mean"),
                median_alignment_share=("purpose_alignment_non_other_word_share", "median"),
                mean_clause_count=("clause_count", "mean"),
                mean_section_count=("complexity_section_count", "mean"),
                mean_section_motive_entropy=("section_motive_entropy", "mean"),
                mean_clause_modality_entropy=("clause_modality_entropy", "mean"),
            )
            .reset_index()
        )
        coherence_complexity_correlation = float(
            coherence_complexity[["complexity_index_exploratory", "purpose_alignment_non_other_word_share"]]
            .corr()
            .iloc[0, 1]
        )
    else:
        coherence_complexity_quartiles = pd.DataFrame(
            columns=[
                "complexity_quartile",
                "ordinance_count",
                "mean_alignment_share",
                "median_alignment_share",
                "mean_clause_count",
                "mean_section_count",
                "mean_section_motive_entropy",
                "mean_clause_modality_entropy",
            ]
        )
        coherence_complexity_correlation = 0.0
    write_parquet(coherence_complexity, coherence_dir / f"coherence_complexity_by_doc__{sample}.parquet")
    write_parquet(coherence_complexity_quartiles, coherence_dir / f"coherence_complexity_quartiles__{sample}.parquet")
    component_corr_rows: list[dict[str, Any]] = []
    if not coherence_complexity.empty:
        component_cols = [
            "complexity_index_exploratory",
            "clause_count",
            "complexity_section_count",
            "section_motive_entropy",
            "clause_modality_entropy",
        ]
        for col in component_cols:
            if col not in coherence_complexity.columns:
                continue
            pair = coherence_complexity[[col, "purpose_alignment_non_other_word_share"]].dropna()
            corr = float(pair.corr().iloc[0, 1]) if len(pair) >= 3 else np.nan
            component_corr_rows.append(
                {
                    "metric_name": col,
                    "row_count": int(len(pair)),
                    "alignment_correlation": corr,
                }
            )
    component_corr_df = pd.DataFrame.from_records(component_corr_rows)
    write_parquet(component_corr_df, coherence_dir / f"coherence_complexity_component_correlations__{sample}.parquet")

    model_input = coherence_complexity[
        [
            "ordinance_doc_id",
            "purpose_alignment_non_other_word_share",
            "purpose_section_filtered_match_flag",
            "complexity_index_exploratory",
            "vintage_bin",
            "jurisdiction_level",
            "jurisdiction_region",
        ]
    ].drop_duplicates("ordinance_doc_id")
    alignment_models, alignment_residuals = _fit_ols_models(
        model_input,
        target_col="purpose_alignment_non_other_word_share",
        complexity_col="complexity_index_exploratory",
        categorical_cols=["vintage_bin", "jurisdiction_level", "jurisdiction_region"],
    )
    match_models, match_residuals = _fit_ols_models(
        model_input.rename(columns={"purpose_section_filtered_match_flag": "filtered_match_flag"}),
        target_col="filtered_match_flag",
        complexity_col="complexity_index_exploratory",
        categorical_cols=["vintage_bin", "jurisdiction_level", "jurisdiction_region"],
    )
    write_parquet(alignment_models, coherence_dir / f"complexity_alignment_models__{sample}.parquet")
    write_parquet(alignment_residuals, coherence_dir / f"complexity_alignment_residuals__{sample}.parquet")
    write_parquet(match_models, coherence_dir / f"complexity_match_models__{sample}.parquet")
    write_parquet(match_residuals, coherence_dir / f"complexity_match_residuals__{sample}.parquet")
    unexpectedly_high_alignment = (
        alignment_residuals[alignment_residuals["model_name"].astype(str) == "vintage_level"]
        .sort_values(["residual", "ordinance_doc_id"], ascending=[False, True])
        .head(15)
        .reset_index(drop=True)
        if not alignment_residuals.empty
        else pd.DataFrame()
    )
    unexpectedly_low_alignment = (
        alignment_residuals[alignment_residuals["model_name"].astype(str) == "vintage_level"]
        .sort_values(["residual", "ordinance_doc_id"], ascending=[True, True])
        .head(15)
        .reset_index(drop=True)
        if not alignment_residuals.empty
        else pd.DataFrame()
    )
    write_parquet(unexpectedly_high_alignment, coherence_dir / f"coherence_positive_residuals__{sample}.parquet")
    write_parquet(unexpectedly_low_alignment, coherence_dir / f"coherence_negative_residuals__{sample}.parquet")
    for direction, df_case in (("positive", unexpectedly_high_alignment), ("negative", unexpectedly_low_alignment)):
        lines = [f"# Coherence residual cases: {direction}", ""]
        if df_case.empty:
            lines.append("No cases available.")
        else:
            for row in df_case.to_dict(orient="records"):
                lines.append(
                    f"- `{clean_optional_str(row['ordinance_doc_id'])}` | residual={float(row['residual']):+.3f} | "
                    f"fitted={float(row['fitted_value']):.3f} | observed={float(row['purpose_alignment_non_other_word_share']):.3f} | "
                    f"complexity={float(row['complexity_index_exploratory']):+.3f} | "
                    f"vintage=`{clean_optional_str(row['vintage_bin'])}` | level=`{clean_optional_str(row['jurisdiction_level'])}` | "
                    f"region=`{clean_optional_str(row['jurisdiction_region'])}`"
                )
        write_text(coherence_dir / f"coherence_{direction}_residuals__{sample}.md", "\n".join(lines))
    adjusted_alignment_row = (
        alignment_models.loc[alignment_models["model_name"].astype(str) == "vintage_level"].head(1)
        if not alignment_models.empty
        else pd.DataFrame()
    )
    adjusted_alignment_coef = float(adjusted_alignment_row["complexity_coef"].iloc[0]) if not adjusted_alignment_row.empty else np.nan
    adjusted_alignment_pvalue = float(adjusted_alignment_row["complexity_pvalue"].iloc[0]) if not adjusted_alignment_row.empty else np.nan

    region_packets: list[dict[str, Any]] = []
    if not top_region_contrasts.empty:
        for region, row in by_region.groupby("jurisdiction_region", dropna=False):
            region_text = clean_optional_str(region)
            cluster_subset = top_region_contrasts[top_region_contrasts["jurisdiction_region"].astype(str) == region_text].head(5)
            adjusted_cluster_subset = top_region_contrasts_adjusted[top_region_contrasts_adjusted["jurisdiction_region"].astype(str) == region_text].head(5)
            motive_subset = opening_motive_region[opening_motive_region["jurisdiction_region"].astype(str) == region_text].head(5)
            prototype_subset = region_prototypes[region_prototypes["jurisdiction_region"].astype(str) == region_text].head(3)
            packet = {
                "jurisdiction_region": region_text,
                "coherence_metrics": row.to_dict(orient="records")[0],
                "top_section_cluster_lifts": cluster_subset.to_dict(orient="records"),
                "top_section_cluster_lifts_vintage_adjusted": adjusted_cluster_subset.to_dict(orient="records"),
                "opening_motives": motive_subset.to_dict(orient="records"),
                "prototype_ordinances": prototype_subset.to_dict(orient="records"),
            }
            region_packets.append(packet)
            lines = [f"# Region contrast packet: {region_text}", ""]
            metrics = packet["coherence_metrics"]
            lines.extend(
                [
                    "## Coherence",
                    "",
                    f"- ordinance_count: {int(metrics['ordinance_count'])}",
                    f"- exact_match_rate: {float(metrics['exact_match_rate']):.3f}",
                    f"- mean_non_other_alignment_share: {float(metrics['mean_non_other_alignment_share']):.3f}",
                    f"- mean_distinct_section_motives: {float(metrics['mean_distinct_section_motives']):.2f}",
                    "",
                    "## Opening Motives",
                    "",
                ]
            )
            for item in packet["opening_motives"]:
                lines.append(f"- {clean_optional_str(item['first_substantive_section_motive'])}: {int(item['ordinance_count'])}")
            lines.extend(["", "## Top Section Cluster Lifts", ""])
            for item in packet["top_section_cluster_lifts"]:
                lines.append(
                    f"- {clean_optional_str(item['cluster_label'])}: lift={float(item['share_lift_vs_overall']):.2f}, "
                    f"share={float(item['share_within_region']):.3f}"
                )
            if packet["top_section_cluster_lifts_vintage_adjusted"]:
                lines.extend(["", "## Top Section Cluster Lifts (Vintage Adjusted)", ""])
                for item in packet["top_section_cluster_lifts_vintage_adjusted"]:
                    lines.append(
                        f"- {clean_optional_str(item['cluster_label'])}: adjusted_lift={float(item['presence_lift_vs_overall_vintage_adjusted']):.2f}, "
                        f"adjusted_presence={float(item['presence_rate_vintage_adjusted']):.3f}, "
                        f"coverage={float(item['vintage_weight_coverage']):.2f}"
                    )
            lines.extend(["", "## Prototype Ordinances", ""])
            for item in packet["prototype_ordinances"]:
                lines.append(
                    f"- `{clean_optional_str(item['ordinance_doc_id'])}` ({clean_optional_str(item['jurisdiction_level'])}) "
                    f"similarity={float(item['cosine_similarity']):.3f}"
                )
            write_text(contrastive_dir / f"region_packet__{region_text}__{sample}.md", "\n".join(lines))
    write_text(contrastive_dir / f"region_contrast_source__{sample}.json", json.dumps(region_packets, indent=2, ensure_ascii=False))

    # Clause-side governance style analysis only when enriched clauses exist.
    governance_summary: dict[str, Any] = {"available": False, "doc_count": 0}
    instrument_summary: dict[str, Any] = {"available": False, "doc_count": 0, "instrument_type_count": 0}
    regulatory_style_summary: dict[str, Any] = {"available": False, "family_count": 0, "top_k": 0, "silhouette_score": 0.0}
    clause_cluster_summary: dict[str, Any] = {"available": False, "cluster_count": 0, "region_contrast_rows": 0}
    style_coherence_summary: dict[str, Any] = {
        "available": False,
        "family_count": 0,
        "row_count": 0,
        "most_coherent_family_label": "",
        "most_fragmented_family_label": "",
    }
    style_policy_bridge_summary: dict[str, Any] = {"available": False, "row_count": 0}
    if not enriched.empty:
        enriched = enriched.merge(
            docs[["ordinance_doc_id", "jurisdiction_region", "jurisdiction_level"]],
            on="ordinance_doc_id",
            how="left",
        )
        for col in [
            "actor",
            "action",
            "regulated_object",
            "condition",
            "exception",
            "threshold_or_quantity",
            "spatial_scope",
            "evidence_quote",
            "enrichment_notes",
        ]:
            if col in enriched.columns:
                enriched[col] = (
                    enriched[col]
                    .astype(str)
                    .replace({"None": "", "nan": "", "<NA>": ""})
                    .fillna("")
                )
        enriched["operative_flag"] = enriched["operative_flag"].fillna(0).astype(int)
        enriched["has_exception"] = enriched["exception"].astype(str).str.strip().ne("").astype(int)
        enriched["has_condition"] = enriched["condition"].astype(str).str.strip().ne("").astype(int)
        enriched["has_threshold"] = enriched["threshold_or_quantity"].astype(str).str.strip().ne("").astype(int)
        enriched["discretion_flag"] = enriched["instrument_type"].astype(str).isin(DISCRETION_INSTRUMENTS).astype(int)
        enriched["structural_flag"] = enriched["instrument_type"].astype(str).isin(STRUCTURAL_INSTRUMENTS).astype(int)
        enriched["direct_rule_flag"] = (
            (enriched["operative_flag"].astype(int) == 1)
            & (enriched["discretion_flag"].astype(int) == 0)
            & (enriched["has_exception"].astype(int) == 0)
            & (enriched["structural_flag"].astype(int) == 0)
        ).astype(int)
        enriched["exception_or_condition_flag"] = (
            (enriched["has_exception"].astype(int) == 1) | (enriched["has_condition"].astype(int) == 1)
        ).astype(int)
        enriched["operative_direct_rule_flag"] = (
            enriched["operative_flag"].astype(int) * enriched["direct_rule_flag"].astype(int)
        ).astype(int)
        enriched["operative_discretion_flag"] = (
            enriched["operative_flag"].astype(int) * enriched["discretion_flag"].astype(int)
        ).astype(int)
        enriched["operative_exception_or_condition_flag"] = (
            enriched["operative_flag"].astype(int) * enriched["exception_or_condition_flag"].astype(int)
        ).astype(int)
        enriched["operative_threshold_flag"] = (
            enriched["operative_flag"].astype(int) * enriched["has_threshold"].astype(int)
        ).astype(int)
        by_doc = (
            enriched.groupby("ordinance_doc_id", dropna=False)
            .agg(
                clause_count=("clause_id", "count"),
                operative_clause_count=("operative_flag", "sum"),
                direct_rule_clause_count=("operative_direct_rule_flag", "sum"),
                discretion_clause_count=("operative_discretion_flag", "sum"),
                exception_or_condition_clause_count=("operative_exception_or_condition_flag", "sum"),
                threshold_clause_count=("operative_threshold_flag", "sum"),
            )
            .reset_index()
        )
        by_doc = by_doc.merge(
            docs[["ordinance_doc_id", "jurisdiction_region", "jurisdiction_level", "census_id_pid6"]],
            on="ordinance_doc_id",
            how="left",
        )
        for stem in ["direct_rule", "discretion", "exception_or_condition", "threshold"]:
            by_doc[f"{stem}_share_of_operative"] = by_doc[f"{stem}_clause_count"] / by_doc["operative_clause_count"].clip(lower=1)
        by_region_gov = (
            by_doc.groupby("jurisdiction_region", dropna=False)
            .agg(
                ordinance_count=("ordinance_doc_id", "nunique"),
                mean_direct_rule_share=("direct_rule_share_of_operative", "mean"),
                mean_discretion_share=("discretion_share_of_operative", "mean"),
                mean_exception_or_condition_share=("exception_or_condition_share_of_operative", "mean"),
                mean_threshold_share=("threshold_share_of_operative", "mean"),
            )
            .reset_index()
        )
        governance_extremes = {
            "direct_rule": by_doc.sort_values(["direct_rule_share_of_operative", "operative_clause_count"], ascending=[False, False]).head(15).reset_index(drop=True),
            "discretion": by_doc.sort_values(["discretion_share_of_operative", "operative_clause_count"], ascending=[False, False]).head(15).reset_index(drop=True),
            "exception_or_condition": by_doc.sort_values(["exception_or_condition_share_of_operative", "operative_clause_count"], ascending=[False, False]).head(15).reset_index(drop=True),
            "threshold": by_doc.sort_values(["threshold_share_of_operative", "operative_clause_count"], ascending=[False, False]).head(15).reset_index(drop=True),
        }
        write_parquet(by_doc, governance_dir / f"governance_style_by_doc__{sample}.parquet")
        write_parquet(by_region_gov, governance_dir / f"governance_style_by_region__{sample}.parquet")
        for name, df_extreme in governance_extremes.items():
            write_parquet(df_extreme, governance_dir / f"governance_style_extreme_{name}__{sample}.parquet")
        governance_lines = ["# Governance style extremes", "", f"- documents covered: {int(len(by_doc))}", ""]
        for name, df_extreme in governance_extremes.items():
            pretty = name.replace("_", " ")
            governance_lines.extend([f"## Highest {pretty} share", ""])
            for row in df_extreme.head(8).to_dict(orient="records"):
                governance_lines.append(
                    f"- `{clean_optional_str(row['ordinance_doc_id'])}` | region=`{clean_optional_str(row['jurisdiction_region'])}` "
                    f"| level=`{clean_optional_str(row['jurisdiction_level'])}` | share={float(row[f'{name}_share_of_operative']):.3f} "
                    f"| operative_clauses={int(row['operative_clause_count'])}"
                )
            governance_lines.append("")
        write_text(governance_dir / f"governance_style_extremes__{sample}.md", "\n".join(governance_lines))

        operative = enriched.loc[enriched["operative_flag"].astype(int) == 1].copy()
        instrument_catalog = (
            operative["instrument_type"]
            .astype(str)
            .replace("", np.nan)
            .dropna()
            .sort_values()
            .unique()
            .tolist()
        )
        instrument_counts = (
            operative.groupby(["ordinance_doc_id", "instrument_type"], dropna=False)["clause_id"]
            .count()
            .reset_index(name="instrument_clause_count")
        )
        instrument_doc_rows: list[dict[str, Any]] = []
        for ordinance_doc_id, group in instrument_counts.groupby("ordinance_doc_id", dropna=False):
            counts = group["instrument_clause_count"].to_numpy()
            total = float(counts.sum()) if len(counts) else 0.0
            ordered = group.sort_values(["instrument_clause_count", "instrument_type"], ascending=[False, True]).reset_index(drop=True)
            instrument_doc_rows.append(
                {
                    "ordinance_doc_id": clean_optional_str(ordinance_doc_id),
                    "distinct_instrument_type_count": int(group["instrument_type"].astype(str).nunique()),
                    "instrument_entropy": _normalized_entropy(counts),
                    "top_instrument_type": clean_optional_str(ordered.iloc[0]["instrument_type"]) if not ordered.empty else "",
                    "top_instrument_share": float(ordered.iloc[0]["instrument_clause_count"] / total) if total > 0 and not ordered.empty else 0.0,
                    "operative_clause_count": int(total),
                }
            )
        instrument_by_doc = pd.DataFrame.from_records(instrument_doc_rows)
        instrument_by_doc = instrument_by_doc.merge(
            docs[["ordinance_doc_id", "jurisdiction_region", "jurisdiction_level", "vintage_bin"]],
            on="ordinance_doc_id",
            how="left",
        )
        instrument_presence = (
            instrument_counts.assign(present=1.0)
            .merge(docs[["ordinance_doc_id", "jurisdiction_region", "jurisdiction_level", "vintage_bin"]], on="ordinance_doc_id", how="left")
        )
        motive_meta = coherence_df[["ordinance_doc_id", "doc_primary_motive", "vintage_bin"]].copy()
        motive_meta["doc_primary_motive"] = motive_meta["doc_primary_motive"].astype(str).replace({"": "unclear"}).fillna("unclear")
        instrument_presence_by_motive = (
            instrument_counts.assign(present=1.0)
            .merge(motive_meta, on="ordinance_doc_id", how="left")
        )
        instrument_by_region = (
            instrument_by_doc.groupby("jurisdiction_region", dropna=False)
            .agg(
                ordinance_count=("ordinance_doc_id", "nunique"),
                mean_distinct_instrument_type_count=("distinct_instrument_type_count", "mean"),
                mean_instrument_entropy=("instrument_entropy", "mean"),
                mean_top_instrument_share=("top_instrument_share", "mean"),
            )
            .reset_index()
            .sort_values("jurisdiction_region")
            .reset_index(drop=True)
        )
        instrument_region_presence = _vintage_adjusted_presence_table(
            instrument_presence,
            group_col="jurisdiction_region",
            vintage_col="vintage_bin",
            category_cols=["instrument_type"],
            value_col="present",
        )
        instrument_motive_presence = _vintage_adjusted_presence_table(
            instrument_presence_by_motive,
            group_col="doc_primary_motive",
            vintage_col="vintage_bin",
            category_cols=["instrument_type"],
            value_col="present",
        )
        instrument_share = instrument_counts.merge(
            instrument_by_doc[["ordinance_doc_id", "operative_clause_count"]],
            on="ordinance_doc_id",
            how="left",
        )
        instrument_share["instrument_clause_share"] = (
            instrument_share["instrument_clause_count"] / instrument_share["operative_clause_count"].clip(lower=1.0)
        )
        covered_docs = instrument_by_doc[
            ["ordinance_doc_id", "jurisdiction_region", "jurisdiction_level", "vintage_bin"]
        ].drop_duplicates()
        covered_docs = covered_docs.merge(
            motive_meta[["ordinance_doc_id", "doc_primary_motive"]].drop_duplicates(),
            on="ordinance_doc_id",
            how="left",
        )
        if instrument_catalog:
            instrument_catalog_df = pd.DataFrame({"instrument_type": instrument_catalog})
            share_grid = (
                covered_docs.assign(_merge_key=1)
                .merge(instrument_catalog_df.assign(_merge_key=1), on="_merge_key", how="inner")
                .drop(columns="_merge_key")
                .merge(
                    instrument_share[["ordinance_doc_id", "instrument_type", "instrument_clause_share"]],
                    on=["ordinance_doc_id", "instrument_type"],
                    how="left",
                )
            )
            share_grid["instrument_clause_share"] = pd.to_numeric(
                share_grid["instrument_clause_share"], errors="coerce"
            ).fillna(0.0)
        else:
            share_grid = pd.DataFrame(
                columns=[
                    "ordinance_doc_id",
                    "jurisdiction_region",
                    "jurisdiction_level",
                    "vintage_bin",
                    "doc_primary_motive",
                    "instrument_type",
                    "instrument_clause_share",
                ]
            )
        instrument_region_share = _vintage_adjusted_category_mean_table(
            share_grid,
            group_col="jurisdiction_region",
            vintage_col="vintage_bin",
            category_cols=["instrument_type"],
            value_col="instrument_clause_share",
        )
        instrument_motive_share = _vintage_adjusted_category_mean_table(
            share_grid,
            group_col="doc_primary_motive",
            vintage_col="vintage_bin",
            category_cols=["instrument_type"],
            value_col="instrument_clause_share",
        )
        top_region_instruments = (
            instrument_region_presence.sort_values(
                ["jurisdiction_region", "presence_lift_vs_overall_vintage_adjusted", "presence_rate_vintage_adjusted"],
                ascending=[True, False, False],
            )
            .groupby("jurisdiction_region", dropna=False)
            .head(10)
            .reset_index(drop=True)
        )
        top_motive_instruments = (
            instrument_motive_presence.sort_values(
                ["doc_primary_motive", "presence_lift_vs_overall_vintage_adjusted", "presence_rate_vintage_adjusted"],
                ascending=[True, False, False],
            )
            .groupby("doc_primary_motive", dropna=False)
            .head(10)
            .reset_index(drop=True)
        )
        top_region_instrument_shares = (
            instrument_region_share.sort_values(
                ["jurisdiction_region", "instrument_clause_share_lift_vs_overall_vintage_adjusted", "instrument_clause_share__vintage_adjusted"],
                ascending=[True, False, False],
            )
            .groupby("jurisdiction_region", dropna=False)
            .head(10)
            .reset_index(drop=True)
        )
        top_motive_instrument_shares = (
            instrument_motive_share.sort_values(
                ["doc_primary_motive", "instrument_clause_share_lift_vs_overall_vintage_adjusted", "instrument_clause_share__vintage_adjusted"],
                ascending=[True, False, False],
            )
            .groupby("doc_primary_motive", dropna=False)
            .head(10)
            .reset_index(drop=True)
        )
        instrument_extremes = {
            "diversity": instrument_by_doc.sort_values(["distinct_instrument_type_count", "instrument_entropy"], ascending=[False, False]).head(15).reset_index(drop=True),
            "entropy": instrument_by_doc.sort_values(["instrument_entropy", "distinct_instrument_type_count"], ascending=[False, False]).head(15).reset_index(drop=True),
            "concentration": instrument_by_doc.sort_values(["top_instrument_share", "operative_clause_count"], ascending=[False, False]).head(15).reset_index(drop=True),
        }
        write_parquet(instrument_by_doc, instruments_dir / f"instrument_complexity_by_doc__{sample}.parquet")
        write_parquet(instrument_by_region, instruments_dir / f"instrument_complexity_by_region__{sample}.parquet")
        write_parquet(instrument_region_presence, instruments_dir / f"instrument_presence_by_region_vintage_adjusted__{sample}.parquet")
        write_parquet(top_region_instruments, instruments_dir / f"top_region_instrument_lifts__{sample}.parquet")
        write_parquet(instrument_motive_presence, instruments_dir / f"instrument_presence_by_doc_primary_motive_vintage_adjusted__{sample}.parquet")
        write_parquet(top_motive_instruments, instruments_dir / f"top_doc_primary_motive_instrument_lifts__{sample}.parquet")
        write_parquet(instrument_region_share, instruments_dir / f"instrument_share_by_region_vintage_adjusted__{sample}.parquet")
        write_parquet(top_region_instrument_shares, instruments_dir / f"top_region_instrument_share_lifts__{sample}.parquet")
        write_parquet(instrument_motive_share, instruments_dir / f"instrument_share_by_doc_primary_motive_vintage_adjusted__{sample}.parquet")
        write_parquet(top_motive_instrument_shares, instruments_dir / f"top_doc_primary_motive_instrument_share_lifts__{sample}.parquet")
        for name, df_extreme in instrument_extremes.items():
            write_parquet(df_extreme, instruments_dir / f"instrument_extreme_{name}__{sample}.parquet")
        instrument_lines = [
            "# Instrument diversity and concentration",
            "",
            f"- documents covered: {int(len(instrument_by_doc))}",
            f"- instrument types observed: {int(len(instrument_catalog))}",
            "",
            "## Regional instrument lifts (vintage adjusted, coverage-limited)",
            "",
        ]
        for row in top_region_instruments.head(20).to_dict(orient="records"):
            instrument_lines.append(
                f"- region=`{clean_optional_str(row['jurisdiction_region'])}` | instrument=`{clean_optional_str(row['instrument_type'])}` "
                f"| adjusted_lift={float(row['presence_lift_vs_overall_vintage_adjusted']):.2f} "
                f"| adjusted_presence={float(row['presence_rate_vintage_adjusted']):.3f}"
            )
        instrument_lines.extend(["", "## Regional instrument share lifts (vintage adjusted, coverage-limited)", ""])
        for row in top_region_instrument_shares.head(20).to_dict(orient="records"):
            instrument_lines.append(
                f"- region=`{clean_optional_str(row['jurisdiction_region'])}` | instrument=`{clean_optional_str(row['instrument_type'])}` "
                f"| adjusted_share_lift={float(row['instrument_clause_share_lift_vs_overall_vintage_adjusted']):.2f} "
                f"| adjusted_share={float(row['instrument_clause_share__vintage_adjusted']):.3f}"
            )
        instrument_lines.extend(["", "## Purpose-to-instrument lifts (vintage adjusted, coverage-limited)", ""])
        for row in top_motive_instruments.head(20).to_dict(orient="records"):
            instrument_lines.append(
                f"- doc_primary_motive=`{clean_optional_str(row['doc_primary_motive'])}` | instrument=`{clean_optional_str(row['instrument_type'])}` "
                f"| adjusted_lift={float(row['presence_lift_vs_overall_vintage_adjusted']):.2f} "
                f"| adjusted_presence={float(row['presence_rate_vintage_adjusted']):.3f}"
            )
        instrument_lines.extend(["", "## Purpose-to-instrument share lifts (vintage adjusted, coverage-limited)", ""])
        for row in top_motive_instrument_shares.head(20).to_dict(orient="records"):
            instrument_lines.append(
                f"- doc_primary_motive=`{clean_optional_str(row['doc_primary_motive'])}` | instrument=`{clean_optional_str(row['instrument_type'])}` "
                f"| adjusted_share_lift={float(row['instrument_clause_share_lift_vs_overall_vintage_adjusted']):.2f} "
                f"| adjusted_share={float(row['instrument_clause_share__vintage_adjusted']):.3f}"
            )
        instrument_lines.extend(["", "## Highest instrument diversity", ""])
        for row in instrument_extremes["diversity"].head(8).to_dict(orient="records"):
            instrument_lines.append(
                f"- `{clean_optional_str(row['ordinance_doc_id'])}` | region=`{clean_optional_str(row['jurisdiction_region'])}` "
                f"| level=`{clean_optional_str(row['jurisdiction_level'])}` | distinct_types={int(row['distinct_instrument_type_count'])} "
                f"| entropy={float(row['instrument_entropy']):.3f}"
            )
        instrument_lines.extend(["", "## Highest instrument concentration", ""])
        for row in instrument_extremes["concentration"].head(8).to_dict(orient="records"):
            instrument_lines.append(
                f"- `{clean_optional_str(row['ordinance_doc_id'])}` | region=`{clean_optional_str(row['jurisdiction_region'])}` "
                f"| level=`{clean_optional_str(row['jurisdiction_level'])}` | top_instrument=`{clean_optional_str(row['top_instrument_type'])}` "
                f"| top_share={float(row['top_instrument_share']):.3f}"
            )
        write_text(instruments_dir / f"instrument_diversity_notes__{sample}.md", "\n".join(instrument_lines))
        instrument_summary = {
            "available": True,
            "doc_count": int(len(instrument_by_doc)),
            "instrument_type_count": int(len(instrument_catalog)),
        }
        governance_summary = {"available": True, "doc_count": int(len(by_doc))}
        style_df = by_doc.merge(
            instrument_by_doc[
                [
                    "ordinance_doc_id",
                    "distinct_instrument_type_count",
                    "instrument_entropy",
                    "top_instrument_type",
                    "top_instrument_share",
                    "operative_clause_count",
                    "vintage_bin",
                ]
            ],
            on=["ordinance_doc_id", "operative_clause_count"],
            how="inner",
            suffixes=("", "_instrument"),
        )
        style_df = style_df.merge(
            coherence_df[["ordinance_doc_id", "doc_primary_motive"]],
            on="ordinance_doc_id",
            how="left",
        )
        style_df["operative_clause_count_log"] = np.log1p(pd.to_numeric(style_df["operative_clause_count"], errors="coerce").fillna(0.0))
        regulatory_style_summary = _build_regulatory_style_family_outputs(
            style_df=style_df,
            style_dir=style_dir,
            sample=sample,
        )

    policy_signature_summary_path = run_root / "clusters" / "policy_signature_kmeans_probe" / "summary.json"
    policy_signature_labels_path = run_root / "clusters" / "policy_signature_kmeans_probe" / "cluster_labels.parquet"
    policy_signature_section_normalized_summary_path = run_root / "clusters" / "policy_signature_section_normalized_clause_share_probe" / "summary.json"
    policy_signature_section_normalized_labels_path = run_root / "clusters" / "policy_signature_section_normalized_clause_share_probe" / "cluster_labels.parquet"
    policy_signature_section_presence_summary_path = run_root / "clusters" / "policy_signature_section_presence_probe" / "summary.json"
    policy_signature_section_presence_labels_path = run_root / "clusters" / "policy_signature_section_presence_probe" / "cluster_labels.parquet"
    policy_signature_summary: dict[str, Any] = {
        "available": False,
        "item_count": 0,
        "cluster_count": 0,
        "reportable_cluster_count": 0,
        "rare_cluster_count": 0,
        "silhouette_score": 0.0,
    }
    policy_signature_top_labels: list[str] = []
    policy_signature_section_normalized_summary: dict[str, Any] = {
        "available": False,
        "item_count": 0,
        "cluster_count": 0,
        "reportable_cluster_count": 0,
        "rare_cluster_count": 0,
        "silhouette_score": 0.0,
    }
    policy_signature_section_normalized_top_labels: list[str] = []
    policy_signature_section_presence_summary: dict[str, Any] = {
        "available": False,
        "item_count": 0,
        "cluster_count": 0,
        "reportable_cluster_count": 0,
        "rare_cluster_count": 0,
        "silhouette_score": 0.0,
    }
    policy_signature_section_presence_top_labels: list[str] = []
    if clause_surface_available and policy_signature_summary_path.is_file():
        try:
            policy_signature_summary = json.loads(policy_signature_summary_path.read_text())
            policy_signature_summary["available"] = True
        except json.JSONDecodeError:
            policy_signature_summary = {
                "available": False,
                "item_count": 0,
                "cluster_count": 0,
                "reportable_cluster_count": 0,
                "rare_cluster_count": 0,
                "silhouette_score": 0.0,
            }
    if clause_surface_available and policy_signature_labels_path.is_file():
        labels_df = pd.read_parquet(policy_signature_labels_path)
        if not labels_df.empty and "cluster_label" in labels_df.columns:
            labels_df = labels_df.sort_values(
                ["reportable_cluster_flag", "cluster_size", "policy_signature_cluster_id"],
                ascending=[False, False, True],
            ).reset_index(drop=True)
            policy_signature_top_labels = [
                clean_optional_str(value)
                for value in labels_df["cluster_label"].astype(str).tolist()
                if clean_optional_str(value)
            ][:3]
    if clause_surface_available and policy_signature_section_normalized_summary_path.is_file():
        try:
            policy_signature_section_normalized_summary = json.loads(policy_signature_section_normalized_summary_path.read_text())
            policy_signature_section_normalized_summary["available"] = True
        except json.JSONDecodeError:
            policy_signature_section_normalized_summary = {
                "available": False,
                "item_count": 0,
                "cluster_count": 0,
                "reportable_cluster_count": 0,
                "rare_cluster_count": 0,
                "silhouette_score": 0.0,
            }
    if clause_surface_available and policy_signature_section_normalized_labels_path.is_file():
        labels_df = pd.read_parquet(policy_signature_section_normalized_labels_path)
        if not labels_df.empty and "cluster_label" in labels_df.columns:
            labels_df = labels_df.sort_values(
                ["reportable_cluster_flag", "cluster_size", "policy_signature_cluster_id"],
                ascending=[False, False, True],
            ).reset_index(drop=True)
            policy_signature_section_normalized_top_labels = [
                clean_optional_str(value)
                for value in labels_df["cluster_label"].astype(str).tolist()
                if clean_optional_str(value)
            ][:3]
    if clause_surface_available and policy_signature_section_presence_summary_path.is_file():
        try:
            policy_signature_section_presence_summary = json.loads(policy_signature_section_presence_summary_path.read_text())
            policy_signature_section_presence_summary["available"] = True
        except json.JSONDecodeError:
            policy_signature_section_presence_summary = {
                "available": False,
                "item_count": 0,
                "cluster_count": 0,
                "reportable_cluster_count": 0,
                "rare_cluster_count": 0,
                "silhouette_score": 0.0,
            }
    if clause_surface_available and policy_signature_section_presence_labels_path.is_file():
        labels_df = pd.read_parquet(policy_signature_section_presence_labels_path)
        if not labels_df.empty and "cluster_label" in labels_df.columns:
            labels_df = labels_df.sort_values(
                ["reportable_cluster_flag", "cluster_size", "policy_signature_cluster_id"],
                ascending=[False, False, True],
            ).reset_index(drop=True)
            policy_signature_section_presence_top_labels = [
                clean_optional_str(value)
                for value in labels_df["cluster_label"].astype(str).tolist()
                if clean_optional_str(value)
            ][:3]

    clause_assignments_path = run_root / "clusters" / "clause_instrument_hdbscan" / "assignments.parquet"
    if clause_surface_available and clause_assignments_path.is_file():
        clause_assignments = pd.read_parquet(clause_assignments_path)
        if not clause_assignments.empty:
            clause_assignments = clause_assignments.loc[clause_assignments["cluster_id"].astype(int) >= 0].copy()
            clause_assignments["cluster_label"] = clause_assignments["cluster_label"].astype(str).map(clean_optional_str)
            clause_assignments = clause_assignments.loc[clause_assignments["cluster_label"] != ""].copy()
            if not clause_assignments.empty:
                if "vintage_bin" not in clause_assignments.columns:
                    clause_assignments = clause_assignments.merge(
                        docs[["ordinance_doc_id", "vintage_bin"]].drop_duplicates(),
                        on="ordinance_doc_id",
                        how="left",
                    )
                covered_clause_docs = clause_assignments[
                    ["ordinance_doc_id", "jurisdiction_region", "jurisdiction_level", "vintage_bin"]
                ].drop_duplicates()
                cluster_catalog = clause_assignments[["cluster_id", "cluster_label"]].drop_duplicates()
                cluster_doc_counts = (
                    clause_assignments.groupby(["ordinance_doc_id", "cluster_id", "cluster_label"], dropna=False)["clause_id"]
                    .count()
                    .reset_index(name="cluster_clause_count")
                )
                cluster_doc_totals = (
                    clause_assignments.groupby("ordinance_doc_id", dropna=False)["clause_id"]
                    .count()
                    .reset_index(name="clustered_clause_total")
                )
                clause_cluster_share = cluster_doc_counts.merge(cluster_doc_totals, on="ordinance_doc_id", how="left")
                clause_cluster_share["cluster_clause_share"] = (
                    clause_cluster_share["cluster_clause_count"] / clause_cluster_share["clustered_clause_total"].clip(lower=1.0)
                )
                clause_cluster_grid = (
                    covered_clause_docs.assign(_merge_key=1)
                    .merge(cluster_catalog.assign(_merge_key=1), on="_merge_key", how="inner")
                    .drop(columns="_merge_key")
                    .merge(
                        clause_cluster_share[["ordinance_doc_id", "cluster_id", "cluster_label", "cluster_clause_share"]],
                        on=["ordinance_doc_id", "cluster_id", "cluster_label"],
                        how="left",
                    )
                )
                clause_cluster_grid["cluster_clause_share"] = pd.to_numeric(
                    clause_cluster_grid["cluster_clause_share"], errors="coerce"
                ).fillna(0.0)
                clause_cluster_region_share = _vintage_adjusted_category_mean_table(
                    clause_cluster_grid,
                    group_col="jurisdiction_region",
                    vintage_col="vintage_bin",
                    category_cols=["cluster_id", "cluster_label"],
                    value_col="cluster_clause_share",
                )
                clause_cluster_region_rank_df = clause_cluster_region_share.loc[
                    clause_cluster_region_share["overall_mean"].fillna(1.0) < 0.15
                ].copy()
                if clause_cluster_region_rank_df.empty:
                    clause_cluster_region_rank_df = clause_cluster_region_share.copy()
                top_region_clause_clusters = (
                    clause_cluster_region_rank_df.sort_values(
                        ["jurisdiction_region", "cluster_clause_share_lift_vs_overall_vintage_adjusted", "cluster_clause_share__vintage_adjusted"],
                        ascending=[True, False, False],
                    )
                    .groupby("jurisdiction_region", dropna=False)
                    .head(8)
                    .reset_index(drop=True)
                )
                top_clause_clusters_global = (
                    clause_assignments.groupby(["cluster_id", "cluster_label"], dropna=False)
                    .agg(clause_count=("clause_id", "count"), ordinance_count=("ordinance_doc_id", "nunique"))
                    .reset_index()
                    .sort_values(["clause_count", "ordinance_count"], ascending=[False, False])
                    .reset_index(drop=True)
                )
                write_parquet(clause_cluster_region_share, clause_clusters_dir / f"clause_cluster_share_by_region_vintage_adjusted__{sample}.parquet")
                write_parquet(top_region_clause_clusters, clause_clusters_dir / f"top_region_clause_cluster_lifts__{sample}.parquet")
                write_parquet(top_clause_clusters_global, clause_clusters_dir / f"largest_clause_clusters__{sample}.parquet")
                clause_lines = [
                    "# Clause cluster regional contrasts",
                    "",
                    f"- clusters covered: {int(clause_assignments['cluster_id'].nunique())}",
                    "",
                    "## Largest clause clusters",
                    "",
                ]
                for row in top_clause_clusters_global.head(15).to_dict(orient="records"):
                    clause_lines.append(
                        f"- cluster {int(row['cluster_id'])}: `{clean_optional_str(row['cluster_label'])}` | "
                        f"clauses={int(row['clause_count'])} | ordinances={int(row['ordinance_count'])}"
                    )
                clause_lines.extend(["", "## Regional clause-cluster lifts (vintage adjusted)", ""])
                for row in top_region_clause_clusters.head(24).to_dict(orient="records"):
                    clause_lines.append(
                        f"- region=`{clean_optional_str(row['jurisdiction_region'])}` | cluster=`{clean_optional_str(row['cluster_label'])}` "
                        f"| adjusted_share_lift={float(row['cluster_clause_share_lift_vs_overall_vintage_adjusted']):.2f} "
                        f"| adjusted_share={float(row['cluster_clause_share__vintage_adjusted']):.3f}"
                    )
                write_text(clause_clusters_dir / f"clause_cluster_notes__{sample}.md", "\n".join(clause_lines))
                clause_cluster_summary = {
                    "available": True,
                    "cluster_count": int(clause_assignments["cluster_id"].nunique()),
                    "region_contrast_rows": int(len(top_region_clause_clusters)),
                }

    style_assignments_path = style_dir / f"regulatory_style_family_assignments__{sample}.parquet"
    policy_assignments_path = run_root / "clusters" / "policy_signature_kmeans_probe" / "assignments.parquet"
    if style_assignments_path.is_file():
        style_assignments = pd.read_parquet(style_assignments_path).copy()
        coherence_cols = [
            "ordinance_doc_id",
            "purpose_alignment_word_share",
            "purpose_alignment_non_other_word_share",
            "doc_primary_motive",
            "section_primary_motive_weighted_excl_other_unclear",
        ]
        style_coherence = style_assignments.merge(
            coherence_df[[col for col in coherence_cols if col in coherence_df.columns]],
            on="ordinance_doc_id",
            how="left",
        )
        if not alignment_residuals.empty:
            residual_cols = ["ordinance_doc_id", "model_name", "fitted_value", "residual"]
            residual_subset = (
                alignment_residuals[[col for col in residual_cols if col in alignment_residuals.columns]]
                .query("model_name == 'vintage_level'")
                .rename(columns={"fitted_value": "alignment_fitted_vintage_level", "residual": "alignment_residual_vintage_level"})
                .drop(columns=["model_name"], errors="ignore")
            )
            style_coherence = style_coherence.merge(residual_subset, on="ordinance_doc_id", how="left")
        style_coherence_summary_df = (
            style_coherence.groupby(["regulatory_style_family_id", "cluster_label"], dropna=False)
            .agg(
                ordinance_count=("ordinance_doc_id", "nunique"),
                mean_alignment_non_other=("purpose_alignment_non_other_word_share", "mean"),
                mean_alignment_raw=("purpose_alignment_word_share", "mean"),
                mean_alignment_residual_vintage_level=("alignment_residual_vintage_level", "mean"),
                mean_direct_rule_share=("direct_rule_share_of_operative", "mean"),
                mean_discretion_share=("discretion_share_of_operative", "mean"),
                mean_exception_or_condition_share=("exception_or_condition_share_of_operative", "mean"),
                mean_threshold_share=("threshold_share_of_operative", "mean"),
                mean_instrument_entropy=("instrument_entropy", "mean"),
                mean_top_instrument_share=("top_instrument_share", "mean"),
            )
            .reset_index()
            .sort_values(["mean_alignment_residual_vintage_level", "ordinance_count"], ascending=[False, False])
            .reset_index(drop=True)
        )
        write_parquet(style_coherence, style_dir / f"regulatory_style_family_coherence_by_doc__{sample}.parquet")
        write_parquet(style_coherence_summary_df, style_dir / f"regulatory_style_family_coherence__{sample}.parquet")
        coherence_lines = [
            "# Regulatory style and coherence",
            "",
            f"- documents covered: {int(style_coherence['ordinance_doc_id'].nunique())}",
            "",
        ]
        for row in style_coherence_summary_df.to_dict(orient="records"):
            coherence_lines.append(
                f"- family {int(row['regulatory_style_family_id'])}: `{clean_optional_str(row['cluster_label'])}` | "
                f"n={int(row['ordinance_count'])} | mean_alignment_non_other={float(row['mean_alignment_non_other']):.3f} | "
                f"mean_alignment_residual={float(row['mean_alignment_residual_vintage_level']):.3f} | "
                f"direct_rule={float(row['mean_direct_rule_share']):.3f} | discretion={float(row['mean_discretion_share']):.3f}"
            )
        write_text(style_dir / f"regulatory_style_family_coherence_notes__{sample}.md", "\n".join(coherence_lines))
        reportable_style_coherence = style_coherence_summary_df.loc[style_coherence_summary_df["ordinance_count"] >= 2].copy()
        if reportable_style_coherence.empty:
            reportable_style_coherence = style_coherence_summary_df.copy()
        most_coherent_family_label = clean_optional_str(reportable_style_coherence.iloc[0]["cluster_label"]) if not reportable_style_coherence.empty else ""
        if len(reportable_style_coherence) >= 2:
            most_fragmented_family_label = clean_optional_str(reportable_style_coherence.iloc[-1]["cluster_label"])
        else:
            most_fragmented_family_label = ""
        style_coherence_summary = {
            "available": True,
            "family_count": int(style_coherence_summary_df["regulatory_style_family_id"].nunique()),
            "row_count": int(len(style_coherence_summary_df)),
            "most_coherent_family_label": most_coherent_family_label,
            "most_fragmented_family_label": most_fragmented_family_label,
        }
        if policy_assignments_path.is_file():
            policy_assignments = pd.read_parquet(policy_assignments_path).copy()
            if not policy_assignments.empty:
                style_policy = (
                    style_assignments[
                        ["ordinance_doc_id", "regulatory_style_family_id", "cluster_label"]
                    ]
                    .rename(columns={"cluster_label": "regulatory_style_family_label"})
                    .merge(
                        policy_assignments[
                            ["ordinance_doc_id", "policy_signature_cluster_id", "cluster_label", "reportable_cluster_flag"]
                        ].rename(columns={"cluster_label": "policy_signature_label"}),
                        on="ordinance_doc_id",
                        how="inner",
                    )
                )
                if not style_policy.empty:
                    style_policy_summary_df = (
                        style_policy.groupby(
                            [
                                "regulatory_style_family_id",
                                "regulatory_style_family_label",
                                "policy_signature_cluster_id",
                                "policy_signature_label",
                                "reportable_cluster_flag",
                            ],
                            dropna=False,
                        )
                        .agg(ordinance_count=("ordinance_doc_id", "nunique"))
                        .reset_index()
                    )
                    style_family_totals = style_policy_summary_df.groupby("regulatory_style_family_id")["ordinance_count"].transform("sum").clip(lower=1)
                    policy_totals = style_policy_summary_df.groupby("policy_signature_cluster_id")["ordinance_count"].transform("sum").clip(lower=1)
                    style_policy_summary_df["share_within_style_family"] = style_policy_summary_df["ordinance_count"] / style_family_totals
                    style_policy_summary_df["share_within_policy_signature"] = style_policy_summary_df["ordinance_count"] / policy_totals
                    write_parquet(style_policy_summary_df, style_dir / f"regulatory_style_vs_policy_signature__{sample}.parquet")
                    style_policy_bridge_summary = {"available": True, "row_count": int(len(style_policy_summary_df))}

    # Research memo and summary.
    low_align_head = low_alignment.head(10)[
        [
            "ordinance_doc_id",
            "jurisdiction_region",
            "jurisdiction_level",
            "doc_primary_motive",
            "section_primary_motive_weighted_excl_other_unclear",
            "purpose_alignment_non_other_word_share",
        ]
    ]
    packet_lines.extend(
        [
            f"- Ordinances in sample: {int(len(docs))}",
            f"- Mean purpose/section alignment share: {float(coherence_df['purpose_alignment_word_share'].mean()):.3f}",
            f"- Mean purpose/section non-other alignment share: {float(coherence_df['purpose_alignment_non_other_word_share'].mean()):.3f}",
            f"- Exact document-vs-section motive match rate: {float(coherence_df['purpose_section_exact_match_flag'].mean()):.3f}",
            "",
            "## Document Purpose Surface",
            "",
        ]
    )
    if not top_weighted_region.empty:
        for region_name, group in top_weighted_region.groupby("jurisdiction_region", dropna=False):
            labels = ", ".join(
                f"`{clean_optional_str(row['motive'])}` ({float(row['purpose_weight_lift_vs_overall_vintage_adjusted']):.2f}x)"
                for row in group.head(3).to_dict(orient="records")
            )
            packet_lines.append(f"- Weighted purpose lifts in `{clean_optional_str(region_name)}`: {labels}")
    else:
        packet_lines.append("- Weighted purpose lifts unavailable.")
    packet_lines.extend(
        [
            "",
            "### Lowest-alignment ordinances",
            "",
        ]
    )
    for row in low_align_head.to_dict(orient="records"):
        packet_lines.append(
            f"- `{clean_optional_str(row['ordinance_doc_id'])}` | region=`{clean_optional_str(row['jurisdiction_region'])}` | "
            f"level=`{clean_optional_str(row['jurisdiction_level'])}` | doc=`{clean_optional_str(row['doc_primary_motive'])}` | "
            f"sections=`{clean_optional_str(row['section_primary_motive_weighted_excl_other_unclear'])}` | "
            f"alignment={float(row['purpose_alignment_non_other_word_share']):.3f}"
        )
    vintage_counts_map = {
        clean_optional_str(row["vintage_bin"]): int(row["ordinance_count"])
        for row in vintage_counts.to_dict(orient="records")
    }
    top_complex_region = None
    if not complexity_region_adjusted.empty and "complexity_index_exploratory__vintage_adjusted" in complexity_region_adjusted.columns:
        top_complex_region = (
            complexity_region_adjusted.sort_values("complexity_index_exploratory__vintage_adjusted", ascending=False)
            .reset_index(drop=True)
            .iloc[0]
        )
    packet_lines.extend(
        [
            "",
            "## Vintage Composition",
            "",
            f"- 1920-1949 ordinances: {int(vintage_counts_map.get('1920_1949', 0))}",
            f"- 1950s ordinances: {int(vintage_counts_map.get('1950s', 0))}",
            f"- 1960s ordinances: {int(vintage_counts_map.get('1960s', 0))}",
            f"- 1970+ ordinances: {int(vintage_counts_map.get('1970_plus', 0))}",
            "- Regional comparisons should be read through vintage-adjusted tables, not only raw contrasts.",
            "",
            "## Complexity",
            "",
            f"- Median clause count per ordinance: {float(complexity_df['clause_count'].median()):.1f}",
            f"- Median clauses per 1,000 words: {float(complexity_df['clauses_per_1000_words'].median()):.2f}",
            f"- Median section-motive entropy: {float(complexity_df['section_motive_entropy'].median()):.3f}",
            f"- Median modality entropy: {float(complexity_df['clause_modality_entropy'].median()):.3f}",
            f"- Highest vintage-adjusted complexity region: `{clean_optional_str(top_complex_region['jurisdiction_region'])}`"
            if top_complex_region is not None
            else "- Highest vintage-adjusted complexity region: unavailable",
            f"- Complexity/alignment correlation: {float(coherence_complexity_correlation):.3f}",
            (
                f"- Vintage+level adjusted alignment slope on complexity: "
                f"{adjusted_alignment_coef:.3f} (p={adjusted_alignment_pvalue:.3f})"
            )
            if not np.isnan(adjusted_alignment_coef)
            else "- Vintage+level adjusted alignment slope on complexity: unavailable",
            (
                f"- Strongest adjusted regional complexity effect: "
                f"`{clean_optional_str(top_complexity_effects[top_complexity_effects['group_name'].astype(str) == 'jurisdiction_region'].iloc[0]['group_value'])}` "
                f"(coef={float(top_complexity_effects[top_complexity_effects['group_name'].astype(str) == 'jurisdiction_region'].iloc[0]['coefficient']):+.3f})"
            )
            if not top_complexity_effects.empty and (top_complexity_effects["group_name"].astype(str) == "jurisdiction_region").any()
            else "- Strongest adjusted regional complexity effect: unavailable",
            (
                f"- Strongest adjusted jurisdiction-type complexity effect: "
                f"`{clean_optional_str(top_complexity_effects[top_complexity_effects['group_name'].astype(str) == 'jurisdiction_level'].iloc[0]['group_value'])}` "
                f"(coef={float(top_complexity_effects[top_complexity_effects['group_name'].astype(str) == 'jurisdiction_level'].iloc[0]['coefficient']):+.3f})"
            )
            if not top_complexity_effects.empty and (top_complexity_effects["group_name"].astype(str) == "jurisdiction_level").any()
            else "- Strongest adjusted jurisdiction-type complexity effect: unavailable",
            "- Interpretation: the raw pilot signal weakens sharply after vintage and jurisdiction-level controls, so the frontier is now less 'complexity directly reduces coherence' and more 'which ordinances are unexpectedly coherent or incoherent given their complexity and vintage?'.",
            "",
            "## Section Architecture",
            "",
            f"- Median section count per ordinance: {float(doc_arch_df['section_count'].median()):.1f}",
            f"- Distinct architecture anchors observed: {int(sections['architecture_anchor'].astype(str).nunique())}",
            f"- Architecture families available: {bool(architecture_family_summary['available'])}",
            f"- Architecture family count: {int(architecture_family_summary.get('family_count', 0))}",
            f"- Architecture family chosen k: {int(architecture_family_summary.get('top_k', 0))}",
            "",
            "## Retrieval Surfaces",
            "",
            f"- Ordinance neighbor rows: {int(len(ordinance_neighbors))}",
            f"- Region prototype rows: {int(len(region_prototypes))}",
            f"- Level prototype rows: {int(len(level_prototypes))}",
            (
                f"- Highest same-region neighbor lift: `{clean_optional_str(neighbor_by_region.sort_values('same_region_neighbor_lift', ascending=False).iloc[0]['jurisdiction_region'])}` "
                f"({float(neighbor_by_region.sort_values('same_region_neighbor_lift', ascending=False).iloc[0]['same_region_neighbor_lift']):.2f}x baseline)"
            )
            if not neighbor_by_region.empty
            else "- Highest same-region neighbor lift: unavailable",
            (
                f"- Most cross-region bridge-like ordinance: `{clean_optional_str(cross_region_bridges.iloc[0]['ordinance_doc_id'])}` "
                f"from `{clean_optional_str(cross_region_bridges.iloc[0]['jurisdiction_region'])}` with dominant neighbor region "
                f"`{clean_optional_str(cross_region_bridges.iloc[0]['dominant_neighbor_region'])}`"
            )
            if not cross_region_bridges.empty
            else "- Most cross-region bridge-like ordinance: unavailable",
            "",
            "## Section Cluster Surface",
            "",
            f"- Available: {bool(section_cluster_summary['available'])}",
            f"- Non-noise section clusters: {int(section_cluster_summary['cluster_count'])}",
            f"- Vintage-adjusted regional contrast rows: {int(section_cluster_summary.get('region_contrast_rows_vintage_adjusted', 0))}",
            (
                f"- Strongest adjusted region-specific section family: "
                f"`{clean_optional_str(top_joint_effect_clusters[top_joint_effect_clusters['group_name'].astype(str) == 'jurisdiction_region'].iloc[0]['cluster_label'])}` "
                f"for `{clean_optional_str(top_joint_effect_clusters[top_joint_effect_clusters['group_name'].astype(str) == 'jurisdiction_region'].iloc[0]['group_value'])}` "
                f"(coef={float(top_joint_effect_clusters[top_joint_effect_clusters['group_name'].astype(str) == 'jurisdiction_region'].iloc[0]['coefficient']):+.3f})"
            )
            if not top_joint_effect_clusters.empty and (top_joint_effect_clusters["group_name"].astype(str) == "jurisdiction_region").any()
            else "- Strongest adjusted region-specific section family: unavailable",
            (
                f"- Strongest adjusted jurisdiction-type section family: "
                f"`{clean_optional_str(top_joint_effect_clusters[top_joint_effect_clusters['group_name'].astype(str) == 'jurisdiction_level'].iloc[0]['cluster_label'])}` "
                f"for `{clean_optional_str(top_joint_effect_clusters[top_joint_effect_clusters['group_name'].astype(str) == 'jurisdiction_level'].iloc[0]['group_value'])}` "
                f"(coef={float(top_joint_effect_clusters[top_joint_effect_clusters['group_name'].astype(str) == 'jurisdiction_level'].iloc[0]['coefficient']):+.3f})"
            )
            if not top_joint_effect_clusters.empty and (top_joint_effect_clusters["group_name"].astype(str) == "jurisdiction_level").any()
            else "- Strongest adjusted jurisdiction-type section family: unavailable",
            "",
            "## Embedding Surface Comparison",
            "",
            f"- Available: {bool(embedding_surface_summary['available'])}",
            f"- Surfaces compared: {int(embedding_surface_summary.get('surface_count', 0))}",
            f"- Current winner: `{clean_optional_str(embedding_surface_summary.get('winner', ''))}`",
            "",
            "## Contrastive Packets",
            "",
            f"- Region packets: {int(len(region_packets))}",
            "",
            "## Clause-side Coverage",
            "",
            f"- Enriched-clause documents covered: {int(coverage_summary['doc_count'])} / {int(len(docs))}",
            f"- Mean clause enrichment share by document: {float(coverage_summary['mean_clause_share']):.3f}",
            f"- Mean section enrichment share by document: {float(coverage_summary['mean_section_share']):.3f}",
            (
                f"- Highest vintage-adjusted enrichment coverage region: "
                f"`{clean_optional_str(coverage_region_adjusted.sort_values('has_clause_enrichment__vintage_adjusted', ascending=False).iloc[0]['jurisdiction_region'])}` "
                f"(coverage={float(coverage_region_adjusted.sort_values('has_clause_enrichment__vintage_adjusted', ascending=False).iloc[0]['has_clause_enrichment__vintage_adjusted']):.3f})"
            )
            if not coverage_region_adjusted.empty
            else "- Highest vintage-adjusted enrichment coverage region: unavailable",
            "- Clause-side governance and enriched-clause typology should be treated as provisional until this coverage broadens materially beyond the current partial surface.",
            "",
            "## Governance Style",
            "",
            f"- Available: {bool(governance_summary['available'])}",
            f"- Documents covered: {int(governance_summary['doc_count'])}",
            "",
            "## Instrument Diversity",
            "",
            f"- Available: {bool(instrument_summary['available'])}",
            f"- Documents covered: {int(instrument_summary['doc_count'])}",
            f"- Instrument types observed: {int(instrument_summary['instrument_type_count'])}",
            "",
            "## Clause Cluster Surface",
            "",
            f"- Available: {bool(clause_cluster_summary['available'])}",
            f"- Clause clusters covered: {int(clause_cluster_summary.get('cluster_count', 0))}",
            f"- Vintage-adjusted regional contrast rows: {int(clause_cluster_summary.get('region_contrast_rows', 0))}",
            "",
            "## Regulatory Style Families",
            "",
            f"- Available: {bool(regulatory_style_summary['available'])}",
            f"- Family count: {int(regulatory_style_summary.get('family_count', 0))}",
            f"- Chosen k: {int(regulatory_style_summary.get('top_k', 0))}",
            f"- Silhouette score: {float(regulatory_style_summary.get('silhouette_score', 0.0)):.3f}",
            (
                f"- Most coherence-aligned family: `{clean_optional_str(style_coherence_summary.get('most_coherent_family_label', ''))}`"
                if style_coherence_summary["available"] and clean_optional_str(style_coherence_summary.get("most_coherent_family_label", ""))
                else "- Most coherence-aligned family: unavailable"
            ),
            (
                f"- Most fragmented family: `{clean_optional_str(style_coherence_summary.get('most_fragmented_family_label', ''))}`"
                if style_coherence_summary["available"] and clean_optional_str(style_coherence_summary.get("most_fragmented_family_label", ""))
                else "- Most fragmented family: unavailable"
            ),
            "",
            "## Policy Signatures",
            "",
            f"- Available: {bool(policy_signature_summary['available'])}",
            f"- Documents covered: {int(policy_signature_summary.get('item_count', 0))}",
            f"- Cluster count: {int(policy_signature_summary.get('cluster_count', 0))}",
            f"- Reportable cluster count: {int(policy_signature_summary.get('reportable_cluster_count', 0))}",
            f"- Silhouette score: {float(policy_signature_summary.get('silhouette_score', 0.0)):.3f}",
            (
                f"- Leading signature labels: {', '.join(f'`{label}`' for label in policy_signature_top_labels)}"
                if policy_signature_top_labels
                else "- Leading signature labels: unavailable"
            ),
            (
                f"- Section-normalized clause-share comparison: silhouette={float(policy_signature_section_normalized_summary.get('silhouette_score', 0.0)):.3f}, "
                f"reportable_clusters={int(policy_signature_section_normalized_summary.get('reportable_cluster_count', 0))}, "
                f"labels={', '.join(f'`{label}`' for label in policy_signature_section_normalized_top_labels)}"
                if policy_signature_section_normalized_summary["available"]
                else "- Section-normalized clause-share comparison: unavailable"
            ),
            (
                f"- Section-presence comparison: silhouette={float(policy_signature_section_presence_summary.get('silhouette_score', 0.0)):.3f}, "
                f"reportable_clusters={int(policy_signature_section_presence_summary.get('reportable_cluster_count', 0))}, "
                f"labels={', '.join(f'`{label}`' for label in policy_signature_section_presence_top_labels)}"
                if policy_signature_section_presence_summary["available"]
                else "- Section-presence comparison: unavailable"
            ),
            (
                "- Interpretation: raw atomic clause shares remain the tightest geometrically, but section-normalized and section-presence aggregations yield less degenerate document families and are less exposed to long use or definition lists."
                if policy_signature_section_normalized_summary["available"] or policy_signature_section_presence_summary["available"]
                else ""
            ),
            "",
        ]
    )
    write_text(research_dir / f"research_memo__{sample}.md", "\n".join(packet_lines))

    summary = {
        "sample": sample,
        "doc_count": int(len(docs)),
        "coherence_doc_count": int(len(coherence_df)),
        "mean_alignment_share": float(coherence_df["purpose_alignment_word_share"].mean()) if not coherence_df.empty else 0.0,
        "mean_non_other_alignment_share": float(coherence_df["purpose_alignment_non_other_word_share"].mean()) if not coherence_df.empty else 0.0,
        "vintage_bin_count": int(docs["vintage_bin"].astype(str).nunique()),
        "purpose_surface_available": not purpose_rows.empty,
        "purpose_primary_doc_count": int(purpose_primary["ordinance_doc_id"].nunique()) if not purpose_primary.empty else 0,
        "purpose_weighted_region_rows": int(len(weighted_region)),
        "purpose_top_weighted_region_rows": int(len(top_weighted_region)),
        "section_kind_count": int(sections["architecture_anchor"].astype(str).nunique()),
        "complexity_doc_count": int(len(complexity_df)),
        "median_clause_count": float(complexity_df["clause_count"].median()) if not complexity_df.empty else 0.0,
        "median_clauses_per_1000_words": float(complexity_df["clauses_per_1000_words"].median()) if not complexity_df.empty else 0.0,
        "complexity_alignment_correlation": float(coherence_complexity_correlation),
        "complexity_alignment_component_rows": int(len(component_corr_df)),
        "complexity_alignment_model_rows": int(len(alignment_models)),
        "complexity_alignment_adjusted_coef": float(adjusted_alignment_coef) if not np.isnan(adjusted_alignment_coef) else 0.0,
        "complexity_alignment_adjusted_pvalue": float(adjusted_alignment_pvalue) if not np.isnan(adjusted_alignment_pvalue) else 1.0,
        "complexity_region_level_effect_rows": int(len(complexity_effects)),
        "section_cluster_region_level_effect_rows": int(len(cluster_presence_effects)),
        "ordinance_neighbor_rows": int(len(ordinance_neighbors)),
        "region_prototype_rows": int(len(region_prototypes)),
        "neighbor_homophily_region_rows": int(len(neighbor_by_region)),
        "cross_region_bridge_rows": int(len(cross_region_bridges)),
        "section_cluster_surface_available": bool(section_cluster_summary["available"]),
        "section_cluster_count": int(section_cluster_summary["cluster_count"]),
        "section_cluster_vintage_adjusted_region_contrast_rows": int(section_cluster_summary.get("region_contrast_rows_vintage_adjusted", 0)),
        "embedding_surface_comparison_available": bool(embedding_surface_summary["available"]),
        "embedding_surface_count": int(embedding_surface_summary.get("surface_count", 0)),
        "embedding_surface_winner": clean_optional_str(embedding_surface_summary.get("winner", "")),
        "architecture_family_surface_available": bool(architecture_family_summary["available"]),
        "architecture_family_count": int(architecture_family_summary.get("family_count", 0)),
        "architecture_family_top_k": int(architecture_family_summary.get("top_k", 0)),
        "architecture_family_region_contrast_rows": int(architecture_family_summary.get("region_contrast_rows", 0)),
        "region_contrast_packet_count": int(len(region_packets)),
        "enriched_clause_surface_available": bool(coverage_summary["available"]),
        "enriched_clause_doc_count": int(coverage_summary["doc_count"]),
        "enriched_clause_row_count": int(coverage_summary["row_count"]),
        "enriched_clause_doc_coverage_rate": float(coverage_summary["doc_count"] / len(docs)) if len(docs) else 0.0,
        "enriched_clause_mean_clause_share": float(coverage_summary["mean_clause_share"]),
        "enriched_clause_mean_section_share": float(coverage_summary["mean_section_share"]),
        "governance_style_available": bool(governance_summary["available"]),
        "governance_style_doc_count": int(governance_summary["doc_count"]),
        "instrument_diversity_available": bool(instrument_summary["available"]),
        "instrument_diversity_doc_count": int(instrument_summary["doc_count"]),
        "instrument_type_count": int(instrument_summary["instrument_type_count"]),
        "clause_cluster_surface_available": bool(clause_cluster_summary["available"]),
        "clause_cluster_count": int(clause_cluster_summary.get("cluster_count", 0)),
        "clause_cluster_region_contrast_rows": int(clause_cluster_summary.get("region_contrast_rows", 0)),
        "regulatory_style_family_available": bool(regulatory_style_summary["available"]),
        "regulatory_style_family_count": int(regulatory_style_summary.get("family_count", 0)),
        "regulatory_style_family_top_k": int(regulatory_style_summary.get("top_k", 0)),
        "regulatory_style_family_silhouette_score": float(regulatory_style_summary.get("silhouette_score", 0.0)),
        "regulatory_style_coherence_available": bool(style_coherence_summary["available"]),
        "regulatory_style_coherence_rows": int(style_coherence_summary.get("row_count", 0)),
        "regulatory_style_most_coherent_family": clean_optional_str(style_coherence_summary.get("most_coherent_family_label", "")),
        "regulatory_style_most_fragmented_family": clean_optional_str(style_coherence_summary.get("most_fragmented_family_label", "")),
        "regulatory_style_policy_bridge_available": bool(style_policy_bridge_summary["available"]),
        "regulatory_style_policy_bridge_rows": int(style_policy_bridge_summary.get("row_count", 0)),
        "policy_signature_available": bool(policy_signature_summary["available"]),
        "policy_signature_item_count": int(policy_signature_summary.get("item_count", 0)),
        "policy_signature_cluster_count": int(policy_signature_summary.get("cluster_count", 0)),
        "policy_signature_reportable_cluster_count": int(policy_signature_summary.get("reportable_cluster_count", 0)),
        "policy_signature_rare_cluster_count": int(policy_signature_summary.get("rare_cluster_count", 0)),
        "policy_signature_silhouette_score": float(policy_signature_summary.get("silhouette_score", 0.0)),
        "policy_signature_top_labels": policy_signature_top_labels,
        "policy_signature_section_normalized_clause_share_available": bool(policy_signature_section_normalized_summary["available"]),
        "policy_signature_section_normalized_clause_share_cluster_count": int(policy_signature_section_normalized_summary.get("cluster_count", 0)),
        "policy_signature_section_normalized_clause_share_reportable_cluster_count": int(policy_signature_section_normalized_summary.get("reportable_cluster_count", 0)),
        "policy_signature_section_normalized_clause_share_rare_cluster_count": int(policy_signature_section_normalized_summary.get("rare_cluster_count", 0)),
        "policy_signature_section_normalized_clause_share_silhouette_score": float(policy_signature_section_normalized_summary.get("silhouette_score", 0.0)),
        "policy_signature_section_normalized_clause_share_top_labels": policy_signature_section_normalized_top_labels,
        "policy_signature_section_presence_available": bool(policy_signature_section_presence_summary["available"]),
        "policy_signature_section_presence_cluster_count": int(policy_signature_section_presence_summary.get("cluster_count", 0)),
        "policy_signature_section_presence_reportable_cluster_count": int(policy_signature_section_presence_summary.get("reportable_cluster_count", 0)),
        "policy_signature_section_presence_silhouette_score": float(policy_signature_section_presence_summary.get("silhouette_score", 0.0)),
        "policy_signature_section_presence_top_labels": policy_signature_section_presence_top_labels,
    }
    write_json(research_dir / f"research_summary__{sample}.json", summary)
    return summary
