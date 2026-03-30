from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import math
import re
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from .io_utils import clean_optional_str, write_json, write_parquet, write_text

LEGAL_COMPLEXITY_RUNS_BASE = Path("artifacts/runs/legal_complexity_fullsample_v1")
DEFAULT_SOURCE_RUN_ROOT = Path("artifacts/runs/frontier_fullcode_lab_v1/full730_20260309_censuslinked_b")
DEFAULT_PILOT_RUN_ROOT = Path("artifacts/runs/frontier_llm_v2/pilot64_20260310_a")
RUN_SUBDIRS = ("inputs", "complexity", "bridge", "figures", "reports", "manifests")
VINTAGE_BIN_ORDER = ["1920_1949", "1950s", "1960s", "1970_plus", "unknown"]

WORD_RE = re.compile(r"[A-Za-z']+")
SENTENCE_SPLIT_RE = re.compile(r"[.!?]+")
MULTISPACE_RE = re.compile(r"\s+")

REFERENCE_PATTERNS: dict[str, re.Pattern[str]] = {
    "formal_reference_mentions": re.compile(
        r"\b(section|sections|article|articles|chapter|chapters|subsection|subsections|paragraph|paragraphs)\s+[0-9ivxlcdm\-]+",
        re.IGNORECASE,
    ),
    "formal_reference_words": re.compile(
        r"\b(section|sections|article|articles|chapter|chapters|subsection|subsections|paragraph|paragraphs)\b",
        re.IGNORECASE,
    ),
    "dependency_trigger_mentions": re.compile(
        r"\b(subject to|except as provided|provided that|in accordance with|pursuant to|notwithstanding|unless otherwise provided|as required by|as provided in|under the provisions of)\b",
        re.IGNORECASE,
    ),
    "map_reference_mentions": re.compile(
        r"\b(zoning map|official zoning map|district map|map attached|map on file|zoning atlas)\b",
        re.IGNORECASE,
    ),
    "board_reference_mentions": re.compile(
        r"\b(board of appeals|board of adjustment|plan commission|planning commission|zoning administrator|building inspector|zoning board)\b",
        re.IGNORECASE,
    ),
}


@dataclass(frozen=True)
class LegalComplexityPaths:
    source_run_root: str = str(DEFAULT_SOURCE_RUN_ROOT)
    pilot_run_root: str = str(DEFAULT_PILOT_RUN_ROOT)


def make_run_id(prefix: str = "legal_complexity_fullsample_v1") -> str:
    stamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{stamp}"


def ensure_legal_complexity_layout(base_dir: Path | str | None = None, run_id: str | None = None) -> dict[str, Path]:
    base = Path(base_dir or LEGAL_COMPLEXITY_RUNS_BASE).expanduser().resolve()
    rid = run_id or make_run_id()
    run_root = base / rid
    run_root.mkdir(parents=True, exist_ok=True)
    layout = {"run_root": run_root}
    for name in RUN_SUBDIRS:
        path = run_root / name
        path.mkdir(parents=True, exist_ok=True)
        layout[name] = path
    return layout


def legal_complexity_layout_from_run_root(run_root: Path | str) -> dict[str, Path]:
    root = Path(run_root).expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"Run root does not exist: {root}")
    layout = {"run_root": root}
    for name in RUN_SUBDIRS:
        path = root / name
        path.mkdir(parents=True, exist_ok=True)
        layout[name] = path
    return layout


def initialize_legal_complexity_manifest(*, layout: dict[str, Path], paths: LegalComplexityPaths | None = None) -> dict[str, Any]:
    payload = {
        "schema_version": "legal_complexity_fullsample_v1",
        "run_id": layout["run_root"].name,
        "created_at_utc": pd.Timestamp.utcnow().isoformat(),
        "run_root": str(layout["run_root"]),
        "paths": {
            "source_run_root": str(Path((paths or LegalComplexityPaths()).source_run_root).expanduser().resolve()),
            "pilot_run_root": str(Path((paths or LegalComplexityPaths()).pilot_run_root).expanduser().resolve()),
        },
    }
    write_json(layout["manifests"] / "run_manifest.json", payload)
    return payload


def _safe_div(numerator: Any, denominator: Any) -> float:
    try:
        denom = float(denominator)
    except Exception:
        return 0.0
    if not math.isfinite(denom) or denom <= 0.0:
        return 0.0
    try:
        num = float(numerator)
    except Exception:
        return 0.0
    if not math.isfinite(num):
        return 0.0
    return num / denom


def _normalized_entropy_from_counts(counts: np.ndarray) -> float:
    arr = np.asarray(counts, dtype=np.float64)
    arr = arr[np.isfinite(arr) & (arr > 0.0)]
    if arr.size <= 1:
        return 0.0
    probs = arr / arr.sum()
    entropy = float(-(probs * np.log2(probs)).sum())
    denom = float(np.log2(arr.size))
    return entropy / denom if denom > 0.0 else 0.0


def _zscore(series: pd.Series) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce").astype(float)
    mean = float(vals.mean())
    std = float(vals.std(ddof=0))
    if not math.isfinite(std) or std <= 0.0:
        return pd.Series(np.zeros(len(vals), dtype=float), index=series.index)
    return ((vals - mean) / std).astype(float)


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
    years = pd.to_datetime(out["issue_date"], errors="coerce").dt.year if "issue_date" in out.columns else pd.Series(np.nan, index=out.index)
    out["issue_year"] = years.astype("Int64")
    out["vintage_bin"] = out["issue_year"].map(_vintage_bin).astype(str)
    return out


def _vintage_adjusted_mean_table(df: pd.DataFrame, *, group_col: str, vintage_col: str, value_cols: list[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=[group_col, "ordinance_count", "vintage_bin_count", "vintage_weight_coverage"])
    overall_weights = (
        df[["ordinance_doc_id", vintage_col]]
        .drop_duplicates()
        [vintage_col]
        .astype(str)
        .value_counts(normalize=True)
        .reindex(VINTAGE_BIN_ORDER, fill_value=0.0)
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


def _fit_group_effect_model(
    df: pd.DataFrame,
    *,
    target_col: str,
    categorical_cols: list[str],
) -> pd.DataFrame:
    required = [target_col, *categorical_cols]
    if df.empty or any(col not in df.columns for col in required):
        return pd.DataFrame()
    model_df = df[["ordinance_doc_id", *required]].copy()
    model_df[target_col] = pd.to_numeric(model_df[target_col], errors="coerce")
    model_df = model_df.dropna(subset=[target_col]).reset_index(drop=True)
    for col in categorical_cols:
        model_df[col] = model_df[col].astype(str).replace({"": "unknown"}).fillna("unknown")
    if len(model_df) < 24:
        return pd.DataFrame()
    formula = f"{target_col} ~ " + " + ".join(f"C({col})" for col in categorical_cols)
    try:
        result = smf.ols(formula, data=model_df).fit(cov_type="HC1")
    except Exception:
        return pd.DataFrame()
    conf = result.conf_int()
    rows: list[dict[str, Any]] = []
    for term, coef in result.params.items():
        if term == "Intercept":
            continue
        match = re.match(r"C\((?P<group>[^)]+)\)\[T\.(?P<value>.+)\]", str(term))
        if not match:
            continue
        rows.append(
            {
                "target_col": target_col,
                "formula": formula,
                "group_name": clean_optional_str(match.group("group")),
                "group_value": clean_optional_str(match.group("value")),
                "coefficient": float(coef),
                "std_error": float(result.bse.get(term, np.nan)),
                "pvalue": float(result.pvalues.get(term, np.nan)),
                "ci_low": float(conf.loc[term, 0]) if term in conf.index else np.nan,
                "ci_high": float(conf.loc[term, 1]) if term in conf.index else np.nan,
                "nobs": int(result.nobs),
                "r_squared": float(getattr(result, "rsquared", np.nan)),
            }
        )
    return pd.DataFrame.from_records(rows)


@lru_cache(maxsize=50000)
def _approx_syllable_count(word: str) -> int:
    token = re.sub(r"[^a-z]", "", str(word or "").lower())
    if not token:
        return 0
    vowels = "aeiouy"
    count = 0
    prev_is_vowel = False
    for ch in token:
        is_vowel = ch in vowels
        if is_vowel and not prev_is_vowel:
            count += 1
        prev_is_vowel = is_vowel
    if token.endswith("e") and count > 1:
        count -= 1
    if token.endswith("le") and len(token) > 2 and token[-3] not in vowels:
        count += 1
    return max(1, count)


def _text_linguistic_metrics(text: str) -> dict[str, float]:
    raw = str(text or "")
    words = WORD_RE.findall(raw.lower())
    sentences = [segment for segment in SENTENCE_SPLIT_RE.split(raw) if WORD_RE.search(segment)]
    sentence_lengths = [len(WORD_RE.findall(sentence)) for sentence in sentences]
    word_count = len(words)
    sentence_count = len(sentence_lengths)
    avg_sentence_length = float(np.mean(sentence_lengths)) if sentence_lengths else 0.0
    avg_word_length = float(np.mean([len(word) for word in words])) if words else 0.0
    long_word_share = _safe_div(sum(len(word) >= 7 for word in words), word_count)
    syllable_count = sum(_approx_syllable_count(word) for word in words)
    avg_syllables_per_word = _safe_div(syllable_count, word_count)
    if word_count and sentence_count:
        flesch_reading_ease_proxy = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
        flesch_kincaid_grade_proxy = (0.39 * avg_sentence_length) + (11.8 * avg_syllables_per_word) - 15.59
    else:
        flesch_reading_ease_proxy = 0.0
        flesch_kincaid_grade_proxy = 0.0
    return {
        "sentence_count": float(sentence_count),
        "avg_sentence_length_words": avg_sentence_length,
        "avg_word_length_chars": avg_word_length,
        "long_word_share_ge7": long_word_share,
        "avg_syllables_per_word_proxy": avg_syllables_per_word,
        "flesch_reading_ease_proxy": float(flesch_reading_ease_proxy),
        "flesch_kincaid_grade_proxy": float(flesch_kincaid_grade_proxy),
    }


def _reference_metrics_for_section_frame(sections: pd.DataFrame) -> pd.DataFrame:
    frame = sections[["section_id", "ordinance_doc_id", "section_kind", "word_count", "text"]].copy()
    text_series = frame["text"].astype(str)
    for col, pattern in REFERENCE_PATTERNS.items():
        frame[col] = text_series.str.count(pattern).astype(int)
    frame["any_reference_flag"] = (
        frame["formal_reference_mentions"]
        + frame["dependency_trigger_mentions"]
        + frame["map_reference_mentions"]
        + frame["board_reference_mentions"]
    ).gt(0).astype(int)
    frame["any_formal_reference_flag"] = frame["formal_reference_mentions"].gt(0).astype(int)
    frame["reference_mentions_total"] = (
        frame["formal_reference_mentions"]
        + frame["dependency_trigger_mentions"]
        + frame["map_reference_mentions"]
        + frame["board_reference_mentions"]
    )
    frame["reference_mentions_per_1000_words"] = frame["reference_mentions_total"] / frame["word_count"].clip(lower=1).astype(float) * 1000.0
    return frame


def _complexity_family_outputs(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    feature_cols = [
        "structural_complexity_index",
        "dependency_complexity_index",
        "linguistic_complexity_index",
    ]
    available = df.dropna(subset=feature_cols).copy()
    if len(available) < 24:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    x = available[feature_cols].to_numpy(dtype=np.float64)
    means = x.mean(axis=0, keepdims=True)
    stds = x.std(axis=0, keepdims=True)
    stds = np.where(stds > 0.0, stds, 1.0)
    x_scaled = np.nan_to_num((x - means) / stds, nan=0.0, posinf=0.0, neginf=0.0)

    sweep_rows: list[dict[str, Any]] = []
    best_score = -np.inf
    best_k = 0
    best_labels: np.ndarray | None = None
    for k in range(3, min(6, len(available) - 1) + 1):
        model = KMeans(n_clusters=int(k), random_state=42, n_init=30)
        labels = model.fit_predict(x_scaled)
        score = float(silhouette_score(x_scaled, labels)) if len(np.unique(labels)) > 1 else np.nan
        sweep_rows.append({"cluster_count": int(k), "silhouette_score": score, "doc_count": int(len(available))})
        if np.isfinite(score) and score > best_score:
            best_score = score
            best_k = int(k)
            best_labels = labels
    sweep_df = pd.DataFrame.from_records(sweep_rows)
    if best_labels is None:
        return sweep_df, pd.DataFrame(), pd.DataFrame()

    assign = available[
        [
            "ordinance_doc_id",
            "issue_id",
            "jurisdiction_region",
            "jurisdiction_level",
            "vintage_bin",
            "census_id_pid6",
        ]
    ].copy()
    assign["complexity_family_id"] = best_labels.astype(int)
    feature_frame = pd.DataFrame(x_scaled, columns=feature_cols, index=assign.index)

    family_rows: list[dict[str, Any]] = []
    display_names = {
        "structural_complexity_index": "structural",
        "dependency_complexity_index": "dependency",
        "linguistic_complexity_index": "linguistic",
    }
    for family_id, group in assign.groupby("complexity_family_id", dropna=False):
        idx = group.index.to_numpy(dtype=int)
        means_series = feature_frame.loc[idx].mean(axis=0).sort_values(ascending=False)
        labels: list[str] = []
        for col, value in means_series.items():
            score = float(value)
            base = display_names.get(col, col.replace("_", " "))
            if score >= 0.55:
                labels.append(f"high {base}")
            elif score <= -0.55:
                labels.append(f"low {base}")
            elif score >= 0.20:
                labels.append(base)
        if not labels:
            strongest = means_series.reindex(means_series.abs().sort_values(ascending=False).index)
            for col, value in strongest.items():
                base = display_names.get(col, col.replace("_", " "))
                labels.append(f"{'high' if float(value) >= 0 else 'low'} {base}")
                if len(labels) >= 3:
                    break
        label = " | ".join(labels[:3])
        family_rows.append(
            {
                "complexity_family_id": int(family_id),
                "ordinance_count": int(group["ordinance_doc_id"].nunique()),
                "cluster_label": label,
                "silhouette_score": float(best_score),
                "mean_structural_complexity_index": float(group.merge(df[["ordinance_doc_id", "structural_complexity_index"]], on="ordinance_doc_id", how="left")["structural_complexity_index"].mean()),
                "mean_dependency_complexity_index": float(group.merge(df[["ordinance_doc_id", "dependency_complexity_index"]], on="ordinance_doc_id", how="left")["dependency_complexity_index"].mean()),
                "mean_linguistic_complexity_index": float(group.merge(df[["ordinance_doc_id", "linguistic_complexity_index"]], on="ordinance_doc_id", how="left")["linguistic_complexity_index"].mean()),
            }
        )
    family_df = pd.DataFrame.from_records(family_rows).sort_values("complexity_family_id").reset_index(drop=True)
    assign = assign.merge(family_df[["complexity_family_id", "cluster_label"]], on="complexity_family_id", how="left")
    return sweep_df, assign, family_df


def _dependency_profile_outputs(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    feature_cols = [
        "formal_reference_mentions_per_1000_words",
        "dependency_trigger_mentions_per_1000_words",
        "map_reference_mentions_per_1000_words",
        "board_reference_mentions_per_1000_words",
        "reference_bearing_section_share",
    ]
    available = df.dropna(subset=feature_cols).copy()
    if len(available) < 24:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    x = available[feature_cols].to_numpy(dtype=np.float64)
    means = x.mean(axis=0, keepdims=True)
    stds = x.std(axis=0, keepdims=True)
    stds = np.where(stds > 0.0, stds, 1.0)
    x_scaled = np.nan_to_num((x - means) / stds, nan=0.0, posinf=0.0, neginf=0.0)

    sweep_rows: list[dict[str, Any]] = []
    best_score = -np.inf
    best_labels: np.ndarray | None = None
    for k in range(3, min(5, len(available) - 1) + 1):
        model = KMeans(n_clusters=int(k), random_state=42, n_init=30)
        labels = model.fit_predict(x_scaled)
        score = float(silhouette_score(x_scaled, labels)) if len(np.unique(labels)) > 1 else np.nan
        sweep_rows.append({"cluster_count": int(k), "silhouette_score": score, "doc_count": int(len(available))})
        if np.isfinite(score) and score > best_score:
            best_score = score
            best_labels = labels
    sweep_df = pd.DataFrame.from_records(sweep_rows)
    if best_labels is None:
        return sweep_df, pd.DataFrame(), pd.DataFrame()

    assign = available[
        [
            "ordinance_doc_id",
            "issue_id",
            "jurisdiction_display_name",
            "jurisdiction_region",
            "jurisdiction_level",
            "vintage_bin",
            "census_id_pid6",
        ]
    ].copy()
    assign["dependency_profile_id"] = best_labels.astype(int)
    feature_frame = pd.DataFrame(x_scaled, columns=feature_cols, index=assign.index)
    display_names = {
        "formal_reference_mentions_per_1000_words": "formal references",
        "dependency_trigger_mentions_per_1000_words": "dependency triggers",
        "map_reference_mentions_per_1000_words": "map references",
        "board_reference_mentions_per_1000_words": "board references",
        "reference_bearing_section_share": "reference-bearing sections",
    }
    summary_rows: list[dict[str, Any]] = []
    for cluster_id, group in assign.groupby("dependency_profile_id", dropna=False):
        idx = group.index.to_numpy(dtype=int)
        means_series = feature_frame.loc[idx].mean(axis=0).sort_values(ascending=False)
        labels: list[str] = []
        for col, value in means_series.items():
            score = float(value)
            if score >= 0.25:
                labels.append(display_names.get(col, col))
        if not labels:
            strongest = means_series.reindex(means_series.abs().sort_values(ascending=False).index)
            for col, value in strongest.items():
                labels.append(
                    f"{'high' if float(value) >= 0 else 'low'} {display_names.get(col, col)}"
                )
                if len(labels) >= 2:
                    break
        summary_rows.append(
            {
                "dependency_profile_id": int(cluster_id),
                "ordinance_count": int(group["ordinance_doc_id"].nunique()),
                "cluster_label": " | ".join(labels[:3]),
                "silhouette_score": float(best_score),
                "mean_formal_reference_mentions_per_1000_words": float(
                    group.merge(
                        df[["ordinance_doc_id", "formal_reference_mentions_per_1000_words"]],
                        on="ordinance_doc_id",
                        how="left",
                    )["formal_reference_mentions_per_1000_words"].mean()
                ),
                "mean_dependency_trigger_mentions_per_1000_words": float(
                    group.merge(
                        df[["ordinance_doc_id", "dependency_trigger_mentions_per_1000_words"]],
                        on="ordinance_doc_id",
                        how="left",
                    )["dependency_trigger_mentions_per_1000_words"].mean()
                ),
                "mean_map_reference_mentions_per_1000_words": float(
                    group.merge(
                        df[["ordinance_doc_id", "map_reference_mentions_per_1000_words"]],
                        on="ordinance_doc_id",
                        how="left",
                    )["map_reference_mentions_per_1000_words"].mean()
                ),
                "mean_board_reference_mentions_per_1000_words": float(
                    group.merge(
                        df[["ordinance_doc_id", "board_reference_mentions_per_1000_words"]],
                        on="ordinance_doc_id",
                        how="left",
                    )["board_reference_mentions_per_1000_words"].mean()
                ),
                "mean_reference_bearing_section_share": float(
                    group.merge(
                        df[["ordinance_doc_id", "reference_bearing_section_share"]],
                        on="ordinance_doc_id",
                        how="left",
                    )["reference_bearing_section_share"].mean()
                ),
            }
        )
    summary_df = pd.DataFrame.from_records(summary_rows).sort_values("dependency_profile_id").reset_index(drop=True)
    assign = assign.merge(summary_df[["dependency_profile_id", "cluster_label"]], on="dependency_profile_id", how="left")
    return sweep_df, assign, summary_df


def _fit_residual_case_model(
    df: pd.DataFrame,
    *,
    target_col: str,
    categorical_cols: list[str],
    output_label: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    select_cols = list(
        dict.fromkeys(
            [
                "ordinance_doc_id",
                "issue_id",
                "issue_date",
                "jurisdiction_display_name",
                "jurisdiction_region",
                "jurisdiction_level",
                "newspaper_slug",
                target_col,
                *categorical_cols,
            ]
        )
    )
    required = ["ordinance_doc_id", target_col, *categorical_cols]
    if df.empty or any(col not in df.columns for col in required):
        return pd.DataFrame(), pd.DataFrame()
    model_df = df[select_cols].copy()
    model_df[target_col] = pd.to_numeric(model_df[target_col], errors="coerce")
    model_df = model_df.dropna(subset=[target_col]).reset_index(drop=True)
    if len(model_df) < 32:
        return pd.DataFrame(), pd.DataFrame()
    for col in categorical_cols:
        model_df[col] = model_df[col].astype(str).replace({"": "unknown"}).fillna("unknown")
    formula = f"{target_col} ~ " + " + ".join(f"C({col})" for col in categorical_cols)
    try:
        result = smf.ols(formula, data=model_df).fit(cov_type="HC1")
    except Exception:
        return pd.DataFrame(), pd.DataFrame()

    fitted = result.predict(model_df)
    residual = model_df[target_col] - fitted
    resid_std = float(np.std(residual.to_numpy(dtype=float), ddof=0))
    model_df["predicted_value"] = fitted.astype(float)
    model_df["residual_value"] = residual.astype(float)
    model_df["residual_z"] = (
        model_df["residual_value"] / resid_std if math.isfinite(resid_std) and resid_std > 0.0 else 0.0
    )
    model_df["target_col"] = target_col
    model_df["model_label"] = output_label
    summary_rows = [
        {
            "model_label": output_label,
            "target_col": target_col,
            "term": str(term),
            "coefficient": float(coef),
            "std_error": float(result.bse.get(term, np.nan)),
            "pvalue": float(result.pvalues.get(term, np.nan)),
            "r_squared": float(getattr(result, "rsquared", np.nan)),
            "nobs": int(result.nobs),
        }
        for term, coef in result.params.items()
    ]
    return pd.DataFrame.from_records(summary_rows), model_df


def _build_component_exemplars(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    metric_cols = [
        "overall_legal_complexity_index",
        "structural_complexity_index",
        "dependency_complexity_index",
        "linguistic_complexity_index",
        "reference_mentions_per_1000_words",
        "dependency_trigger_mentions_per_1000_words",
        "flesch_kincaid_grade_proxy",
    ]
    available = [col for col in metric_cols if col in df.columns]
    records: list[pd.DataFrame] = []
    base_cols = [
        "ordinance_doc_id",
        "issue_id",
        "issue_date",
        "jurisdiction_display_name",
        "jurisdiction_region",
        "jurisdiction_level",
        "newspaper_slug",
        "title_line",
    ]
    for metric in available:
        sub = (
            df[base_cols + [metric]]
            .copy()
            .sort_values(metric, ascending=False)
            .head(10)
            .reset_index(drop=True)
        )
        sub["metric_name"] = metric
        sub["rank"] = np.arange(1, len(sub) + 1)
        sub["metric_value"] = pd.to_numeric(sub[metric], errors="coerce").astype(float)
        records.append(sub.drop(columns=[metric]))
    return pd.concat(records, ignore_index=True) if records else pd.DataFrame()


def _fit_bridge_linear_models(
    df: pd.DataFrame,
    *,
    outcomes: dict[str, str],
    feature_cols: list[str],
    categorical_cols: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for target_col, model_label in outcomes.items():
        if target_col not in df.columns:
            continue
        model_df = df.dropna(subset=[target_col]).copy()
        if len(model_df) < 24:
            continue
        for col in categorical_cols:
            if col in model_df.columns:
                model_df[col] = model_df[col].astype(str).replace({"": "unknown"}).fillna("unknown")
        formula = (
            f"{target_col} ~ "
            + " + ".join(feature_cols)
            + " + "
            + " + ".join(f"C({col})" for col in categorical_cols)
        )
        try:
            result = smf.ols(formula, data=model_df).fit(cov_type="HC1")
        except Exception:
            continue
        for term, coef in result.params.items():
            rows.append(
                {
                    "model_label": model_label,
                    "target_col": target_col,
                    "formula": formula,
                    "term": str(term),
                    "coefficient": float(coef),
                    "std_error": float(result.bse.get(term, np.nan)),
                    "pvalue": float(result.pvalues.get(term, np.nan)),
                    "r_squared": float(getattr(result, "rsquared", np.nan)),
                    "nobs": int(result.nobs),
                }
            )
    return pd.DataFrame.from_records(rows)


def _plot_grouped_component_bars(summary_df: pd.DataFrame, *, group_col: str, output_path: Path, title: str) -> None:
    if summary_df.empty:
        return
    value_cols = [
        "structural_complexity_index__vintage_adjusted",
        "dependency_complexity_index__vintage_adjusted",
        "linguistic_complexity_index__vintage_adjusted",
        "overall_legal_complexity_index__vintage_adjusted",
    ]
    available = [col for col in value_cols if col in summary_df.columns]
    if not available:
        return
    labels = summary_df[group_col].astype(str).tolist()
    x = np.arange(len(labels))
    width = 0.18
    fig, ax = plt.subplots(figsize=(10, 5.6))
    display_names = {
        "structural_complexity_index__vintage_adjusted": "Structural",
        "dependency_complexity_index__vintage_adjusted": "Dependency",
        "linguistic_complexity_index__vintage_adjusted": "Linguistic",
        "overall_legal_complexity_index__vintage_adjusted": "Overall",
    }
    for idx, col in enumerate(available):
        offset = (idx - (len(available) - 1) / 2.0) * width
        vals = pd.to_numeric(summary_df[col], errors="coerce").fillna(0.0).to_numpy()
        ax.bar(x + offset, vals, width=width, label=display_names.get(col, col))
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.legend(frameon=False)
    ax.set_ylabel("Mean index")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_component_correlation_heatmap(corr_df: pd.DataFrame, output_path: Path) -> None:
    if corr_df.empty:
        return
    metric_order = corr_df["metric_name"].astype(str).tolist()
    corr_matrix = corr_df.set_index("metric_name").reindex(metric_order)[metric_order]
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(corr_matrix.to_numpy(dtype=float), cmap="coolwarm", vmin=-1.0, vmax=1.0)
    ax.set_xticks(np.arange(len(metric_order)))
    ax.set_yticks(np.arange(len(metric_order)))
    ax.set_xticklabels(metric_order, rotation=45, ha="right")
    ax.set_yticklabels(metric_order)
    ax.set_title("Legal complexity component correlations")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_bridge_scatter(bridge_df: pd.DataFrame, output_path: Path) -> None:
    if bridge_df.empty or "overall_legal_complexity_index" not in bridge_df.columns or "purpose_alignment_non_other_word_share" not in bridge_df.columns:
        return
    fig, ax = plt.subplots(figsize=(7.5, 5.6))
    colors = {"northeast": "#1f77b4", "midwest": "#2ca02c", "south": "#d62728", "west": "#ff7f0e"}
    for region, group in bridge_df.groupby("jurisdiction_region", dropna=False):
        ax.scatter(
            group["overall_legal_complexity_index"],
            group["purpose_alignment_non_other_word_share"],
            s=40,
            alpha=0.8,
            label=str(region),
            color=colors.get(str(region), "#7f7f7f"),
        )
    ax.set_xlabel("Overall legal complexity index")
    ax.set_ylabel("Purpose / section non-other alignment")
    ax.set_title("Pilot coherence against full-sample legal complexity")
    ax.legend(frameon=False)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def build_fullsample_legal_complexity(
    *,
    run_root: Path,
    source_run_root: Path | str = DEFAULT_SOURCE_RUN_ROOT,
    pilot_run_root: Path | str = DEFAULT_PILOT_RUN_ROOT,
) -> dict[str, Any]:
    source_root = Path(source_run_root).expanduser().resolve()
    pilot_root = Path(pilot_run_root).expanduser().resolve()
    complexity_dir = run_root / "complexity"
    bridge_dir = run_root / "bridge"
    figures_dir = run_root / "figures"
    reports_dir = run_root / "reports"
    manifests_dir = run_root / "manifests"
    for path in (complexity_dir, bridge_dir, figures_dir, reports_dir, manifests_dir):
        path.mkdir(parents=True, exist_ok=True)

    ordinances = pd.read_parquet(source_root / "corpus" / "ordinances.parquet")
    ordinances = ordinances[
        ordinances["jurisdiction_resolution_status"].astype(str).eq("resolved_with_census")
        & ordinances["clean_text"].astype(str).str.strip().astype(bool)
    ].copy()
    ordinances = _add_vintage_fields(ordinances)
    sections = pd.read_parquet(source_root / "structure" / "sections.parquet")
    sections = sections[sections["ordinance_doc_id"].astype(str).isin(set(ordinances["ordinance_doc_id"].astype(str)))].copy()

    doc_base = ordinances[
        [
            "ordinance_doc_id",
            "issue_id",
            "issue_date",
            "issue_year",
            "vintage_bin",
            "census_id_pid6",
            "jurisdiction_display_name",
            "jurisdiction_region",
            "jurisdiction_level",
            "newspaper_slug",
            "word_count",
            "char_count",
            "line_count",
            "title_line",
            "clean_text",
        ]
    ].copy()

    # Structural layer from sections.
    section_counts = sections.groupby("ordinance_doc_id", dropna=False)["section_id"].nunique().rename("section_count")
    section_lengths = sections.groupby("ordinance_doc_id", dropna=False)["word_count"].agg(
        mean_section_word_count="mean",
        median_section_word_count="median",
        max_section_word_count="max",
        std_section_word_count="std",
    )
    long_section_flags = sections.assign(
        section_over_750=(pd.to_numeric(sections["word_count"], errors="coerce").fillna(0.0) >= 750).astype(int),
        section_over_1500=(pd.to_numeric(sections["word_count"], errors="coerce").fillna(0.0) >= 1500).astype(int),
    )
    long_section_summary = long_section_flags.groupby("ordinance_doc_id", dropna=False).agg(
        sections_over_750_share=("section_over_750", "mean"),
        sections_over_1500_share=("section_over_1500", "mean"),
    )
    kind_pivot = sections.pivot_table(index="ordinance_doc_id", columns="section_kind", values="section_id", aggfunc="count", fill_value=0)
    kind_summary = pd.DataFrame(
        {
            "distinct_section_kind_count": (kind_pivot > 0).sum(axis=1),
            "section_kind_entropy": kind_pivot.apply(lambda row: _normalized_entropy_from_counts(row.to_numpy(dtype=float)), axis=1),
        }
    )

    # Dependency / reference layer.
    section_ref = _reference_metrics_for_section_frame(sections)
    ref_doc = section_ref.groupby("ordinance_doc_id", dropna=False).agg(
        formal_reference_mentions=("formal_reference_mentions", "sum"),
        formal_reference_words=("formal_reference_words", "sum"),
        dependency_trigger_mentions=("dependency_trigger_mentions", "sum"),
        map_reference_mentions=("map_reference_mentions", "sum"),
        board_reference_mentions=("board_reference_mentions", "sum"),
        reference_mentions_total=("reference_mentions_total", "sum"),
        reference_bearing_section_share=("any_reference_flag", "mean"),
        formal_reference_section_share=("any_formal_reference_flag", "mean"),
        max_reference_mentions_in_section=("reference_mentions_total", "max"),
        mean_reference_mentions_in_section=("reference_mentions_total", "mean"),
        max_formal_reference_mentions_in_section=("formal_reference_mentions", "max"),
    )

    # Linguistic layer from full ordinance text.
    linguistic_rows = []
    for row in doc_base[["ordinance_doc_id", "clean_text"]].itertuples(index=False):
        metrics = _text_linguistic_metrics(str(row.clean_text))
        metrics["ordinance_doc_id"] = clean_optional_str(row.ordinance_doc_id)
        linguistic_rows.append(metrics)
    linguistic_df = pd.DataFrame.from_records(linguistic_rows)

    complexity_df = (
        doc_base.drop(columns=["clean_text"])
        .merge(section_counts.reset_index(), on="ordinance_doc_id", how="left")
        .merge(section_lengths.reset_index(), on="ordinance_doc_id", how="left")
        .merge(long_section_summary.reset_index(), on="ordinance_doc_id", how="left")
        .merge(kind_summary.reset_index(), on="ordinance_doc_id", how="left")
        .merge(ref_doc.reset_index(), on="ordinance_doc_id", how="left")
        .merge(linguistic_df, on="ordinance_doc_id", how="left")
        .fillna(0.0)
    )

    complexity_df["section_length_cv"] = complexity_df.apply(
        lambda row: _safe_div(row["std_section_word_count"], row["mean_section_word_count"]),
        axis=1,
    )
    complexity_df["reference_mentions_per_1000_words"] = (
        complexity_df["reference_mentions_total"] / complexity_df["word_count"].clip(lower=1).astype(float) * 1000.0
    )
    complexity_df["formal_reference_mentions_per_1000_words"] = (
        complexity_df["formal_reference_mentions"] / complexity_df["word_count"].clip(lower=1).astype(float) * 1000.0
    )
    complexity_df["dependency_trigger_mentions_per_1000_words"] = (
        complexity_df["dependency_trigger_mentions"] / complexity_df["word_count"].clip(lower=1).astype(float) * 1000.0
    )
    complexity_df["map_reference_mentions_per_1000_words"] = (
        complexity_df["map_reference_mentions"] / complexity_df["word_count"].clip(lower=1).astype(float) * 1000.0
    )
    complexity_df["board_reference_mentions_per_1000_words"] = (
        complexity_df["board_reference_mentions"] / complexity_df["word_count"].clip(lower=1).astype(float) * 1000.0
    )

    # Component indices.
    structural_features = {
        "log_word_count": np.log1p(pd.to_numeric(complexity_df["word_count"], errors="coerce").fillna(0.0)),
        "log_section_count": np.log1p(pd.to_numeric(complexity_df["section_count"], errors="coerce").fillna(0.0)),
        "section_kind_entropy": pd.to_numeric(complexity_df["section_kind_entropy"], errors="coerce").fillna(0.0),
        "section_length_cv": pd.to_numeric(complexity_df["section_length_cv"], errors="coerce").fillna(0.0),
        "sections_over_750_share": pd.to_numeric(complexity_df["sections_over_750_share"], errors="coerce").fillna(0.0),
        "max_section_word_count_log": np.log1p(pd.to_numeric(complexity_df["max_section_word_count"], errors="coerce").fillna(0.0)),
    }
    dependency_features = {
        "reference_mentions_per_1000_words": pd.to_numeric(complexity_df["reference_mentions_per_1000_words"], errors="coerce").fillna(0.0),
        "formal_reference_mentions_per_1000_words": pd.to_numeric(complexity_df["formal_reference_mentions_per_1000_words"], errors="coerce").fillna(0.0),
        "dependency_trigger_mentions_per_1000_words": pd.to_numeric(complexity_df["dependency_trigger_mentions_per_1000_words"], errors="coerce").fillna(0.0),
        "map_reference_mentions_per_1000_words": pd.to_numeric(complexity_df["map_reference_mentions_per_1000_words"], errors="coerce").fillna(0.0),
        "board_reference_mentions_per_1000_words": pd.to_numeric(complexity_df["board_reference_mentions_per_1000_words"], errors="coerce").fillna(0.0),
        "reference_bearing_section_share": pd.to_numeric(complexity_df["reference_bearing_section_share"], errors="coerce").fillna(0.0),
    }
    linguistic_features = {
        "avg_sentence_length_words": pd.to_numeric(complexity_df["avg_sentence_length_words"], errors="coerce").fillna(0.0),
        "avg_word_length_chars": pd.to_numeric(complexity_df["avg_word_length_chars"], errors="coerce").fillna(0.0),
        "long_word_share_ge7": pd.to_numeric(complexity_df["long_word_share_ge7"], errors="coerce").fillna(0.0),
        "avg_syllables_per_word_proxy": pd.to_numeric(complexity_df["avg_syllables_per_word_proxy"], errors="coerce").fillna(0.0),
        "flesch_kincaid_grade_proxy": pd.to_numeric(complexity_df["flesch_kincaid_grade_proxy"], errors="coerce").fillna(0.0),
    }
    for name, series in {**structural_features, **dependency_features, **linguistic_features}.items():
        complexity_df[name] = pd.to_numeric(series, errors="coerce").fillna(0.0)

    complexity_df["structural_complexity_index"] = pd.concat([_zscore(series) for series in structural_features.values()], axis=1).mean(axis=1)
    complexity_df["dependency_complexity_index"] = pd.concat([_zscore(series) for series in dependency_features.values()], axis=1).mean(axis=1)
    complexity_df["linguistic_complexity_index"] = pd.concat([_zscore(series) for series in linguistic_features.values()], axis=1).mean(axis=1)
    complexity_df["overall_legal_complexity_index"] = complexity_df[
        ["structural_complexity_index", "dependency_complexity_index", "linguistic_complexity_index"]
    ].mean(axis=1)

    write_parquet(complexity_df.sort_values("ordinance_doc_id").reset_index(drop=True), complexity_dir / "complexity_by_doc.parquet")

    component_cols = [
        "structural_complexity_index",
        "dependency_complexity_index",
        "linguistic_complexity_index",
        "overall_legal_complexity_index",
        "section_kind_entropy",
        "reference_mentions_per_1000_words",
        "avg_sentence_length_words",
    ]
    corr = complexity_df[component_cols].corr(numeric_only=True)
    corr_df = corr.reset_index().rename(columns={"index": "metric_name"})
    write_parquet(corr_df, complexity_dir / "complexity_component_correlations.parquet")

    region_summary = _vintage_adjusted_mean_table(
        complexity_df,
        group_col="jurisdiction_region",
        vintage_col="vintage_bin",
        value_cols=[
            "structural_complexity_index",
            "dependency_complexity_index",
            "linguistic_complexity_index",
            "overall_legal_complexity_index",
        ],
    )
    level_summary = _vintage_adjusted_mean_table(
        complexity_df,
        group_col="jurisdiction_level",
        vintage_col="vintage_bin",
        value_cols=[
            "structural_complexity_index",
            "dependency_complexity_index",
            "linguistic_complexity_index",
            "overall_legal_complexity_index",
        ],
    )
    write_parquet(region_summary, complexity_dir / "complexity_by_region_vintage_adjusted.parquet")
    write_parquet(level_summary, complexity_dir / "complexity_by_level_vintage_adjusted.parquet")

    dependency_component_cols = [
        "formal_reference_mentions_per_1000_words",
        "dependency_trigger_mentions_per_1000_words",
        "map_reference_mentions_per_1000_words",
        "board_reference_mentions_per_1000_words",
        "reference_bearing_section_share",
    ]
    dependency_region_summary = _vintage_adjusted_mean_table(
        complexity_df,
        group_col="jurisdiction_region",
        vintage_col="vintage_bin",
        value_cols=dependency_component_cols,
    )
    dependency_level_summary = _vintage_adjusted_mean_table(
        complexity_df,
        group_col="jurisdiction_level",
        vintage_col="vintage_bin",
        value_cols=dependency_component_cols,
    )
    write_parquet(dependency_region_summary, complexity_dir / "dependency_components_by_region_vintage_adjusted.parquet")
    write_parquet(dependency_level_summary, complexity_dir / "dependency_components_by_level_vintage_adjusted.parquet")

    effect_rows: list[pd.DataFrame] = []
    for target_col in [
        "structural_complexity_index",
        "dependency_complexity_index",
        "linguistic_complexity_index",
        "overall_legal_complexity_index",
    ]:
        effect_df = _fit_group_effect_model(
            complexity_df,
            target_col=target_col,
            categorical_cols=["vintage_bin", "jurisdiction_region", "jurisdiction_level"],
        )
        if not effect_df.empty:
            effect_rows.append(effect_df)
    effect_table = pd.concat(effect_rows, ignore_index=True) if effect_rows else pd.DataFrame()
    write_parquet(effect_table, complexity_dir / "complexity_region_level_effects.parquet")
    if not effect_table.empty:
        top_effects = effect_table.sort_values(["target_col", "pvalue", "coefficient"], ascending=[True, True, False]).groupby("target_col").head(8).reset_index(drop=True)
    else:
        top_effects = pd.DataFrame()
    write_parquet(top_effects, complexity_dir / "top_complexity_region_level_effects.parquet")

    dependency_effect_rows: list[pd.DataFrame] = []
    for target_col in dependency_component_cols:
        effect_df = _fit_group_effect_model(
            complexity_df,
            target_col=target_col,
            categorical_cols=["vintage_bin", "jurisdiction_region", "jurisdiction_level"],
        )
        if not effect_df.empty:
            dependency_effect_rows.append(effect_df)
    dependency_effect_table = pd.concat(dependency_effect_rows, ignore_index=True) if dependency_effect_rows else pd.DataFrame()
    write_parquet(dependency_effect_table, complexity_dir / "dependency_component_region_level_effects.parquet")
    if not dependency_effect_table.empty:
        dependency_top_effects = (
            dependency_effect_table.sort_values(["target_col", "pvalue", "coefficient"], ascending=[True, True, False])
            .groupby("target_col")
            .head(8)
            .reset_index(drop=True)
        )
    else:
        dependency_top_effects = pd.DataFrame()
    write_parquet(dependency_top_effects, complexity_dir / "top_dependency_component_effects.parquet")

    exemplar_df = _build_component_exemplars(complexity_df)
    write_parquet(exemplar_df, complexity_dir / "complexity_component_exemplars.parquet")

    residual_models: list[pd.DataFrame] = []
    residual_frames: list[pd.DataFrame] = []
    for target_col, label in [
        ("overall_legal_complexity_index", "overall_complexity"),
        ("dependency_complexity_index", "dependency_complexity"),
        ("linguistic_complexity_index", "linguistic_complexity"),
    ]:
        model_rows, residual_df = _fit_residual_case_model(
            complexity_df,
            target_col=target_col,
            categorical_cols=["vintage_bin", "jurisdiction_region", "jurisdiction_level"],
            output_label=label,
        )
        if not model_rows.empty:
            residual_models.append(model_rows)
        if not residual_df.empty:
            residual_frames.append(residual_df)
            write_parquet(
                residual_df.sort_values("residual_value", ascending=False).head(15).reset_index(drop=True),
                complexity_dir / f"{label}__top_positive_residuals.parquet",
            )
            write_parquet(
                residual_df.sort_values("residual_value", ascending=True).head(15).reset_index(drop=True),
                complexity_dir / f"{label}__top_negative_residuals.parquet",
            )
    residual_model_df = pd.concat(residual_models, ignore_index=True) if residual_models else pd.DataFrame()
    residual_case_df = pd.concat(residual_frames, ignore_index=True) if residual_frames else pd.DataFrame()
    write_parquet(residual_model_df, complexity_dir / "complexity_residual_models.parquet")
    write_parquet(residual_case_df, complexity_dir / "complexity_residual_cases.parquet")

    sweep_df, family_assignments, family_summary = _complexity_family_outputs(complexity_df)
    write_parquet(sweep_df, complexity_dir / "complexity_family_sweep.parquet")
    write_parquet(family_assignments, complexity_dir / "complexity_family_assignments.parquet")
    write_parquet(family_summary, complexity_dir / "complexity_family_summary.parquet")
    if not family_assignments.empty:
        family_presence = family_assignments[["ordinance_doc_id", "complexity_family_id", "cluster_label", "jurisdiction_region", "jurisdiction_level", "vintage_bin"]].copy()
        family_presence["present"] = 1.0
        family_region = (
            family_presence.groupby(["jurisdiction_region", "complexity_family_id", "cluster_label"], dropna=False)["present"]
            .mean()
            .reset_index(name="presence_rate_raw")
        )
        family_level = (
            family_presence.groupby(["jurisdiction_level", "complexity_family_id", "cluster_label"], dropna=False)["present"]
            .mean()
            .reset_index(name="presence_rate_raw")
        )
    else:
        family_region = pd.DataFrame()
        family_level = pd.DataFrame()
    write_parquet(family_region, complexity_dir / "complexity_family_by_region.parquet")
    write_parquet(family_level, complexity_dir / "complexity_family_by_level.parquet")

    dependency_sweep, dependency_assignments, dependency_summary = _dependency_profile_outputs(complexity_df)
    write_parquet(dependency_sweep, complexity_dir / "dependency_profile_sweep.parquet")
    write_parquet(dependency_assignments, complexity_dir / "dependency_profile_assignments.parquet")
    write_parquet(dependency_summary, complexity_dir / "dependency_profile_summary.parquet")
    if not dependency_assignments.empty:
        dep_region = (
            dependency_assignments.assign(present=1.0)
            .groupby(["jurisdiction_region", "dependency_profile_id", "cluster_label"], dropna=False)["present"]
            .mean()
            .reset_index(name="presence_rate_raw")
        )
        dep_level = (
            dependency_assignments.assign(present=1.0)
            .groupby(["jurisdiction_level", "dependency_profile_id", "cluster_label"], dropna=False)["present"]
            .mean()
            .reset_index(name="presence_rate_raw")
        )
    else:
        dep_region = pd.DataFrame()
        dep_level = pd.DataFrame()
    write_parquet(dep_region, complexity_dir / "dependency_profile_by_region.parquet")
    write_parquet(dep_level, complexity_dir / "dependency_profile_by_level.parquet")

    # Pilot bridge to governance/coherence.
    bridge_summary: dict[str, Any] = {
        "available": False,
        "doc_count": 0,
        "coherence_model_rows": 0,
        "governance_model_rows": 0,
    }
    if pilot_root.is_dir():
        coherence_path = pilot_root / "research" / "coherence" / "document_section_motive_alignment__pilot.parquet"
        gov_path = pilot_root / "research" / "governance_style" / "governance_style_by_doc__pilot.parquet"
        instrument_complexity_path = pilot_root / "research" / "instruments" / "instrument_complexity_by_doc__pilot.parquet"
        if coherence_path.is_file():
            coherence_df = pd.read_parquet(coherence_path)
            bridge_df = complexity_df.merge(
                coherence_df[
                    [
                        "ordinance_doc_id",
                        "jurisdiction_region",
                        "jurisdiction_level",
                        "vintage_bin",
                        "doc_primary_motive",
                        "purpose_alignment_word_share",
                        "purpose_alignment_non_other_word_share",
                        "purpose_section_exact_match_flag",
                    ]
                ],
                on=["ordinance_doc_id", "jurisdiction_region", "jurisdiction_level", "vintage_bin"],
                how="inner",
            )
            if gov_path.is_file():
                gov_df = pd.read_parquet(gov_path)
                bridge_df = bridge_df.merge(gov_df, on=["ordinance_doc_id", "jurisdiction_region", "jurisdiction_level"], how="left")
            if instrument_complexity_path.is_file():
                instrument_df = pd.read_parquet(instrument_complexity_path)
                bridge_df = bridge_df.merge(
                    instrument_df[
                        [
                            "ordinance_doc_id",
                            "distinct_instrument_type_count",
                            "instrument_entropy",
                            "top_instrument_share",
                        ]
                    ],
                    on="ordinance_doc_id",
                    how="left",
                )
            write_parquet(bridge_df, bridge_dir / "pilot_complexity_bridge.parquet")
            coherence_models = pd.DataFrame()
            governance_models = pd.DataFrame()
            subcomponent_models = pd.DataFrame()
            if not bridge_df.empty:
                try:
                    coherence_model = smf.ols(
                        "purpose_alignment_non_other_word_share ~ structural_complexity_index + dependency_complexity_index + linguistic_complexity_index + C(vintage_bin) + C(jurisdiction_level)",
                        data=bridge_df,
                    ).fit(cov_type="HC1")
                    coherence_models = pd.DataFrame.from_records(
                        [
                            {
                                "term": str(term),
                                "coefficient": float(coef),
                                "std_error": float(coherence_model.bse.get(term, np.nan)),
                                "pvalue": float(coherence_model.pvalues.get(term, np.nan)),
                                "r_squared": float(getattr(coherence_model, "rsquared", np.nan)),
                                "nobs": int(coherence_model.nobs),
                            }
                            for term, coef in coherence_model.params.items()
                        ]
                    )
                except Exception:
                    coherence_models = pd.DataFrame()
                subcomponent_feature_cols = [
                    "formal_reference_mentions_per_1000_words",
                    "dependency_trigger_mentions_per_1000_words",
                    "map_reference_mentions_per_1000_words",
                    "board_reference_mentions_per_1000_words",
                    "section_kind_entropy",
                    "flesch_kincaid_grade_proxy",
                ]
                subcomponent_outcomes = {
                    "purpose_alignment_non_other_word_share": "coherence_subcomponents",
                    "direct_rule_share_of_operative": "direct_rule_subcomponents",
                    "discretion_share_of_operative": "discretion_subcomponents",
                    "exception_or_condition_share_of_operative": "exception_condition_subcomponents",
                    "threshold_share_of_operative": "threshold_subcomponents",
                    "distinct_instrument_type_count": "instrument_variety_subcomponents",
                    "instrument_entropy": "instrument_entropy_subcomponents",
                    "top_instrument_share": "instrument_concentration_subcomponents",
                }
                subcomponent_models = _fit_bridge_linear_models(
                    bridge_df,
                    outcomes=subcomponent_outcomes,
                    feature_cols=subcomponent_feature_cols,
                    categorical_cols=["vintage_bin", "jurisdiction_level"],
                )
                if "direct_rule_share_of_operative" in bridge_df.columns:
                    gov_targets = {
                        "direct_rule_share_of_operative": "direct_rule",
                        "discretion_share_of_operative": "discretion",
                        "exception_or_condition_share_of_operative": "exception_condition",
                    }
                    gov_rows: list[pd.DataFrame] = []
                    for target_col, label in gov_targets.items():
                        try:
                            result = smf.ols(
                                f"{target_col} ~ structural_complexity_index + dependency_complexity_index + linguistic_complexity_index + C(vintage_bin) + C(jurisdiction_level)",
                                data=bridge_df.dropna(subset=[target_col]),
                            ).fit(cov_type="HC1")
                        except Exception:
                            continue
                        gov_rows.append(
                            pd.DataFrame.from_records(
                                [
                                    {
                                        "model_label": label,
                                        "target_col": target_col,
                                        "term": str(term),
                                        "coefficient": float(coef),
                                        "std_error": float(result.bse.get(term, np.nan)),
                                        "pvalue": float(result.pvalues.get(term, np.nan)),
                                        "r_squared": float(getattr(result, "rsquared", np.nan)),
                                        "nobs": int(result.nobs),
                                    }
                                    for term, coef in result.params.items()
                                ]
                            )
                        )
                    governance_models = pd.concat(gov_rows, ignore_index=True) if gov_rows else pd.DataFrame()
            write_parquet(coherence_models, bridge_dir / "pilot_coherence_models.parquet")
            write_parquet(governance_models, bridge_dir / "pilot_governance_models.parquet")
            write_parquet(subcomponent_models, bridge_dir / "pilot_subcomponent_bridge_models.parquet")
            bridge_summary = {
                "available": True,
                "doc_count": int(bridge_df["ordinance_doc_id"].nunique()) if not bridge_df.empty else 0,
                "coherence_model_rows": int(len(coherence_models)),
                "governance_model_rows": int(len(governance_models)),
                "subcomponent_model_rows": int(len(subcomponent_models)),
            }
        else:
            write_parquet(pd.DataFrame(), bridge_dir / "pilot_complexity_bridge.parquet")
            write_parquet(pd.DataFrame(), bridge_dir / "pilot_coherence_models.parquet")
            write_parquet(pd.DataFrame(), bridge_dir / "pilot_governance_models.parquet")
            write_parquet(pd.DataFrame(), bridge_dir / "pilot_subcomponent_bridge_models.parquet")

    # Figures.
    _plot_grouped_component_bars(region_summary.sort_values("jurisdiction_region"), group_col="jurisdiction_region", output_path=figures_dir / "complexity_by_region_vintage_adjusted.png", title="Legal complexity by region")
    _plot_grouped_component_bars(level_summary.sort_values("jurisdiction_level"), group_col="jurisdiction_level", output_path=figures_dir / "complexity_by_level_vintage_adjusted.png", title="Legal complexity by jurisdiction type")
    _plot_component_correlation_heatmap(corr_df, figures_dir / "complexity_component_correlation_heatmap.png")
    bridge_scatter_path = figures_dir / "pilot_complexity_coherence_scatter.png"
    if (bridge_dir / "pilot_complexity_bridge.parquet").is_file():
        _plot_bridge_scatter(pd.read_parquet(bridge_dir / "pilot_complexity_bridge.parquet"), bridge_scatter_path)

    # Top-line memo.
    top_region = (
        region_summary.sort_values("overall_legal_complexity_index__vintage_adjusted", ascending=False).head(1)["jurisdiction_region"].astype(str).tolist()
    )
    top_level = (
        level_summary.sort_values("overall_legal_complexity_index__vintage_adjusted", ascending=False).head(1)["jurisdiction_level"].astype(str).tolist()
    )
    most_complex = complexity_df.sort_values("overall_legal_complexity_index", ascending=False).head(10)
    least_complex = complexity_df.sort_values("overall_legal_complexity_index", ascending=True).head(10)
    memo_lines = [
        "# Full-sample legal complexity memo",
        "",
        f"- Source sample: {int(len(complexity_df))} clean-text, census-linked ordinances from `{source_root}`.",
        f"- Complexity frame: structural + dependency/reference + linguistic/readability controls.",
        f"- Highest vintage-adjusted overall complexity region: `{clean_optional_str(top_region[0]) if top_region else ''}`.",
        f"- Highest vintage-adjusted overall complexity jurisdiction type: `{clean_optional_str(top_level[0]) if top_level else ''}`.",
        "",
        "## Structural layer",
        f"- Mean section count: {float(complexity_df['section_count'].mean()):.2f}",
        f"- Median section count: {float(complexity_df['section_count'].median()):.2f}",
        f"- Mean section-kind entropy: {float(complexity_df['section_kind_entropy'].mean()):.3f}",
        "",
        "## Dependency layer",
        f"- Mean formal-reference mentions per 1,000 words: {float(complexity_df['formal_reference_mentions_per_1000_words'].mean()):.3f}",
        f"- Mean dependency-trigger mentions per 1,000 words: {float(complexity_df['dependency_trigger_mentions_per_1000_words'].mean()):.3f}",
        f"- Mean map-reference mentions per 1,000 words: {float(complexity_df['map_reference_mentions_per_1000_words'].mean()):.3f}",
        f"- Mean board-reference mentions per 1,000 words: {float(complexity_df['board_reference_mentions_per_1000_words'].mean()):.3f}",
        f"- Mean reference-bearing section share: {float(complexity_df['reference_bearing_section_share'].mean()):.3f}",
        "",
        "## Linguistic layer",
        f"- Mean sentence length: {float(complexity_df['avg_sentence_length_words'].mean()):.3f} words",
        f"- Mean long-word share (>=7 chars): {float(complexity_df['long_word_share_ge7'].mean()):.3f}",
        f"- Mean Flesch-Kincaid grade proxy: {float(complexity_df['flesch_kincaid_grade_proxy'].mean()):.3f}",
        "",
        "## Most complex ordinances",
    ]
    for row in most_complex.itertuples(index=False):
        memo_lines.append(
            f"- `{clean_optional_str(getattr(row, 'ordinance_doc_id'))}` | region={clean_optional_str(getattr(row, 'jurisdiction_region'))} | level={clean_optional_str(getattr(row, 'jurisdiction_level'))} | overall={float(getattr(row, 'overall_legal_complexity_index')):.3f}"
        )
    memo_lines.extend(["", "## Least complex ordinances"])
    for row in least_complex.itertuples(index=False):
        memo_lines.append(
            f"- `{clean_optional_str(getattr(row, 'ordinance_doc_id'))}` | region={clean_optional_str(getattr(row, 'jurisdiction_region'))} | level={clean_optional_str(getattr(row, 'jurisdiction_level'))} | overall={float(getattr(row, 'overall_legal_complexity_index')):.3f}"
        )
    if not residual_case_df.empty:
        overall_resid = residual_case_df[residual_case_df["model_label"].astype(str).eq("overall_complexity")].copy()
        memo_lines.extend(["", "## Unexpectedly complex given vintage, region, and level"])
        for row in overall_resid.sort_values("residual_value", ascending=False).head(8).itertuples(index=False):
            memo_lines.append(
                f"- `{clean_optional_str(getattr(row, 'ordinance_doc_id'))}` | {clean_optional_str(getattr(row, 'jurisdiction_display_name'))} | region={clean_optional_str(getattr(row, 'jurisdiction_region'))} | level={clean_optional_str(getattr(row, 'jurisdiction_level'))} | residual={float(getattr(row, 'residual_value')):.3f}"
            )
        memo_lines.extend(["", "## Unexpectedly simple given vintage, region, and level"])
        for row in overall_resid.sort_values("residual_value", ascending=True).head(8).itertuples(index=False):
            memo_lines.append(
                f"- `{clean_optional_str(getattr(row, 'ordinance_doc_id'))}` | {clean_optional_str(getattr(row, 'jurisdiction_display_name'))} | region={clean_optional_str(getattr(row, 'jurisdiction_region'))} | level={clean_optional_str(getattr(row, 'jurisdiction_level'))} | residual={float(getattr(row, 'residual_value')):.3f}"
            )
    if not dependency_summary.empty:
        memo_lines.extend(["", "## Dependency-profile families"])
        for row in dependency_summary.sort_values("ordinance_count", ascending=False).itertuples(index=False):
            memo_lines.append(
                f"- profile {int(getattr(row, 'dependency_profile_id'))}: {clean_optional_str(getattr(row, 'cluster_label'))} | docs={int(getattr(row, 'ordinance_count'))} | formal={float(getattr(row, 'mean_formal_reference_mentions_per_1000_words')):.2f} | triggers={float(getattr(row, 'mean_dependency_trigger_mentions_per_1000_words')):.2f} | boards={float(getattr(row, 'mean_board_reference_mentions_per_1000_words')):.2f}"
            )
    if not coherence_models.empty:
        memo_lines.extend(["", "## Bridge to pilot semantics"])
        bridge_terms = coherence_models[
            coherence_models["term"].astype(str).isin(
                ["structural_complexity_index", "dependency_complexity_index", "linguistic_complexity_index"]
            )
        ].copy()
        for row in bridge_terms.sort_values("pvalue").itertuples(index=False):
            memo_lines.append(
                f"- coherence bridge: term={clean_optional_str(getattr(row, 'term'))} | coef={float(getattr(row, 'coefficient')):.3f} | p={float(getattr(row, 'pvalue')):.4f}"
            )
    if not subcomponent_models.empty:
        memo_lines.extend(["", "## Bridge subcomponents"])
        bridge_sub = subcomponent_models[
            ~subcomponent_models["term"].astype(str).str.startswith(("Intercept", "C("))
            & (subcomponent_models["pvalue"] < 0.10)
        ].copy()
        bridge_sub["target_rank"] = bridge_sub.groupby("target_col")["pvalue"].rank(method="first")
        bridge_sub = bridge_sub[bridge_sub["target_rank"] <= 2].sort_values(["target_col", "pvalue", "term"])
        for row in bridge_sub.itertuples(index=False):
            memo_lines.append(
                f"- subcomponent bridge: target={clean_optional_str(getattr(row, 'target_col'))} | term={clean_optional_str(getattr(row, 'term'))} | coef={float(getattr(row, 'coefficient')):.3f} | p={float(getattr(row, 'pvalue')):.4f}"
            )
    write_text(reports_dir / "legal_complexity_memo.md", "\n".join(memo_lines))

    case_note_lines = [
        "# Legal complexity casebook",
        "",
        "This note collects concrete high-end exemplars and residual cases from the full-sample legal-complexity surface.",
        "",
        "## Top component exemplars",
    ]
    if not exemplar_df.empty:
        for metric_name, group in exemplar_df.groupby("metric_name", dropna=False):
            case_note_lines.extend(["", f"### {clean_optional_str(metric_name)}"])
            for row in group.sort_values("rank").head(5).itertuples(index=False):
                case_note_lines.append(
                    f"- rank {int(getattr(row, 'rank'))}: `{clean_optional_str(getattr(row, 'ordinance_doc_id'))}` | {clean_optional_str(getattr(row, 'jurisdiction_display_name'))} | region={clean_optional_str(getattr(row, 'jurisdiction_region'))} | level={clean_optional_str(getattr(row, 'jurisdiction_level'))} | value={float(getattr(row, 'metric_value')):.3f} | title={clean_optional_str(getattr(row, 'title_line'))}"
                )
    if not residual_case_df.empty:
        case_note_lines.extend(["", "## Residual complexity cases"])
        for label in ["overall_complexity", "dependency_complexity", "linguistic_complexity"]:
            sub = residual_case_df[residual_case_df["model_label"].astype(str).eq(label)].copy()
            if sub.empty:
                continue
            case_note_lines.extend(["", f"### {clean_optional_str(label)}: most above expectation"])
            for row in sub.sort_values("residual_value", ascending=False).head(5).itertuples(index=False):
                case_note_lines.append(
                    f"- `{clean_optional_str(getattr(row, 'ordinance_doc_id'))}` | {clean_optional_str(getattr(row, 'jurisdiction_display_name'))} | region={clean_optional_str(getattr(row, 'jurisdiction_region'))} | level={clean_optional_str(getattr(row, 'jurisdiction_level'))} | residual={float(getattr(row, 'residual_value')):.3f} | predicted={float(getattr(row, 'predicted_value')):.3f}"
                )
            case_note_lines.extend(["", f"### {clean_optional_str(label)}: most below expectation"])
            for row in sub.sort_values("residual_value", ascending=True).head(5).itertuples(index=False):
                case_note_lines.append(
                    f"- `{clean_optional_str(getattr(row, 'ordinance_doc_id'))}` | {clean_optional_str(getattr(row, 'jurisdiction_display_name'))} | region={clean_optional_str(getattr(row, 'jurisdiction_region'))} | level={clean_optional_str(getattr(row, 'jurisdiction_level'))} | residual={float(getattr(row, 'residual_value')):.3f} | predicted={float(getattr(row, 'predicted_value')):.3f}"
                )
    if not dependency_summary.empty:
        case_note_lines.extend(["", "## Dependency profile families"])
        for row in dependency_summary.sort_values("ordinance_count", ascending=False).itertuples(index=False):
            case_note_lines.append(
                f"- profile {int(getattr(row, 'dependency_profile_id'))}: {clean_optional_str(getattr(row, 'cluster_label'))} | docs={int(getattr(row, 'ordinance_count'))} | formal={float(getattr(row, 'mean_formal_reference_mentions_per_1000_words')):.2f} | triggers={float(getattr(row, 'mean_dependency_trigger_mentions_per_1000_words')):.2f} | map={float(getattr(row, 'mean_map_reference_mentions_per_1000_words')):.2f} | board={float(getattr(row, 'mean_board_reference_mentions_per_1000_words')):.2f}"
            )
    write_text(reports_dir / "legal_complexity_casebook.md", "\n".join(case_note_lines))

    bridge_note_lines = [
        "# Legal complexity bridge findings",
        "",
        "This note records how the full-sample legal-complexity backbone links to the LLM-native pilot's semantic coherence and governance-style surfaces.",
        "",
        f"- Bridge coverage: docs={int(bridge_summary.get('doc_count', 0))}",
        f"- Coherence model rows: {int(bridge_summary.get('coherence_model_rows', 0))}",
        f"- Governance model rows: {int(bridge_summary.get('governance_model_rows', 0))}",
        f"- Subcomponent model rows: {int(bridge_summary.get('subcomponent_model_rows', 0))}",
    ]
    if not coherence_models.empty:
        bridge_note_lines.extend(["", "## Decomposed complexity to coherence"])
        for row in coherence_models[
            coherence_models["term"].astype(str).isin(
                ["structural_complexity_index", "dependency_complexity_index", "linguistic_complexity_index"]
            )
        ].sort_values("pvalue").itertuples(index=False):
            bridge_note_lines.append(
                f"- term={clean_optional_str(getattr(row, 'term'))} | coef={float(getattr(row, 'coefficient')):.3f} | p={float(getattr(row, 'pvalue')):.4f} | r2={float(getattr(row, 'r_squared')):.3f}"
            )
    if not governance_models.empty:
        bridge_note_lines.extend(["", "## Decomposed complexity to governance style"])
        for target_col in [
            "direct_rule_share_of_operative",
            "discretion_share_of_operative",
            "exception_or_condition_share_of_operative",
        ]:
            sub = governance_models[
                (governance_models["target_col"].astype(str) == target_col)
                & governance_models["term"].astype(str).isin(
                    ["structural_complexity_index", "dependency_complexity_index", "linguistic_complexity_index"]
                )
            ].sort_values("pvalue")
            for row in sub.itertuples(index=False):
                bridge_note_lines.append(
                    f"- target={clean_optional_str(getattr(row, 'target_col'))} | term={clean_optional_str(getattr(row, 'term'))} | coef={float(getattr(row, 'coefficient')):.3f} | p={float(getattr(row, 'pvalue')):.4f}"
                )
    if not subcomponent_models.empty:
        bridge_note_lines.extend(["", "## Dependency, architecture, and readability subcomponents"])
        bridge_sub = subcomponent_models[
            ~subcomponent_models["term"].astype(str).str.startswith(("Intercept", "C("))
            & (subcomponent_models["pvalue"] < 0.10)
        ].copy()
        bridge_sub["target_rank"] = bridge_sub.groupby("target_col")["pvalue"].rank(method="first")
        bridge_sub = bridge_sub[bridge_sub["target_rank"] <= 2].sort_values(["target_col", "pvalue", "term"])
        for row in bridge_sub.itertuples(index=False):
            bridge_note_lines.append(
                f"- target={clean_optional_str(getattr(row, 'target_col'))} | term={clean_optional_str(getattr(row, 'term'))} | coef={float(getattr(row, 'coefficient')):.3f} | p={float(getattr(row, 'pvalue')):.4f} | r2={float(getattr(row, 'r_squared')):.3f}"
            )
    write_text(reports_dir / "legal_complexity_bridge_findings.md", "\n".join(bridge_note_lines))

    summary = {
        "schema_version": "legal_complexity_fullsample_v1",
        "doc_count": int(len(complexity_df)),
        "source_run_root": str(source_root),
        "pilot_run_root": str(pilot_root),
        "mean_section_count": float(complexity_df["section_count"].mean()),
        "mean_reference_mentions_per_1000_words": float(complexity_df["reference_mentions_per_1000_words"].mean()),
        "mean_dependency_mentions_per_1000_words": float(complexity_df["dependency_trigger_mentions_per_1000_words"].mean()),
        "mean_flesch_kincaid_grade_proxy": float(complexity_df["flesch_kincaid_grade_proxy"].mean()),
        "top_region_by_overall_complexity": clean_optional_str(top_region[0]) if top_region else "",
        "top_level_by_overall_complexity": clean_optional_str(top_level[0]) if top_level else "",
        "complexity_family_count": int(family_summary["complexity_family_id"].nunique()) if not family_summary.empty else 0,
        "complexity_family_silhouette_score": float(family_summary["silhouette_score"].iloc[0]) if not family_summary.empty else 0.0,
        "dependency_profile_count": int(dependency_summary["dependency_profile_id"].nunique()) if not dependency_summary.empty else 0,
        "dependency_profile_silhouette_score": float(dependency_summary["silhouette_score"].iloc[0]) if not dependency_summary.empty else 0.0,
        "residual_case_available": not residual_case_df.empty,
        "residual_case_count": int(len(residual_case_df)),
        "bridge_available": bool(bridge_summary["available"]),
        "bridge_doc_count": int(bridge_summary["doc_count"]),
        "bridge_coherence_model_rows": int(bridge_summary["coherence_model_rows"]),
        "bridge_governance_model_rows": int(bridge_summary["governance_model_rows"]),
        "bridge_subcomponent_model_rows": int(bridge_summary.get("subcomponent_model_rows", 0)),
    }
    write_json(manifests_dir / "summary.json", summary)
    return summary
