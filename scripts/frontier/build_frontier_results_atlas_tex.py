#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import re
import shutil
import textwrap
from datetime import date
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PURPOSE_ORDER = [
    "public_health_safety",
    "growth_management",
    "externality_control",
    "infrastructure_coordination",
    "aesthetic_design_control",
    "environmental_protection",
    "exclusion",
    "value_capture",
    "fiscal_zoning",
]
REGION_ORDER = ["northeast", "midwest", "south", "west"]
LEVEL_ORDER = ["city", "village", "borough", "town", "township", "county", "city_and_county"]
REGION_COLORS = {
    "northeast": "#1f77b4",
    "midwest": "#2ca02c",
    "south": "#d62728",
    "west": "#ff7f0e",
}
SECTION_PURPOSE_STACK_ORDER = [
    "externality_control",
    "public_health_safety",
    "growth_management",
    "exclusion",
    "aesthetic_design_control",
    "infrastructure_coordination",
    "environmental_protection",
    "other_purpose_families",
]
SECTION_PURPOSE_COLORS = {
    "externality_control": "#c44e52",
    "public_health_safety": "#4c72b0",
    "growth_management": "#55a868",
    "exclusion": "#8172b3",
    "aesthetic_design_control": "#dd8452",
    "infrastructure_coordination": "#64b5cd",
    "environmental_protection": "#8c8c3c",
    "other_purpose_families": "#b5b5b5",
}
STYLE_DISCRETION_INSTRUMENTS = {"conditional_use", "special_exception", "variance", "appeal", "permit_administration"}
STYLE_STRUCTURAL_INSTRUMENTS = {"district_framework", "zoning_map", "definition", "purpose"}
STYLE_FAMILY_SHORT_LABELS = {
    "high instrument variety | exception-and-condition-heavy | discretion-heavy": "Mixed-toolkit carve-out code",
    "direct-rule-heavy | threshold-heavy | top-instrument concentration": "Rule-heavy threshold code",
    "direct-rule-heavy | large operative scale | high instrument variety": "Long-form mixed rule code",
    "discretion-heavy | top-instrument concentration | exception-and-condition-heavy": "Approval-and-carve-out code",
    "top-instrument concentration | direct-rule-heavy | threshold-heavy": "Single-tool rule code",
}
PURPOSE_BUNDLE_EXAMPLE_ISSUES = [
    "red-bank-register__1949-04-28",
    "biloxi-daily-herald__1973-04-14",
    "orem-geneva-times__1957-12-26",
]
STYLE_FAMILY_READING_GUIDE = {
    "high instrument variety | exception-and-condition-heavy | discretion-heavy": "wide instrument mix, many carve-outs, and frequent routing through approvals or administrative review",
    "direct-rule-heavy | threshold-heavy | top-instrument concentration": "direct standards dominate, numerical thresholds show up often, and a narrow set of instrument types carries most of the ordinance",
    "direct-rule-heavy | large operative scale | high instrument variety": "still rule-heavy, but spread across a long ordinance with many different instrument types",
    "discretion-heavy | top-instrument concentration | exception-and-condition-heavy": "many operative clauses route through permits or variances, and many hinge on conditions or carve-outs",
    "top-instrument concentration | direct-rule-heavy | threshold-heavy": "almost all operative clauses are doing one kind of rule work, with very little administrative routing",
}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.is_file() else pd.DataFrame()


def _tex_escape(value: Any) -> str:
    text = str(value or "")
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def _fmt_int(value: Any) -> str:
    try:
        return f"{int(value):,}"
    except Exception:
        return "0"


def _fmt_float(value: Any, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return "0.000"


def _clip(text: Any, limit: int) -> str:
    raw = str(text or "").strip()
    if len(raw) <= limit:
        return raw
    return raw[: max(0, limit - 3)].rstrip() + "..."


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _issue_display_label(issue_id: Any) -> str:
    raw = str(issue_id or "").strip()
    if not raw:
        return ""
    if "__" not in raw:
        return _category_label(raw)
    publication, issue_date = raw.split("__", 1)
    label = _category_label(publication)
    return f"{label} ({issue_date})" if issue_date else label


def _style_example_ordinance_label(
    *,
    issue_id: Any,
    publication_key: Any,
    jurisdiction_name: Any,
    jurisdiction_state_abbr: Any,
    jurisdiction_level: Any,
    issue_year: Any,
) -> str:
    publication = str(publication_key or "").strip().replace("-", " ").replace("_", " ").title()
    place = _category_label(jurisdiction_name)
    level = str(jurisdiction_level or "").strip().lower()
    state = str(jurisdiction_state_abbr or "").upper().strip()
    year = ""
    if pd.notna(issue_year):
        try:
            year = str(int(issue_year))
        except Exception:
            year = ""
    if level == "county" and place and "County" not in place:
        place = f"{place} County"
    elif level == "city_and_county" and place and "County" not in place:
        place = f"{place} City and County"
    if publication and place and state and year:
        return f"{publication} / {place}, {state} ({year})"
    if place and state and year:
        return f"{place}, {state} ({year})"
    if publication and year:
        return f"{publication} ({year})"
    return _issue_display_label(issue_id)


def _latex_label_slug(value: Any) -> str:
    raw = str(value or "").strip().lower()
    if not raw:
        return ""
    slug = "".join(ch if ch.isalnum() else "-" for ch in raw).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug


def _appendix_ref_label(issue_id: Any) -> str:
    slug = _latex_label_slug(issue_id)
    return f"app:issue:{slug}" if slug else ""


def _appendix_hyperlink(issue_id: Any, text: Any) -> str:
    rendered = _tex_escape(text)
    label = _appendix_ref_label(issue_id)
    if not label:
        return rendered
    return rf"\hyperref[{label}]{{{rendered}}}"


def _issue_page_glob_prefix(issue_id: Any) -> str:
    raw = str(issue_id or "").strip()
    if not raw or "__" not in raw:
        return ""
    publication, issue_date = raw.split("__", 1)
    try:
        year, month, day = issue_date.split("-")
        month_num = int(month)
    except Exception:
        return ""
    month_map = {
        1: "jan",
        2: "feb",
        3: "mar",
        4: "apr",
        5: "may",
        6: "jun",
        7: "jul",
        8: "aug",
        9: "sep",
        10: "oct",
        11: "nov",
        12: "dec",
    }
    month_slug = month_map.get(month_num)
    if not month_slug:
        return ""
    return f"{publication}-{month_slug}-{day}-{year}-p-"


def _page_num_from_image_path(path: Path) -> int:
    try:
        return int(path.stem.rsplit("-p-", 1)[1])
    except Exception:
        return 10**9


def _find_issue_first_page_image(issue_id: Any, *, repo_root: Path) -> Path | None:
    prefix = _issue_page_glob_prefix(issue_id)
    if not prefix:
        return None
    search_roots = [
        repo_root / "artifacts/scratch" / "referenced_issue_first_pages",
        repo_root / "artifacts/scratch" / "desmond_newspaper_ordinances_archive_20260311" / "archived_duplicates" / "raw",
        repo_root / "artifacts/scratch" / "desmond_newspaper_ordinances_archive_20260311" / "previous_duplicate_archive" / "raw",
        repo_root / "reports" / "runs" / "pi_v1_run60_fullnewspaper" / "fullnewspaper_v3_run_20260221_scale25_strictfull" / "reports" / "fullnewspaper_v3" / "figures" / "appendix_pages",
    ]
    matches: list[Path] = []
    for root in search_roots:
        if not root.exists():
            continue
        matches.extend(sorted(root.glob(f"*{prefix}*.png")))
    if not matches:
        return None
    first = min(matches, key=_page_num_from_image_path)
    return first.resolve()


def _copy_appendix_page_asset(local_path: Path, *, asset_dir: Path, issue_id: str) -> Path:
    asset_dir.mkdir(parents=True, exist_ok=True)
    issue_slug = _latex_label_slug(issue_id) or local_path.stem
    suffix = local_path.suffix or ".png"
    target = asset_dir / f"{issue_slug}{suffix}"
    shutil.copy2(local_path, target)
    return target.resolve()


def _titleish(text: str) -> str:
    raw = text.replace("_", " ")
    mapping = {
        "1920 1949": "1920--1949",
        "1950s": "1950s",
        "1960s": "1960s",
        "1970 plus": "1970+",
        "city and county": "City and county",
    }
    lowered = raw.lower()
    if lowered in mapping:
        return mapping[lowered]
    return raw.title()


def _bridge_term_label(term: Any) -> str:
    raw = str(term or "")
    replacements = {
        "structural_complexity_index": "Structural complexity index (1 SD)",
        "dependency_complexity_index": "Dependency complexity index (1 SD)",
        "linguistic_complexity_index": "Linguistic complexity index (1 SD)",
        "section_kind_entropy": "Section-kind entropy",
        "flesch_kincaid_grade_proxy": "Flesch--Kincaid grade proxy",
        "formal_reference_mentions_per_1000_words": "Formal references per 1,000 words",
        "dependency_trigger_mentions_per_1000_words": "Dependency triggers per 1,000 words",
        "map_reference_mentions_per_1000_words": "Map references per 1,000 words",
        "board_reference_mentions_per_1000_words": "Board references per 1,000 words",
    }
    if raw == "Intercept":
        return "Intercept (borough, 1920--1949 baseline)"
    if raw in replacements:
        return replacements[raw]
    if raw.startswith("C(vintage_bin)[T.") and raw.endswith("]"):
        value = raw[len("C(vintage_bin)[T.") : -1]
        return f"Vintage: {_titleish(value)} vs 1920--1949"
    if raw.startswith("C(jurisdiction_level)[T.") and raw.endswith("]"):
        value = raw[len("C(jurisdiction_level)[T.") : -1]
        return f"Jurisdiction type: {_titleish(value)} vs Borough"
    if raw.startswith("C(jurisdiction_region)[T.") and raw.endswith("]"):
        value = raw[len("C(jurisdiction_region)[T.") : -1]
        return f"Region: {_titleish(value)} vs Midwest"
    return _titleish(raw)


def _bridge_model_label(label: Any) -> str:
    raw = str(label or "")
    mapping = {
        "direct_rule": "Direct-rule share",
        "discretion": "Discretion share",
        "exception_or_condition": "Exception/condition share",
        "coherence_subcomponents": "Coherence model",
        "direct_rule_subcomponents": "Direct-rule model",
        "threshold_subcomponents": "Threshold share model",
        "exception_or_condition_subcomponents": "Exception/condition model",
        "discretion_subcomponents": "Discretion model",
    }
    return mapping.get(raw, _titleish(raw))


def _table_block(
    *,
    caption: str,
    label: str,
    colspec: str,
    header: list[str],
    rows: list[list[str]],
) -> str:
    body = "\n".join(" & ".join(row) + r" \\" for row in rows) if rows else r"\multicolumn{99}{c}{No rows available.} \\"
    uses_x = "X" in colspec
    env = "tabularx" if uses_x else "tabular"
    env_open = rf"\begin{{{env}}}{{\textwidth}}{{{colspec}}}" if uses_x else rf"\begin{{{env}}}{{{colspec}}}"
    env_close = rf"\end{{{env}}}"
    return rf"""
\begin{{table}}[H]
\centering
\footnotesize
\setlength{{\tabcolsep}}{{4pt}}
\renewcommand{{\arraystretch}}{{0.98}}
{env_open}
\toprule
{' & '.join(_tex_escape(cell) for cell in header)} \\\\
\midrule
{body}
\bottomrule
{env_close}
\caption{{{caption}}}
\label{{{label}}}
\end{{table}}
""".strip()


def _figure_block(path: str, caption: str, width: str = "0.88\\textwidth") -> str:
    return rf"""
\IfFileExists{{{path}}}{{
\begin{{figure}}[H]
  \centering
  \includegraphics[width={width}]{{{path}}}
  \caption{{{caption}}}
\end{{figure}}
}}{{}}
""".strip()


def _paragraph_block(paragraphs: list[str], *, needspace: str | None = None) -> str:
    rendered = [rf"\noindent {paragraph.strip()}" for paragraph in paragraphs if str(paragraph).strip()]
    body = "\n\n".join(rendered)
    if needspace:
        return rf"\Needspace{{{needspace}}}" + "\n" + body
    return body


def _rows_from_df(df: pd.DataFrame, *, columns: list[str], formatters: dict[str, Any] | None = None) -> list[list[str]]:
    formatters = formatters or {}
    rows: list[list[str]] = []
    for row in df.to_dict(orient="records"):
        rendered: list[str] = []
        for col in columns:
            value = row.get(col, "")
            if col in formatters:
                value = formatters[col](value)
            else:
                value = _tex_escape(value)
            rendered.append(str(value))
        rows.append(rendered)
    return rows


def _display_label(value: Any) -> str:
    return str(value or "").strip().replace("_", " ")


def _category_label(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if text == "city_and_county":
        return "City and County"
    return text.replace("_", " ").title()


def _purpose_short_label(value: Any) -> str:
    text = str(value or "").strip()
    mapping = {
        "public_health_safety": "Public Health",
        "growth_management": "Growth Mgmt.",
        "externality_control": "Externality",
        "infrastructure_coordination": "Infrastructure",
        "aesthetic_design_control": "Aesthetic",
        "environmental_protection": "Environmental",
        "exclusion": "Exclusion",
        "value_capture": "Value Capture",
        "fiscal_zoning": "Fiscal",
        "other_purpose_families": "Other Purposes",
    }
    return mapping.get(text, _category_label(text))


def _build_document_purpose_weights(document_purposes: pd.DataFrame) -> pd.DataFrame:
    required_cols = {"ordinance_doc_id", "motive", "confidence"}
    if document_purposes.empty or not required_cols.issubset(document_purposes.columns):
        return pd.DataFrame()
    out = document_purposes[["ordinance_doc_id", "motive", "confidence"]].copy()
    out["motive"] = out["motive"].fillna("").astype(str)
    out["confidence"] = pd.to_numeric(out["confidence"], errors="coerce").fillna(0.0)
    out = out[out["motive"].isin(PURPOSE_ORDER)].copy()
    if out.empty:
        return pd.DataFrame()
    total_conf = out.groupby("ordinance_doc_id", dropna=False)["confidence"].transform("sum")
    count_conf = out.groupby("ordinance_doc_id", dropna=False)["confidence"].transform("count").clip(lower=1)
    out["purpose_weight"] = np.where(
        total_conf.gt(0),
        out["confidence"] / total_conf.clip(lower=1e-9),
        1.0 / count_conf,
    )
    return out.sort_values(["ordinance_doc_id", "purpose_weight", "motive"], ascending=[True, False, True]).reset_index(drop=True)


def _summarize_overall_purpose_presence(weights: pd.DataFrame) -> pd.DataFrame:
    required_cols = {"ordinance_doc_id", "motive", "purpose_weight"}
    if weights.empty or not required_cols.issubset(weights.columns):
        return pd.DataFrame()
    total_docs = max(int(weights["ordinance_doc_id"].nunique()), 1)
    out = (
        weights.groupby("motive", dropna=False)
        .agg(
            ordinance_count=("ordinance_doc_id", "nunique"),
        )
        .reset_index()
    )
    out["ordinance_share"] = out["ordinance_count"] / float(total_docs)
    return out.sort_values(["ordinance_share", "motive"], ascending=[False, True]).reset_index(drop=True)


def _summarize_document_purpose_concentration(weights: pd.DataFrame) -> dict[str, float]:
    required_cols = {"ordinance_doc_id", "motive", "purpose_weight"}
    if weights.empty or not required_cols.issubset(weights.columns):
        return {"doc_count": 0, "median_label_count": 0.0, "median_top_weight": 0.0, "median_top_gap": 0.0}
    rows: list[dict[str, float]] = []
    for _, group in weights.groupby("ordinance_doc_id", dropna=False):
        ordered = group.sort_values(["purpose_weight", "motive"], ascending=[False, True]).reset_index(drop=True)
        top_weight = float(ordered.loc[0, "purpose_weight"]) if not ordered.empty else 0.0
        second_weight = float(ordered.loc[1, "purpose_weight"]) if len(ordered) > 1 else 0.0
        rows.append(
            {
                "label_count": float(len(ordered)),
                "top_weight": top_weight,
                "top_gap": top_weight - second_weight,
            }
        )
    summary = pd.DataFrame.from_records(rows)
    return {
        "doc_count": float(len(summary)),
        "median_label_count": float(summary["label_count"].median()),
        "median_top_weight": float(summary["top_weight"].median()),
        "median_top_gap": float(summary["top_gap"].median()),
    }


def _build_purpose_cooccurrence_matrix(weights: pd.DataFrame, *, motive_order: list[str] | None = None) -> pd.DataFrame:
    required_cols = {"ordinance_doc_id", "motive"}
    if weights.empty or not required_cols.issubset(weights.columns):
        return pd.DataFrame()
    motive_order = motive_order or PURPOSE_ORDER
    presence = (
        weights.assign(present=1)
        .pivot_table(
            index="ordinance_doc_id",
            columns="motive",
            values="present",
            aggfunc="max",
            fill_value=0,
            observed=False,
        )
        .reindex(columns=[m for m in motive_order if m in set(weights["motive"].astype(str))], fill_value=0)
    )
    if presence.empty:
        return pd.DataFrame()
    total_docs = max(len(presence), 1)
    values = pd.DataFrame(index=presence.columns, columns=presence.columns, dtype=float)
    for a, b in itertools.product(presence.columns, presence.columns):
        values.loc[a, b] = float(((presence[a] > 0) & (presence[b] > 0)).sum()) / float(total_docs)
    return values


def _build_purpose_bundle_examples(
    *,
    document_purposes: pd.DataFrame,
    docs: pd.DataFrame,
    issue_ids: list[str] | None = None,
) -> pd.DataFrame:
    required_doc_purpose_cols = {"ordinance_doc_id", "motive", "confidence", "evidence"}
    required_doc_cols = {"ordinance_doc_id", "issue_id", "jurisdiction_city_name", "jurisdiction_state_abbr", "issue_date"}
    if (
        document_purposes.empty
        or docs.empty
        or not required_doc_purpose_cols.issubset(document_purposes.columns)
        or not required_doc_cols.issubset(docs.columns)
    ):
        return pd.DataFrame()
    issue_ids = issue_ids or PURPOSE_BUNDLE_EXAMPLE_ISSUES
    weights = _build_document_purpose_weights(document_purposes)
    if weights.empty:
        return pd.DataFrame()

    docs_meta = docs[["ordinance_doc_id", "issue_id", "jurisdiction_city_name", "jurisdiction_state_abbr", "issue_date"]].copy()
    docs_meta["issue_year"] = docs_meta["issue_date"].fillna("").astype(str).str.slice(0, 4)

    merged = (
        document_purposes[["ordinance_doc_id", "motive", "confidence", "evidence"]]
        .merge(weights[["ordinance_doc_id", "motive", "purpose_weight"]], on=["ordinance_doc_id", "motive"], how="left")
        .merge(docs_meta, on="ordinance_doc_id", how="left")
    )
    merged["confidence"] = pd.to_numeric(merged["confidence"], errors="coerce").fillna(0.0)
    merged["purpose_weight"] = pd.to_numeric(merged["purpose_weight"], errors="coerce").fillna(0.0)

    rows: list[dict[str, str]] = []
    for issue_id in issue_ids:
        example = merged[merged["issue_id"].astype(str) == str(issue_id)].copy()
        if example.empty:
            continue
        example = example.sort_values(["purpose_weight", "confidence", "motive"], ascending=[False, False, True]).reset_index(drop=True)
        meta = example.iloc[0]
        city = _category_label(meta.get("jurisdiction_city_name", ""))
        state = str(meta.get("jurisdiction_state_abbr", "") or "").upper().strip()
        year = str(meta.get("issue_year", "") or "").strip()
        ordinance = f"{city}, {state} ({year})" if city and state and year else f"{city}, {state}".strip(", ")
        bundle = "; ".join(
            f"{_purpose_short_label(row['motive'])} {float(row['purpose_weight']) * 100.0:.1f}%"
            for _, row in example.head(5).iterrows()
        )
        top_quotes = []
        for _, row in example.head(2).iterrows():
            evidence = _clip(str(row.get("evidence", "") or ""), 95)
            if not evidence:
                continue
            top_quotes.append(f"{_purpose_short_label(row['motive'])}: {evidence}")
        rows.append(
            {
                "ordinance": ordinance,
                "bundle": bundle or "—",
                "language": "; ".join(top_quotes) if top_quotes else "—",
                "issue_id": str(meta.get("issue_id", "") or ""),
            }
        )
    return pd.DataFrame(rows)


def _style_family_short_label(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    if raw in STYLE_FAMILY_SHORT_LABELS:
        return STYLE_FAMILY_SHORT_LABELS[raw]
    parts = [part.strip() for part in raw.split("|") if part.strip()]
    if not parts:
        return _category_label(raw)
    readable = {
        "direct-rule-heavy": "Rule-heavy",
        "discretion-heavy": "Approval-heavy",
        "exception-and-condition-heavy": "Carve-out-heavy",
        "threshold-heavy": "Threshold-heavy",
        "high instrument variety": "Mixed-toolkit",
        "high instrument diversity": "Mixed-toolkit",
        "top-instrument concentration": "Single-tool",
        "large operative scale": "Long-form",
    }
    phrase = " ".join(readable.get(part, _category_label(part)) for part in parts[:2]).strip()
    return f"{phrase} code".strip()


def _style_family_reading_guide(value: Any) -> str:
    raw = str(value or "").strip()
    if raw in STYLE_FAMILY_READING_GUIDE:
        return STYLE_FAMILY_READING_GUIDE[raw]
    return "similar ordinances share the same balance between rules, approvals, carve-outs, and clause mix"


def _style_clause_flags(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    for col in ["instrument_type", "requirement", "evidence_quote", "exception", "condition"]:
        out[col] = out[col].fillna("").astype(str)
    out["operative_flag"] = pd.to_numeric(out.get("operative_flag", 0), errors="coerce").fillna(0).astype(int)
    out["has_exception"] = out["exception"].str.strip().ne("")
    out["has_condition"] = out["condition"].str.strip().ne("")
    out["discretion_flag"] = out["instrument_type"].isin(STYLE_DISCRETION_INSTRUMENTS)
    out["structural_flag"] = out["instrument_type"].isin(STYLE_STRUCTURAL_INSTRUMENTS)
    out["direct_rule_flag"] = (
        out["operative_flag"].eq(1)
        & ~out["discretion_flag"]
        & ~out["has_exception"]
        & ~out["structural_flag"]
    )
    out["exception_or_condition_flag"] = out["has_exception"] | out["has_condition"]
    return out


def _style_clause_text(row: pd.Series) -> str:
    requirement = " ".join(str(row.get("requirement", "") or "").split()).strip()
    evidence = " ".join(str(row.get("evidence_quote", "") or "").split()).strip().strip('"')
    text = requirement or evidence
    return _clip(text, 140)


def _pick_style_clause_example(df: pd.DataFrame, bucket: str, *, exclude_texts: set[str] | None = None) -> str:
    if df.empty:
        return ""
    exclude_texts = exclude_texts or set()
    if bucket == "direct_rule":
        mask = df["direct_rule_flag"]
        priorities = {
            "height_standard": 0,
            "parking_standard": 1,
            "setback_standard": 2,
            "lot_area_standard": 3,
            "environmental_protection": 4,
            "nonconforming_rule": 5,
            "use_permission": 6,
            "general_rule": 7,
            "other": 10,
        }
    elif bucket == "discretion":
        mask = df["operative_flag"].eq(1) & df["discretion_flag"]
        priorities = {
            "permit_administration": 0,
            "conditional_use": 1,
            "variance": 2,
            "special_exception": 3,
            "appeal": 4,
        }
    else:
        mask = df["operative_flag"].eq(1) & df["exception_or_condition_flag"]
        priorities = {
            "general_rule": 0,
            "use_permission": 1,
            "setback_standard": 2,
            "parking_standard": 3,
            "permit_administration": 4,
            "other": 8,
        }
    candidates = df[mask].copy()
    if candidates.empty:
        return ""
    candidates["example_text"] = candidates.apply(_style_clause_text, axis=1)
    if exclude_texts:
        candidates = candidates[~candidates["example_text"].isin(exclude_texts)].copy()
    if candidates.empty:
        return ""
    candidates["text_len"] = candidates["example_text"].str.len()
    keyword_mask = candidates["example_text"].str.contains(r"\b(?:except|provided|unless|only|if|when)\b", case=False, regex=True)
    candidates["keyword_bonus"] = np.where(keyword_mask, 0, 1)
    candidates["instrument_priority"] = candidates["instrument_type"].map(priorities).fillna(7)
    filtered = candidates[candidates["text_len"].between(40, 150)].copy()
    if filtered.empty:
        filtered = candidates.copy()
    filtered = filtered.sort_values(
        ["keyword_bonus", "instrument_priority", "text_len", "section_title"],
        ascending=[True, True, True, True],
    )
    return str(filtered.iloc[0]["example_text"])


def _effect_group_label(value: Any) -> str:
    text = str(value or "").strip()
    mapping = {
        "jurisdiction_region": "Region",
        "jurisdiction_level": "Jurisdiction type",
        "vintage_bin": "Vintage bin",
    }
    return mapping.get(text, _category_label(text))


def _target_label(value: Any) -> str:
    text = str(value or "").strip()
    mapping = {
        "overall_legal_complexity_index": "Overall legal complexity index",
        "structural_complexity_index": "Structural complexity index",
        "dependency_complexity_index": "Dependency complexity index",
        "linguistic_complexity_index": "Linguistic complexity index",
    }
    return mapping.get(text, _category_label(text))


def _top_purpose_lifts(df: pd.DataFrame, *, group_col: str, top_n: int = 2) -> pd.DataFrame:
    if df.empty:
        return df
    out = df[df["motive"].astype(str).isin(PURPOSE_ORDER)].copy()
    out["abs_lift_deviation"] = (pd.to_numeric(out["purpose_weight_lift_vs_overall_vintage_adjusted"], errors="coerce") - 1.0).abs()
    out = out.sort_values(
        [group_col, "abs_lift_deviation", "purpose_weight_lift_vs_overall_vintage_adjusted", "purpose_weight__vintage_adjusted", "motive"],
        ascending=[True, False, False, False, True],
    )
    return out.groupby(group_col, dropna=False).head(top_n).reset_index(drop=True)


def _purpose_alignment_summary(df: pd.DataFrame, *, min_docs: int = 8) -> pd.DataFrame:
    if df.empty:
        return df
    out = df[df["doc_primary_motive"].astype(str).isin(PURPOSE_ORDER)].copy()
    out = out[out["ordinance_count"].fillna(0) >= min_docs].copy()
    return out.sort_values(
        ["mean_non_other_alignment_share", "ordinance_count", "doc_primary_motive"],
        ascending=[False, False, True],
    ).reset_index(drop=True)


def _leading_text_line(value: Any) -> str:
    lines = str(value or "").splitlines()
    for line in lines[:8]:
        compact = " ".join(line.split()).strip()
        if len(compact) >= 12:
            return compact
    return ""


def _best_section_label(title: Any, section_kind: Any) -> str:
    title_text = " ".join(str(title or "").split()).strip()
    if title_text:
        return title_text
    kind_text = str(section_kind or "").strip().replace("-", " ")
    return _titleish(kind_text) if kind_text else ""


def _build_section_purpose_mix(
    *,
    section_motives: pd.DataFrame,
    sections: pd.DataFrame,
    doc_primary_motives: pd.DataFrame,
    min_docs: int = 8,
) -> pd.DataFrame:
    if section_motives.empty or sections.empty or doc_primary_motives.empty:
        return pd.DataFrame()
    required_section_cols = {"section_id", "ordinance_doc_id", "primary_motive"}
    required_structure_cols = {"section_id", "ordinance_doc_id", "word_count"}
    required_doc_cols = {"ordinance_doc_id", "primary_motive"}
    if not required_section_cols.issubset(section_motives.columns):
        return pd.DataFrame()
    if not required_structure_cols.issubset(sections.columns):
        return pd.DataFrame()
    if not required_doc_cols.issubset(doc_primary_motives.columns):
        return pd.DataFrame()

    doc_primary = (
        doc_primary_motives[["ordinance_doc_id", "primary_motive"]]
        .rename(columns={"primary_motive": "doc_primary_motive"})
        .copy()
    )
    doc_primary["doc_primary_motive"] = doc_primary["doc_primary_motive"].fillna("").astype(str)
    doc_primary = doc_primary[
        doc_primary["doc_primary_motive"].isin(PURPOSE_ORDER)
        & ~doc_primary["doc_primary_motive"].isin(["other", "unclear"])
    ].drop_duplicates()
    if doc_primary.empty:
        return pd.DataFrame()

    merged = (
        section_motives[["section_id", "ordinance_doc_id", "primary_motive"]]
        .rename(columns={"primary_motive": "section_primary_motive"})
        .merge(
            sections[["section_id", "ordinance_doc_id", "word_count"]],
            on=["section_id", "ordinance_doc_id"],
            how="left",
        )
        .merge(doc_primary, on="ordinance_doc_id", how="left")
    )
    merged["doc_primary_motive"] = merged["doc_primary_motive"].fillna("").astype(str)
    merged["section_primary_motive"] = merged["section_primary_motive"].fillna("").astype(str)
    merged["word_count"] = pd.to_numeric(merged["word_count"], errors="coerce").fillna(0.0)
    merged = merged[
        merged["doc_primary_motive"].isin(PURPOSE_ORDER)
        & merged["doc_primary_motive"].ne("other")
        & merged["doc_primary_motive"].ne("unclear")
        & merged["section_primary_motive"].ne("")
        & ~merged["section_primary_motive"].isin(["other", "unclear"])
    ].copy()
    if merged.empty:
        return pd.DataFrame()

    doc_counts = (
        doc_primary[["ordinance_doc_id", "doc_primary_motive"]]
        .groupby("doc_primary_motive", dropna=False)["ordinance_doc_id"]
        .nunique()
        .rename("ordinance_count")
        .reset_index()
    )
    keep_motives = doc_counts.loc[doc_counts["ordinance_count"] >= min_docs, "doc_primary_motive"].tolist()
    if not keep_motives:
        return pd.DataFrame()
    merged = merged[merged["doc_primary_motive"].isin(keep_motives)].copy()
    merged["section_primary_motive"] = merged["section_primary_motive"].where(
        merged["section_primary_motive"].isin(SECTION_PURPOSE_STACK_ORDER[:-1]),
        "other_purpose_families",
    )

    mix = (
        merged.groupby(["doc_primary_motive", "section_primary_motive"], dropna=False)
        .agg(
            section_words=("word_count", "sum"),
            section_count=("section_id", "count"),
        )
        .reset_index()
        .merge(doc_counts, on="doc_primary_motive", how="left")
    )
    mix["share_within_doc_family"] = mix.groupby("doc_primary_motive", dropna=False)["section_words"].transform(
        lambda series: series / series.sum() if float(series.sum()) > 0 else 0.0
    )
    mix["doc_primary_motive"] = pd.Categorical(mix["doc_primary_motive"], categories=PURPOSE_ORDER, ordered=True)
    mix["section_primary_motive"] = pd.Categorical(
        mix["section_primary_motive"],
        categories=SECTION_PURPOSE_STACK_ORDER,
        ordered=True,
    )
    return mix.sort_values(["doc_primary_motive", "section_primary_motive"]).reset_index(drop=True)


def _fmt_pct_diff_from_lift(value: Any) -> str:
    try:
        pct = (float(value) - 1.0) * 100.0
        return _tex_escape(f"{pct:+.1f}%")
    except Exception:
        return _tex_escape("+0.0%")


def _purpose_region_matrix_table(df: pd.DataFrame) -> str:
    if df.empty:
        rows = [r"\multicolumn{10}{c}{No rows available.} \\"]
    else:
        pivot = (
            df[df["motive"].astype(str).isin(PURPOSE_ORDER)]
            .pivot_table(
                index="jurisdiction_region",
                columns="motive",
                values="purpose_weight_lift_vs_overall_vintage_adjusted",
                aggfunc="first",
            )
            .reindex(index=REGION_ORDER, columns=PURPOSE_ORDER)
        )
        rows = []
        for region in pivot.index:
            rendered = [_tex_escape(_category_label(region))]
            for motive in pivot.columns:
                value = pivot.loc[region, motive]
                if pd.isna(value):
                    rendered.append(_tex_escape("—"))
                else:
                    rendered.append(_fmt_pct_diff_from_lift(value))
            rows.append(" & ".join(rendered) + r" \\")
        if not rows:
            rows = [r"\multicolumn{10}{c}{No rows available.} \\"]
    header = " & ".join(
        [_tex_escape("Region")] + [_tex_escape(_purpose_short_label(m)) for m in PURPOSE_ORDER]
    )
    body = "\n".join(rows)
    return rf"""
\begin{{table}}[H]
\centering
\footnotesize
\setlength{{\tabcolsep}}{{4pt}}
\renewcommand{{\arraystretch}}{{0.98}}
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{lrrrrrrrrr}}
\toprule
{header} \\\\
\midrule
{body}
\bottomrule
\end{{tabular}}%
}}
\caption{{Document-purpose deviations by region on the full sample. Each cell compares a region's mean full-text purpose weight for that purpose with the full-sample mean for the same purpose, then reports how far above or below that benchmark the region sits.}}
\label{{tab:purpose-region}}
    \end{{table}}
    """.strip()


def _section_purpose_profile_table(df: pd.DataFrame) -> str:
    if df.empty:
        return _table_block(
            caption="Section-purpose composition by document purpose.",
            label="tab:section-purpose-profile",
            colspec=r"lrll>{\raggedright\arraybackslash}X",
            header=["Headline purpose", "Docs", "Most section text", "Second-most", "How to read it"],
            rows=[],
        )
    ranked = (
        df.sort_values(
            ["doc_primary_motive", "share_within_doc_family", "section_primary_motive"],
            ascending=[True, False, True],
        )
        .groupby("doc_primary_motive", dropna=False, observed=False)
        .head(2)
        .copy()
    )
    rows: list[list[str]] = []
    for motive in [item for item in PURPOSE_ORDER if item in set(df["doc_primary_motive"].astype(str))]:
        group = ranked[ranked["doc_primary_motive"].astype(str) == motive].reset_index(drop=True)
        if group.empty:
            continue
        first = group.iloc[0]
        second = group.iloc[1] if len(group) > 1 else None
        headline_label = _purpose_short_label(motive)
        first_label = _purpose_short_label(first["section_primary_motive"])
        second_label = _purpose_short_label(second["section_primary_motive"]) if second is not None else "—"
        if str(first["section_primary_motive"]) == motive:
            interpretation = f"The headline purpose is also the main section-level focus ({float(first['share_within_doc_family']) * 100.0:.1f}%)."
        elif second is not None and str(second["section_primary_motive"]) == motive:
            interpretation = (
                f"The headline purpose stays visible, but more section text is spent on {first_label} rules "
                f"({float(first['share_within_doc_family']) * 100.0:.1f}%) than on {headline_label} language."
            )
        else:
            interpretation = (
                f"The headline purpose shows up mostly in framing; the operative text is led by "
                f"{first_label} and {second_label} sections."
            )
        rows.append(
            [
                _tex_escape(headline_label),
                _fmt_int(first["ordinance_count"]),
                _tex_escape(f"{first_label} ({float(first['share_within_doc_family']) * 100.0:.1f}%)"),
                _tex_escape(f"{second_label} ({float(second['share_within_doc_family']) * 100.0:.1f}%)") if second is not None else _tex_escape("—"),
                _tex_escape(interpretation),
            ]
        )
    return _table_block(
        caption="Section-purpose composition by document purpose. `Most section text` and `Second-most` report the two largest section-purpose buckets within each document-purpose row, and the last column gives a short reading of that pattern.",
        label="tab:section-purpose-profile",
        colspec=r"lrll>{\raggedright\arraybackslash}X",
        header=["Headline purpose", "Docs", "Most section text", "Second-most", "How to read it"],
        rows=rows,
    )


def _build_section_purpose_examples(
    *,
    doc_primary_motives: pd.DataFrame,
    document_purposes: pd.DataFrame,
    docs: pd.DataFrame,
    section_motives: pd.DataFrame,
    sections: pd.DataFrame,
) -> pd.DataFrame:
    required_doc_cols = {"ordinance_doc_id", "primary_motive"}
    required_doc_purpose_cols = {"ordinance_doc_id", "motive", "evidence", "confidence"}
    required_docs_meta = {"ordinance_doc_id", "issue_id", "jurisdiction_city_name", "jurisdiction_state_abbr", "issue_date", "word_count", "text"}
    required_section_cols = {"section_id", "ordinance_doc_id", "primary_motive"}
    required_structure_cols = {"section_id", "ordinance_doc_id", "word_count"}
    if (
        doc_primary_motives.empty
        or document_purposes.empty
        or docs.empty
        or section_motives.empty
        or sections.empty
        or not required_doc_cols.issubset(doc_primary_motives.columns)
        or not required_doc_purpose_cols.issubset(document_purposes.columns)
        or not required_docs_meta.issubset(docs.columns)
        or not required_section_cols.issubset(section_motives.columns)
        or not required_structure_cols.issubset(sections.columns)
    ):
        return pd.DataFrame()

    example_motives = [
        "public_health_safety",
        "growth_management",
        "externality_control",
        "environmental_protection",
    ]

    doc_primary = (
        doc_primary_motives[["ordinance_doc_id", "primary_motive"]]
        .rename(columns={"primary_motive": "doc_primary_motive"})
        .copy()
    )
    doc_primary["doc_primary_motive"] = doc_primary["doc_primary_motive"].fillna("").astype(str)
    doc_primary = doc_primary[
        doc_primary["doc_primary_motive"].isin(PURPOSE_ORDER)
        & ~doc_primary["doc_primary_motive"].isin(["other", "unclear"])
    ].drop_duplicates()
    if doc_primary.empty:
        return pd.DataFrame()

    merged = (
        section_motives[["section_id", "ordinance_doc_id", "primary_motive"]]
        .rename(columns={"primary_motive": "section_primary_motive"})
        .merge(
            sections[["section_id", "ordinance_doc_id", "word_count"]],
            on=["section_id", "ordinance_doc_id"],
            how="left",
        )
        .merge(doc_primary, on="ordinance_doc_id", how="inner")
    )
    merged["section_primary_motive"] = merged["section_primary_motive"].fillna("").astype(str)
    merged["word_count"] = pd.to_numeric(merged["word_count"], errors="coerce").fillna(0.0)
    merged = merged[
        merged["section_primary_motive"].ne("")
        & ~merged["section_primary_motive"].isin(["other", "unclear"])
    ].copy()
    if merged.empty:
        return pd.DataFrame()

    merged["section_primary_motive"] = merged["section_primary_motive"].where(
        merged["section_primary_motive"].isin(SECTION_PURPOSE_STACK_ORDER[:-1]),
        "other_purpose_families",
    )
    doc_mix = (
        merged.groupby(["ordinance_doc_id", "doc_primary_motive", "section_primary_motive"], dropna=False)["word_count"]
        .sum()
        .reset_index()
    )
    doc_mix["share_within_doc"] = doc_mix.groupby("ordinance_doc_id", dropna=False)["word_count"].transform(
        lambda series: series / series.sum() if float(series.sum()) > 0 else 0.0
    )
    pivot = (
        doc_mix.pivot_table(
            index=["ordinance_doc_id", "doc_primary_motive"],
            columns="section_primary_motive",
            values="share_within_doc",
            fill_value=0.0,
            observed=False,
        )
        .reindex(columns=SECTION_PURPOSE_STACK_ORDER, fill_value=0.0)
    )
    if pivot.empty:
        return pd.DataFrame()

    docs_meta = docs[
        ["ordinance_doc_id", "issue_id", "jurisdiction_city_name", "jurisdiction_state_abbr", "issue_date", "word_count", "text"]
    ].copy()
    docs_meta["issue_date"] = docs_meta["issue_date"].fillna("").astype(str)
    docs_meta["issue_year"] = docs_meta["issue_date"].str.slice(0, 4)
    docs_meta["word_count"] = pd.to_numeric(docs_meta["word_count"], errors="coerce").fillna(0.0)
    docs_meta["title_guess"] = docs_meta["text"].map(_leading_text_line)
    docs_meta = docs_meta.drop(columns=["text"])

    candidates = document_purposes.merge(docs_meta, on="ordinance_doc_id", how="left")
    candidates["confidence"] = pd.to_numeric(candidates["confidence"], errors="coerce").fillna(0.0)

    rows: list[dict[str, Any]] = []
    for motive in example_motives:
        if motive not in set(pivot.index.get_level_values(1).astype(str)):
            continue
        group = pivot[pivot.index.get_level_values(1).astype(str) == motive]
        if group.empty:
            continue
        centroid = group.mean(axis=0).to_numpy(dtype=float)
        group_matrix = group.to_numpy(dtype=float)
        distances = np.sqrt(((group_matrix - centroid) ** 2).sum(axis=1))
        best_idx = int(distances.argmin())
        doc_id = str(group.index[best_idx][0])
        example = (
            candidates[
                (candidates["ordinance_doc_id"].astype(str) == doc_id)
                & (candidates["motive"].astype(str) == motive)
            ]
            .sort_values(["confidence", "word_count"], ascending=[False, False])
            .head(1)
        )
        if example.empty:
            continue
        example_row = example.iloc[0]
        top_section = (
            doc_mix[doc_mix["ordinance_doc_id"].astype(str) == doc_id]
            .sort_values(["share_within_doc", "section_primary_motive"], ascending=[False, True])
            .head(3)
        )
        section_summary = "; ".join(
            f"{_purpose_short_label(row['section_primary_motive'])} {float(row['share_within_doc']) * 100.0:.1f}%"
            for _, row in top_section.iterrows()
        )
        rows.append(
            {
                "headline_purpose": _purpose_short_label(motive),
                "ordinance": f"{_category_label(example_row['jurisdiction_city_name'])}, {str(example_row['jurisdiction_state_abbr'] or '').upper()} ({example_row['issue_year']})",
                "language": _clip(str(example_row["evidence"] or ""), 180),
                "section_pattern": section_summary or "—",
                "issue_id": str(example_row.get("issue_id", "") or ""),
            }
        )
    return pd.DataFrame(rows)


def _section_purpose_examples_table(df: pd.DataFrame) -> str:
    table_df = df.copy()
    if not table_df.empty:
        table_df["ordinance_link"] = table_df.apply(
            lambda row: _appendix_hyperlink(row.get("issue_id", ""), row.get("ordinance", "")),
            axis=1,
        )
    return _table_block(
        caption="Section-purpose examples. Each row pairs one document-purpose group with an ordinance, direct purpose language, and the main section purposes in that ordinance. Ordinance names link to the appendix.",
        label="tab:section-purpose-examples",
        colspec=r"l l >{\raggedright\arraybackslash}X >{\raggedright\arraybackslash}X",
        header=["Headline purpose", "Example ordinance", "Explicit ordinance language", "Main section purposes"],
        rows=_rows_from_df(
            table_df,
            columns=["headline_purpose", "ordinance_link", "language", "section_pattern"],
            formatters={"ordinance_link": lambda x: x},
        ),
    )


def _build_style_family_examples(
    *,
    pilot_style: pd.DataFrame,
    style_prototypes: pd.DataFrame,
    docs: pd.DataFrame,
    enriched_clauses: pd.DataFrame,
    min_docs: int = 3,
) -> pd.DataFrame:
    required_style_cols = {
        "regulatory_style_family_id",
        "cluster_label",
        "ordinance_count",
        "mean_direct_rule_share",
        "mean_discretion_share",
        "mean_exception_or_condition_share",
    }
    required_proto_cols = {"regulatory_style_family_id", "prototype_rank", "ordinance_doc_id"}
    required_doc_cols = {
        "ordinance_doc_id",
        "issue_id",
        "publication_key",
        "jurisdiction_city_name",
        "jurisdiction_state_abbr",
        "jurisdiction_level",
        "issue_date",
    }
    required_clause_cols = {
        "ordinance_doc_id",
        "instrument_type",
        "requirement",
        "evidence_quote",
        "operative_flag",
        "exception",
        "condition",
    }
    if (
        pilot_style.empty
        or style_prototypes.empty
        or docs.empty
        or enriched_clauses.empty
        or not required_style_cols.issubset(pilot_style.columns)
        or not required_proto_cols.issubset(style_prototypes.columns)
        or not required_doc_cols.issubset(docs.columns)
        or not required_clause_cols.issubset(enriched_clauses.columns)
    ):
        return pd.DataFrame()

    style_df = (
        pilot_style[pilot_style["ordinance_count"].fillna(0) >= min_docs]
        .sort_values(["ordinance_count", "regulatory_style_family_id"], ascending=[False, True])
        .copy()
    )
    if style_df.empty:
        return pd.DataFrame()

    prototypes = (
        style_prototypes[style_prototypes["prototype_rank"].fillna(0).astype(int) == 1][
            ["regulatory_style_family_id", "ordinance_doc_id"]
        ]
        .drop_duplicates("regulatory_style_family_id")
        .copy()
    )
    docs_meta = docs[
        ["ordinance_doc_id", "issue_id", "publication_key", "jurisdiction_city_name", "jurisdiction_state_abbr", "jurisdiction_level", "issue_date"]
    ].copy()
    docs_meta["issue_year"] = pd.to_datetime(docs_meta["issue_date"], errors="coerce").dt.year
    clause_df = _style_clause_flags(
        enriched_clauses[
            [
                "ordinance_doc_id",
                "section_title",
                "instrument_type",
                "requirement",
                "evidence_quote",
                "operative_flag",
                "exception",
                "condition",
            ]
        ].copy()
    )

    rows: list[dict[str, str]] = []
    for family in style_df.to_dict(orient="records"):
        family_id = int(family["regulatory_style_family_id"])
        prototype = prototypes[prototypes["regulatory_style_family_id"].astype(int) == family_id]
        if prototype.empty:
            continue
        doc_id = str(prototype.iloc[0]["ordinance_doc_id"])
        doc_meta = docs_meta[docs_meta["ordinance_doc_id"].astype(str) == doc_id].head(1)
        if doc_meta.empty:
            continue
        doc_row = doc_meta.iloc[0]
        ordinance = _style_example_ordinance_label(
            issue_id=doc_row.get("issue_id", ""),
            publication_key=doc_row.get("publication_key", ""),
            jurisdiction_name=doc_row.get("jurisdiction_city_name", ""),
            jurisdiction_state_abbr=doc_row.get("jurisdiction_state_abbr", ""),
            jurisdiction_level=doc_row.get("jurisdiction_level", ""),
            issue_year=doc_row.get("issue_year", np.nan),
        )
        doc_clauses = clause_df[clause_df["ordinance_doc_id"].astype(str) == doc_id].copy()
        chosen_texts: set[str] = set()
        direct_example = _pick_style_clause_example(doc_clauses, "direct_rule")
        if direct_example:
            chosen_texts.add(direct_example)
        discretion_example = _pick_style_clause_example(doc_clauses, "discretion", exclude_texts=chosen_texts)
        if discretion_example:
            chosen_texts.add(discretion_example)
        carve_out_example = _pick_style_clause_example(doc_clauses, "exception_or_condition", exclude_texts=chosen_texts)
        example_parts = []
        if direct_example:
            example_parts.append(f"Rule: {direct_example}")
        if discretion_example:
            example_parts.append(f"Discretion: {discretion_example}")
        if carve_out_example:
            example_parts.append(f"Carve-out: {carve_out_example}")
        pattern = (
            f"Rules {float(family['mean_direct_rule_share']) * 100.0:.0f}%, "
            f"discretion {float(family['mean_discretion_share']) * 100.0:.0f}%, "
            f"carve-outs {float(family['mean_exception_or_condition_share']) * 100.0:.0f}%; "
            f"{_style_family_reading_guide(family['cluster_label'])}."
        )
        rows.append(
            {
                "style_name": _style_family_short_label(family["cluster_label"]),
                "ordinance": ordinance,
                "pattern": pattern,
                "clause_examples": " ".join(example_parts) if example_parts else "—",
                "issue_id": str(doc_row.get("issue_id", "") or ""),
            }
        )
    return pd.DataFrame(rows)


def _style_family_examples_table(df: pd.DataFrame) -> str:
    table_df = df.copy()
    if not table_df.empty:
        table_df["ordinance_link"] = table_df.apply(
            lambda row: _appendix_hyperlink(row.get("issue_id", ""), row.get("ordinance", "")),
            axis=1,
        )
    rows = _rows_from_df(
        table_df,
        columns=["style_name", "ordinance_link", "pattern", "clause_examples"],
        formatters={"ordinance_link": lambda x: x},
    )
    body = "\n".join(" & ".join(row) + r" \\" for row in rows) if rows else r"\multicolumn{4}{c}{No rows available.} \\"
    header = "Style & Prototype ordinance & What it looks like & Representative clauses \\\\"
    caption = (
        "Governance-style examples. Each row shows one representative ordinance from that style together with one rule clause, one discretion clause, and one carve-out clause from the same ordinance. Ordinance names link to the appendix."
    )
    return rf"""
\begin{{table}}[H]
\centering
\footnotesize
\setlength{{\tabcolsep}}{{4pt}}
\renewcommand{{\arraystretch}}{{0.98}}
\begin{{tabular}}{{>{{\raggedright\arraybackslash}}p{{0.16\textwidth}}>{{\raggedright\arraybackslash}}p{{0.15\textwidth}}>{{\raggedright\arraybackslash}}p{{0.25\textwidth}}>{{\raggedright\arraybackslash}}p{{0.34\textwidth}}}}
\toprule
{header}
\midrule
{body}
\bottomrule
\end{{tabular}}
\caption{{{caption}}}
\label{{tab:pilot-style-examples}}
\end{{table}}
""".strip()


def _opening_move_takeaway(df: pd.DataFrame) -> str:
    required = {"jurisdiction_region", "first_substantive_section_kind", "ordinance_count"}
    if df.empty or not required.issubset(df.columns):
        return "The opening-move table shows how ordinances usually enter their first substantive section."
    top = (
        df[df["jurisdiction_region"].astype(str).isin(REGION_ORDER)]
        .copy()
        .sort_values(
            ["jurisdiction_region", "ordinance_count", "first_substantive_section_kind"],
            ascending=[True, False, True],
        )
        .groupby("jurisdiction_region", dropna=False, observed=False)
        .head(3)
        .copy()
    )
    if top.empty:
        return "The opening-move table shows how ordinances usually enter their first substantive section."
    keyword_mask = top["first_substantive_section_kind"].fillna("").astype(str).str.contains(
        "preamble|purpose|general|intent|finding|title",
        case=False,
        regex=True,
    )
    if float(keyword_mask.mean()) >= 0.5:
        return (
            "The opening-move table and the companion examples show that many ordinances begin with "
            "preamble, purpose, or general-provisions language, so the dominant opening motive is often "
            "framing language rather than district standards."
        )
    return "The opening-move table and the companion examples show how ordinances usually enter their first substantive section."


def _build_opening_move_examples(
    *,
    opening_motive_region: pd.DataFrame,
    docs: pd.DataFrame,
    sections: pd.DataFrame,
    section_motives: pd.DataFrame,
) -> pd.DataFrame:
    required_opening_cols = {"jurisdiction_region", "first_substantive_section_motive", "ordinance_count"}
    required_docs_cols = {"ordinance_doc_id", "issue_id", "jurisdiction_region", "jurisdiction_city_name", "jurisdiction_state_abbr", "issue_date"}
    required_section_cols = {"section_id", "ordinance_doc_id", "section_number", "title", "section_kind", "word_count"}
    required_motive_cols = {"section_id", "ordinance_doc_id", "primary_motive", "evidence", "confidence"}
    if (
        opening_motive_region.empty
        or docs.empty
        or sections.empty
        or section_motives.empty
        or not required_opening_cols.issubset(opening_motive_region.columns)
        or not required_docs_cols.issubset(docs.columns)
        or not required_section_cols.issubset(sections.columns)
        or not required_motive_cols.issubset(section_motives.columns)
    ):
        return pd.DataFrame()

    merged = (
        sections[["section_id", "ordinance_doc_id", "section_number", "title", "section_kind", "word_count"]]
        .merge(
            section_motives[["section_id", "ordinance_doc_id", "primary_motive", "evidence", "confidence"]],
            on=["section_id", "ordinance_doc_id"],
            how="left",
        )
        .merge(
            docs[["ordinance_doc_id", "issue_id", "jurisdiction_region", "jurisdiction_city_name", "jurisdiction_state_abbr", "issue_date"]],
            on="ordinance_doc_id",
            how="left",
        )
    )
    merged["primary_motive"] = merged["primary_motive"].fillna("").astype(str)
    merged["section_kind"] = merged["section_kind"].fillna("").astype(str)
    merged["word_count"] = pd.to_numeric(merged["word_count"], errors="coerce").fillna(0.0)
    merged["confidence"] = pd.to_numeric(merged["confidence"], errors="coerce").fillna(0.0)
    merged["issue_year"] = pd.to_datetime(merged["issue_date"], errors="coerce").dt.year
    merged = merged[
        merged["primary_motive"].ne("")
        & ~merged["primary_motive"].isin(["other", "unclear"])
    ].copy()
    if merged.empty:
        return pd.DataFrame()

    first = (
        merged.sort_values(["ordinance_doc_id", "section_number", "section_id"])
        .groupby("ordinance_doc_id", dropna=False, observed=False)
        .head(1)
        .copy()
    )
    if first.empty:
        return pd.DataFrame()

    opening_rank = (
        opening_motive_region[
            opening_motive_region["jurisdiction_region"].astype(str).isin(REGION_ORDER)
            & opening_motive_region["first_substantive_section_motive"].fillna("").astype(str).ne("")
            & ~opening_motive_region["first_substantive_section_motive"].fillna("").astype(str).isin(["other", "unclear"])
        ]
        .copy()
        .sort_values(
            ["jurisdiction_region", "ordinance_count", "first_substantive_section_motive"],
            ascending=[True, False, True],
        )
    )

    rows: list[dict[str, str]] = []
    for region in REGION_ORDER:
        region_rank = opening_rank[opening_rank["jurisdiction_region"].astype(str) == region].head(1)
        lead_motive = (
            str(region_rank.iloc[0]["first_substantive_section_motive"])
            if not region_rank.empty
            else ""
        )
        candidates = first[first["jurisdiction_region"].astype(str) == region].copy()
        if lead_motive:
            candidates = candidates[candidates["primary_motive"].astype(str) == lead_motive].copy()
        if candidates.empty:
            continue
        candidates["kind_count"] = candidates.groupby("section_kind", dropna=False)["ordinance_doc_id"].transform("count")
        candidates["city_state_key"] = (
            candidates["jurisdiction_city_name"].fillna("").astype(str).str.lower()
            + "|"
            + candidates["jurisdiction_state_abbr"].fillna("").astype(str).str.lower()
        )
        candidates = (
            candidates.sort_values(
                ["kind_count", "confidence", "word_count", "issue_year", "jurisdiction_city_name"],
                ascending=[False, False, False, True, True],
            )
            .drop_duplicates("city_state_key")
            .reset_index(drop=True)
        )
        if candidates.empty:
            continue
        example = candidates.iloc[0]
        city = _category_label(example["jurisdiction_city_name"])
        state = str(example["jurisdiction_state_abbr"] or "").upper()
        year = ""
        if pd.notna(example["issue_year"]):
            try:
                year = str(int(example["issue_year"]))
            except Exception:
                year = ""
        ordinance = f"{city}, {state} ({year})" if city and state and year else f"{city}, {state}".strip(", ")
        rows.append(
            {
                "region": _category_label(region),
                "ordinance": ordinance,
                "opening_section": _clip(_best_section_label(example["title"], example["section_kind"]), 52),
                "language": _clip(str(example["evidence"] or ""), 180),
                "issue_id": str(example.get("issue_id", "") or ""),
            }
        )
    return pd.DataFrame(rows)


def _opening_move_examples_table(df: pd.DataFrame) -> str:
    table_df = df.copy()
    if not table_df.empty:
        table_df["ordinance_link"] = table_df.apply(
            lambda row: _appendix_hyperlink(row.get("issue_id", ""), row.get("ordinance", "")),
            axis=1,
        )
    return _table_block(
        caption="Opening-move examples. These openings are often preamble, purpose, or general-provisions sections rather than district-standard sections. Ordinance names link to the appendix.",
        label="tab:opening-move-examples",
        colspec=r"l l l >{\raggedright\arraybackslash}X",
        header=["Region", "Example ordinance", "First section", "Opening language"],
        rows=_rows_from_df(
            table_df,
            columns=["region", "ordinance_link", "opening_section", "language"],
            formatters={"ordinance_link": lambda x: x},
        ),
    )


def _opening_move_table(df: pd.DataFrame) -> str:
    if df.empty:
        return _table_block(
            caption="Opening substantive section motives by region. Each ordinance contributes one motive: the motive assigned to its first substantive section. The table reports the two most common opening motives within each region.",
            label="tab:opening-move-region",
            colspec=r"lrlrlr",
            header=["Region", "Most common", "Share", "Second-most", "Share", "Docs"],
            rows=[],
        )
    out = df.copy()
    out["first_substantive_section_motive"] = out["first_substantive_section_motive"].fillna("").astype(str)
    out = out[
        out["jurisdiction_region"].astype(str).isin(REGION_ORDER)
        & out["first_substantive_section_motive"].ne("")
        & ~out["first_substantive_section_motive"].isin(["other", "unclear"])
    ].copy()
    if out.empty:
        return _table_block(
            caption="Opening substantive section motives by region. Each ordinance contributes one motive: the motive assigned to its first substantive section. The table reports the two most common opening motives within each region.",
            label="tab:opening-move-region",
            colspec=r"lrlrlr",
            header=["Region", "Most common", "Share", "Second-most", "Share", "Docs"],
            rows=[],
        )
    out["share_within_region"] = out.groupby("jurisdiction_region", dropna=False)["ordinance_count"].transform(
        lambda series: series / series.sum() if float(series.sum()) > 0 else 0.0
    )
    ranked = (
        out.sort_values(
            ["jurisdiction_region", "share_within_region", "ordinance_count", "first_substantive_section_motive"],
            ascending=[True, False, False, True],
        )
        .groupby("jurisdiction_region", dropna=False, observed=False)
        .head(2)
        .copy()
    )
    rows: list[list[str]] = []
    for region in REGION_ORDER:
        group = ranked[ranked["jurisdiction_region"].astype(str) == region].reset_index(drop=True)
        if group.empty:
            continue
        first = group.iloc[0]
        second = group.iloc[1] if len(group) > 1 else None
        rows.append(
            [
                _tex_escape(_category_label(region)),
                _tex_escape(_purpose_short_label(first["first_substantive_section_motive"])),
                _tex_escape(f"{float(first['share_within_region']) * 100.0:.1f}%"),
                _tex_escape(_purpose_short_label(second["first_substantive_section_motive"])) if second is not None else _tex_escape("—"),
                _tex_escape(f"{float(second['share_within_region']) * 100.0:.1f}%") if second is not None else _tex_escape("—"),
                _fmt_int(out.loc[out["jurisdiction_region"].astype(str) == region, "ordinance_count"].sum()),
            ]
        )
    return _table_block(
        caption="Opening substantive section motives by region. Each ordinance contributes one motive: the motive assigned to its first substantive section. The table reports the two most common opening motives within each region.",
        label="tab:opening-move-region",
        colspec=r"lrlrlr",
        header=["Region", "Most common", "Share", "Second-most", "Share", "Docs"],
        rows=rows,
    )


def _build_purpose_example_rows(
    *,
    purpose_region: pd.DataFrame,
    document_purposes: pd.DataFrame,
    docs: pd.DataFrame,
    section_motives: pd.DataFrame,
    sections: pd.DataFrame,
) -> pd.DataFrame:
    required_purpose_cols = {"jurisdiction_region", "motive", "purpose_weight_lift_vs_overall_vintage_adjusted"}
    required_doc_purpose_cols = {"ordinance_doc_id", "motive", "evidence", "confidence"}
    required_doc_cols = {"ordinance_doc_id", "issue_id", "jurisdiction_region", "jurisdiction_city_name", "jurisdiction_state_abbr", "issue_date"}
    required_section_cols = {"section_id", "ordinance_doc_id", "primary_motive"}
    required_structure_cols = {"section_id", "ordinance_doc_id", "word_count"}
    if (
        purpose_region.empty
        or document_purposes.empty
        or docs.empty
        or section_motives.empty
        or sections.empty
        or not required_purpose_cols.issubset(purpose_region.columns)
        or not required_doc_purpose_cols.issubset(document_purposes.columns)
        or not required_doc_cols.issubset(docs.columns)
        or not required_section_cols.issubset(section_motives.columns)
        or not required_structure_cols.issubset(sections.columns)
    ):
        return pd.DataFrame()

    specs = [
        {"region": "northeast", "motive": "value_capture"},
        {"region": "midwest", "motive": "fiscal_zoning"},
        {"region": "south", "motive": "environmental_protection"},
        {"region": "west", "motive": "growth_management"},
    ]

    plot_signals = purpose_region[
        purpose_region["jurisdiction_region"].astype(str).isin(REGION_ORDER)
        & purpose_region["motive"].astype(str).isin(PURPOSE_ORDER)
    ].copy()
    plot_signals["pct_diff"] = (
        pd.to_numeric(plot_signals["purpose_weight_lift_vs_overall_vintage_adjusted"], errors="coerce").fillna(1.0) - 1.0
    ) * 100.0

    docs_meta = docs[
        ["ordinance_doc_id", "issue_id", "jurisdiction_region", "jurisdiction_city_name", "jurisdiction_state_abbr", "issue_date", "word_count", "text"]
    ].copy()
    docs_meta["issue_date"] = docs_meta["issue_date"].fillna("").astype(str)
    docs_meta["issue_year"] = docs_meta["issue_date"].str.slice(0, 4)
    docs_meta["title_guess"] = docs_meta["text"].map(_leading_text_line)
    docs_meta = docs_meta.drop(columns=["text"])

    candidates = document_purposes.merge(docs_meta, on="ordinance_doc_id", how="left")
    candidates["confidence"] = pd.to_numeric(candidates["confidence"], errors="coerce").fillna(0.0)
    candidates["word_count"] = pd.to_numeric(candidates["word_count"], errors="coerce").fillna(0.0)

    section_mix = (
        section_motives[["section_id", "ordinance_doc_id", "primary_motive"]]
        .merge(
            sections[["section_id", "ordinance_doc_id", "word_count"]],
            on=["section_id", "ordinance_doc_id"],
            how="left",
        )
        .copy()
    )
    section_mix["word_count"] = pd.to_numeric(section_mix["word_count"], errors="coerce").fillna(0.0)
    section_mix["primary_motive"] = section_mix["primary_motive"].fillna("").astype(str)
    section_mix = section_mix[
        section_mix["primary_motive"].ne("")
        & ~section_mix["primary_motive"].isin(["other", "unclear"])
    ].copy()
    section_mix = (
        section_mix.groupby(["ordinance_doc_id", "primary_motive"], dropna=False)["word_count"]
        .sum()
        .reset_index()
    )
    section_mix["share_within_doc"] = section_mix.groupby("ordinance_doc_id", dropna=False)["word_count"].transform(
        lambda series: series / series.sum() if float(series.sum()) > 0 else 0.0
    )

    rows: list[dict[str, Any]] = []
    for spec in specs:
        region = spec["region"]
        motive = spec["motive"]
        signal = plot_signals[
            (plot_signals["jurisdiction_region"].astype(str) == region)
            & (plot_signals["motive"].astype(str) == motive)
        ]
        example_pool = candidates[
            (candidates["jurisdiction_region"].astype(str) == region)
            & (candidates["motive"].astype(str) == motive)
        ].sort_values(["confidence", "word_count", "ordinance_doc_id"], ascending=[False, False, True])
        if signal.empty or example_pool.empty:
            continue
        example = example_pool.iloc[0]
        doc_id = str(example["ordinance_doc_id"])
        top_section = (
            section_mix[section_mix["ordinance_doc_id"].astype(str) == doc_id]
            .sort_values(["share_within_doc", "primary_motive"], ascending=[False, True])
            .head(3)
        )
        section_summary = "; ".join(
            f"{_purpose_short_label(row['primary_motive'])} {float(row['share_within_doc']) * 100.0:.1f}%"
            for _, row in top_section.iterrows()
        )
        rows.append(
            {
                "signal": f"{_category_label(region)}: {_purpose_short_label(motive)} ({float(signal.iloc[0]['pct_diff']):+.1f}%)",
                "ordinance": f"{_category_label(example['jurisdiction_city_name'])}, {str(example['jurisdiction_state_abbr'] or '').upper()} ({example['issue_year']})",
                "language": _clip(str(example["evidence"] or ""), 180),
                "section_pattern": section_summary or "—",
                "issue_id": str(example.get("issue_id", "") or ""),
            }
        )
    return pd.DataFrame(rows)


def _purpose_examples_table(df: pd.DataFrame) -> str:
    table_df = df.copy()
    if not table_df.empty:
        table_df["ordinance_link"] = table_df.apply(
            lambda row: _appendix_hyperlink(row.get("issue_id", ""), row.get("ordinance", "")),
            axis=1,
        )
    return _table_block(
        caption=(
            "Purpose-bundle examples. Each row keeps the full returned purpose bundle for one ordinance instead of collapsing it to a single headline label. Ordinance names link to the appendix."
        ),
        label="tab:purpose-examples",
        colspec=r"l >{\raggedright\arraybackslash}X >{\raggedright\arraybackslash}X",
        header=["Example ordinance", "Returned purpose bundle", "Selected ordinance language"],
        rows=_rows_from_df(
            table_df,
            columns=["ordinance_link", "bundle", "language"],
            formatters={"ordinance_link": lambda x: x},
        ),
    )


def _build_clause_family_examples(
    *,
    largest_clause_clusters: pd.DataFrame,
    clause_cluster_exemplars: pd.DataFrame,
    docs: pd.DataFrame,
    repo_root: Path,
) -> pd.DataFrame:
    required_family_cols = {"cluster_id", "cluster_label", "clause_count", "ordinance_count"}
    required_exemplar_cols = {
        "cluster_id",
        "issue_id",
        "ordinance_doc_id",
        "section_title",
        "clause_text",
        "centroid_similarity",
    }
    required_doc_cols = {
        "ordinance_doc_id",
        "issue_id",
        "publication_key",
        "jurisdiction_city_name",
        "jurisdiction_state_abbr",
        "jurisdiction_level",
        "issue_date",
    }
    if (
        largest_clause_clusters.empty
        or clause_cluster_exemplars.empty
        or docs.empty
        or not required_family_cols.issubset(largest_clause_clusters.columns)
        or not required_exemplar_cols.issubset(clause_cluster_exemplars.columns)
        or not required_doc_cols.issubset(docs.columns)
    ):
        return pd.DataFrame()

    target_cluster_ids = [161, 6, 211, 209]
    family_df = (
        largest_clause_clusters[largest_clause_clusters["cluster_id"].isin(target_cluster_ids)]
        .copy()
        .drop_duplicates("cluster_id")
    )
    if family_df.empty:
        return pd.DataFrame()

    docs_meta = docs[
        [
            "ordinance_doc_id",
            "issue_id",
            "publication_key",
            "jurisdiction_city_name",
            "jurisdiction_state_abbr",
            "jurisdiction_level",
            "issue_date",
        ]
    ].copy()
    docs_meta["issue_year"] = pd.to_datetime(docs_meta["issue_date"], errors="coerce").dt.year

    merged = (
        clause_cluster_exemplars[
            [
                "cluster_id",
                "issue_id",
                "ordinance_doc_id",
                "section_title",
                "clause_text",
                "centroid_similarity",
            ]
        ]
        .merge(
            family_df[["cluster_id", "cluster_label", "clause_count", "ordinance_count"]],
            on="cluster_id",
            how="inner",
        )
        .merge(docs_meta, on=["ordinance_doc_id", "issue_id"], how="left")
    )
    merged["centroid_similarity"] = pd.to_numeric(merged["centroid_similarity"], errors="coerce").fillna(0.0)
    merged["cluster_id"] = pd.to_numeric(merged["cluster_id"], errors="coerce").astype("Int64")

    rows: list[dict[str, str]] = []
    for cluster_id in target_cluster_ids:
        family_rows = (
            merged[merged["cluster_id"] == cluster_id]
            .sort_values(["centroid_similarity", "issue_id"], ascending=[False, True])
            .copy()
        )
        if family_rows.empty:
            continue
        chosen = None
        for _, candidate in family_rows.iterrows():
            if _find_issue_first_page_image(candidate.get("issue_id", ""), repo_root=repo_root) is not None:
                chosen = candidate
                break
        if chosen is None:
            chosen = family_rows.iloc[0]
        publication = " ".join(str(chosen.get("publication_key", "") or "").replace("-", " ").replace("_", " ").split()).title()
        place = _category_label(chosen.get("jurisdiction_city_name", ""))
        state = str(chosen.get("jurisdiction_state_abbr", "") or "").upper().strip()
        year = ""
        if pd.notna(chosen.get("issue_year", np.nan)):
            try:
                year = str(int(chosen.get("issue_year")))
            except Exception:
                year = ""
        if publication and place and state and year:
            ordinance = f"{publication} / {place}, {state} ({year})"
        elif place and state and year:
            ordinance = f"{place}, {state} ({year})"
        else:
            ordinance = _issue_display_label(chosen.get("issue_id", ""))
        rows.append(
            {
                "clause_family": str(chosen.get("cluster_label", "") or ""),
                "ordinance": ordinance,
                "section": _clip(_best_section_label(chosen.get("section_title", ""), ""), 56),
                "clause_text": _clip(str(chosen.get("clause_text", "") or ""), 220),
                "issue_id": str(chosen.get("issue_id", "") or ""),
            }
        )
    return pd.DataFrame(rows)


def _clause_family_examples_table(df: pd.DataFrame) -> str:
    table_df = df.copy()
    if not table_df.empty:
        table_df["ordinance_link"] = table_df.apply(
            lambda row: _appendix_hyperlink(row.get("issue_id", ""), row.get("ordinance", "")),
            axis=1,
        )
    return _table_block(
        caption=(
            "Clause-family examples. Each row shows one representative clause from one of the larger recurring clause families listed above. "
            "Ordinance names link to the appendix."
        ),
        label="tab:clause-family-examples",
        colspec=r">{\raggedright\arraybackslash}p{0.23\textwidth}>{\raggedright\arraybackslash}p{0.19\textwidth}>{\raggedright\arraybackslash}p{0.22\textwidth}>{\raggedright\arraybackslash}p{0.28\textwidth}",
        header=["Clause family", "Example ordinance", "Section", "Representative clause"],
        rows=_rows_from_df(
            table_df,
            columns=["clause_family", "ordinance_link", "section", "clause_text"],
            formatters={"ordinance_link": lambda x: x},
        ),
    )


def _compact_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _pick_complexity_excerpt(text: Any, mode: str) -> str:
    compact = _compact_text(text)
    if not compact:
        return ""
    pattern_map = {
        "dependency_high": [
            r"It is not intended by this ordinance[^.]{0,340}\.",
            r"rules, regulations, or permits previously adopted[^.]{0,260}\.",
            r"subject to[^.]{0,220}\.",
        ],
        "linguistic_high": [
            r"General Statutes, as amended,[^.]{0,360}\.",
            r"to provide adequate light and air[^.]{0,260}\.",
            r"to prevent the overcrowding of land[^.]{0,240}\.",
        ],
        "direct_rule": [
            r"No building shall exceed[^.]{0,180}\.",
            r"no building or part thereof shall be erected[^.]{0,220}\.",
            r"Maximum height of building[^.;]{0,120}(?:[.;]|$)",
            r"Minimum lot area[^.;]{0,120}(?:[.;]|$)",
        ],
    }
    patterns = pattern_map.get(mode, [])
    for pattern in patterns:
        match = re.search(pattern, compact, flags=re.IGNORECASE)
        if match:
            return _clip(match.group(0), 220)

    fragments = [fragment.strip(" ;:") for fragment in re.split(r"(?<=[.!?;])\s+", compact) if fragment.strip()]
    if mode == "dependency_high":
        preferred = [frag for frag in fragments if re.search(r"\b(?:subject to|permit|board|commission|administrator|regulation|ordinance)\b", frag, flags=re.I)]
    elif mode == "linguistic_high":
        preferred = [frag for frag in fragments if len(frag.split()) >= 18]
    else:
        preferred = [frag for frag in fragments if re.search(r"\b(?:shall|minimum|maximum|required|prohibited)\b", frag, flags=re.I)]
    if preferred:
        return _clip(preferred[0], 220)
    return _clip(compact, 220)


def _excerpt_around_needle(text: Any, needle: str, *, limit: int = 220) -> str:
    compact = _compact_text(text)
    if not compact or not needle:
        return ""
    idx = compact.lower().find(needle.lower())
    if idx == -1:
        return ""
    window = compact[idx : idx + max(limit * 2, 420)]
    sentence_end = re.search(r"[.!?]", window[max(0, len(needle)) :])
    if sentence_end:
        end_idx = max(0, len(needle)) + sentence_end.start() + 1
        return _clip(window[:end_idx], limit)
    return _clip(window, limit)


def _build_complexity_examples(
    *,
    complexity_by_doc: pd.DataFrame,
    docs: pd.DataFrame,
) -> pd.DataFrame:
    required_complexity_cols = {
        "ordinance_doc_id",
        "issue_id",
        "issue_year",
        "jurisdiction_region",
        "structural_complexity_index",
        "dependency_complexity_index",
        "linguistic_complexity_index",
        "overall_legal_complexity_index",
        "reference_mentions_per_1000_words",
        "flesch_kincaid_grade_proxy",
        "avg_sentence_length_words",
        "section_count",
    }
    required_docs_cols = {
        "ordinance_doc_id",
        "text",
        "jurisdiction_city_name",
        "jurisdiction_state_abbr",
    }
    if (
        complexity_by_doc.empty
        or docs.empty
        or not required_complexity_cols.issubset(complexity_by_doc.columns)
        or not required_docs_cols.issubset(docs.columns)
    ):
        return pd.DataFrame()

    specs = [
        {
            "signal": "Northeast: heavier dependency",
            "issue_id": "chester-times__1945-11-20",
            "mode": "dependency_high",
            "needle": "It is not intended by this ordinance",
            "profile": lambda row: (
                f"Dependency {float(row['dependency_complexity_index']):+.2f}; "
                f"{float(row['reference_mentions_per_1000_words']):.1f} references / 1,000 words; "
                f"{_fmt_int(row['section_count'])} sections."
            ),
        },
        {
            "signal": "Midwest: lighter dependency",
            "issue_id": "cherokee-daily-times__1976-10-01",
            "mode": "direct_rule",
            "needle": "no building or part thereof shall be erected",
            "profile": lambda row: (
                f"Dependency {float(row['dependency_complexity_index']):+.2f}; "
                f"{float(row['reference_mentions_per_1000_words']):.1f} references / 1,000 words; "
                f"{_fmt_int(row['section_count'])} sections."
            ),
        },
        {
            "signal": "South: denser language",
            "issue_id": "statesville-record-and-landmark__1971-09-30",
            "mode": "linguistic_high",
            "needle": "General Statutes, as amended",
            "profile": lambda row: (
                f"Linguistic {float(row['linguistic_complexity_index']):+.2f}; "
                f"grade {float(row['flesch_kincaid_grade_proxy']):.1f}; "
                f"{float(row['avg_sentence_length_words']):.1f} words / sentence."
            ),
        },
        {
            "signal": "West: lighter overall",
            "issue_id": "rigby-star__1971-04-22",
            "mode": "direct_rule",
            "needle": "No building shall exceed two (2) stories in height",
            "profile": lambda row: (
                f"Overall {float(row['overall_legal_complexity_index']):+.2f}; "
                f"dependency {float(row['dependency_complexity_index']):+.2f}; "
                f"{float(row['reference_mentions_per_1000_words']):.1f} references / 1,000 words."
            ),
        },
    ]

    merged = complexity_by_doc.merge(
        docs[["ordinance_doc_id", "text", "jurisdiction_city_name", "jurisdiction_state_abbr"]],
        on="ordinance_doc_id",
        how="left",
        suffixes=("", "_doc"),
    ).copy()

    rows: list[dict[str, str]] = []
    for spec in specs:
        example = merged[merged["issue_id"].astype(str) == spec["issue_id"]].head(1)
        if example.empty:
            continue
        row = example.iloc[0]
        year = ""
        if pd.notna(row.get("issue_year")):
            try:
                year = str(int(row["issue_year"]))
            except Exception:
                year = ""
        city = _category_label(row.get("jurisdiction_city_name", ""))
        state = str(row.get("jurisdiction_state_abbr", "") or "").upper().strip()
        ordinance = f"{city}, {state} ({year})" if city and state and year else f"{city}, {state}".strip(", ")
        rows.append(
            {
                "signal": spec["signal"],
                "ordinance": ordinance,
                "language": (
                    _excerpt_around_needle(row.get("text", ""), str(spec.get("needle", "") or ""))
                    or _pick_complexity_excerpt(row.get("text", ""), spec["mode"])
                ),
                "score_profile": spec["profile"](row),
                "issue_id": str(row.get("issue_id", "") or ""),
            }
        )
    return pd.DataFrame(rows)


def _complexity_examples_table(df: pd.DataFrame) -> str:
    table_df = df.copy()
    if not table_df.empty:
        table_df["ordinance_link"] = table_df.apply(
            lambda row: _appendix_hyperlink(row.get("issue_id", ""), row.get("ordinance", "")),
            axis=1,
        )
    return _table_block(
        caption="Legal-complexity examples. Each row ties one regional pattern to ordinance language and the feature values driving that score. Ordinance names link to the appendix.",
        label="tab:complexity-examples",
        colspec=r"l l >{\raggedright\arraybackslash}X >{\raggedright\arraybackslash}X",
        header=["Signal", "Example ordinance", "Ordinance language", "What is driving it"],
        rows=_rows_from_df(
            table_df,
            columns=["signal", "ordinance_link", "language", "score_profile"],
            formatters={"ordinance_link": lambda x: x},
        ),
    )


def _build_referenced_page_appendix(
    *,
    purpose_examples: pd.DataFrame,
    clause_family_examples: pd.DataFrame,
    section_purpose_examples: pd.DataFrame,
    opening_move_examples: pd.DataFrame,
    style_family_examples: pd.DataFrame,
    complexity_examples: pd.DataFrame,
    repo_root: Path,
    asset_dir: Path,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    if not purpose_examples.empty and {"ordinance", "language", "issue_id"}.issubset(purpose_examples.columns):
        context_col = "bundle" if "bundle" in purpose_examples.columns else ("signal" if "signal" in purpose_examples.columns else None)
        if context_col:
            frames.append(
                purpose_examples.assign(
                    appendix_group="Purpose-bundle example",
                    context=lambda d: d[context_col].fillna("").astype(str),
                    excerpt=lambda d: d["language"].fillna("").astype(str),
                )[["appendix_group", "context", "ordinance", "excerpt", "issue_id"]]
            )
    if not clause_family_examples.empty and {"clause_family", "ordinance", "clause_text", "issue_id"}.issubset(clause_family_examples.columns):
        frames.append(
            clause_family_examples.assign(
                appendix_group="Clause-family example",
                context=lambda d: d["clause_family"].fillna("").astype(str),
                excerpt=lambda d: d["clause_text"].fillna("").astype(str),
            )[["appendix_group", "context", "ordinance", "excerpt", "issue_id"]]
        )
    if not section_purpose_examples.empty and {"headline_purpose", "ordinance", "language", "issue_id"}.issubset(section_purpose_examples.columns):
        frames.append(
            section_purpose_examples.assign(
                appendix_group="Section-purpose example",
                context=lambda d: d["headline_purpose"].fillna("").astype(str),
                excerpt=lambda d: d["language"].fillna("").astype(str),
            )[["appendix_group", "context", "ordinance", "excerpt", "issue_id"]]
        )
    if not opening_move_examples.empty and {"region", "ordinance", "language", "issue_id"}.issubset(opening_move_examples.columns):
        frames.append(
            opening_move_examples.assign(
                appendix_group="Opening-move example",
                context=lambda d: d["region"].fillna("").astype(str),
                excerpt=lambda d: d["language"].fillna("").astype(str),
            )[["appendix_group", "context", "ordinance", "excerpt", "issue_id"]]
        )
    if not style_family_examples.empty and {"style_name", "ordinance", "clause_examples", "issue_id"}.issubset(style_family_examples.columns):
        frames.append(
            style_family_examples.assign(
                appendix_group="Governance-style example",
                context=lambda d: d["style_name"].fillna("").astype(str),
                excerpt=lambda d: d["clause_examples"].fillna("").astype(str),
            )[["appendix_group", "context", "ordinance", "excerpt", "issue_id"]]
        )
    if not complexity_examples.empty and {"signal", "ordinance", "language", "issue_id"}.issubset(complexity_examples.columns):
        frames.append(
            complexity_examples.assign(
                appendix_group="Legal-complexity example",
                context=lambda d: d["signal"].fillna("").astype(str),
                excerpt=lambda d: d["language"].fillna("").astype(str),
            )[["appendix_group", "context", "ordinance", "excerpt", "issue_id"]]
        )
    if not frames:
        return pd.DataFrame()

    appendix = pd.concat(frames, ignore_index=True)
    appendix["issue_id"] = appendix["issue_id"].fillna("").astype(str)
    appendix = appendix[appendix["issue_id"].ne("")].drop_duplicates(subset=["issue_id"], keep="first").copy()
    if appendix.empty:
        return pd.DataFrame()

    appendix["issue_label"] = appendix["issue_id"].map(_issue_display_label)
    local_paths: list[str] = []
    embedded_assets: list[str] = []
    statuses: list[str] = []
    for row in appendix.to_dict(orient="records"):
        issue_id = str(row.get("issue_id", "") or "")
        local_image = _find_issue_first_page_image(issue_id, repo_root=repo_root)
        if local_image is None:
            local_paths.append("")
            embedded_assets.append("")
            statuses.append("missing_issue_png")
            continue
        embedded_asset = _copy_appendix_page_asset(local_image, asset_dir=asset_dir, issue_id=issue_id)
        local_paths.append(str(local_image))
        embedded_assets.append(str(embedded_asset))
        statuses.append("embedded")
    appendix["local_image_path"] = local_paths
    appendix["embedded_asset_path"] = embedded_assets
    appendix["asset_status"] = statuses
    return appendix


def _referenced_page_appendix_block(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    blocks = [
        r"\appendix",
        r"\section*{Appendix: Referenced Newspaper Issues}",
        (
            "This appendix collects the newspaper issues cited in the main text. "
            "Each ordinance link in the main text jumps here. "
            "For each issue, the appendix reproduces the excerpt quoted in the memo."
        ),
    ]
    for row in df.to_dict(orient="records"):
        title = str(row.get("issue_label") or row.get("issue_id") or row.get("ordinance") or "Referenced issue")
        appendix_label = _appendix_ref_label(row.get("issue_id", ""))
        blocks.append(r"\clearpage")
        if appendix_label:
            blocks.append(r"\phantomsection")
        blocks.append(rf"\subsection*{{{_tex_escape(title)}}}")
        if appendix_label:
            blocks.append(rf"\label{{{appendix_label}}}")
        blocks.append(r"\small")
        blocks.append(rf"\textbf{{Used in:}} {_tex_escape(row.get('appendix_group', ''))} ({_tex_escape(row.get('context', ''))})\\")
        blocks.append(rf"\textbf{{Ordinance:}} {_tex_escape(row.get('ordinance', ''))}\\")
        blocks.append(r"\normalsize")
        excerpt = str(row.get("excerpt", "") or "").strip()
        if excerpt:
            blocks.append(r"\textbf{Excerpt used in the main text:}")
            blocks.append(r"\begin{quote}\small")
            blocks.append(_tex_escape(excerpt))
            blocks.append(r"\end{quote}")
        embedded_asset = str(row.get("embedded_asset_path", "") or "").strip()
        if embedded_asset:
            blocks.append(r"\begin{center}")
            blocks.append(rf"\includegraphics[width=\textwidth,height=0.82\textheight,keepaspectratio]{{{embedded_asset}}}")
            blocks.append(r"\end{center}")
        else:
            status_note = "Issue image not available in this environment."
            blocks.append(
                r"\fbox{\parbox{0.96\textwidth}{"
                + r"\textbf{Page image unavailable locally.} "
                + _tex_escape(status_note)
                + r"}}"
            )
    return "\n\n".join(blocks)


def _plot_purpose_lift_heatmap(df: pd.DataFrame, *, group_col: str, out_path: Path, title: str) -> None:
    if df.empty:
        return
    plot_df = df[df["motive"].astype(str).isin(PURPOSE_ORDER)].copy()
    if plot_df.empty:
        return
    pivot = plot_df.pivot_table(
        index=group_col,
        columns="motive",
        values="purpose_weight_lift_vs_overall_vintage_adjusted",
        aggfunc="first",
    )
    desired_rows = REGION_ORDER if group_col == "jurisdiction_region" else LEVEL_ORDER
    row_order = [item for item in desired_rows if item in pivot.index]
    col_order = [item for item in PURPOSE_ORDER if item in pivot.columns]
    pivot = pivot.reindex(index=row_order, columns=col_order)
    if pivot.empty:
        return
    lift_values = pivot.to_numpy(dtype=float)
    values = (lift_values - 1.0) * 100.0
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return
    delta = max(abs(float(finite.min())), abs(float(finite.max())), 1.0)
    fig, ax = plt.subplots(figsize=(max(8, 0.85 * len(col_order) + 2), max(4.5, 0.62 * len(row_order) + 1.5)))
    im = ax.imshow(values, aspect="auto", cmap="coolwarm", vmin=-delta, vmax=delta)
    ax.set_xticks(range(len(col_order)))
    ax.set_xticklabels([_category_label(x) for x in col_order], rotation=35, ha="right")
    ax.set_yticks(range(len(row_order)))
    ax.set_yticklabels([_category_label(x) for x in row_order])
    ax.set_title(title)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            val = values[i, j]
            if not np.isfinite(val):
                continue
            color = "white" if abs(val) > delta * 0.55 else "#1a202c"
            ax.text(j, i, f"{val:+.1f}", ha="center", va="center", fontsize=8, color=color)
    cbar = fig.colorbar(im, ax=ax, shrink=0.88)
    cbar.set_label("Percent difference vs full-sample baseline")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_overall_purpose_presence(df: pd.DataFrame, *, out_path: Path) -> None:
    required_cols = {"motive", "ordinance_share"}
    if df.empty or not required_cols.issubset(df.columns):
        return
    plot_df = df.copy().sort_values(["ordinance_share", "motive"], ascending=[True, True]).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8.8, max(4.8, 0.52 * len(plot_df) + 1.8)))
    y = np.arange(len(plot_df))
    ax.barh(y, plot_df["ordinance_share"], color="#4c72b0", edgecolor="white", linewidth=0.8, height=0.72)
    ax.set_yticks(y)
    ax.set_yticklabels([_purpose_short_label(value) for value in plot_df["motive"]], fontsize=10)
    ax.set_xlim(0.0, 1.05)
    xticks = np.linspace(0.0, 1.0, 6)
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{tick:.0%}" for tick in xticks])
    ax.set_xlabel("Share of ordinances whose returned bundle includes that purpose")
    ax.set_title("How often each purpose appears across ordinances")
    ax.grid(axis="x", color="#cbd5e0", linewidth=0.6, alpha=0.35)
    for idx, row in plot_df.iterrows():
        value = float(row["ordinance_share"])
        ax.text(min(value + 0.012, 1.01), idx, f"{value:.1%}", va="center", ha="left", fontsize=9, color="#1a202c")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_purpose_cooccurrence_heatmap(df: pd.DataFrame, *, out_path: Path) -> None:
    if df.empty:
        return
    values = df.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(8.6, 7.4))
    im = ax.imshow(values, cmap="YlGnBu", vmin=0.0, vmax=max(float(values.max()), 0.35))
    ax.set_xticks(range(len(df.columns)))
    ax.set_xticklabels([_purpose_short_label(x) for x in df.columns], rotation=35, ha="right")
    ax.set_yticks(range(len(df.index)))
    ax.set_yticklabels([_purpose_short_label(x) for x in df.index])
    ax.set_title("Purpose co-occurrence across ordinances")
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            val = float(values[i, j])
            color = "white" if val > (values.max() * 0.55) else "#1a202c"
            ax.text(j, i, f"{val:.0%}", ha="center", va="center", fontsize=8, color=color)
    cbar = fig.colorbar(im, ax=ax, shrink=0.88)
    cbar.set_label("Share of ordinances containing that purpose pair")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_ranked_purpose_region_dotplot(df: pd.DataFrame, *, out_path: Path) -> None:
    if df.empty:
        return
    plot_df = df[df["motive"].astype(str).isin(PURPOSE_ORDER)].copy()
    if plot_df.empty:
        return
    plot_df["pct_diff"] = (
        pd.to_numeric(plot_df["purpose_weight_lift_vs_overall_vintage_adjusted"], errors="coerce").fillna(1.0) - 1.0
    ) * 100.0
    regions = [region for region in REGION_ORDER if region in set(plot_df["jurisdiction_region"].astype(str))]
    if not regions:
        return
    delta = max(float(plot_df["pct_diff"].abs().max()), 1.0)
    fig, axes = plt.subplots(len(regions), 1, figsize=(9.2, max(7.2, 2.2 * len(regions))), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, region in zip(axes, regions, strict=False):
        group = (
            plot_df[plot_df["jurisdiction_region"].astype(str) == region]
            .sort_values(["pct_diff", "motive"], ascending=[False, True])
            .reset_index(drop=True)
        )
        y = np.arange(len(group))
        color = REGION_COLORS.get(region, "#4a5568")
        ax.axvline(0.0, color="#4a5568", linewidth=0.9, linestyle="--", alpha=0.8)
        ax.hlines(y, 0.0, group["pct_diff"], color=color, linewidth=2.2, alpha=0.35)
        ax.scatter(
            group["pct_diff"],
            y,
            s=52,
            color=color,
            edgecolor="white",
            linewidth=0.7,
            zorder=3,
        )
        for idx, row in group.iterrows():
            value = float(row["pct_diff"])
            x_offset = 0.22 if value >= 0.0 else -0.22
            ha = "left" if value >= 0.0 else "right"
            ax.text(value + x_offset, idx, f"{value:+.1f}", va="center", ha=ha, fontsize=8, color="#1a202c")
        ax.set_yticks(y)
        ax.set_yticklabels([_purpose_short_label(value) for value in group["motive"]], fontsize=9)
        ax.invert_yaxis()
        ax.set_title(_category_label(region), loc="left", fontsize=11, color=color)
        ax.grid(axis="x", color="#cbd5e0", linewidth=0.6, alpha=0.35)
        ax.set_xlim(-(delta * 1.28), delta * 1.28)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    axes[-1].set_xlabel("Percent difference vs full-sample benchmark")
    fig.suptitle("Ranked regional purpose deviations", y=0.995, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_section_purpose_mix_chart(df: pd.DataFrame, *, out_path: Path) -> None:
    if df.empty:
        return
    pivot = (
        df.pivot_table(
            index="doc_primary_motive",
            columns="section_primary_motive",
            values="share_within_doc_family",
            aggfunc="first",
            fill_value=0.0,
            observed=False,
        )
        .reindex(index=[motive for motive in PURPOSE_ORDER if motive in set(df["doc_primary_motive"].astype(str))])
        .reindex(columns=SECTION_PURPOSE_STACK_ORDER, fill_value=0.0)
    )
    if pivot.empty:
        return
    count_map = (
        df[["doc_primary_motive", "ordinance_count"]]
        .drop_duplicates()
        .assign(doc_primary_motive=lambda d: d["doc_primary_motive"].astype(str))
        .set_index("doc_primary_motive")["ordinance_count"]
        .to_dict()
    )

    fig, ax = plt.subplots(figsize=(9.4, max(4.8, 0.62 * len(pivot.index) + 1.8)))
    y = np.arange(len(pivot.index))
    left = np.zeros(len(pivot.index))
    for motive in SECTION_PURPOSE_STACK_ORDER:
        values = pivot[motive].to_numpy(dtype=float)
        if np.allclose(values, 0.0):
            continue
        ax.barh(
            y,
            values,
            left=left,
            height=0.72,
            color=SECTION_PURPOSE_COLORS.get(motive, "#a0aec0"),
            edgecolor="white",
            linewidth=0.7,
            label=_purpose_short_label(motive),
        )
        left += values

    labels = [
        f"{_purpose_short_label(motive)} (n={_fmt_int(count_map.get(str(motive), 0))})"
        for motive in pivot.index.astype(str)
    ]
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0.0, 1.0)
    xticks = np.linspace(0.0, 1.0, 6)
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{tick:.0%}" for tick in xticks])
    ax.set_xlabel("Share of substantive section text")
    ax.set_ylabel("Primary document purpose")
    ax.set_title("How section text is distributed within each document purpose")
    ax.grid(axis="x", color="#cbd5e0", linewidth=0.6, alpha=0.35)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        deduped: dict[str, Any] = {}
        for handle, label in zip(handles, labels, strict=False):
            deduped.setdefault(label, handle)
        ax.legend(
            deduped.values(),
            deduped.keys(),
            title="Section purpose",
            frameon=False,
            bbox_to_anchor=(1.02, 1.0),
            loc="upper left",
        )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _clause_cluster_centroids(df: pd.DataFrame) -> pd.DataFrame:
    required = {"cluster_id", "cluster_label", "umap_x", "umap_y"}
    if df.empty or not required.issubset(df.columns):
        return pd.DataFrame(columns=["cluster_id", "cluster_label", "cluster_size", "umap_x", "umap_y"])
    plot_df = df.copy()
    plot_df["cluster_id"] = pd.to_numeric(plot_df["cluster_id"], errors="coerce")
    plot_df = plot_df[plot_df["cluster_id"].ge(0)].copy()
    plot_df = plot_df[np.isfinite(plot_df["umap_x"]) & np.isfinite(plot_df["umap_y"])].copy()
    if plot_df.empty:
        return pd.DataFrame(columns=["cluster_id", "cluster_label", "cluster_size", "umap_x", "umap_y"])
    rows: list[dict[str, Any]] = []
    for cluster_id, group in plot_df.groupby("cluster_id", dropna=False):
        label_series = group["cluster_label"].fillna("").astype(str).str.strip()
        label = next((value for value in label_series.tolist() if value), f"cluster {int(cluster_id)}")
        rows.append(
            {
                "cluster_id": int(cluster_id),
                "cluster_label": label,
                "cluster_size": int(len(group)),
                "umap_x": float(group["umap_x"].median()),
                "umap_y": float(group["umap_y"].median()),
            }
        )
    return pd.DataFrame.from_records(rows).sort_values(["cluster_size", "cluster_id"], ascending=[False, True]).reset_index(drop=True)


def _select_clause_zoom_centroids(
    centroids: pd.DataFrame,
    *,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    limit: int = 18,
) -> pd.DataFrame:
    if centroids.empty:
        return centroids
    inside = centroids[
        centroids["umap_x"].between(xlim[0], xlim[1]) & centroids["umap_y"].between(ylim[0], ylim[1])
    ].copy()
    if inside.empty:
        return inside
    x_span = max(float(xlim[1] - xlim[0]), 1e-6)
    y_span = max(float(ylim[1] - ylim[0]), 1e-6)
    min_sep = 0.085
    selected_rows: list[int] = []
    selected_points: list[tuple[float, float]] = []
    for idx, row in inside.sort_values(["cluster_size", "cluster_id"], ascending=[False, True]).iterrows():
        x = float(row["umap_x"])
        y = float(row["umap_y"])
        too_close = False
        for sx, sy in selected_points:
            dx = (x - sx) / x_span
            dy = (y - sy) / y_span
            if (dx * dx + dy * dy) ** 0.5 < min_sep:
                too_close = True
                break
        if too_close:
            continue
        selected_rows.append(idx)
        selected_points.append((x, y))
        if len(selected_rows) >= limit:
            break
    return inside.loc[selected_rows].reset_index(drop=True)


def _plot_clause_umap_zoom(df: pd.DataFrame, *, out_path: Path) -> None:
    required = {"cluster_id", "cluster_label", "umap_x", "umap_y"}
    if df.empty or not required.issubset(df.columns):
        return
    plot_df = df.copy()
    plot_df["cluster_id"] = pd.to_numeric(plot_df["cluster_id"], errors="coerce")
    plot_df = plot_df[np.isfinite(plot_df["umap_x"]) & np.isfinite(plot_df["umap_y"])].copy()
    if plot_df.empty:
        return

    x = plot_df["umap_x"].to_numpy(dtype=float)
    y = plot_df["umap_y"].to_numpy(dtype=float)
    xbins = np.linspace(float(np.nanmin(x)), float(np.nanmax(x)), 7)
    ybins = np.linspace(float(np.nanmin(y)), float(np.nanmax(y)), 7)
    hist, xedges, yedges = np.histogram2d(x, y, bins=[xbins, ybins])
    ix, iy = np.unravel_index(int(np.argmax(hist)), hist.shape)
    x0, x1 = float(xedges[ix]), float(xedges[ix + 1])
    y0, y1 = float(yedges[iy]), float(yedges[iy + 1])
    x_margin = (x1 - x0) * 0.45
    y_margin = (y1 - y0) * 0.45
    xlim = (x0 - x_margin, x1 + x_margin)
    ylim = (y0 - y_margin, y1 + y_margin)

    view_df = plot_df[
        plot_df["umap_x"].between(xlim[0], xlim[1]) & plot_df["umap_y"].between(ylim[0], ylim[1])
    ].copy()
    if view_df.empty:
        return

    fig, ax = plt.subplots(figsize=(9.4, 7.6))
    noise = view_df[view_df["cluster_id"].lt(0)].copy()
    non_noise = view_df[view_df["cluster_id"].ge(0)].copy()
    if not noise.empty:
        ax.scatter(
            noise["umap_x"],
            noise["umap_y"],
            color="#cbd5e0",
            s=10,
            alpha=0.35,
            linewidths=0.0,
            zorder=1,
        )
    cluster_ids = sorted(int(value) for value in non_noise["cluster_id"].dropna().astype(int).unique().tolist())
    cmap = plt.get_cmap("tab20", max(1, len(cluster_ids)))
    for idx, cluster_id in enumerate(cluster_ids):
        group = non_noise[non_noise["cluster_id"].astype(int) == cluster_id]
        ax.scatter(
            group["umap_x"],
            group["umap_y"],
            color=cmap(idx),
            s=14,
            alpha=0.72,
            linewidths=0.0,
            zorder=2,
        )

    centroids = _clause_cluster_centroids(plot_df)
    labeled = _select_clause_zoom_centroids(centroids, xlim=xlim, ylim=ylim, limit=18)
    offsets = [
        (14, 14), (14, -14), (-14, 14), (-14, -14),
        (20, 0), (-20, 0), (0, 20), (0, -20),
        (24, 12), (24, -12), (-24, 12), (-24, -12),
        (12, 24), (12, -24), (-12, 24), (-12, -24),
    ]
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    placed_bboxes = []
    for _, row in labeled.iterrows():
        x0 = float(row["umap_x"])
        y0 = float(row["umap_y"])
        label_text = textwrap.fill(str(row["cluster_label"]).replace("_", " "), width=24)
        ax.scatter([x0], [y0], color="black", s=22, marker="x", linewidths=0.75, zorder=4)
        for dx, dy in offsets:
            ann = ax.annotate(
                label_text,
                xy=(x0, y0),
                xytext=(dx, dy),
                textcoords="offset points",
                fontsize=6.8,
                ha="left" if dx >= 0 else "right",
                va="bottom" if dy >= 0 else "top",
                bbox={"boxstyle": "round,pad=0.22", "facecolor": "white", "edgecolor": "#1a202c", "alpha": 0.94, "linewidth": 0.45},
                arrowprops={"arrowstyle": "-", "color": "#1a202c", "linewidth": 0.40, "alpha": 0.55},
                zorder=5,
            )
            fig.canvas.draw()
            bbox = ann.get_window_extent(renderer=renderer).expanded(1.03, 1.10)
            if any(bbox.overlaps(other) for other in placed_bboxes):
                ann.remove()
                continue
            placed_bboxes.append(bbox)
            break

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_title("Clause-level view: dense middle of the manifold")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.grid(alpha=0.12, linewidth=0.4)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _coverage_rows(audit: dict[str, Any]) -> list[list[str]]:
    rows: list[list[str]] = []
    for row in audit.get("surfaces", []):
        rows.append(
            [
                _tex_escape(row.get("surface", "")),
                _tex_escape(row.get("status", "")),
                _tex_escape(_clip(row.get("rationale", ""), 90)),
            ]
        )
    return rows


def _drop_placeholder_clusters(df: pd.DataFrame, *, label_col: str = "cluster_label") -> pd.DataFrame:
    if df.empty or label_col not in df.columns:
        return df
    label = df[label_col].fillna("").astype(str).str.strip()
    mask = label.ne("") & ~label.str.fullmatch(r"cluster \d+") & label.str.lower().ne("noise")
    filtered = df.loc[mask].copy()
    return filtered if not filtered.empty else df


def build_results_atlas(
    *,
    frontier_run_root: Path,
    complexity_run_root: Path,
    output_path: Path | None = None,
) -> Path:
    repo_root = _repo_root()
    frontier_run_root = frontier_run_root.resolve()
    complexity_run_root = complexity_run_root.resolve()
    update_date = date.today().isoformat()
    reports_dir = frontier_run_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    output_path = (output_path or (reports_dir / "frontier_results_atlas.tex")).resolve()

    frontier_research = frontier_run_root / "research"
    frontier_reports = frontier_run_root / "reports"
    complexity_reports = complexity_run_root / "reports"
    complexity_root = complexity_run_root / "complexity"
    complexity_bridge_root = complexity_run_root / "bridge"

    full_summary = _load_json(frontier_research / "research_summary__full.json")
    pilot_summary = _load_json(frontier_research / "research_summary__pilot.json")
    complexity_summary = _load_json(complexity_run_root / "manifests" / "summary.json")
    coverage_audit = _load_json(frontier_research / "coverage" / "research_surface_coverage_audit.json")
    coverage_summary = coverage_audit.get("summary", {})
    stable_surface_count = int(coverage_summary.get("stable_surface_count", 0))
    emerging_surface_count = int(coverage_summary.get("emerging_surface_count", 0))
    partial_surface_count = int(coverage_summary.get("partial_surface_count", 0))
    deferred_surface_count = int(coverage_summary.get("deferred_surface_count", 0))

    full_region = _read_parquet(frontier_research / "section_clusters" / "top_region_contrasts_vintage_adjusted__full.parquet")
    full_level = _read_parquet(frontier_research / "section_clusters" / "top_level_contrasts_vintage_adjusted__full.parquet")
    purpose_region = _read_parquet(frontier_research / "purposes" / "weighted_purpose_by_region_vintage_adjusted__full.parquet")
    document_purposes_full = _read_parquet(frontier_run_root / "semantics" / "document_purposes__full.parquet")
    docs_full = _read_parquet(frontier_run_root / "corpus" / "docs.parquet")
    full_complexity_models = _read_parquet(frontier_research / "coherence" / "complexity_alignment_models__full.parquet")
    surface_comparison = _read_parquet(frontier_research / "embedding_comparison" / "surface_comparison__full.parquet")
    positive_residuals = _read_parquet(frontier_research / "coherence" / "coherence_positive_residuals__full.parquet")
    negative_residuals = _read_parquet(frontier_research / "coherence" / "coherence_negative_residuals__full.parquet")

    legal_region = _read_parquet(complexity_root / "complexity_by_region_vintage_adjusted.parquet")
    legal_level = _read_parquet(complexity_root / "complexity_by_level_vintage_adjusted.parquet")
    complexity_by_doc = _read_parquet(complexity_root / "complexity_by_doc.parquet")
    legal_effects = _read_parquet(complexity_root / "top_complexity_region_level_effects.parquet")
    legal_bridge_coherence = _read_parquet(complexity_bridge_root / "pilot_coherence_models.parquet")
    legal_bridge_governance = _read_parquet(complexity_bridge_root / "pilot_governance_models.parquet")
    legal_bridge_subcomponents = _read_parquet(complexity_bridge_root / "pilot_subcomponent_bridge_models.parquet")

    pilot_style = _read_parquet(frontier_research / "style_families" / "regulatory_style_family_summary__pilot.parquet")
    pilot_style_prototypes = _read_parquet(frontier_research / "style_families" / "regulatory_style_family_prototypes__pilot.parquet")
    pilot_largest_clause = _read_parquet(frontier_research / "clause_clusters" / "largest_clause_clusters__pilot.parquet")
    clause_cluster_exemplars = _read_parquet(frontier_run_root / "clusters" / "clause_instrument_hdbscan" / "cluster_exemplars.parquet")
    clause_assignments = _read_parquet(frontier_run_root / "clusters" / "clause_instrument_hdbscan" / "assignments.parquet")
    clause_umap = _read_parquet(frontier_run_root / "embeddings" / "clause" / "umap_2d.parquet")
    enriched_clauses_pilot = _read_parquet(frontier_run_root / "semantics" / "enriched_clauses__pilot.parquet")

    policy_raw = _load_json(frontier_run_root / "clusters" / "policy_signature_kmeans_probe" / "summary.json")
    policy_norm = _load_json(frontier_run_root / "clusters" / "policy_signature_section_normalized_clause_share_probe" / "summary.json")
    policy_presence = _load_json(frontier_run_root / "clusters" / "policy_signature_section_presence_probe" / "summary.json")

    full_region = _drop_placeholder_clusters(full_region)
    full_region = (
        full_region.sort_values(
            ["jurisdiction_region", "presence_lift_vs_overall_vintage_adjusted", "presence_rate_vintage_adjusted"],
            ascending=[True, False, False],
        )
        .groupby("jurisdiction_region", dropna=False)
        .head(2)
        .reset_index(drop=True)
    )
    full_level = _drop_placeholder_clusters(full_level)
    full_level = full_level[full_level["ordinance_count"].fillna(0) >= 5].copy()
    full_level = (
        full_level.sort_values(
            ["jurisdiction_level", "presence_lift_vs_overall_vintage_adjusted", "presence_rate_vintage_adjusted"],
            ascending=[True, False, False],
        )
        .groupby("jurisdiction_level", dropna=False)
        .head(2)
        .reset_index(drop=True)
    )
    purpose_weights = _build_document_purpose_weights(document_purposes_full)
    overall_purpose_summary = _summarize_overall_purpose_presence(purpose_weights)
    purpose_concentration = _summarize_document_purpose_concentration(purpose_weights)
    cooccurrence_order = overall_purpose_summary["motive"].tolist() if not overall_purpose_summary.empty else PURPOSE_ORDER
    purpose_cooccurrence = _build_purpose_cooccurrence_matrix(purpose_weights, motive_order=cooccurrence_order)
    purpose_examples = _build_purpose_bundle_examples(
        document_purposes=document_purposes_full,
        docs=docs_full,
    )
    clause_family_examples = _build_clause_family_examples(
        largest_clause_clusters=pilot_largest_clause,
        clause_cluster_exemplars=clause_cluster_exemplars,
        docs=docs_full,
        repo_root=repo_root,
    )
    _plot_overall_purpose_presence(
        overall_purpose_summary,
        out_path=frontier_reports / "figures" / "overall_purpose_presence__full.png",
    )
    _plot_purpose_cooccurrence_heatmap(
        purpose_cooccurrence,
        out_path=frontier_reports / "figures" / "purpose_cooccurrence_heatmap__full.png",
    )
    clause_umap_plot_df = (
        clause_assignments.merge(clause_umap[["item_id", "umap_x", "umap_y"]], on="item_id", how="left")
        if not clause_assignments.empty and not clause_umap.empty and {"item_id", "umap_x", "umap_y"}.issubset(clause_umap.columns)
        else pd.DataFrame()
    )
    _plot_clause_umap_zoom(
        clause_umap_plot_df,
        out_path=frontier_reports / "figures" / "clause_instrument_hdbscan_umap_zoom.png",
    )

    if not positive_residuals.empty:
        positive_residuals = positive_residuals.head(6).copy()
    if not negative_residuals.empty:
        negative_residuals = negative_residuals.head(6).copy()

    legal_effects = legal_effects.loc[legal_effects["group_name"].astype(str) != "vintage_bin"].copy()
    legal_effects = legal_effects.sort_values(["pvalue", "coefficient"], ascending=[True, False]).head(8).copy()

    if not legal_bridge_coherence.empty:
        legal_bridge_coherence = (
            legal_bridge_coherence[legal_bridge_coherence["term"].astype(str) != "Intercept"]
            .loc[lambda d: ~d["term"].astype(str).str.startswith("C(jurisdiction_level)[T.")]
            .sort_values(["pvalue", "coefficient"], ascending=[True, False])
            .head(8)
            .copy()
        )
    if not legal_bridge_governance.empty:
        legal_bridge_governance = (
            legal_bridge_governance[legal_bridge_governance["term"].astype(str) != "Intercept"]
            .loc[lambda d: ~d["term"].astype(str).str.startswith("C(jurisdiction_level)[T.")]
            .sort_values(["pvalue", "coefficient"], ascending=[True, False])
            .head(10)
            .copy()
        )
    if not legal_bridge_subcomponents.empty:
        legal_bridge_subcomponents = (
            legal_bridge_subcomponents[legal_bridge_subcomponents["term"].astype(str) != "Intercept"]
            .loc[lambda d: ~d["term"].astype(str).str.startswith("C(jurisdiction_level)[T.")]
            .sort_values(["pvalue", "coefficient"], ascending=[True, False])
            .head(10)
            .copy()
        )

    if not pilot_largest_clause.empty and "cluster_label" in pilot_largest_clause.columns:
        mask = (
            pilot_largest_clause["cluster_label"].fillna("").str.strip().ne("")
            & ~pilot_largest_clause["cluster_label"].fillna("").str.fullmatch(r"cluster \d+")
            & pilot_largest_clause["cluster_label"].fillna("").str.lower().ne("noise")
        )
        filtered = pilot_largest_clause[mask].copy()
        if not filtered.empty:
            pilot_largest_clause = filtered
    pilot_largest_clause = pilot_largest_clause.head(10).copy()

    full_backbone_table = _table_block(
        caption="Section-architecture backbone summary.",
        label="tab:full-backbone",
        colspec=r">{\raggedright\arraybackslash}Xr",
        header=["Metric", "Value"],
        rows=[
            ["Documents in section-architecture corpus", _fmt_int(full_summary["doc_count"])],
            ["Section semantic families", _fmt_int(full_summary["section_cluster_count"])],
            ["Mean purpose/section alignment share", _fmt_float(full_summary["mean_alignment_share"])],
            ["Mean non-`other` alignment share", _fmt_float(full_summary["mean_non_other_alignment_share"])],
            ["Complexity/alignment raw correlation", _fmt_float(full_summary["complexity_alignment_correlation"])],
            ["Complexity/alignment adjusted coefficient", _fmt_float(full_summary["complexity_alignment_adjusted_coef"])],
            ["Complexity/alignment adjusted p-value", _fmt_float(full_summary["complexity_alignment_adjusted_pvalue"], 4)],
            ["Architecture families", _fmt_int(full_summary["architecture_family_count"])],
        ],
    )

    purpose_intro = _paragraph_block(
        [
            "We pass the full ordinance text to the model and ask it to identify the main regulatory purposes stated in the ordinance. The model can return up to five purposes for a single ordinance, so the output is usually a small purpose bundle rather than one label.",
            f"We use the returned scores as weights and normalize them into a within-ordinance document-purpose mix. In the current run, the median ordinance has {int(round(purpose_concentration.get('median_label_count', 0.0)))} returned purposes, and the strongest one accounts for about {float(purpose_concentration.get('median_top_weight', 0.0)) * 100.0:.1f}\\% of the total weight.",
        ]
    )
    purpose_label_table = _table_block(
        caption="Document-purpose labels used in this section.",
        label="tab:purpose-labels",
        colspec=r">{\raggedright\arraybackslash}X>{\raggedright\arraybackslash}X",
        header=["Short label", "Label definition"],
        rows=[
            ["Public Health", "Protecting health, safety, welfare, and related public-risk concerns."],
            ["Growth Mgmt.", "Managing orderly growth, land development, density, and planning."],
            ["Externality", "Limiting nuisance, congestion, hazards, and conflicts between incompatible uses."],
            ["Infrastructure", "Coordinating development with streets, utilities, water, sewer, transportation, schools, and public services."],
            ["Aesthetic", "Regulating appearance, design, signs, landscaping, and visual harmony."],
            ["Environmental", "Protecting floodplains, wetlands, drainage systems, natural resources, and other environmental conditions."],
            ["Exclusion", "Preserving residential character or separating uses by keeping certain uses out."],
            ["Value Capture", "Protecting property values, land values, building values, or the tax base."],
            ["Fiscal", "Raising, protecting, or managing taxes, assessments, fees, rents, or revenue."],
            ["Other", "A stated regulatory purpose that does not fit one of the main categories."],
            ["Unclear", "Too little signal in the text to assign a substantive purpose with confidence."],
        ],
    )
    purpose_weights_intro = _paragraph_block(
        [
            "The first chart asks how often each purpose appears anywhere in the returned bundle. The denominator is all ordinances in the run. Public health and growth language appear in most ordinances, while fiscal and value-capture language appear much less often.",
        ],
        needspace="0.62\\textheight",
    )
    purpose_cooccurrence_intro = _paragraph_block(
        [
            "The next chart uses the same full-document purpose bundles and asks which purposes tend to appear together in the same ordinance. Public health, growth, infrastructure, and externality-control language often appear in the same document. That pattern matches what shows up when you read the ordinances themselves: these texts usually stack several goals together instead of making a single narrow claim.",
        ],
        needspace="0.62\\textheight",
    )
    purpose_examples_intro = _paragraph_block(
        [
            "The examples table below keeps that full bundle visible. Each row shows the returned purpose mix for one ordinance together with a couple of short evidence lines from the same text. The rows are bundle summaries for individual ordinances.",
        ],
        needspace="0.52\\textheight",
    )
    clause_intro = _paragraph_block(
        [
            "For the pilot subset, we start from the section text and process one section at a time. A clause-extraction pass reads that section and returns atomic regulatory statements rather than full paragraphs. For each extracted clause, it outputs the verbatim clause text, a normalized requirement, one motive label, one modality label, and one short instrument hint. If the section is mostly purpose language, publication boilerplate, signatures, or other non-operative text, that pass can return no clauses at all.",
            "A second enrichment pass then takes each extracted clause together with its section context and adds structured fields such as actor, action, condition, exception, threshold, spatial scope, instrument type, and an operative flag. We keep the clauses marked operative for the clause-layer results. That produces 28,748 enriched operative clauses across 61 ordinances in the pilot subset. The table counts the most common recurring clause types across those clauses, and the chart places those clause types by clause-embedding similarity.",
        ],
    )
    clause_family_intro = _paragraph_block(
        [
            "Taken together, they show both which clause patterns recur most often and which ones sit near each other in semantic space.",
        ],
        needspace="0.55\\textheight",
    )
    clause_examples_intro = _paragraph_block(
        [
            "The next table makes those family labels more concrete. Each row shows one representative clause from a recurring family, together with the ordinance and section where it appears.",
        ],
        needspace="0.52\\textheight",
    )
    clause_zoom_intro = _paragraph_block(
        [
            "The zoom below stays on the same clause map and focuses on its crowded middle, where many mid-sized recurring clause families sit close together.",
        ],
        needspace="0.56\\textheight",
    )
    purpose_examples_table = _purpose_examples_table(purpose_examples)
    clause_family_examples_table = _clause_family_examples_table(clause_family_examples)
    referenced_page_appendix = _build_referenced_page_appendix(
        purpose_examples=purpose_examples,
        clause_family_examples=clause_family_examples,
        section_purpose_examples=pd.DataFrame(),
        opening_move_examples=pd.DataFrame(),
        style_family_examples=pd.DataFrame(),
        complexity_examples=pd.DataFrame(),
        repo_root=repo_root,
        asset_dir=frontier_reports / "figures" / "appendix_referenced_pages",
    )
    referenced_page_appendix_block = _referenced_page_appendix_block(referenced_page_appendix)

    region_table = _table_block(
        caption="Top two section-family lifts by region. `Presence` is the region-specific section presence rate and `Lift` is that presence divided by the overall benchmark rate, so values above one indicate section families that are overrepresented in that region.",
        label="tab:full-region-contrasts",
        colspec=r"l>{\raggedright\arraybackslash}Xrr",
        header=["Region", "Section family", "Presence", "Lift"],
        rows=_rows_from_df(
            full_region,
            columns=["jurisdiction_region", "cluster_label", "presence_rate_vintage_adjusted", "presence_lift_vs_overall_vintage_adjusted"],
            formatters={
                "jurisdiction_region": _tex_escape,
                "cluster_label": lambda x: _tex_escape(_clip(x, 58)),
                "presence_rate_vintage_adjusted": lambda x: _fmt_float(x, 3),
                "presence_lift_vs_overall_vintage_adjusted": lambda x: _fmt_float(x, 3),
            },
        ),
    )

    level_table = _table_block(
        caption="Top two section-family lifts per jurisdiction type on the full sample. This is the jurisdiction-type analog to Table~\\ref{tab:full-region-contrasts}. `Presence` is the jurisdiction-type-specific section presence rate and `Lift` compares that presence to the full-sample baseline. Values above one indicate section families that are overrepresented in that jurisdiction type.",
        label="tab:full-level-contrasts",
        colspec=r"l>{\raggedright\arraybackslash}Xrrr",
        header=["Level", "Section family", "Docs", "Presence", "Lift"],
        rows=_rows_from_df(
            full_level,
            columns=["jurisdiction_level", "cluster_label", "ordinance_count", "presence_rate_vintage_adjusted", "presence_lift_vs_overall_vintage_adjusted"],
            formatters={
                "jurisdiction_level": _tex_escape,
                "cluster_label": lambda x: _tex_escape(_clip(x, 52)),
                "ordinance_count": _fmt_int,
                "presence_rate_vintage_adjusted": lambda x: _fmt_float(x, 3),
                "presence_lift_vs_overall_vintage_adjusted": lambda x: _fmt_float(x, 3),
            },
        ),
    )

    residual_table = _table_block(
        caption="Extreme coherence residual cases. Residuals come from models that predict document-to-section semantic alignment using ordinance complexity while accounting for institutional composition in the background. Positive residuals are ordinances that are more coherent than the model predicts; negative residuals are more fragmented than the model predicts.",
        label="tab:coherence-residuals",
        colspec=r"l>{\raggedright\arraybackslash}Xlr",
        header=["Sign", "Ordinance", "Region", "Residual"],
        rows=(
            _rows_from_df(
                positive_residuals.assign(sign="+"),
                columns=["sign", "ordinance_doc_id", "jurisdiction_region", "residual"],
                formatters={
                    "sign": _tex_escape,
                    "ordinance_doc_id": lambda x: _tex_escape(_clip(x, 38)),
                    "jurisdiction_region": _tex_escape,
                    "residual": lambda x: _fmt_float(x, 3),
                },
            )
            + _rows_from_df(
                negative_residuals.assign(sign="-"),
                columns=["sign", "ordinance_doc_id", "jurisdiction_region", "residual"],
                formatters={
                    "sign": _tex_escape,
                    "ordinance_doc_id": lambda x: _tex_escape(_clip(x, 38)),
                    "jurisdiction_region": _tex_escape,
                    "residual": lambda x: _fmt_float(x, 3),
                },
            )
        ),
    )

    legal_summary_table = _table_block(
        caption="Legal-complexity backbone summary from the dedicated complexity run.",
        label="tab:legal-summary",
        colspec=r">{\raggedright\arraybackslash}Xr",
        header=["Metric", "Value"],
        rows=[
            ["Documents in complexity backbone", _fmt_int(complexity_summary["doc_count"])],
            ["Complexity families", _fmt_int(complexity_summary["complexity_family_count"])],
            ["Dependency profiles", _fmt_int(complexity_summary["dependency_profile_count"])],
            ["Mean section count", _fmt_float(complexity_summary["mean_section_count"], 2)],
            ["Mean dependency mentions per 1,000 words", _fmt_float(complexity_summary["mean_dependency_mentions_per_1000_words"], 3)],
            ["Mean reference mentions per 1,000 words", _fmt_float(complexity_summary["mean_reference_mentions_per_1000_words"], 3)],
            ["Mean grade proxy", _fmt_float(complexity_summary["mean_flesch_kincaid_grade_proxy"], 3)],
            ["Highest-complexity region", _tex_escape(complexity_summary["top_region_by_overall_complexity"])],
        ],
    )

    legal_region_table = _table_block(
        caption="Legal-complexity means by region. Each component is standardized, so positive values mean above-average complexity on that dimension and negative values mean below-average complexity. `Overall` is the combined index used for the main legal-complexity comparisons.",
        label="tab:legal-region",
        colspec="lrrrr",
        header=["Region", "Structural", "Dependency", "Linguistic", "Overall"],
        rows=_rows_from_df(
            legal_region.sort_values("overall_legal_complexity_index__vintage_adjusted", ascending=False),
            columns=[
                "jurisdiction_region",
                "structural_complexity_index__vintage_adjusted",
                "dependency_complexity_index__vintage_adjusted",
                "linguistic_complexity_index__vintage_adjusted",
                "overall_legal_complexity_index__vintage_adjusted",
            ],
            formatters={
                "jurisdiction_region": _tex_escape,
                "structural_complexity_index__vintage_adjusted": lambda x: _fmt_float(x, 3),
                "dependency_complexity_index__vintage_adjusted": lambda x: _fmt_float(x, 3),
                "linguistic_complexity_index__vintage_adjusted": lambda x: _fmt_float(x, 3),
                "overall_legal_complexity_index__vintage_adjusted": lambda x: _fmt_float(x, 3),
            },
        ),
    )

    bridge_coherence_table = _table_block(
        caption="Bridge from the legal-complexity backbone into semantic coherence. Sample note: this bridge uses the clause/semantic subset where both legal-complexity and semantic outcomes are observed.",
        label="tab:bridge-coherence",
        colspec=r">{\raggedright\arraybackslash}Xrrr",
        header=["Term", "Coef.", "p", "$R^2$"],
        rows=_rows_from_df(
            legal_bridge_coherence,
            columns=["term", "coefficient", "pvalue", "r_squared"],
            formatters={
                "term": lambda x: _tex_escape(_clip(_bridge_term_label(x), 52)),
                "coefficient": lambda x: _fmt_float(x, 3),
                "pvalue": lambda x: _fmt_float(x, 4),
                "r_squared": lambda x: _fmt_float(x, 3),
            },
        ),
    )

    bridge_governance_table = _table_block(
        caption="Bridge from legal complexity into governance-style outcomes. Sample note: this bridge uses the clause/semantic subset where both legal-complexity and governance outcomes are observed.",
        label="tab:bridge-governance",
        colspec=r"llrrr",
        header=["Model", "Term", "Coef.", "p", "$R^2$"],
        rows=_rows_from_df(
            legal_bridge_governance,
            columns=["model_label", "term", "coefficient", "pvalue", "r_squared"],
            formatters={
                "model_label": lambda x: _tex_escape(_bridge_model_label(x)),
                "term": lambda x: _tex_escape(_clip(_bridge_term_label(x), 42)),
                "coefficient": lambda x: _fmt_float(x, 3),
                "pvalue": lambda x: _fmt_float(x, 4),
                "r_squared": lambda x: _fmt_float(x, 3),
            },
        ),
    )

    bridge_subcomponent_table = _table_block(
        caption="Most informative subcomponent-to-semantic bridge terms from the legal-complexity package. Read this as a diagnostic table showing which specific legal features, such as section-kind entropy or references per 1,000 words, line up most strongly with coherence and governance outcomes. Sample note: this bridge uses the clause/semantic subset where both legal-complexity and semantic outcomes are observed.",
        label="tab:bridge-subcomponents",
        colspec=r"llrrr",
        header=["Model", "Term", "Coef.", "p", "$R^2$"],
        rows=_rows_from_df(
            legal_bridge_subcomponents,
            columns=["model_label", "term", "coefficient", "pvalue", "r_squared"],
            formatters={
                "model_label": lambda x: _tex_escape(_bridge_model_label(x)),
                "term": lambda x: _tex_escape(_clip(_bridge_term_label(x), 42)),
                "coefficient": lambda x: _fmt_float(x, 3),
                "pvalue": lambda x: _fmt_float(x, 4),
                "r_squared": lambda x: _fmt_float(x, 3),
            },
        ),
    )

    pilot_summary_table = _table_block(
        caption="Clause/governance layer summary. Sample note: this block uses the clause-covered subset rather than the national corpus, because enriched clause semantics are not yet available everywhere.",
        label="tab:pilot-summary",
        colspec=r">{\raggedright\arraybackslash}Xr",
        header=["Metric", "Value"],
        rows=[
            ["Documents in clause/governance subset", _fmt_int(pilot_summary["doc_count"])],
            ["Enriched-clause documents", _fmt_int(pilot_summary["enriched_clause_doc_count"])],
            ["Coverage rate", _fmt_float(pilot_summary["enriched_clause_doc_coverage_rate"], 3)],
            ["Enriched clauses", _fmt_int(pilot_summary["enriched_clause_row_count"])],
            ["Instrument types", _fmt_int(pilot_summary["instrument_type_count"])],
            ["Clause clusters", _fmt_int(pilot_summary["clause_cluster_count"])],
            ["Governance-style families", _fmt_int(pilot_summary["regulatory_style_family_count"])],
            ["Raw policy-signature clusters", _fmt_int(pilot_summary["policy_signature_cluster_count"])],
            ["Section-normalized policy-signature clusters", _fmt_int(pilot_summary["policy_signature_section_normalized_clause_share_cluster_count"])],
        ],
    )

    style_table = _table_block(
        caption="Pilot governance styles in the pilot clause subset. Each row is a recurring ordinance-level clause mix.",
        label="tab:pilot-style",
        colspec=r">{\raggedright\arraybackslash}Xlrrrr",
        header=["Style name", "Most common headline purpose", "Docs", "Direct-rule share", "Discretion share", "Exception/condition share"],
        rows=_rows_from_df(
            pilot_style.sort_values("ordinance_count", ascending=False).head(5),
            columns=[
                "cluster_label",
                "dominant_doc_primary_motive",
                "ordinance_count",
                "mean_direct_rule_share",
                "mean_discretion_share",
                "mean_exception_or_condition_share",
            ],
            formatters={
                "cluster_label": lambda x: _tex_escape(_clip(_style_family_short_label(x), 40)),
                "dominant_doc_primary_motive": lambda x: _tex_escape(_purpose_short_label(x)),
                "ordinance_count": _fmt_int,
                "mean_direct_rule_share": lambda x: _fmt_float(x, 3),
                "mean_discretion_share": lambda x: _fmt_float(x, 3),
                "mean_exception_or_condition_share": lambda x: _fmt_float(x, 3),
            },
        ),
    )

    clause_family_table = _table_block(
        caption="Largest recurring clause types in the pilot clause subset.",
        label="tab:pilot-clause-families",
        colspec=r"r>{\raggedright\arraybackslash}Xrr",
        header=["ID", "Clause family", "Clauses", "Ordinances"],
        rows=_rows_from_df(
            pilot_largest_clause,
            columns=["cluster_id", "cluster_label", "clause_count", "ordinance_count"],
            formatters={
                "cluster_id": _fmt_int,
                "cluster_label": lambda x: _tex_escape(_clip(x, 58)),
                "clause_count": _fmt_int,
                "ordinance_count": _fmt_int,
            },
        ),
    )

    policy_compare_table = _table_block(
        caption="Three ordinance-level clause aggregation strategies for document typology. `Reportable` counts clusters large enough to interpret as real document families. The comparison matters because raw atomic clause counts can be dominated by long use lists or definition sections, while section-normalized surfaces trade geometric tightness for more balanced document-level composition.",
        label="tab:policy-aggregation",
        colspec=r">{\raggedright\arraybackslash}Xrrrr",
        header=["Aggregation surface", "Items", "Clusters", "Reportable", "Silhouette"],
        rows=[
            ["Raw atomic clause shares", _fmt_int(policy_raw.get("item_count", 0)), _fmt_int(policy_raw.get("cluster_count", 0)), _fmt_int(policy_raw.get("reportable_cluster_count", 0)), _fmt_float(policy_raw.get("silhouette_score", 0.0), 3)],
            ["Section-normalized clause shares", _fmt_int(policy_norm.get("item_count", 0)), _fmt_int(policy_norm.get("cluster_count", 0)), _fmt_int(policy_norm.get("reportable_cluster_count", 0)), _fmt_float(policy_norm.get("silhouette_score", 0.0), 3)],
            ["Section presence only", _fmt_int(policy_presence.get("item_count", 0)), _fmt_int(policy_presence.get("cluster_count", 0)), _fmt_int(policy_presence.get("reportable_cluster_count", 0)), _fmt_float(policy_presence.get("silhouette_score", 0.0), 3)],
        ],
    )

    tex = rf"""
\documentclass[11pt]{{article}}
\usepackage[margin=1in]{{geometry}}
\usepackage{{booktabs}}
\usepackage{{array}}
\usepackage{{tabularx}}
\usepackage{{float}}
\usepackage{{graphicx}}
\usepackage{{hyperref}}
\usepackage{{setspace}}
\usepackage{{caption}}
\usepackage{{pdflscape}}
\usepackage{{needspace}}
\captionsetup[table]{{position=bottom}}
\captionsetup[figure]{{position=bottom}}
\setlength{{\parskip}}{{0.6em}}
\setlength{{\parindent}}{{0pt}}
\begin{{document}}

\begin{{center}}
{{\LARGE Update: {update_date}}}
\end{{center}}

\section*{{Purpose Language in Ordinances}}
{purpose_intro}

{purpose_label_table}

{purpose_weights_intro}

{_figure_block(str((frontier_reports / 'figures' / 'overall_purpose_presence__full.png').resolve()), 'How often each purpose appears across the full sample. Each bar is the share of ordinances whose returned purpose bundle includes that purpose anywhere.') }

{purpose_cooccurrence_intro}

{_figure_block(str((frontier_reports / 'figures' / 'purpose_cooccurrence_heatmap__full.png').resolve()), 'Purpose co-occurrence across the full sample. The cells are built from the returned full-document purpose bundles. The diagonal shows the share of ordinances containing that purpose anywhere in the bundle, and the off-diagonal cells show the share containing both purposes.') }

{purpose_examples_intro}

{purpose_examples_table}

\section*{{Clause Layer}}
{clause_intro}

{clause_family_intro}

{clause_family_table}

{clause_examples_intro}

{clause_family_examples_table}

{_figure_block(str((frontier_reports / 'figures' / 'clause_instrument_hdbscan_umap.png').resolve()), 'Clause-level view of the pilot subset. Each point is one enriched clause. Nearby points usually use similar language and do similar regulatory work. Labels mark the largest recurring clause types.')}

{clause_zoom_intro}

{_figure_block(str((frontier_reports / 'figures' / 'clause_instrument_hdbscan_umap_zoom.png').resolve()), 'Zoom into the crowded middle of the same clause manifold. This view surfaces additional recurring clause families that are compressed in the full plot.')}

{referenced_page_appendix_block}

\end{{document}}
""".strip() + "\n"

    output_path.write_text(tex, encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a tables-and-figures-focused LaTeX atlas across the frontier and legal-complexity research surfaces.")
    parser.add_argument(
        "--frontier-run-root",
        default="artifacts/runs/frontier_llm_v2/pilot64_20260310_a",
        help="Run root for the frontier_llm_v2 research surface.",
    )
    parser.add_argument(
        "--complexity-run-root",
        default="artifacts/runs/legal_complexity_fullsample_v1/full728_20260311_a",
        help="Run root for the legal-complexity full-sample package.",
    )
    parser.add_argument(
        "--output-path",
        default=None,
        help="Optional explicit output path for the generated .tex file.",
    )
    args = parser.parse_args()

    output = build_results_atlas(
        frontier_run_root=Path(args.frontier_run_root),
        complexity_run_root=Path(args.complexity_run_root),
        output_path=Path(args.output_path) if args.output_path else None,
    )
    print(output)


if __name__ == "__main__":
    main()
