from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from newsvlm_analysis.zoning_mechanics import extract_mechanics

from .io_utils import clean_optional_str, safe_float, write_json, write_parquet


REGULATORY_MOTIVES = (
    "externality_control",
    "exclusion",
    "value_capture",
    "growth_management",
    "fiscal_zoning",
    "public_health_safety",
    "aesthetic_design_control",
    "infrastructure_coordination",
    "environmental_protection",
    "other",
    "unclear",
)

MOTIVE_PATTERNS: dict[str, list[str]] = {
    "externality_control": ["nuisance", "noise", "smoke", "odor", "hazard", "congestion", "incompatible uses"],
    "exclusion": ["single-family", "residential character", "noxious", "separate uses", "exclusive"],
    "value_capture": ["property values", "value of buildings", "value of land", "tax base"],
    "growth_management": ["orderly growth", "development of land", "comprehensive plan", "population", "density"],
    "fiscal_zoning": ["tax", "assessment", "fee", "fees", "rent", "revenue"],
    "public_health_safety": ["health", "safety", "welfare", "fire", "light and air", "danger"],
    "aesthetic_design_control": ["appearance", "aesthetic", "harmony", "landscaping", "sign", "design"],
    "infrastructure_coordination": ["street", "sewer", "water", "utility", "utilities", "schools", "transportation", "public services"],
    "environmental_protection": ["floodplain", "erosion", "wetland", "natural resource", "drainage", "environment"],
}

ACTOR_PATTERNS = [
    ("planning_commission", re.compile(r"\bplanning commission\b", re.IGNORECASE)),
    ("city_council", re.compile(r"\b(city council|board of trustees|board of supervisors)\b", re.IGNORECASE)),
    ("zoning_board", re.compile(r"\b(board of appeals|board of adjustment|zoning board)\b", re.IGNORECASE)),
    ("zoning_administrator", re.compile(r"\b(zoning administrator|building inspector|administrative official)\b", re.IGNORECASE)),
    ("property_owner", re.compile(r"\b(owner|developer|applicant|occupant)\b", re.IGNORECASE)),
]

TARGET_PATTERNS = [
    ("lot", re.compile(r"\blot\b", re.IGNORECASE)),
    ("building", re.compile(r"\b(building|structure)\b", re.IGNORECASE)),
    ("use", re.compile(r"\buse\b", re.IGNORECASE)),
    ("district", re.compile(r"\bdistrict\b", re.IGNORECASE)),
    ("sign", re.compile(r"\bsign\b", re.IGNORECASE)),
    ("parking", re.compile(r"\bparking\b", re.IGNORECASE)),
    ("subdivision", re.compile(r"\b(subdivision|plat)\b", re.IGNORECASE)),
]

CONDITION_RE = re.compile(r"\b(if|unless|provided that|when)\b(.+?)(?:[.;]|$)", re.IGNORECASE)
EXCEPTION_RE = re.compile(r"\b(except|provided however|but)\b(.+?)(?:[.;]|$)", re.IGNORECASE)
PURPOSE_RE = re.compile(r"\b(purpose|intent|policy)\b", re.IGNORECASE)
PREAMBLE_RE = re.compile(r"^\s*(whereas|be it ordained|it is the purpose|it is the intent)\b", re.IGNORECASE)
MAP_BOUNDARY_RE = re.compile(
    r"\b(beginning at|thence|boundary line|boundary lines|center line|official zoning map|shown upon the map|metes and bounds)\b",
    re.IGNORECASE,
)
BOILERPLATE_RE = re.compile(
    r"\b(short title|effective date|severability|repeal(?:er)?|publication|this ordinance shall take effect|ordinance number)\b",
    re.IGNORECASE,
)
NUMERIC_PATTERNS = {
    "minimum_lot_area": re.compile(r"\b(?:minimum\s+)?lot\s+(?:area|size).{0,30}?(\d[\d,]*(?:\.\d+)?)\s*(sq\.?\s*ft\.?|square feet|acres?)", re.IGNORECASE),
    "minimum_lot_width": re.compile(r"\blot\s+width.{0,24}?(\d[\d,]*(?:\.\d+)?)\s*(feet|ft\.?)", re.IGNORECASE),
    "front_setback": re.compile(r"\bfront\s+yard.{0,24}?(\d[\d,]*(?:\.\d+)?)\s*(feet|ft\.?)", re.IGNORECASE),
    "side_setback": re.compile(r"\bside\s+yard.{0,24}?(\d[\d,]*(?:\.\d+)?)\s*(feet|ft\.?)", re.IGNORECASE),
    "rear_setback": re.compile(r"\brear\s+yard.{0,24}?(\d[\d,]*(?:\.\d+)?)\s*(feet|ft\.?)", re.IGNORECASE),
    "height_limit": re.compile(r"\bheight.{0,24}?(\d[\d,]*(?:\.\d+)?)\s*(feet|ft\.?|stories)", re.IGNORECASE),
    "parking_spaces": re.compile(r"\bparking.{0,24}?(\d[\d,]*(?:\.\d+)?)\s*(spaces?|stalls?)", re.IGNORECASE),
    "penalty_amount": re.compile(r"\b(?:fine|penalty).{0,18}?\$?(\d[\d,]*(?:\.\d+)?)", re.IGNORECASE),
}

OPERATIVE_ADMIN_INSTRUMENTS = {
    "permit_administration",
    "appeal",
    "variance",
    "conditional_use",
    "special_exception",
    "plat_approval",
    "subdivision_standard",
    "zoning_map",
}

NON_OPERATIVE_INSTRUMENTS = {
    "purpose",
    "definition",
}


def _motive_scores(text: str) -> Counter[str]:
    sample = text.lower()
    scores: Counter[str] = Counter()
    for motive, needles in MOTIVE_PATTERNS.items():
        for needle in needles:
            if needle in sample:
                scores[motive] += 1
    if not scores and PURPOSE_RE.search(sample):
        scores["other"] += 1
    if not scores:
        scores["unclear"] += 1
    return scores


def _top_motives(text: str) -> tuple[str, list[str], float]:
    scores = _motive_scores(text)
    ordered = [m for m, _ in scores.most_common()]
    primary = ordered[0]
    secondary = ordered[1:3]
    confidence = min(0.95, 0.35 + 0.15 * scores[primary])
    return primary, secondary, confidence


def _extract_purposes(doc_id: str, text: str) -> list[dict[str, Any]]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    purpose_lines = [line for line in lines if PURPOSE_RE.search(line) or "general welfare" in line.lower()]
    if not purpose_lines:
        purpose_lines = lines[:5]
    purposes: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line in purpose_lines[:8]:
        primary, _, confidence = _top_motives(line)
        key = f"{primary}|{line[:80]}"
        if key in seen:
            continue
        seen.add(key)
        purposes.append(
            {
                "doc_id": doc_id,
                "motive": primary,
                "description": line[:600],
                "evidence": line[:300],
                "confidence": float(confidence),
            }
        )
    return purposes


def _extract_actor(text: str) -> str:
    for label, pattern in ACTOR_PATTERNS:
        if pattern.search(text):
            return label
    return "unknown"


def _extract_target(text: str) -> str:
    for label, pattern in TARGET_PATTERNS:
        if pattern.search(text):
            return label
    return "general"


def _extract_condition(text: str) -> str:
    m = CONDITION_RE.search(text)
    return clean_optional_str(m.group(2))[:240] if m else ""


def _extract_exception(text: str) -> str:
    m = EXCEPTION_RE.search(text)
    return clean_optional_str(m.group(2))[:240] if m else ""


def _instrument_type(text: str, hint: str) -> str:
    sample = f"{hint} {text}".lower()
    mapping = [
        ("setback", "setback_standard"),
        ("lot area", "lot_area_standard"),
        ("lot width", "lot_width_standard"),
        ("height", "height_standard"),
        ("parking", "parking_standard"),
        ("conditional use", "conditional_use"),
        ("special exception", "special_exception"),
        ("variance", "variance"),
        ("appeal", "appeal"),
        ("district", "district_framework"),
        ("map", "zoning_map"),
        ("permit", "permit_administration"),
        ("subdivision", "subdivision_standard"),
        ("plat", "plat_approval"),
        ("penalty", "penalty_enforcement"),
        ("nonconforming", "nonconforming_rule"),
        ("definition", "definition"),
        ("purpose", "purpose"),
    ]
    for needle, label in mapping:
        if needle in sample:
            return label
    return "general_rule"


def _classify_clause_role(text: str, section_kind: str, modality: str, instrument_type: str) -> str:
    sample = text.lower()
    if section_kind == "definitions" or modality == "definition" or instrument_type == "definition":
        return "definition"
    if section_kind == "purpose" or PREAMBLE_RE.search(text) or "general welfare" in sample:
        return "purpose_preamble"
    if section_kind == "district_map" or MAP_BOUNDARY_RE.search(text):
        return "map_boundary"
    if BOILERPLATE_RE.search(text) or section_kind == "closing":
        return "closing_boilerplate"
    if instrument_type in OPERATIVE_ADMIN_INSTRUMENTS:
        return "administrative_procedure"
    return "substantive_standard"


def _operative_clause_flag(*, clause_role: str, modality: str, instrument_type: str) -> int:
    if clause_role in {"definition", "purpose_preamble", "map_boundary", "closing_boilerplate"}:
        return 0
    if instrument_type in NON_OPERATIVE_INSTRUMENTS and modality not in {"must", "must_not", "may"}:
        return 0
    return 1


def build_semantics_outputs(*, run_root: Path) -> dict[str, Any]:
    corpus_dir = run_root / "corpus"
    structure_dir = run_root / "structure"
    semantics_dir = run_root / "semantics"

    ordinances = pd.read_parquet(corpus_dir / "ordinances.parquet")
    if "jurisdiction_resolution_status" in ordinances.columns:
        ordinances = ordinances[ordinances["jurisdiction_resolution_status"].astype(str) == "resolved_with_census"].copy()
    ordinances = ordinances[ordinances["clean_text"].astype(str).str.strip().astype(bool)].copy()
    sections = pd.read_parquet(structure_dir / "sections.parquet")
    clauses = pd.read_parquet(structure_dir / "clauses.parquet")

    doc_rows: list[dict[str, Any]] = []
    purpose_rows: list[dict[str, Any]] = []
    section_rows: list[dict[str, Any]] = []
    clause_rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []

    for row in ordinances.itertuples(index=False):
        ordinance_doc_id = clean_optional_str(getattr(row, "ordinance_doc_id", ""))
        text = clean_optional_str(getattr(row, "clean_text", ""))
        primary, secondary, confidence = _top_motives(text)
        mechanics = extract_mechanics(text)
        doc_rows.append(
            {
                "ordinance_doc_id": ordinance_doc_id,
                "issue_id": clean_optional_str(getattr(row, "issue_id", "")),
                "jurisdiction_key": clean_optional_str(getattr(row, "jurisdiction_key", "")),
                "publication_key": clean_optional_str(getattr(row, "publication_key", "")),
                "primary_motive": primary,
                "secondary_motives": "|".join(secondary),
                "motive_confidence": float(confidence),
                "document_purpose_present": int(bool(PURPOSE_RE.search(text))),
                "action_tags": "|".join(mechanics.action_tags),
                "dimension_tags": "|".join(mechanics.dimension_tags),
                "instrument_tags": "|".join(mechanics.instrument_tags),
                "district_tokens": "|".join(mechanics.district_tokens),
            }
        )
        purpose_rows.extend(_extract_purposes(ordinance_doc_id, text))

    for row in sections.itertuples(index=False):
        text = clean_optional_str(getattr(row, "text", ""))
        primary, secondary, confidence = _top_motives(text)
        mechanics = extract_mechanics(text)
        section_rows.append(
            {
                "section_id": clean_optional_str(getattr(row, "section_id", "")),
                "ordinance_doc_id": clean_optional_str(getattr(row, "ordinance_doc_id", "")),
                "section_kind": clean_optional_str(getattr(row, "section_kind", "")),
                "heading": clean_optional_str(getattr(row, "heading", "")),
                "primary_motive": primary,
                "secondary_motives": "|".join(secondary),
                "motive_confidence": float(confidence),
                "section_purpose_flag": int(bool(PURPOSE_RE.search(text)) or clean_optional_str(getattr(row, "section_kind", "")) == "purpose"),
                "action_tags": "|".join(mechanics.action_tags),
                "dimension_tags": "|".join(mechanics.dimension_tags),
                "instrument_tags": "|".join(mechanics.instrument_tags),
            }
        )

    for row in clauses.itertuples(index=False):
        text = clean_optional_str(getattr(row, "clause_text", ""))
        primary, secondary, confidence = _top_motives(text)
        modality = clean_optional_str(getattr(row, "modality", ""))
        instrument_hint = clean_optional_str(getattr(row, "instrument_hint", ""))
        section_kind = clean_optional_str(getattr(row, "section_kind", ""))
        instrument_type = _instrument_type(text, instrument_hint)
        clause_role = _classify_clause_role(text, section_kind, modality, instrument_type)
        operative_clause_flag = _operative_clause_flag(
            clause_role=clause_role,
            modality=modality,
            instrument_type=instrument_type,
        )
        clause_id = clean_optional_str(getattr(row, "clause_id", ""))
        section_id = clean_optional_str(getattr(row, "section_id", ""))
        ordinance_doc_id = clean_optional_str(getattr(row, "ordinance_doc_id", ""))

        clause_rows.append(
            {
                "clause_id": clause_id,
                "section_id": section_id,
                "ordinance_doc_id": ordinance_doc_id,
                "primary_motive": primary,
                "secondary_motives": "|".join(secondary),
                "motive_confidence": float(confidence),
                "modality": modality,
                "actor": _extract_actor(text),
                "target": _extract_target(text),
                "condition_text": _extract_condition(text),
                "exception_text": _extract_exception(text),
                "instrument_type": instrument_type,
                "instrument_hint": instrument_hint,
                "clause_role": clause_role,
                "operative_clause_flag": operative_clause_flag,
                "clause_text": text,
            }
        )

        for feature_name, pattern in NUMERIC_PATTERNS.items():
            for match in pattern.finditer(text):
                feature_rows.append(
                    {
                        "clause_id": clause_id,
                        "section_id": section_id,
                        "ordinance_doc_id": ordinance_doc_id,
                        "feature_name": feature_name,
                        "numeric_value": safe_float(match.group(1)),
                        "unit": clean_optional_str(match.group(2)) if match.lastindex and match.lastindex >= 2 else "",
                        "clause_role": clause_role,
                        "operative_clause_flag": operative_clause_flag,
                        "evidence_text": text[:300],
                    }
                )

        mechanics = extract_mechanics(text)
        for token in mechanics.district_tokens:
            feature_rows.append(
                {
                    "clause_id": clause_id,
                    "section_id": section_id,
                    "ordinance_doc_id": ordinance_doc_id,
                    "feature_name": "district_token",
                    "numeric_value": None,
                    "unit": "",
                    "clause_role": clause_role,
                    "operative_clause_flag": operative_clause_flag,
                    "evidence_text": token,
                }
            )

    doc_df = pd.DataFrame.from_records(doc_rows).sort_values("ordinance_doc_id").reset_index(drop=True)
    purposes_df = pd.DataFrame.from_records(purpose_rows).sort_values(["doc_id", "motive"]).reset_index(drop=True)
    section_df = pd.DataFrame.from_records(section_rows).sort_values("section_id").reset_index(drop=True)
    clause_df = pd.DataFrame.from_records(clause_rows).sort_values("clause_id").reset_index(drop=True)
    features_df = pd.DataFrame.from_records(feature_rows).reset_index(drop=True)
    operative_clause_df = clause_df[clause_df["operative_clause_flag"].astype(int) == 1].reset_index(drop=True) if not clause_df.empty else clause_df.copy()
    operative_features_df = (
        features_df[features_df["operative_clause_flag"].astype(int) == 1].reset_index(drop=True) if not features_df.empty else features_df.copy()
    )

    if not clause_df.empty:
        counts = clause_df.groupby("ordinance_doc_id").agg(
            clause_count=("clause_id", "count"),
            dominant_instrument=("instrument_type", lambda s: s.value_counts().index[0]),
            must_clause_count=("modality", lambda s: int((s == "must").sum())),
            must_not_clause_count=("modality", lambda s: int((s == "must_not").sum())),
            definition_clause_count=("modality", lambda s: int((s == "definition").sum())),
        )
        doc_df = doc_df.merge(counts, on="ordinance_doc_id", how="left")
        operative_counts = operative_clause_df.groupby("ordinance_doc_id").agg(
            operative_clause_count=("clause_id", "count"),
            operative_dominant_instrument=("instrument_type", lambda s: s.value_counts().index[0]),
            operative_must_clause_count=("modality", lambda s: int((s == "must").sum())),
            operative_must_not_clause_count=("modality", lambda s: int((s == "must_not").sum())),
        )
        doc_df = doc_df.merge(operative_counts, on="ordinance_doc_id", how="left")
    else:
        for col in ("clause_count", "dominant_instrument", "must_clause_count", "must_not_clause_count", "definition_clause_count"):
            doc_df[col] = 0 if col.endswith("_count") else ""
        for col in ("operative_clause_count", "operative_dominant_instrument", "operative_must_clause_count", "operative_must_not_clause_count"):
            doc_df[col] = 0 if col.endswith("_count") else ""

    for count_col in (
        "clause_count",
        "must_clause_count",
        "must_not_clause_count",
        "definition_clause_count",
        "operative_clause_count",
        "operative_must_clause_count",
        "operative_must_not_clause_count",
    ):
        if count_col in doc_df.columns:
            doc_df[count_col] = doc_df[count_col].fillna(0).astype(int)
    for text_col in ("dominant_instrument", "operative_dominant_instrument"):
        if text_col in doc_df.columns:
            doc_df[text_col] = doc_df[text_col].fillna("").astype(str)

    write_parquet(doc_df, semantics_dir / "document_semantics.parquet")
    write_parquet(purposes_df, semantics_dir / "document_purposes.parquet")
    write_parquet(section_df, semantics_dir / "section_semantics.parquet")
    write_parquet(clause_df, semantics_dir / "clause_semantics.parquet")
    write_parquet(operative_clause_df, semantics_dir / "operative_clause_semantics.parquet")
    write_parquet(features_df, semantics_dir / "hard_feature_observations.parquet")
    write_parquet(operative_features_df, semantics_dir / "operative_hard_feature_observations.parquet")
    write_json(
        semantics_dir / "semantics_summary.json",
        {
            "document_rows": int(len(doc_df)),
            "document_purpose_rows": int(len(purposes_df)),
            "section_rows": int(len(section_df)),
            "clause_rows": int(len(clause_df)),
            "operative_clause_rows": int(len(operative_clause_df)),
            "feature_rows": int(len(features_df)),
            "operative_feature_rows": int(len(operative_features_df)),
            "clause_role_counts": clause_df["clause_role"].value_counts(dropna=False).to_dict() if not clause_df.empty else {},
            "top_primary_motives": doc_df["primary_motive"].value_counts(dropna=False).head(10).to_dict() if not doc_df.empty else {},
        },
    )
    return {
        "document_rows": int(len(doc_df)),
        "section_rows": int(len(section_df)),
        "clause_rows": int(len(clause_df)),
        "operative_clause_rows": int(len(operative_clause_df)),
        "feature_rows": int(len(features_df)),
        "operative_feature_rows": int(len(operative_features_df)),
    }
