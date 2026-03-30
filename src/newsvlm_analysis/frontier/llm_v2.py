from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import re
import shutil
import subprocess
import textwrap
import time
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import hdbscan
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, field_validator

from .clusters import _assign_exemplars
from .contracts import FrontierExperiment
from .embeddings import _dense_embedding_matrix, _safe_pca, _safe_umap
from .io_utils import clean_optional_str, iter_jsonl, normalize_for_fingerprint, slugify, write_json, write_jsonl, write_parquet
from .llm_v2_research import build_llm_v2_research_outputs
from .policy_signature import build_policy_signature_outputs
from .report import build_report_bundle
from .story import build_story_outputs


csv.field_size_limit(2**31 - 1)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_LLM_V2_RUNS_BASE = REPO_ROOT / "artifacts" / "runs" / "frontier_llm_v2"
DEFAULT_BASELINE_RUN_ROOT = REPO_ROOT / "artifacts" / "runs" / "frontier_fullcode_lab_v1" / "full730_20260309_censuslinked_b"
DEFAULT_PROMPT_DIR = REPO_ROOT / "prompts" / "frontier" / "llm_v2"
LLM_V2_RUN_SUBDIRS = (
    "corpus",
    "geo",
    "structure",
    "requests",
    "results",
    "semantics",
    "embeddings",
    "clusters",
    "review",
    "story",
    "reports",
    "manifests",
)
SECTION_CLUSTER_EXPERIMENT = "section_semantic_hdbscan"
SECTION_LOCAL_CLUSTER_EXPERIMENT = "section_semantic_hdbscan_local_baseline"
CLAUSE_CLUSTER_EXPERIMENT = "clause_instrument_hdbscan"
REVIEW_SAMPLE_SIZES = {
    "section_segmentation": 20,
    "section_motive": 80,
    "clause": 120,
    "section_embedding_comparison": 30,
    "policy_signature": 20,
}
MOTIVES = (
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
INSTRUMENT_TYPES = (
    "setback_standard",
    "lot_area_standard",
    "lot_width_standard",
    "height_standard",
    "parking_standard",
    "conditional_use",
    "special_exception",
    "variance",
    "appeal",
    "district_framework",
    "zoning_map",
    "permit_administration",
    "subdivision_standard",
    "plat_approval",
    "penalty_enforcement",
    "nonconforming_rule",
    "use_permission",
    "density_standard",
    "signage_design_standard",
    "environmental_protection",
    "definition",
    "purpose",
    "general_rule",
    "other",
)
RESPONSE_STAGE_PROMPTS = {
    "section_segmentation": "section_segmentation_prompt.txt",
    "document_purpose": "document_purpose_prompt.txt",
    "section_motive": "section_motive_prompt.txt",
    "clause_extraction": "clause_extraction_prompt.txt",
    "clause_enrichment": "clause_enrichment_prompt.txt",
    "section_cluster_label": "section_cluster_label_prompt.txt",
    "section_local_cluster_label": "section_cluster_label_prompt.txt",
    "clause_cluster_label": "clause_cluster_label_prompt.txt",
}
RESPONSE_STAGE_MODELS = {
    "section_segmentation": "gpt-5-mini",
    "document_purpose": "gpt-5-mini",
    "section_motive": "gpt-5-mini",
    "clause_extraction": "gpt-5-mini",
    "clause_enrichment": "gpt-5-mini",
    "section_cluster_label": "gpt-5-mini",
    "section_local_cluster_label": "gpt-5-mini",
    "clause_cluster_label": "gpt-5-mini",
}
SECTION_EMBEDDING_MODEL = "text-embedding-3-large"
SECTION_EMBEDDING_MAX_CHARS = 12_000
SECTION_CLUSTER_MIN_SIZE = 12
CLAUSE_CLUSTER_MIN_SIZE = 20
REPORTABLE_CLUSTER_MIN_SIZE = 10


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _split_text_for_embedding(text: str, *, max_chars: int = SECTION_EMBEDDING_MAX_CHARS) -> list[str]:
    text = clean_optional_str(text).strip()
    if not text:
        return [""]

    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    if not paragraphs:
        paragraphs = [text]

    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        candidate = paragraph if not current else f"{current}\n\n{paragraph}"
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            chunks.append(current)
            current = ""
        remaining = paragraph
        while len(remaining) > max_chars:
            split_at = remaining.rfind(" ", 0, max_chars)
            if split_at < max_chars // 2:
                split_at = max_chars
            piece = remaining[:split_at].strip()
            if piece:
                chunks.append(piece)
            remaining = remaining[split_at:].strip()
        if remaining:
            current = remaining
    if current:
        chunks.append(current)
    return chunks or [text]


def _load_prompt(prompt_dir: Path, prompt_name: str) -> tuple[str, str]:
    path = prompt_dir / prompt_name
    if not path.is_file():
        raise SystemExit(f"Prompt file not found: {path}")
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise SystemExit(f"Prompt file is empty: {path}")
    return text, _sha256_text(text)


def _coerce_json(text: str) -> dict[str, Any]:
    stripped = clean_optional_str(text)
    if not stripped:
        raise ValueError("empty response text")
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError as exc:
        import re

        cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", stripped)
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError as exc2:
            if "Invalid control character" in str(exc):
                parsed = json.loads(cleaned.replace("\n", "\\n"))
            elif "Invalid \\escape" in str(exc2):
                def _escape_invalid_backslashes(s: str) -> str:
                    out: list[str] = []
                    hexdigits = set("0123456789abcdefABCDEF")
                    i = 0
                    n = len(s)
                    while i < n:
                        ch = s[i]
                        if ch != "\\":
                            out.append(ch)
                            i += 1
                            continue
                        if i + 1 >= n:
                            out.append("\\\\")
                            i += 1
                            continue
                        nxt = s[i + 1]
                        if nxt in {'"', "\\", "/", "b", "f", "n", "r", "t"}:
                            out.append("\\")
                            i += 1
                            continue
                        if nxt == "u":
                            if i + 5 < n and all(c in hexdigits for c in s[i + 2 : i + 6]):
                                out.append("\\")
                                i += 1
                                continue
                            out.append("\\\\")
                            i += 1
                            continue
                        out.append("\\\\")
                        i += 1
                    return "".join(out)

                parsed = json.loads(_escape_invalid_backslashes(cleaned))
            else:
                raise
    if not isinstance(parsed, dict):
        raise ValueError("response did not parse to an object")
    return parsed


def _extract_output_text(body: dict[str, Any]) -> str:
    out = body.get("output")
    if not isinstance(out, list):
        return ""
    parts: list[str] = []
    for item in out:
        if not isinstance(item, dict):
            continue
        for content in item.get("content") or []:
            if isinstance(content, dict) and content.get("type") == "output_text":
                parts.append(str(content.get("text") or ""))
    return "".join(parts)


def _extract_embedding(body: dict[str, Any]) -> list[float] | None:
    data = body.get("data")
    if not isinstance(data, list) or not data:
        return None
    first = data[0]
    if not isinstance(first, dict):
        return None
    emb = first.get("embedding")
    if not isinstance(emb, list) or not emb:
        return None
    if not all(isinstance(x, (int, float)) for x in emb):
        return None
    return [float(x) for x in emb]


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return default
        return int(value)
    except Exception:
        return default


@dataclass(slots=True)
class LLMV2Paths:
    baseline_run_root: str = str(DEFAULT_BASELINE_RUN_ROOT)
    prompt_dir: str = str(DEFAULT_PROMPT_DIR)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class LLMV2RunConfig:
    schema_version: str
    run_id: str
    created_at_utc: str
    run_root: str
    scope_label: str
    baseline_alias: str
    pilot_size: int
    full_doc_count: int
    section_model: str
    semantic_model: str
    section_embedding_model: str
    execution_mode: str
    paths: LLMV2Paths

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["paths"] = self.paths.to_dict()
        return payload


class PurposeItem(BaseModel):
    motive: str
    description: str = Field(min_length=1, max_length=600)
    evidence: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("motive")
    @classmethod
    def _validate_motive(cls, value: str) -> str:
        out = clean_optional_str(value)
        if out not in set(MOTIVES):
            raise ValueError(f"invalid motive: {out}")
        return out

    @field_validator("evidence")
    @classmethod
    def _trim_evidence(cls, value: str | None) -> str | None:
        if value is None:
            return None
        out = clean_optional_str(value)
        return out[:300] or None


class DocumentPurposeOut(BaseModel):
    doc_id: str = Field(min_length=1)
    purposes: list[PurposeItem] = Field(default_factory=list)
    notes: str = ""


class SectionSpec(BaseModel):
    section_number: int = Field(ge=1)
    title: str = Field(min_length=1, max_length=160)
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)


class SectionSegmentationOut(BaseModel):
    doc_id: str = Field(min_length=1)
    sections: list[SectionSpec] = Field(min_length=1, max_length=80)
    notes: str = ""


class SectionMotiveOut(BaseModel):
    section_id: str = Field(min_length=1)
    primary_motive: str
    secondary_motives: list[str] = Field(default_factory=list)
    evidence: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    notes: str = ""

    @field_validator("primary_motive")
    @classmethod
    def _validate_primary_motive(cls, value: str) -> str:
        out = clean_optional_str(value)
        if out not in set(MOTIVES):
            raise ValueError(f"invalid motive: {out}")
        return out

    @field_validator("secondary_motives")
    @classmethod
    def _validate_secondary_motives(cls, value: list[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for item in value or []:
            motive = clean_optional_str(item)
            if motive not in set(MOTIVES) or motive in seen:
                continue
            seen.add(motive)
            out.append(motive)
        return out[:3]

    @field_validator("evidence")
    @classmethod
    def _trim_section_evidence(cls, value: str | None) -> str | None:
        if value is None:
            return None
        out = clean_optional_str(value)
        return out[:300] or None


def _validate_section_specs_strict(sections: list[SectionSpec], *, last_line: int) -> None:
    if not sections:
        raise ValueError("no sections")
    if sections[0].start_line != 1:
        raise ValueError("first section must start at line 1")
    prev_end = 0
    for sec in sections:
        if sec.start_line != prev_end + 1:
            raise ValueError("non-contiguous sections")
        if sec.end_line < sec.start_line:
            raise ValueError("end_line before start_line")
        prev_end = sec.end_line
    if prev_end != last_line:
        raise ValueError(f"final section must end at LAST_LINE={last_line}, got {prev_end}")


def _repair_section_specs(sections: list[SectionSpec], *, last_line: int) -> tuple[list[SectionSpec], bool]:
    ordered = sorted(sections, key=lambda s: (s.section_number, s.start_line, s.end_line, s.title))
    try:
        _validate_section_specs_strict(ordered, last_line=last_line)
        return ordered, False
    except Exception:
        pass

    by_start = sorted(sections, key=lambda s: (s.start_line, s.end_line, s.section_number, s.title))
    normalized: list[SectionSpec] = []
    repair_used = False
    prev_end = 0
    for idx, sec in enumerate(by_start, start=1):
        start_line = int(sec.start_line)
        end_line = int(sec.end_line)
        if end_line < start_line:
            raise ValueError("end_line before start_line")
        if idx == 1:
            if start_line != 1:
                if start_line <= 3:
                    start_line = 1
                    repair_used = True
                else:
                    raise ValueError("first section must start at line 1")
        else:
            gap = start_line - prev_end - 1
            if gap < 0:
                raise ValueError("overlapping sections")
            if gap == 0:
                pass
            elif gap <= 1:
                normalized[-1] = SectionSpec(
                    section_number=normalized[-1].section_number,
                    title=normalized[-1].title,
                    start_line=normalized[-1].start_line,
                    end_line=start_line - 1,
                )
                repair_used = True
            else:
                raise ValueError("non-contiguous sections")
        normalized.append(
            SectionSpec(
                section_number=idx,
                title=sec.title,
                start_line=start_line,
                end_line=end_line,
            )
        )
        prev_end = end_line

    if normalized[-1].end_line != last_line:
        tail_gap = last_line - normalized[-1].end_line
        if tail_gap < 0:
            raise ValueError(f"final section exceeds LAST_LINE={last_line}")
        if tail_gap <= 1:
            normalized[-1] = SectionSpec(
                section_number=normalized[-1].section_number,
                title=normalized[-1].title,
                start_line=normalized[-1].start_line,
                end_line=last_line,
            )
            repair_used = True
        else:
            raise ValueError(f"final section must end at LAST_LINE={last_line}, got {normalized[-1].end_line}")

    _validate_section_specs_strict(normalized, last_line=last_line)
    return normalized, repair_used


class ClauseItem(BaseModel):
    clause_number: int = Field(ge=1)
    clause_text: str = Field(min_length=1, max_length=900)
    requirement: str = Field(min_length=1, max_length=900)
    instrument_hint: str = Field(min_length=1, max_length=120)
    motive: str
    modality: Literal["must", "must_not", "may", "definition", "other"]
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("motive")
    @classmethod
    def _validate_clause_motive(cls, value: str) -> str:
        out = clean_optional_str(value)
        if out not in set(MOTIVES):
            raise ValueError(f"invalid motive: {out}")
        return out


class ClauseExtractionOut(BaseModel):
    section_id: str = Field(min_length=1)
    clauses: list[ClauseItem] = Field(default_factory=list)
    notes: str = ""

    @field_validator("clauses")
    @classmethod
    def _unique_clause_numbers(cls, value: list[ClauseItem]) -> list[ClauseItem]:
        nums = [item.clause_number for item in value]
        if len(nums) != len(set(nums)):
            raise ValueError("duplicate clause_number values")
        return value


class ClauseEnrichmentOut(BaseModel):
    clause_id: str = Field(min_length=1)
    actor: str | None = None
    action: str | None = None
    regulated_object: str | None = None
    condition: str | None = None
    exception: str | None = None
    threshold_or_quantity: str | None = None
    spatial_scope: str | None = None
    instrument_type: str
    operative_flag: int = Field(ge=0, le=1)
    evidence_quote: str = Field(min_length=1, max_length=300)
    confidence: float = Field(ge=0.0, le=1.0)
    notes: str = ""

    @field_validator("instrument_type")
    @classmethod
    def _validate_instrument_type(cls, value: str) -> str:
        out = clean_optional_str(value)
        if out not in set(INSTRUMENT_TYPES):
            raise ValueError(f"invalid instrument_type: {out}")
        return out


class ClusterLabelOut(BaseModel):
    cluster_id: int
    cluster_label: str = Field(min_length=1, max_length=120)
    cluster_description: str = Field(min_length=1, max_length=900)
    keywords: list[str] = Field(min_length=3, max_length=25)
    notes: str = ""

    @field_validator("keywords")
    @classmethod
    def _normalize_keywords(cls, value: list[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for item in value or []:
            text = clean_optional_str(item)
            if not text:
                continue
            lowered = text.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            out.append(text)
        if len(out) < 3:
            raise ValueError("need at least 3 keywords")
        return out[:25]


def section_segmentation_openai_text_config() -> dict[str, Any]:
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "doc_id": {"type": "string", "minLength": 1},
            "sections": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "section_number": {"type": "integer", "minimum": 1},
                        "title": {"type": "string", "minLength": 1, "maxLength": 160},
                        "start_line": {"type": "integer", "minimum": 1},
                        "end_line": {"type": "integer", "minimum": 1},
                    },
                    "required": ["section_number", "title", "start_line", "end_line"],
                },
                "minItems": 1,
                "maxItems": 80,
            },
            "notes": {"type": "string"},
        },
        "required": ["doc_id", "sections", "notes"],
    }
    return {"format": {"type": "json_schema", "name": "frontier_llm_v2_section_segmentation", "schema": schema, "strict": True}}


def document_purpose_openai_text_config() -> dict[str, Any]:
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "doc_id": {"type": "string", "minLength": 1},
            "purposes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "motive": {"type": "string", "enum": list(MOTIVES)},
                        "description": {"type": "string", "minLength": 1, "maxLength": 600},
                        "evidence": {"type": ["string", "null"], "maxLength": 300},
                        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    },
                    "required": ["motive", "description", "evidence", "confidence"],
                },
                "maxItems": 12,
            },
            "notes": {"type": "string"},
        },
        "required": ["doc_id", "purposes", "notes"],
    }
    return {"format": {"type": "json_schema", "name": "frontier_llm_v2_document_purpose", "schema": schema, "strict": True}}


def section_motive_openai_text_config() -> dict[str, Any]:
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "section_id": {"type": "string", "minLength": 1},
            "primary_motive": {"type": "string", "enum": list(MOTIVES)},
            "secondary_motives": {
                "type": "array",
                "items": {"type": "string", "enum": list(MOTIVES)},
                "maxItems": 3,
            },
            "evidence": {"type": ["string", "null"], "maxLength": 300},
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "notes": {"type": "string"},
        },
        "required": ["section_id", "primary_motive", "secondary_motives", "evidence", "confidence", "notes"],
    }
    return {"format": {"type": "json_schema", "name": "frontier_llm_v2_section_motive", "schema": schema, "strict": True}}


def clause_extraction_openai_text_config() -> dict[str, Any]:
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "section_id": {"type": "string", "minLength": 1},
            "clauses": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "clause_number": {"type": "integer", "minimum": 1},
                        "clause_text": {"type": "string", "minLength": 1, "maxLength": 900},
                        "requirement": {"type": "string", "minLength": 1, "maxLength": 900},
                        "instrument_hint": {"type": "string", "minLength": 1, "maxLength": 120},
                        "motive": {"type": "string", "enum": list(MOTIVES)},
                        "modality": {"type": "string", "enum": ["must", "must_not", "may", "definition", "other"]},
                        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    },
                    "required": [
                        "clause_number",
                        "clause_text",
                        "requirement",
                        "instrument_hint",
                        "motive",
                        "modality",
                        "confidence",
                    ],
                },
            },
            "notes": {"type": "string"},
        },
        "required": ["section_id", "clauses", "notes"],
    }
    return {"format": {"type": "json_schema", "name": "frontier_llm_v2_clause_extraction", "schema": schema, "strict": True}}


def clause_enrichment_openai_text_config() -> dict[str, Any]:
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "clause_id": {"type": "string", "minLength": 1},
            "actor": {"type": ["string", "null"], "maxLength": 160},
            "action": {"type": ["string", "null"], "maxLength": 200},
            "regulated_object": {"type": ["string", "null"], "maxLength": 200},
            "condition": {"type": ["string", "null"], "maxLength": 300},
            "exception": {"type": ["string", "null"], "maxLength": 300},
            "threshold_or_quantity": {"type": ["string", "null"], "maxLength": 200},
            "spatial_scope": {"type": ["string", "null"], "maxLength": 200},
            "instrument_type": {"type": "string", "enum": list(INSTRUMENT_TYPES)},
            "operative_flag": {"type": "integer", "enum": [0, 1]},
            "evidence_quote": {"type": "string", "minLength": 1, "maxLength": 300},
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "notes": {"type": "string"},
        },
        "required": [
            "clause_id",
            "actor",
            "action",
            "regulated_object",
            "condition",
            "exception",
            "threshold_or_quantity",
            "spatial_scope",
            "instrument_type",
            "operative_flag",
            "evidence_quote",
            "confidence",
            "notes",
        ],
    }
    return {"format": {"type": "json_schema", "name": "frontier_llm_v2_clause_enrichment", "schema": schema, "strict": True}}


def cluster_label_openai_text_config(name: str) -> dict[str, Any]:
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "cluster_id": {"type": "integer"},
            "cluster_label": {"type": "string", "minLength": 1, "maxLength": 120},
            "cluster_description": {"type": "string", "minLength": 1, "maxLength": 900},
            "keywords": {"type": "array", "items": {"type": "string"}, "minItems": 3, "maxItems": 25},
            "notes": {"type": "string"},
        },
        "required": ["cluster_id", "cluster_label", "cluster_description", "keywords", "notes"],
    }
    return {"format": {"type": "json_schema", "name": name, "schema": schema, "strict": True}}


def _make_run_id(prefix: str = "frontier_llm_v2") -> str:
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def ensure_llm_v2_run_layout(base_dir: Path | str | None = None, run_id: str | None = None) -> dict[str, Path]:
    base = Path(base_dir or DEFAULT_LLM_V2_RUNS_BASE).expanduser().resolve()
    rid = run_id or _make_run_id()
    run_root = base / rid
    run_root.mkdir(parents=True, exist_ok=True)
    layout = {"run_root": run_root}
    for name in LLM_V2_RUN_SUBDIRS:
        path = run_root / name
        path.mkdir(parents=True, exist_ok=True)
        layout[name] = path
    return layout


def llm_v2_layout_from_run_root(run_root: Path | str) -> dict[str, Path]:
    root = Path(run_root).expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"Run root does not exist: {root}")
    layout = {"run_root": root}
    for name in LLM_V2_RUN_SUBDIRS:
        path = root / name
        path.mkdir(parents=True, exist_ok=True)
        layout[name] = path
    return layout


def initialize_llm_v2_manifest(
    *,
    layout: dict[str, Path],
    baseline_run_root: Path | str = DEFAULT_BASELINE_RUN_ROOT,
    prompt_dir: Path | str = DEFAULT_PROMPT_DIR,
    pilot_size: int = 64,
    full_doc_count: int = 0,
    execution_mode: str = "batch_and_live_pilot",
) -> dict[str, Any]:
    config = LLMV2RunConfig(
        schema_version="frontier_llm_v2",
        run_id=layout["run_root"].name,
        created_at_utc=_utc_now_iso(),
        run_root=str(layout["run_root"]),
        scope_label="code_publication_full_issue_pid6_linked",
        baseline_alias="baseline_local",
        pilot_size=int(pilot_size),
        full_doc_count=int(full_doc_count),
        section_model="gpt-5-mini",
        semantic_model="gpt-5-mini",
        section_embedding_model=SECTION_EMBEDDING_MODEL,
        execution_mode=execution_mode,
        paths=LLMV2Paths(
            baseline_run_root=str(Path(baseline_run_root).expanduser().resolve()),
            prompt_dir=str(Path(prompt_dir).expanduser().resolve()),
        ),
    )
    payload = config.to_dict()
    write_json(layout["manifests"] / "run_manifest.json", payload)
    return payload


def _read_manifest(run_root: Path) -> dict[str, Any]:
    manifest_path = run_root / "manifests" / "run_manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"Missing run manifest: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _update_manifest(run_root: Path, key: str, value: Any) -> None:
    payload = _read_manifest(run_root)
    payload[key] = value
    write_json(run_root / "manifests" / "run_manifest.json", payload)


def _baseline_run_root(run_root: Path) -> Path:
    manifest = _read_manifest(run_root)
    raw = ((manifest.get("paths") or {}).get("baseline_run_root")) if isinstance(manifest.get("paths"), dict) else None
    root = Path(raw or DEFAULT_BASELINE_RUN_ROOT).expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"Baseline run root does not exist: {root}")
    return root


def _story_report_sample(run_root: Path) -> Literal["pilot", "full"]:
    """Only promote story/report to full once a coherent full research surface exists."""
    return "full" if (run_root / "research" / "research_summary__full.json").is_file() else "pilot"


def _prompt_dir(run_root: Path) -> Path:
    manifest = _read_manifest(run_root)
    raw = ((manifest.get("paths") or {}).get("prompt_dir")) if isinstance(manifest.get("paths"), dict) else None
    prompt_dir = Path(raw or DEFAULT_PROMPT_DIR).expanduser().resolve()
    if not prompt_dir.is_dir():
        raise SystemExit(f"Prompt dir does not exist: {prompt_dir}")
    return prompt_dir


def _copy_baseline_frame(src: Path, dest: Path) -> None:
    if not src.is_file():
        raise SystemExit(f"Missing baseline artifact: {src}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    if src.suffix == ".parquet":
        write_parquet(pd.read_parquet(src), dest)
    elif src.suffix == ".json":
        write_json(dest, json.loads(src.read_text(encoding="utf-8")))
    else:
        shutil.copy2(src, dest)


def _with_length_quartile(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if out.empty:
        out["length_quartile"] = pd.Series(dtype=int)
        return out
    series = out["char_count"].rank(method="first")
    out["length_quartile"] = pd.qcut(series, q=4, labels=[1, 2, 3, 4]).astype(int)
    return out


def _pilot_pick_region(df: pd.DataFrame, quota: int) -> list[str]:
    working = df.copy().sort_values(["issue_date", "ordinance_doc_id"]).reset_index(drop=True)
    selected: list[str] = []
    selected_cluster_counts: Counter[str] = Counter()
    selected_level_counts: Counter[str] = Counter()
    selected_quartile_counts: Counter[int] = Counter()
    while len(selected) < min(quota, len(working)):
        best_idx: int | None = None
        best_score: tuple[float, float, float, float, str] | None = None
        for idx, row in working.iterrows():
            doc_id = clean_optional_str(row["ordinance_doc_id"])
            if doc_id in selected:
                continue
            cluster = clean_optional_str(row.get("baseline_policy_signature_label", "unknown")) or "unknown"
            level = clean_optional_str(row.get("jurisdiction_level", "unknown")) or "unknown"
            quartile = _safe_int(row.get("length_quartile"), 0)
            coverage = 0.0
            coverage += 4.0 if selected_level_counts[level] == 0 else 1.0 / (1.0 + selected_level_counts[level])
            coverage += 3.0 if selected_quartile_counts[quartile] == 0 else 1.0 / (1.0 + selected_quartile_counts[quartile])
            coverage += 3.0 if selected_cluster_counts[cluster] == 0 else 1.0 / (1.0 + selected_cluster_counts[cluster])
            char_bonus = float(row.get("char_count", 0)) / 100000.0
            score = (coverage, -selected_cluster_counts[cluster], -selected_level_counts[level], -selected_quartile_counts[quartile], doc_id)
            score = (coverage + char_bonus, score[1], score[2], score[3], score[4])
            if best_score is None or score > best_score:
                best_score = score
                best_idx = int(idx)
        if best_idx is None:
            break
        chosen = working.iloc[best_idx]
        doc_id = clean_optional_str(chosen["ordinance_doc_id"])
        selected.append(doc_id)
        selected_cluster_counts[clean_optional_str(chosen.get("baseline_policy_signature_label", "unknown")) or "unknown"] += 1
        selected_level_counts[clean_optional_str(chosen.get("jurisdiction_level", "unknown")) or "unknown"] += 1
        selected_quartile_counts[_safe_int(chosen.get("length_quartile"), 0)] += 1
    return selected


def prepare_llm_v2_documents(
    *,
    run_root: Path,
    pilot_size: int = 64,
    pilot_region_quotas: dict[str, int] | None = None,
) -> dict[str, Any]:
    baseline_root = _baseline_run_root(run_root)
    corpus_dir = run_root / "corpus"
    geo_dir = run_root / "geo"

    for name in ("publication_issues.parquet", "ordinances.parquet", "pages.parquet", "geo_links.parquet", "dedup_groups.parquet", "timeline_scaffold.parquet"):
        _copy_baseline_frame(baseline_root / "corpus" / name, corpus_dir / name)
    for name in ("jurisdiction_audit.parquet", "by_jurisdiction_geo.parquet", "by_publication_geo.parquet", "high_mismatch_publications.parquet"):
        _copy_baseline_frame(baseline_root / "geo" / name, geo_dir / name)

    ordinances = pd.read_parquet(corpus_dir / "ordinances.parquet")
    ordinances = ordinances[
        ordinances["jurisdiction_resolution_status"].astype(str).eq("resolved_with_census")
        & ordinances["clean_text"].astype(str).str.strip().astype(bool)
    ].copy()
    policy_assign_path = baseline_root / "clusters" / "policy_signature_kmeans_probe" / "assignments.parquet"
    policy_assign = pd.read_parquet(policy_assign_path)[["ordinance_doc_id", "policy_signature_cluster_id", "cluster_label"]].rename(
        columns={"cluster_label": "baseline_policy_signature_label"}
    )
    docs = ordinances.merge(policy_assign, on="ordinance_doc_id", how="left")
    docs["baseline_policy_signature_label"] = docs["baseline_policy_signature_label"].astype(str).replace("nan", "").fillna("")
    docs["char_count"] = docs["clean_text"].astype(str).map(len)
    docs["word_count"] = docs["clean_text"].astype(str).str.split().str.len()
    docs = _with_length_quartile(docs)
    docs["text"] = docs["clean_text"].astype(str)
    docs["doc_id"] = docs["ordinance_doc_id"].astype(str)
    docs_out_cols = [
        "doc_id",
        "ordinance_doc_id",
        "issue_id",
        "issue_date",
        "census_id_pid6",
        "publication_key",
        "publication_city_name",
        "publication_state_abbr",
        "publication_region",
        "jurisdiction_key",
        "jurisdiction_city_name",
        "jurisdiction_state_abbr",
        "jurisdiction_region",
        "jurisdiction_level",
        "dedup_group_id",
        "baseline_policy_signature_label",
        "policy_signature_cluster_id",
        "char_count",
        "word_count",
        "length_quartile",
        "text",
    ]
    docs = docs[docs_out_cols].sort_values(["jurisdiction_region", "issue_date", "doc_id"]).reset_index(drop=True)
    write_parquet(docs, corpus_dir / "docs.parquet")
    write_jsonl(
        corpus_dir / "docs.jsonl",
        [
            {
                **{k: ("" if pd.isna(v) else v) for k, v in row.items()},
            }
            for row in docs.to_dict(orient="records")
        ],
    )

    quotas = pilot_region_quotas or {"northeast": 16, "midwest": 16, "south": 16, "west": 16}
    selected_ids: list[str] = []
    for region, quota in quotas.items():
        region_df = docs[docs["jurisdiction_region"].astype(str) == region].copy()
        if len(region_df) < quota:
            raise SystemExit(f"Insufficient docs for region quota {region}={quota}; available={len(region_df)}")
        selected_ids.extend(_pilot_pick_region(region_df, quota))
    pilot_df = docs[docs["ordinance_doc_id"].isin(selected_ids)].copy().sort_values(["jurisdiction_region", "issue_date", "doc_id"]).reset_index(drop=True)
    full_df = docs.copy()

    write_parquet(pilot_df, corpus_dir / "pilot_sample.parquet")
    write_parquet(full_df, corpus_dir / "full_sample.parquet")
    write_jsonl(corpus_dir / "pilot_sample.jsonl", pilot_df.to_dict(orient="records"))
    write_jsonl(corpus_dir / "full_sample.jsonl", full_df.to_dict(orient="records"))
    (corpus_dir / "pilot_doc_ids.txt").write_text("\n".join(pilot_df["ordinance_doc_id"].astype(str).tolist()) + "\n", encoding="utf-8")
    (corpus_dir / "full_doc_ids.txt").write_text("\n".join(full_df["ordinance_doc_id"].astype(str).tolist()) + "\n", encoding="utf-8")

    summary = {
        "baseline_run_root": str(baseline_root),
        "full_doc_count": int(len(full_df)),
        "pilot_doc_count": int(len(pilot_df)),
        "requested_pilot_size": int(pilot_size),
        "pilot_region_quotas": {str(k): int(v) for k, v in quotas.items()},
        "pilot_region_counts": pilot_df["jurisdiction_region"].value_counts().to_dict(),
        "pilot_level_counts": pilot_df["jurisdiction_level"].value_counts().to_dict(),
        "pilot_length_quartiles": pilot_df["length_quartile"].value_counts().sort_index().to_dict(),
        "baseline_policy_signature_cluster_count": int(policy_assign["policy_signature_cluster_id"].nunique(dropna=True)),
    }
    write_json(corpus_dir / "document_prep_summary.json", summary)
    _update_manifest(run_root, "pilot_size", int(len(pilot_df)))
    _update_manifest(run_root, "full_doc_count", int(len(full_df)))
    return summary


def _sample_docs(run_root: Path, sample: Literal["pilot", "full"]) -> pd.DataFrame:
    path = run_root / "corpus" / f"{sample}_sample.parquet"
    if not path.is_file():
        raise SystemExit(f"Missing sample parquet: {path}. Run document prep first.")
    return pd.read_parquet(path)


def _requests_stage_dir(run_root: Path, sample: str, stage: str) -> Path:
    path = run_root / "requests" / sample / stage
    path.mkdir(parents=True, exist_ok=True)
    return path


def _results_stage_dir(run_root: Path, sample: str, stage: str) -> Path:
    path = run_root / "results" / sample / stage
    path.mkdir(parents=True, exist_ok=True)
    return path


def _rotate_files_if_needed(
    *,
    next_line: str,
    requests_in_shard: int,
    bytes_in_shard: int,
    req_limit: int,
    max_bytes: int,
) -> bool:
    if requests_in_shard <= 0:
        return False
    if requests_in_shard >= req_limit:
        return True
    if max_bytes > 0 and bytes_in_shard + len(next_line.encode("utf-8")) > max_bytes:
        return True
    return False


def _clear_request_stage_dir(stage_dir: Path) -> None:
    stage_dir.mkdir(parents=True, exist_ok=True)
    for pattern in ("openai_requests_shard*.jsonl", "mapping_shard*.jsonl"):
        for path in stage_dir.glob(pattern):
            path.unlink()
    export_summary = stage_dir / "export_summary.json"
    if export_summary.exists():
        export_summary.unlink()


def _next_request_shard_index(stage_dir: Path) -> int:
    indices: set[int] = set()
    for pattern in ("openai_requests_shard*.jsonl", "mapping_shard*.jsonl"):
        for path in stage_dir.glob(pattern):
            token = path.stem.rsplit("shard", 1)[-1]
            if token.isdigit():
                indices.add(int(token))
    return (max(indices) + 1) if indices else 0


def _open_request_shard(stage_dir: Path, shard_idx: int) -> tuple[Any, Any]:
    openai_f = (stage_dir / f"openai_requests_shard{shard_idx:03d}.jsonl").open("w", encoding="utf-8")
    mapping_f = (stage_dir / f"mapping_shard{shard_idx:03d}.jsonl").open("w", encoding="utf-8")
    return openai_f, mapping_f


def _response_request_line(*, custom_id: str, model: str, prompt: str, text_cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/responses",
        "body": {
            "model": model,
            "input": [{"role": "user", "content": [{"type": "input_text", "text": prompt}]}],
            "text": text_cfg,
        },
    }


def _embedding_request_line(*, custom_id: str, model: str, text: str) -> dict[str, Any]:
    return {"custom_id": custom_id, "method": "POST", "url": "/v1/embeddings", "body": {"model": model, "input": text}}


def _wrap_lines(text: str, width: int = 120) -> list[str]:
    out: list[str] = []
    for raw in str(text or "").splitlines():
        if not raw.strip():
            out.append("")
            continue
        wrapped = textwrap.wrap(
            raw,
            width=max(40, int(width)),
            break_long_words=False,
            break_on_hyphens=False,
            replace_whitespace=False,
            drop_whitespace=False,
        )
        out.extend(wrapped or [raw])
    return out or [""]


def _line_numbered(lines: list[str]) -> str:
    return "\n".join(f"L{i:04d}: {line}".rstrip() for i, line in enumerate(lines, start=1))


def _response_stage_text_config(stage: str) -> dict[str, Any]:
    if stage == "section_segmentation":
        return section_segmentation_openai_text_config()
    if stage == "document_purpose":
        return document_purpose_openai_text_config()
    if stage == "section_motive":
        return section_motive_openai_text_config()
    if stage == "clause_extraction":
        return clause_extraction_openai_text_config()
    if stage == "clause_enrichment":
        return clause_enrichment_openai_text_config()
    if stage == "section_cluster_label":
        return cluster_label_openai_text_config("frontier_llm_v2_section_cluster_label")
    if stage == "section_local_cluster_label":
        return cluster_label_openai_text_config("frontier_llm_v2_section_local_cluster_label")
    if stage == "clause_cluster_label":
        return cluster_label_openai_text_config("frontier_llm_v2_clause_cluster_label")
    raise SystemExit(f"Unknown response stage: {stage}")


def _load_filter_values(path: Path, *, column: str) -> set[str]:
    if not path.is_file():
        raise SystemExit(f"Source filter file not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        df = pd.read_parquet(path)
    elif suffix in {".csv", ".tsv"}:
        df = pd.read_csv(path, sep="\t" if suffix == ".tsv" else ",")
    elif suffix == ".jsonl":
        df = pd.read_json(path, lines=True)
    elif suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            df = pd.DataFrame(payload)
        else:
            raise SystemExit(f"JSON filter file must contain a list of objects: {path}")
    else:
        raise SystemExit(f"Unsupported source filter format: {path}")
    if column not in df.columns:
        raise SystemExit(f"Filter column '{column}' not found in {path}")
    values = {
        clean_optional_str(value)
        for value in df[column].astype(str).tolist()
        if clean_optional_str(value)
    }
    if not values:
        raise SystemExit(f"Filter file {path} produced no usable values for column '{column}'")
    return values


def _stage_source_df(run_root: Path, sample: Literal["pilot", "full"], stage: str) -> pd.DataFrame:
    if stage in {"section_segmentation", "document_purpose"}:
        return _sample_docs(run_root, sample)
    if stage == "section_motive":
        path = run_root / "structure" / f"sections__{sample}.parquet"
        if not path.is_file():
            raise SystemExit(f"Missing sections parquet for sample={sample}: {path}")
        return pd.read_parquet(path)
    if stage == "clause_extraction":
        sec_path = run_root / "structure" / f"sections__{sample}.parquet"
        mot_path = run_root / "semantics" / f"section_motives__{sample}.parquet"
        if not sec_path.is_file():
            raise SystemExit(f"Missing sections parquet for sample={sample}: {sec_path}")
        if not mot_path.is_file():
            raise SystemExit(f"Missing section motives parquet for sample={sample}: {mot_path}")
        return pd.read_parquet(sec_path).merge(pd.read_parquet(mot_path)[["section_id", "primary_motive"]], on="section_id", how="left")
    if stage == "clause_enrichment":
        path = run_root / "semantics" / f"clauses__{sample}.parquet"
        if not path.is_file():
            raise SystemExit(f"Missing clauses parquet for sample={sample}: {path}")
        return pd.read_parquet(path)
    if stage == "section_cluster_label":
        return _cluster_label_source_df(run_root, SECTION_CLUSTER_EXPERIMENT)
    if stage == "section_local_cluster_label":
        return _cluster_label_source_df(run_root, SECTION_LOCAL_CLUSTER_EXPERIMENT)
    if stage == "clause_cluster_label":
        return _cluster_label_source_df(run_root, CLAUSE_CLUSTER_EXPERIMENT)
    if stage == "section_embedding":
        path = run_root / "structure" / f"sections__{sample}.parquet"
        if not path.is_file():
            raise SystemExit(f"Missing sections parquet for sample={sample}: {path}")
        return pd.read_parquet(path)
    raise SystemExit(f"Unknown stage: {stage}")


def _cluster_label_source_df(run_root: Path, experiment_name: str) -> pd.DataFrame:
    assign_path = run_root / "clusters" / experiment_name / "assignments.parquet"
    if not assign_path.is_file():
        raise SystemExit(f"Missing cluster assignments for {experiment_name}: {assign_path}")
    return pd.read_parquet(assign_path)


def export_llm_v2_requests(
    *,
    run_root: Path,
    sample: Literal["pilot", "full"],
    stage: str,
    requests_per_shard: int = 1000,
    max_bytes_per_shard: int = 180_000_000,
    source_filter_path: Path | None = None,
    source_filter_column: str | None = None,
    append: bool = False,
) -> dict[str, Any]:
    prompt_dir = _prompt_dir(run_root)
    stage_dir = _requests_stage_dir(run_root, sample, stage)
    stage_dir.mkdir(parents=True, exist_ok=True)
    prompt_name = RESPONSE_STAGE_PROMPTS.get(stage)
    prompt_text = ""
    prompt_sha = ""
    if prompt_name:
        prompt_text, prompt_sha = _load_prompt(prompt_dir, prompt_name)
    model = RESPONSE_STAGE_MODELS.get(stage, "gpt-5-mini")
    source_df = _stage_source_df(run_root, sample, stage)
    filter_values: set[str] | None = None
    if (source_filter_path is None) != (source_filter_column is None):
        raise SystemExit("source_filter_path and source_filter_column must be provided together.")
    if source_filter_path is not None and source_filter_column is not None:
        filter_values = _load_filter_values(source_filter_path, column=source_filter_column)
        if source_filter_column not in source_df.columns:
            raise SystemExit(
                f"Source filter column '{source_filter_column}' is not present in stage source for stage='{stage}'."
            )
        source_df = source_df[
            source_df[source_filter_column].astype(str).map(clean_optional_str).isin(filter_values)
        ].copy()
    source_row_count = int(len(source_df))
    if not append:
        _clear_request_stage_dir(stage_dir)
    shard_idx = _next_request_shard_index(stage_dir) if append else 0
    requests_in_shard = 0
    bytes_in_shard = 0
    openai_f, mapping_f = _open_request_shard(stage_dir, shard_idx)
    exported = 0
    text_cfg = _response_stage_text_config(stage) if stage != "section_embedding" else None
    for row in source_df.to_dict(orient="records"):
        if stage == "section_segmentation":
            doc_id = clean_optional_str(row["doc_id"])
            text = clean_optional_str(row["text"])
            lines = _wrap_lines(text, 120)
            prompt = "\n".join(
                [
                    prompt_text,
                    "",
                    f"DOC_ID: {doc_id}",
                    f"LAST_LINE: {len(lines)}",
                    "DOC_TEXT_LINES:",
                    _line_numbered(lines),
                ]
            ).strip()
            custom_id = f"doc_sections::{doc_id}"
            req = _response_request_line(custom_id=custom_id, model=model, prompt=prompt, text_cfg=text_cfg or {})
            mapping = {
                "custom_id": custom_id,
                "stage": stage,
                "sample": sample,
                "doc_id": doc_id,
                "ordinance_doc_id": clean_optional_str(row["ordinance_doc_id"]),
                "issue_id": clean_optional_str(row["issue_id"]),
                "prompt_path": str(prompt_dir / prompt_name),
                "prompt_sha256": prompt_sha,
                "model": model,
                "wrap_width": 120,
                "n_lines": len(lines),
                "doc_chars": len(text),
            }
        elif stage == "document_purpose":
            doc_id = clean_optional_str(row["doc_id"])
            prompt = "\n".join([prompt_text, "", f"DOC_ID: {doc_id}", "DOC_TEXT:", "<<<", clean_optional_str(row["text"]), ">>>"]).strip()
            custom_id = f"doc_purpose::{doc_id}"
            req = _response_request_line(custom_id=custom_id, model=model, prompt=prompt, text_cfg=text_cfg or {})
            mapping = {
                "custom_id": custom_id,
                "stage": stage,
                "sample": sample,
                "doc_id": doc_id,
                "ordinance_doc_id": clean_optional_str(row["ordinance_doc_id"]),
                "issue_id": clean_optional_str(row["issue_id"]),
                "prompt_path": str(prompt_dir / prompt_name),
                "prompt_sha256": prompt_sha,
                "model": model,
            }
        elif stage == "section_motive":
            section_id = clean_optional_str(row["section_id"])
            prompt = "\n".join(
                [
                    prompt_text,
                    "",
                    f"SECTION_ID: {section_id}",
                    f"DOC_ID: {clean_optional_str(row['ordinance_doc_id'])}",
                    f"SECTION_TITLE: {clean_optional_str(row['title'])}",
                    "SECTION_TEXT:",
                    "<<<",
                    clean_optional_str(row["text"]),
                    ">>>",
                ]
            ).strip()
            custom_id = f"section_motive::{section_id}"
            req = _response_request_line(custom_id=custom_id, model=model, prompt=prompt, text_cfg=text_cfg or {})
            mapping = {
                "custom_id": custom_id,
                "stage": stage,
                "sample": sample,
                "section_id": section_id,
                "doc_id": clean_optional_str(row["ordinance_doc_id"]),
                "prompt_path": str(prompt_dir / prompt_name),
                "prompt_sha256": prompt_sha,
                "model": model,
            }
        elif stage == "clause_extraction":
            section_id = clean_optional_str(row["section_id"])
            motive_guess = clean_optional_str(row.get("primary_motive", ""))
            prompt_parts = [
                prompt_text,
                "",
                f"SECTION_ID: {section_id}",
                f"DOC_ID: {clean_optional_str(row['ordinance_doc_id'])}",
                f"SECTION_TITLE: {clean_optional_str(row['title'])}",
            ]
            if motive_guess:
                prompt_parts.append(f"PRIMARY_MOTIVE_GUESS: {motive_guess}")
            prompt_parts.extend(["SECTION_TEXT:", "<<<", clean_optional_str(row["text"]), ">>>"])
            prompt = "\n".join(prompt_parts).strip()
            custom_id = f"section_clauses::{section_id}"
            req = _response_request_line(custom_id=custom_id, model=model, prompt=prompt, text_cfg=text_cfg or {})
            mapping = {
                "custom_id": custom_id,
                "stage": stage,
                "sample": sample,
                "section_id": section_id,
                "doc_id": clean_optional_str(row["ordinance_doc_id"]),
                "section_title": clean_optional_str(row["title"]),
                "primary_motive_guess": motive_guess,
                "prompt_path": str(prompt_dir / prompt_name),
                "prompt_sha256": prompt_sha,
                "model": model,
            }
        elif stage == "clause_enrichment":
            clause_id = clean_optional_str(row["clause_id"])
            prompt = "\n".join(
                [
                    prompt_text,
                    "",
                    f"CLAUSE_ID: {clause_id}",
                    f"DOC_ID: {clean_optional_str(row['ordinance_doc_id'])}",
                    f"SECTION_ID: {clean_optional_str(row['section_id'])}",
                    f"SECTION_TITLE: {clean_optional_str(row['section_title'])}",
                    f"CLAUSE_TEXT: {clean_optional_str(row['clause_text'])}",
                    f"REQUIREMENT: {clean_optional_str(row['requirement'])}",
                    f"MODALITY: {clean_optional_str(row['modality'])}",
                    f"MOTIVE: {clean_optional_str(row['motive'])}",
                    f"INSTRUMENT_HINT: {clean_optional_str(row['instrument_hint'])}",
                ]
            ).strip()
            custom_id = f"clause_enrichment::{clause_id}"
            req = _response_request_line(custom_id=custom_id, model=model, prompt=prompt, text_cfg=text_cfg or {})
            mapping = {
                "custom_id": custom_id,
                "stage": stage,
                "sample": sample,
                "clause_id": clause_id,
                "section_id": clean_optional_str(row["section_id"]),
                "doc_id": clean_optional_str(row["ordinance_doc_id"]),
                "prompt_path": str(prompt_dir / prompt_name),
                "prompt_sha256": prompt_sha,
                "model": model,
            }
        elif stage == "section_embedding":
            section_id = clean_optional_str(row["section_id"])
            full_text = clean_optional_str(row["text"])
            segments = _split_text_for_embedding(full_text)
            segment_count = len(segments)
            for seg_idx, seg_text in enumerate(segments, start=1):
                custom_id = f"{section_id}::segment::{seg_idx:03d}"
                req = _embedding_request_line(custom_id=custom_id, model=SECTION_EMBEDDING_MODEL, text=seg_text)
                mapping = {
                    "custom_id": custom_id,
                    "stage": stage,
                    "sample": sample,
                    "section_id": section_id,
                    "doc_id": clean_optional_str(row["ordinance_doc_id"]),
                    "section_title": clean_optional_str(row["title"]),
                    "model": SECTION_EMBEDDING_MODEL,
                    "segment_index": int(seg_idx),
                    "segment_count": int(segment_count),
                    "segment_char_count": int(len(seg_text)),
                    "section_char_count": int(len(full_text)),
                    "text_sha256": _sha256_text(seg_text),
                }
                line = json.dumps(req, ensure_ascii=False)
                projected_requests = requests_in_shard + 1
                projected_bytes = bytes_in_shard + len(line.encode("utf-8")) + 1
                if projected_requests > requests_per_shard or projected_bytes > max_bytes_per_shard:
                    openai_f.close()
                    mapping_f.close()
                    shard_idx += 1
                    requests_in_shard = 0
                    bytes_in_shard = 0
                    openai_f, mapping_f = _open_request_shard(stage_dir, shard_idx)
                openai_f.write(line + "\n")
                mapping_f.write(json.dumps(mapping, ensure_ascii=False) + "\n")
                requests_in_shard += 1
                bytes_in_shard += len(line.encode("utf-8")) + 1
                exported += 1
            continue
        elif stage in {"section_cluster_label", "section_local_cluster_label", "clause_cluster_label"}:
            cluster_id = _safe_int(row["cluster_id"])
            prompt_lines = [
                prompt_text,
                "",
                f"CLUSTER_ID: {cluster_id}",
                f"CLUSTER_SIZE: {_safe_int(row['cluster_size'])}",
                f"REPRESENTATION_LEVEL: {clean_optional_str(row['representation_level'])}",
            ]
            if stage in {"section_cluster_label", "section_local_cluster_label"}:
                prompt_lines.append(f"SECTION_TITLE_COUNTS: {clean_optional_str(row['section_title_counts'])}")
            else:
                prompt_lines.append(f"INSTRUMENT_TYPE_COUNTS: {clean_optional_str(row['instrument_type_counts'])}")
                prompt_lines.append(f"MODALITY_COUNTS: {clean_optional_str(row['modality_counts'])}")
            prompt_lines.append(f"MOTIVE_COUNTS: {clean_optional_str(row['motive_counts'])}")
            prompt_lines.append("REPRESENTATIVE_TEXTS:")
            for idx, sample_text in enumerate(str(row["representative_texts"]).split("\n\n<<<NEXT>>>\n\n"), start=1):
                prompt_lines.extend([f"[TEXT {idx}]", sample_text.strip()])
            prompt = "\n".join(prompt_lines).strip()
            custom_id = f"{stage}::{cluster_id}"
            req = _response_request_line(custom_id=custom_id, model=model, prompt=prompt, text_cfg=text_cfg or {})
            mapping = {
                "custom_id": custom_id,
                "stage": stage,
                "sample": sample,
                "cluster_id": cluster_id,
                "experiment_name": clean_optional_str(row["experiment_name"]),
                "representation_level": clean_optional_str(row["representation_level"]),
                "cluster_size": _safe_int(row["cluster_size"]),
                "prompt_path": str(prompt_dir / prompt_name),
                "prompt_sha256": prompt_sha,
                "model": model,
            }
        else:
            raise SystemExit(f"Unsupported export stage: {stage}")

        req_s = json.dumps(req, ensure_ascii=False) + "\n"
        if _rotate_files_if_needed(
            next_line=req_s,
            requests_in_shard=requests_in_shard,
            bytes_in_shard=bytes_in_shard,
            req_limit=max(1, int(requests_per_shard)),
            max_bytes=max(0, int(max_bytes_per_shard)),
        ):
            openai_f.close()
            mapping_f.close()
            shard_idx += 1
            requests_in_shard = 0
            bytes_in_shard = 0
            openai_f, mapping_f = _open_request_shard(stage_dir, shard_idx)
        openai_f.write(req_s)
        mapping_f.write(json.dumps(mapping, ensure_ascii=False) + "\n")
        exported += 1
        requests_in_shard += 1
        bytes_in_shard += len(req_s.encode("utf-8"))

    openai_f.close()
    mapping_f.close()
    summary = {
        "stage": stage,
        "sample": sample,
        "source_row_count": int(source_row_count),
        "exported_requests": int(exported),
        "shard_count": int(shard_idx + 1),
        "model": model if stage != "section_embedding" else SECTION_EMBEDDING_MODEL,
        "append": bool(append),
    }
    if source_filter_path is not None and source_filter_column is not None:
        summary["source_filter_path"] = str(source_filter_path)
        summary["source_filter_column"] = str(source_filter_column)
        summary["source_filter_value_count"] = int(len(filter_values or set()))
    write_json(stage_dir / "export_summary.json", summary)
    return summary


def _load_mapping_by_custom_id(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for obj in iter_jsonl(path):
        cid = clean_optional_str(obj.get("custom_id") or obj.get("id"))
        if cid:
            out[cid] = obj
    return out


def _discover_request_shards(request_dir: Path) -> list[int]:
    shards: set[int] = set()
    for path in request_dir.glob("mapping_shard*.jsonl"):
        token = path.name.removeprefix("mapping_shard").removesuffix(".jsonl")
        if token.isdigit():
            shards.add(int(token))
    return sorted(shards)


def _write_invalid_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    write_jsonl(path, rows)


def rehydrate_llm_v2_results(
    *,
    run_root: Path,
    sample: Literal["pilot", "full"],
    stage: str,
    allow_partial_results: bool = False,
    allow_errors: bool = False,
) -> dict[str, Any]:
    request_dir = _requests_stage_dir(run_root, sample, stage)
    results_dir = _results_stage_dir(run_root, sample, stage)
    shards = _discover_request_shards(request_dir)
    if not shards:
        raise SystemExit(f"No mapping shards found under {request_dir}")
    mapping_by_cid: dict[str, dict[str, Any]] = {}
    results_by_cid: dict[str, dict[str, Any]] = {}
    errors_by_cid: dict[str, dict[str, Any]] = {}
    for shard in shards:
        mapping_by_cid.update(_load_mapping_by_custom_id(request_dir / f"mapping_shard{shard:03d}.jsonl"))
        results_path = results_dir / f"openai_results_shard{shard:03d}.jsonl"
        errors_path = results_dir / f"openai_errors_shard{shard:03d}.jsonl"
        if results_path.is_file():
            results_by_cid.update(_load_mapping_by_custom_id(results_path))
        elif not allow_partial_results:
            raise SystemExit(f"Missing results shard: {results_path}")
        if errors_path.is_file():
            errors_by_cid.update(_load_mapping_by_custom_id(errors_path))

    invalid_rows: list[dict[str, Any]] = []
    stage_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    stamp = _utc_now_iso()

    if stage == "section_segmentation":
        docs = {clean_optional_str(r["doc_id"]): r for r in _sample_docs(run_root, sample).to_dict(orient="records")}
        for cid, mapping in mapping_by_cid.items():
            res = results_by_cid.get(cid)
            if res is None:
                invalid_rows.append({"custom_id": cid, "doc_id": mapping["doc_id"], "error": "missing_result", "has_error_line": cid in errors_by_cid})
                continue
            body = (res.get("response") or {}).get("body") or {}
            text = _extract_output_text(body)
            if not text:
                invalid_rows.append({"custom_id": cid, "doc_id": mapping["doc_id"], "error": "empty_output_text"})
                continue
            try:
                parsed = _coerce_json(text)
                payload = SectionSegmentationOut.model_validate(parsed)
            except Exception as exc:
                invalid_rows.append({"custom_id": cid, "doc_id": mapping["doc_id"], "error": "validation_failed", "message": str(exc), "raw_text": text[:4000]})
                continue
            doc_id = clean_optional_str(mapping["doc_id"])
            if payload.doc_id != doc_id:
                invalid_rows.append({"custom_id": cid, "doc_id": doc_id, "error": "doc_id_mismatch", "model_doc_id": payload.doc_id})
                continue
            doc = docs.get(doc_id)
            if doc is None:
                invalid_rows.append({"custom_id": cid, "doc_id": doc_id, "error": "missing_doc"})
                continue
            lines = _wrap_lines(clean_optional_str(doc["text"]), width=_safe_int(mapping.get("wrap_width"), 120))
            last_line = len(lines)
            try:
                ordered, repair_used = _repair_section_specs(payload.sections, last_line=last_line)
            except Exception as exc:
                invalid_rows.append({"custom_id": cid, "doc_id": doc_id, "error": "section_coverage_invalid", "message": str(exc)})
                continue
            for sec in ordered:
                text_slice = "\n".join(lines[sec.start_line - 1 : sec.end_line]).strip()
                stage_rows.append(
                    {
                        "section_id": f"{doc_id}::section::{sec.section_number:03d}",
                        "section_number": int(sec.section_number),
                        "ordinance_doc_id": doc_id,
                        "issue_id": clean_optional_str(doc["issue_id"]),
                        "census_id_pid6": clean_optional_str(doc["census_id_pid6"]),
                        "jurisdiction_region": clean_optional_str(doc["jurisdiction_region"]),
                        "jurisdiction_level": clean_optional_str(doc["jurisdiction_level"]),
                        "title": sec.title,
                        "title_slug": slugify(sec.title),
                        "section_kind": slugify(sec.title),
                        "start_line": int(sec.start_line),
                        "end_line": int(sec.end_line),
                        "text": text_slice,
                        "char_count": len(text_slice),
                        "word_count": len(text_slice.split()),
                        "sample": sample,
                        "custom_id": cid,
                        "coverage_repair_used": int(repair_used),
                        "notes": payload.notes,
                    }
                )
            manifest_rows.append(
                {
                    "doc_id": doc_id,
                    "section_count": len(ordered),
                    "sample": sample,
                    "custom_id": cid,
                    "coverage_repair_used": int(repair_used),
                    "rehydrated_at": stamp,
                }
            )
        stage_df = pd.DataFrame.from_records(stage_rows).sort_values(["ordinance_doc_id", "section_number"]).reset_index(drop=True) if stage_rows else pd.DataFrame()
        write_parquet(stage_df, run_root / "structure" / f"sections__{sample}.parquet")
        write_jsonl(run_root / "structure" / f"sections__{sample}.jsonl", stage_rows)
    elif stage == "document_purpose":
        for cid, mapping in mapping_by_cid.items():
            res = results_by_cid.get(cid)
            if res is None:
                invalid_rows.append({"custom_id": cid, "doc_id": mapping["doc_id"], "error": "missing_result", "has_error_line": cid in errors_by_cid})
                continue
            text = _extract_output_text((res.get("response") or {}).get("body") or {})
            if not text:
                invalid_rows.append({"custom_id": cid, "doc_id": mapping["doc_id"], "error": "empty_output_text"})
                continue
            try:
                payload = DocumentPurposeOut.model_validate(_coerce_json(text))
            except Exception as exc:
                invalid_rows.append({"custom_id": cid, "doc_id": mapping["doc_id"], "error": "validation_failed", "message": str(exc), "raw_text": text[:4000]})
                continue
            for item in payload.purposes:
                stage_rows.append(
                    {
                        "ordinance_doc_id": payload.doc_id,
                        "motive": item.motive,
                        "description": item.description,
                        "evidence": item.evidence,
                        "confidence": float(item.confidence),
                        "sample": sample,
                        "custom_id": cid,
                        "notes": payload.notes,
                    }
                )
            manifest_rows.append({"doc_id": payload.doc_id, "purpose_count": len(payload.purposes), "sample": sample, "custom_id": cid, "rehydrated_at": stamp})
        stage_df = pd.DataFrame.from_records(stage_rows).sort_values(["ordinance_doc_id", "confidence"], ascending=[True, False]).reset_index(drop=True) if stage_rows else pd.DataFrame()
        write_parquet(stage_df, run_root / "semantics" / f"document_purposes__{sample}.parquet")
        write_jsonl(run_root / "semantics" / f"document_purposes__{sample}.jsonl", stage_rows)
    elif stage == "section_motive":
        for cid, mapping in mapping_by_cid.items():
            res = results_by_cid.get(cid)
            if res is None:
                invalid_rows.append({"custom_id": cid, "section_id": mapping["section_id"], "error": "missing_result", "has_error_line": cid in errors_by_cid})
                continue
            text = _extract_output_text((res.get("response") or {}).get("body") or {})
            if not text:
                invalid_rows.append({"custom_id": cid, "section_id": mapping["section_id"], "error": "empty_output_text"})
                continue
            try:
                payload = SectionMotiveOut.model_validate(_coerce_json(text))
            except Exception as exc:
                invalid_rows.append({"custom_id": cid, "section_id": mapping["section_id"], "error": "validation_failed", "message": str(exc), "raw_text": text[:4000]})
                continue
            stage_rows.append(
                {
                    "section_id": payload.section_id,
                    "ordinance_doc_id": clean_optional_str(mapping["doc_id"]),
                    "primary_motive": payload.primary_motive,
                    "secondary_motives": "|".join(payload.secondary_motives),
                    "evidence": payload.evidence,
                    "confidence": float(payload.confidence),
                    "notes": payload.notes,
                    "sample": sample,
                    "custom_id": cid,
                }
            )
        stage_df = pd.DataFrame.from_records(stage_rows).sort_values("section_id").reset_index(drop=True) if stage_rows else pd.DataFrame()
        write_parquet(stage_df, run_root / "semantics" / f"section_motives__{sample}.parquet")
        write_jsonl(run_root / "semantics" / f"section_motives__{sample}.jsonl", stage_rows)
    elif stage == "clause_extraction":
        section_lookup = pd.read_parquet(run_root / "structure" / f"sections__{sample}.parquet").set_index("section_id")
        for cid, mapping in mapping_by_cid.items():
            res = results_by_cid.get(cid)
            if res is None:
                invalid_rows.append({"custom_id": cid, "section_id": mapping["section_id"], "error": "missing_result", "has_error_line": cid in errors_by_cid})
                continue
            text = _extract_output_text((res.get("response") or {}).get("body") or {})
            if not text:
                invalid_rows.append({"custom_id": cid, "section_id": mapping["section_id"], "error": "empty_output_text"})
                continue
            try:
                payload = ClauseExtractionOut.model_validate(_coerce_json(text))
            except Exception as exc:
                invalid_rows.append({"custom_id": cid, "section_id": mapping["section_id"], "error": "validation_failed", "message": str(exc), "raw_text": text[:4000]})
                continue
            section_id = clean_optional_str(mapping["section_id"])
            if payload.section_id != section_id:
                invalid_rows.append({"custom_id": cid, "section_id": section_id, "error": "section_id_mismatch", "model_section_id": payload.section_id})
                continue
            if section_id not in section_lookup.index:
                invalid_rows.append({"custom_id": cid, "section_id": section_id, "error": "missing_section_lookup"})
                continue
            sec = section_lookup.loc[section_id]
            for item in payload.clauses:
                stage_rows.append(
                    {
                        "clause_id": f"{section_id}::clause::{item.clause_number:03d}",
                        "clause_number": int(item.clause_number),
                        "section_id": section_id,
                        "section_title": clean_optional_str(sec["title"]),
                        "ordinance_doc_id": clean_optional_str(sec["ordinance_doc_id"]),
                        "issue_id": clean_optional_str(sec["issue_id"]),
                        "census_id_pid6": clean_optional_str(sec["census_id_pid6"]),
                        "jurisdiction_region": clean_optional_str(sec["jurisdiction_region"]),
                        "jurisdiction_level": clean_optional_str(sec["jurisdiction_level"]),
                        "clause_text": item.clause_text,
                        "requirement": item.requirement,
                        "instrument_hint": item.instrument_hint,
                        "motive": item.motive,
                        "modality": item.modality,
                        "confidence": float(item.confidence),
                        "sample": sample,
                        "custom_id": cid,
                        "notes": payload.notes,
                    }
                )
            manifest_rows.append({"section_id": section_id, "clause_count": len(payload.clauses), "sample": sample, "custom_id": cid, "rehydrated_at": stamp})
        stage_df = pd.DataFrame.from_records(stage_rows).sort_values(["ordinance_doc_id", "section_id", "clause_number"]).reset_index(drop=True) if stage_rows else pd.DataFrame()
        write_parquet(stage_df, run_root / "semantics" / f"clauses__{sample}.parquet")
        write_jsonl(run_root / "semantics" / f"clauses__{sample}.jsonl", stage_rows)
    elif stage == "clause_enrichment":
        clause_lookup = pd.read_parquet(run_root / "semantics" / f"clauses__{sample}.parquet").set_index("clause_id")
        for cid, mapping in mapping_by_cid.items():
            res = results_by_cid.get(cid)
            if res is None:
                invalid_rows.append({"custom_id": cid, "clause_id": mapping["clause_id"], "error": "missing_result", "has_error_line": cid in errors_by_cid})
                continue
            text = _extract_output_text((res.get("response") or {}).get("body") or {})
            if not text:
                invalid_rows.append({"custom_id": cid, "clause_id": mapping["clause_id"], "error": "empty_output_text"})
                continue
            try:
                payload = ClauseEnrichmentOut.model_validate(_coerce_json(text))
            except Exception as exc:
                invalid_rows.append({"custom_id": cid, "clause_id": mapping["clause_id"], "error": "validation_failed", "message": str(exc), "raw_text": text[:4000]})
                continue
            clause_id = clean_optional_str(mapping["clause_id"])
            if payload.clause_id != clause_id:
                invalid_rows.append({"custom_id": cid, "clause_id": clause_id, "error": "clause_id_mismatch", "model_clause_id": payload.clause_id})
                continue
            if clause_id not in clause_lookup.index:
                invalid_rows.append({"custom_id": cid, "clause_id": clause_id, "error": "missing_clause_lookup"})
                continue
            clause = clause_lookup.loc[clause_id]
            stage_rows.append(
                {
                    **{k: ("" if pd.isna(v) else v) for k, v in clause.to_dict().items()},
                    "clause_id": clause_id,
                    "actor": clean_optional_str(payload.actor),
                    "action": clean_optional_str(payload.action),
                    "regulated_object": clean_optional_str(payload.regulated_object),
                    "condition": clean_optional_str(payload.condition),
                    "exception": clean_optional_str(payload.exception),
                    "threshold_or_quantity": clean_optional_str(payload.threshold_or_quantity),
                    "spatial_scope": clean_optional_str(payload.spatial_scope),
                    "instrument_type": payload.instrument_type,
                    "operative_flag": int(payload.operative_flag),
                    "evidence_quote": clean_optional_str(payload.evidence_quote),
                    "enrichment_confidence": float(payload.confidence),
                    "enrichment_notes": clean_optional_str(payload.notes),
                    "sample": sample,
                    "custom_id": cid,
                }
            )
        stage_df = pd.DataFrame.from_records(stage_rows).sort_values("clause_id").reset_index(drop=True) if stage_rows else pd.DataFrame()
        write_parquet(stage_df, run_root / "semantics" / f"enriched_clauses__{sample}.parquet")
        write_jsonl(run_root / "semantics" / f"enriched_clauses__{sample}.jsonl", stage_rows)
    elif stage == "section_embedding":
        rows: list[dict[str, Any]] = []
        vectors: list[list[float]] = []
        section_lookup = pd.read_parquet(run_root / "structure" / f"sections__{sample}.parquet").set_index("section_id")
        segment_rows: list[dict[str, Any]] = []
        expected_segments: dict[str, int] = {}
        for cid, mapping in mapping_by_cid.items():
            section_id = clean_optional_str(mapping["section_id"])
            expected_segments[section_id] = max(expected_segments.get(section_id, 0), _safe_int(mapping.get("segment_count", 1)))
            res = results_by_cid.get(cid)
            if res is None:
                invalid_rows.append({"custom_id": cid, "section_id": section_id, "error": "missing_result", "has_error_line": cid in errors_by_cid})
                continue
            emb = _extract_embedding((res.get("response") or {}).get("body") or {})
            if emb is None:
                invalid_rows.append({"custom_id": cid, "section_id": section_id, "error": "missing_embedding"})
                continue
            if section_id not in section_lookup.index:
                invalid_rows.append({"custom_id": cid, "section_id": section_id, "error": "missing_section_lookup"})
                continue
            segment_rows.append(
                {
                    "section_id": section_id,
                    "segment_index": _safe_int(mapping.get("segment_index", 1)),
                    "segment_count": _safe_int(mapping.get("segment_count", 1)),
                    "segment_char_count": max(1, _safe_int(mapping.get("segment_char_count", 0))),
                    "embedding": emb,
                }
            )
        segment_df = pd.DataFrame.from_records(segment_rows)
        if not segment_df.empty:
            for section_id, group in segment_df.groupby("section_id", dropna=False):
                expected = expected_segments.get(str(section_id), 0)
                observed = int(len(group))
                observed_idxs = sorted(group["segment_index"].astype(int).tolist())
                if expected and observed != expected:
                    invalid_rows.append(
                        {
                            "custom_id": f"section_embedding::{section_id}",
                            "section_id": section_id,
                            "error": "segment_count_mismatch",
                            "expected_segments": int(expected),
                            "observed_segments": int(observed),
                        }
                    )
                    continue
                if observed_idxs != list(range(1, observed + 1)):
                    invalid_rows.append(
                        {
                            "custom_id": f"section_embedding::{section_id}",
                            "section_id": section_id,
                            "error": "segment_index_invalid",
                            "observed_segment_indexes": observed_idxs,
                        }
                    )
                    continue
                section = section_lookup.loc[str(section_id)]
                row = {k: ("" if pd.isna(v) else v) for k, v in section.to_dict().items()}
                row["section_id"] = str(section_id)
                row["item_id"] = str(section_id)
                row["embedding_model"] = SECTION_EMBEDDING_MODEL
                row["embedding_source"] = "segment_weighted_mean"
                row["embedding_segment_count"] = int(observed)
                row["embedding_segment_char_count_total"] = int(group["segment_char_count"].sum())
                rows.append(row)
                seg_matrix = np.asarray(group["embedding"].tolist(), dtype=np.float32)
                weights = group["segment_char_count"].to_numpy(dtype=np.float32)
                weights = weights / weights.sum()
                vectors.append(np.average(seg_matrix, axis=0, weights=weights).astype(np.float32))
        if vectors:
            matrix = np.asarray(vectors, dtype=np.float32)
            meta_df = pd.DataFrame.from_records(rows)
            meta_df["embedding_row_index"] = np.arange(len(meta_df))
            pca_coords = _safe_pca(matrix, 10)
            umap_2d = _safe_umap(matrix, 2)
            umap_5d = _safe_umap(matrix, 5)
            out_dir = run_root / "embeddings" / "section"
            out_dir.mkdir(parents=True, exist_ok=True)
            np.save(out_dir / "embedding_matrix.npy", matrix)
            np.save(out_dir / "umap_5d.npy", umap_5d)
            write_parquet(meta_df, out_dir / "metadata.parquet")
            write_parquet(pd.DataFrame({"item_id": meta_df["item_id"], "umap_x": umap_2d[:, 0], "umap_y": umap_2d[:, 1]}), out_dir / "umap_2d.parquet")
            pca_df = pd.DataFrame({"item_id": meta_df["item_id"]})
            for idx in range(pca_coords.shape[1]):
                pca_df[f"pc{idx+1}"] = pca_coords[:, idx]
            write_parquet(pca_df, out_dir / "pca_coords.parquet")
            write_json(out_dir / "summary.json", {"item_count": int(len(meta_df)), "embedding_model": SECTION_EMBEDDING_MODEL, "sample": sample})
            stage_df = meta_df
        else:
            stage_df = pd.DataFrame()
    elif stage in {"section_cluster_label", "section_local_cluster_label", "clause_cluster_label"}:
        if stage == "section_cluster_label":
            target_experiment = SECTION_CLUSTER_EXPERIMENT
        elif stage == "section_local_cluster_label":
            target_experiment = SECTION_LOCAL_CLUSTER_EXPERIMENT
        else:
            target_experiment = CLAUSE_CLUSTER_EXPERIMENT
        target_dir = run_root / "clusters" / target_experiment
        for cid, mapping in mapping_by_cid.items():
            res = results_by_cid.get(cid)
            if res is None:
                invalid_rows.append({"custom_id": cid, "cluster_id": mapping["cluster_id"], "error": "missing_result", "has_error_line": cid in errors_by_cid})
                continue
            text = _extract_output_text((res.get("response") or {}).get("body") or {})
            if not text:
                invalid_rows.append({"custom_id": cid, "cluster_id": mapping["cluster_id"], "error": "empty_output_text"})
                continue
            try:
                payload = ClusterLabelOut.model_validate(_coerce_json(text))
            except Exception as exc:
                invalid_rows.append({"custom_id": cid, "cluster_id": mapping["cluster_id"], "error": "validation_failed", "message": str(exc), "raw_text": text[:4000]})
                continue
            cluster_id = _safe_int(mapping["cluster_id"])
            if int(payload.cluster_id) != cluster_id:
                invalid_rows.append({"custom_id": cid, "cluster_id": cluster_id, "error": "cluster_id_mismatch", "model_cluster_id": payload.cluster_id})
                continue
            stage_rows.append(
                {
                    "cluster_id": cluster_id,
                    "cluster_label": payload.cluster_label,
                    "cluster_label_slug": slugify(payload.cluster_label),
                    "cluster_description": payload.cluster_description,
                    "keywords": "|".join(payload.keywords),
                    "label_taxonomy": "semantic" if stage in {"section_cluster_label", "section_local_cluster_label"} else "instrument",
                    "custom_id": cid,
                    "sample": sample,
                    "notes": payload.notes,
                }
            )
        stage_df = pd.DataFrame.from_records(stage_rows).sort_values("cluster_id").reset_index(drop=True) if stage_rows else pd.DataFrame()
        if not stage_df.empty:
            write_parquet(stage_df, target_dir / "cluster_labels_llm.parquet")
        if stage_df.empty and not allow_errors:
            raise SystemExit(f"No cluster labels rehydrated for stage={stage}")
    else:
        raise SystemExit(f"Unsupported rehydrate stage: {stage}")

    invalid_path = run_root / "results" / sample / stage / "invalid_rows.jsonl"
    _write_invalid_rows(invalid_path, invalid_rows)
    if manifest_rows:
        write_jsonl(run_root / "results" / sample / stage / "manifest.jsonl", manifest_rows)
    ok_count = int(len(stage_rows))
    if stage == "section_segmentation":
        ok_count = int(len(manifest_rows))
    elif stage == "section_embedding":
        ok_count = int(len(stage_df))
    summary = {
        "stage": stage,
        "sample": sample,
        "ok_count": ok_count,
        "invalid_count": int(len(invalid_rows)),
        "allow_partial_results": bool(allow_partial_results),
    }
    write_json(run_root / "results" / sample / stage / "rehydrate_summary.json", summary)
    if invalid_rows and not allow_errors:
        raise SystemExit(f"Rehydration had failures for stage={stage} sample={sample}: invalid_count={len(invalid_rows)}")
    return summary


def build_llm_v2_semantics(*, run_root: Path, sample: Literal["pilot", "full"]) -> dict[str, Any]:
    docs = _sample_docs(run_root, sample)
    purposes_path = run_root / "semantics" / f"document_purposes__{sample}.parquet"
    motives_path = run_root / "semantics" / f"section_motives__{sample}.parquet"
    clauses_path = run_root / "semantics" / f"clauses__{sample}.parquet"
    enriched_path = run_root / "semantics" / f"enriched_clauses__{sample}.parquet"
    sections_path = run_root / "structure" / f"sections__{sample}.parquet"
    for path in (purposes_path, motives_path, clauses_path, enriched_path, sections_path):
        if not path.is_file():
            raise SystemExit(f"Missing required semantic artifact: {path}")
    purposes = pd.read_parquet(purposes_path)
    section_motives = pd.read_parquet(motives_path)
    clauses = pd.read_parquet(clauses_path)
    enriched = pd.read_parquet(enriched_path)
    sections = pd.read_parquet(sections_path)

    enriched = enriched.merge(
        section_motives[["section_id", "primary_motive", "confidence"]].rename(columns={"primary_motive": "section_primary_motive", "confidence": "section_motive_confidence"}),
        on="section_id",
        how="left",
    )
    clause_semantics = enriched.copy()
    clause_semantics["clause_role"] = np.where(
        clause_semantics["instrument_type"].astype(str).eq("definition"),
        "definition",
        np.where(
            clause_semantics["instrument_type"].astype(str).eq("purpose"),
            "purpose_preamble",
            np.where(clause_semantics["operative_flag"].astype(int).eq(1), "substantive_or_administrative", "non_operative"),
        ),
    )
    clause_semantics["operative_clause_flag"] = clause_semantics["operative_flag"].astype(int)
    operative_clause_semantics = clause_semantics[clause_semantics["operative_clause_flag"].astype(int) == 1].reset_index(drop=True)

    feature_rows: list[dict[str, Any]] = []
    for row in clause_semantics.itertuples(index=False):
        base = {
            "ordinance_doc_id": clean_optional_str(getattr(row, "ordinance_doc_id", "")),
            "section_id": clean_optional_str(getattr(row, "section_id", "")),
            "clause_id": clean_optional_str(getattr(row, "clause_id", "")),
            "instrument_type": clean_optional_str(getattr(row, "instrument_type", "")),
            "sample": sample,
            "operative_clause_flag": _safe_int(getattr(row, "operative_clause_flag", 0)),
        }
        if clean_optional_str(getattr(row, "condition", "")):
            feature_rows.append({**base, "feature_name": "has_condition", "feature_value": 1.0})
        if clean_optional_str(getattr(row, "exception", "")):
            feature_rows.append({**base, "feature_name": "has_exception", "feature_value": 1.0})
        if clean_optional_str(getattr(row, "threshold_or_quantity", "")):
            feature_rows.append({**base, "feature_name": "has_threshold", "feature_value": 1.0})
        if clean_optional_str(getattr(row, "spatial_scope", "")):
            feature_rows.append({**base, "feature_name": "has_spatial_scope", "feature_value": 1.0})
        actor = clean_optional_str(getattr(row, "actor", ""))
        if actor:
            feature_rows.append({**base, "feature_name": f"actor__{slugify(actor)}", "feature_value": 1.0})
        instrument = clean_optional_str(getattr(row, "instrument_type", ""))
        if instrument:
            feature_rows.append({**base, "feature_name": f"instrument_attr__{instrument}", "feature_value": 1.0})
    feature_df = pd.DataFrame.from_records(feature_rows).drop_duplicates(["clause_id", "feature_name"]) if feature_rows else pd.DataFrame(columns=["ordinance_doc_id", "section_id", "clause_id", "instrument_type", "sample", "operative_clause_flag", "feature_name", "feature_value"])
    operative_feature_df = feature_df[feature_df["operative_clause_flag"].astype(int) == 1].reset_index(drop=True) if not feature_df.empty else feature_df.copy()

    purpose_grouped = purposes.groupby("ordinance_doc_id", dropna=False)
    section_grouped = sections.merge(
        section_motives[["section_id", "primary_motive", "secondary_motives", "evidence", "confidence"]],
        on="section_id",
        how="left",
    ).groupby("ordinance_doc_id", dropna=False)
    clause_grouped = clause_semantics.groupby("ordinance_doc_id", dropna=False)
    doc_rows: list[dict[str, Any]] = []
    purpose_rows: list[dict[str, Any]] = []
    for row in docs.itertuples(index=False):
        doc_id = clean_optional_str(getattr(row, "ordinance_doc_id"))
        doc_purpose = purpose_grouped.get_group(doc_id) if doc_id in purpose_grouped.groups else pd.DataFrame(columns=purposes.columns)
        doc_sections = section_grouped.get_group(doc_id) if doc_id in section_grouped.groups else pd.DataFrame(columns=["primary_motive", "word_count"])
        doc_clauses = clause_grouped.get_group(doc_id) if doc_id in clause_grouped.groups else pd.DataFrame(columns=clause_semantics.columns)
        operative_doc_clauses = doc_clauses[doc_clauses["operative_clause_flag"].astype(int) == 1].copy() if not doc_clauses.empty else doc_clauses.copy()

        purpose_rows.extend(doc_purpose.to_dict(orient="records"))
        if not doc_purpose.empty:
            top_purpose = doc_purpose.sort_values(["confidence", "motive"], ascending=[False, True]).iloc[0]
            document_primary_motive = clean_optional_str(top_purpose["motive"])
            document_primary_confidence = float(top_purpose["confidence"])
            purpose_notes = "|".join(sorted(set(doc_purpose["description"].astype(str).head(4).tolist())))
        else:
            document_primary_motive = "unclear"
            document_primary_confidence = 0.0
            purpose_notes = ""

        section_weighted: Counter[str] = Counter()
        for sec in doc_sections.itertuples(index=False):
            motive = clean_optional_str(getattr(sec, "primary_motive", "")) or "unclear"
            weight = max(1, _safe_int(getattr(sec, "word_count", 0), 1))
            section_weighted[motive] += weight
        section_primary_motive_weighted = section_weighted.most_common(1)[0][0] if section_weighted else "unclear"
        filtered_section_weighted = Counter({k: v for k, v in section_weighted.items() if k not in {"other", "unclear"}})
        section_primary_motive_weighted_excluding_other_unclear = filtered_section_weighted.most_common(1)[0][0] if filtered_section_weighted else section_primary_motive_weighted

        clause_motive_distribution = clause_semantics.loc[clause_semantics["ordinance_doc_id"].astype(str) == doc_id, "motive"].value_counts().to_dict()
        operative_clause_motive_distribution = operative_doc_clauses["motive"].value_counts().to_dict() if not operative_doc_clauses.empty else {}
        dominant_clause_motive = next(iter(operative_clause_motive_distribution.keys()), next(iter(clause_motive_distribution.keys()), "unclear"))
        dominant_instrument = clean_optional_str(doc_clauses["instrument_type"].mode().iloc[0]) if not doc_clauses.empty and not doc_clauses["instrument_type"].mode().empty else ""
        operative_dominant_instrument = clean_optional_str(operative_doc_clauses["instrument_type"].mode().iloc[0]) if not operative_doc_clauses.empty and not operative_doc_clauses["instrument_type"].mode().empty else dominant_instrument
        doc_rows.append(
            {
                "ordinance_doc_id": doc_id,
                "issue_id": clean_optional_str(getattr(row, "issue_id")),
                "census_id_pid6": clean_optional_str(getattr(row, "census_id_pid6")),
                "jurisdiction_region": clean_optional_str(getattr(row, "jurisdiction_region")),
                "jurisdiction_level": clean_optional_str(getattr(row, "jurisdiction_level")),
                "primary_motive": document_primary_motive,
                "primary_motive_confidence": float(document_primary_confidence),
                "document_purpose_present": int(not doc_purpose.empty),
                "document_purpose_count": int(len(doc_purpose)),
                "section_count": int(len(doc_sections)),
                "clause_count": int(len(doc_clauses)),
                "operative_clause_count": int(len(operative_doc_clauses)),
                "must_clause_count": int((doc_clauses["modality"].astype(str) == "must").sum()) if not doc_clauses.empty else 0,
                "must_not_clause_count": int((doc_clauses["modality"].astype(str) == "must_not").sum()) if not doc_clauses.empty else 0,
                "definition_clause_count": int((doc_clauses["instrument_type"].astype(str) == "definition").sum()) if not doc_clauses.empty else 0,
                "operative_must_clause_count": int((operative_doc_clauses["modality"].astype(str) == "must").sum()) if not operative_doc_clauses.empty else 0,
                "operative_must_not_clause_count": int((operative_doc_clauses["modality"].astype(str) == "must_not").sum()) if not operative_doc_clauses.empty else 0,
                "dominant_instrument": dominant_instrument,
                "operative_dominant_instrument": operative_dominant_instrument,
                "section_primary_motive_weighted": section_primary_motive_weighted,
                "section_primary_motive_weighted_excluding_other_unclear": section_primary_motive_weighted_excluding_other_unclear,
                "dominant_clause_motive": dominant_clause_motive,
                "flag_document_vs_section_mismatch": int(document_primary_motive not in {"", "unclear"} and section_primary_motive_weighted_excluding_other_unclear not in {"", "unclear"} and document_primary_motive != section_primary_motive_weighted_excluding_other_unclear),
                "flag_section_vs_clause_mismatch": int(section_primary_motive_weighted_excluding_other_unclear not in {"", "unclear"} and dominant_clause_motive not in {"", "unclear"} and section_primary_motive_weighted_excluding_other_unclear != dominant_clause_motive),
                "section_motive_weighted_share_json": json.dumps(dict(section_weighted), sort_keys=True),
                "clause_motive_distribution_json": json.dumps(clause_motive_distribution, sort_keys=True),
                "operative_clause_motive_distribution_json": json.dumps(operative_clause_motive_distribution, sort_keys=True),
                "document_purpose_notes": purpose_notes,
                "sample": sample,
            }
        )

    document_semantics = pd.DataFrame.from_records(doc_rows).sort_values("ordinance_doc_id").reset_index(drop=True) if doc_rows else pd.DataFrame()
    semantics_dir = run_root / "semantics"
    write_parquet(document_semantics, semantics_dir / "document_semantics.parquet")
    write_parquet(pd.DataFrame.from_records(purpose_rows).sort_values(["ordinance_doc_id", "confidence"], ascending=[True, False]).reset_index(drop=True) if purpose_rows else pd.DataFrame(), semantics_dir / "document_purposes.parquet")
    write_parquet(section_motives.sort_values("section_id").reset_index(drop=True), semantics_dir / "section_motives.parquet")
    write_parquet(clause_semantics.sort_values("clause_id").reset_index(drop=True), semantics_dir / "clause_semantics.parquet")
    write_parquet(operative_clause_semantics.sort_values("clause_id").reset_index(drop=True), semantics_dir / "operative_clause_semantics.parquet")
    write_parquet(feature_df.sort_values(["ordinance_doc_id", "clause_id", "feature_name"]).reset_index(drop=True), semantics_dir / "hard_feature_observations.parquet")
    write_parquet(operative_feature_df.sort_values(["ordinance_doc_id", "clause_id", "feature_name"]).reset_index(drop=True), semantics_dir / "operative_hard_feature_observations.parquet")
    summary = {
        "sample": sample,
        "document_rows": int(len(document_semantics)),
        "document_purpose_rows": int(len(purposes)),
        "section_rows": int(len(section_motives)),
        "clause_rows": int(len(clause_semantics)),
        "operative_clause_rows": int(len(operative_clause_semantics)),
        "feature_rows": int(len(feature_df)),
        "operative_feature_rows": int(len(operative_feature_df)),
        "top_primary_motives": document_semantics["primary_motive"].value_counts().head(10).to_dict() if not document_semantics.empty else {},
    }
    write_json(semantics_dir / "semantics_summary.json", summary)
    return summary


def _section_ids_for_sample(run_root: Path, sample: Literal["pilot", "full"]) -> set[str]:
    sections = pd.read_parquet(run_root / "structure" / f"sections__{sample}.parquet")
    return set(sections["section_id"].astype(str).tolist())


def build_llm_v2_embeddings(*, run_root: Path, sample: Literal["pilot", "full"]) -> dict[str, Any]:
    section_dir = run_root / "embeddings" / "section"
    if not (section_dir / "metadata.parquet").is_file():
        raise SystemExit(f"Missing OpenAI section embeddings under {section_dir}")
    section_meta = pd.read_parquet(section_dir / "metadata.parquet")
    section_matrix = np.load(section_dir / "embedding_matrix.npy")
    sample_section_ids = _section_ids_for_sample(run_root, sample)
    if sample == "pilot":
        keep_mask = section_meta["section_id"].astype(str).isin(sample_section_ids).to_numpy()
        section_meta = section_meta.loc[keep_mask].reset_index(drop=True)
        section_matrix = section_matrix[keep_mask]
        section_meta["embedding_row_index"] = np.arange(len(section_meta))
        np.save(section_dir / "embedding_matrix.npy", section_matrix)
        write_parquet(section_meta, section_dir / "metadata.parquet")
        umap_2d = _safe_umap(section_matrix, 2)
        umap_5d = _safe_umap(section_matrix, 5)
        pca_coords = _safe_pca(section_matrix, 10)
        np.save(section_dir / "umap_5d.npy", umap_5d)
        write_parquet(pd.DataFrame({"item_id": section_meta["item_id"], "umap_x": umap_2d[:, 0], "umap_y": umap_2d[:, 1]}), section_dir / "umap_2d.parquet")
        pca_df = pd.DataFrame({"item_id": section_meta["item_id"]})
        for idx in range(pca_coords.shape[1]):
            pca_df[f"pc{idx+1}"] = pca_coords[:, idx]
        write_parquet(pca_df, section_dir / "pca_coords.parquet")

    # Local section baseline.
    local_dir = run_root / "embeddings" / "section_local_baseline"
    local_dir.mkdir(parents=True, exist_ok=True)
    section_texts = [clean_optional_str(x) for x in section_meta["text"].tolist()]
    local_matrix, local_meta = _dense_embedding_matrix(section_texts)
    local_umap_2d = _safe_umap(local_matrix, 2)
    local_umap_5d = _safe_umap(local_matrix, 5)
    local_pca = _safe_pca(local_matrix, 10)
    np.save(local_dir / "embedding_matrix.npy", local_matrix)
    np.save(local_dir / "umap_5d.npy", local_umap_5d)
    local_meta_df = section_meta.copy()
    local_meta_df["embedding_row_index"] = np.arange(len(local_meta_df))
    write_parquet(local_meta_df, local_dir / "metadata.parquet")
    write_parquet(pd.DataFrame({"item_id": local_meta_df["item_id"], "umap_x": local_umap_2d[:, 0], "umap_y": local_umap_2d[:, 1]}), local_dir / "umap_2d.parquet")
    local_pca_df = pd.DataFrame({"item_id": local_meta_df["item_id"]})
    for idx in range(local_pca.shape[1]):
        local_pca_df[f"pc{idx+1}"] = local_pca[:, idx]
    write_parquet(local_pca_df, local_dir / "pca_coords.parquet")
    write_json(local_dir / "summary.json", {"item_count": int(len(local_meta_df)), **local_meta})

    # Ordinance embeddings = mean of section embeddings.
    ord_dir = run_root / "embeddings" / "ordinance"
    ord_dir.mkdir(parents=True, exist_ok=True)
    ordinance_vectors: list[np.ndarray] = []
    ordinance_rows: list[dict[str, Any]] = []
    ordinances = pd.read_parquet(run_root / "corpus" / "ordinances.parquet")
    ordinances = ordinances[ordinances["ordinance_doc_id"].astype(str).isin(section_meta["ordinance_doc_id"].astype(str))].copy()
    section_group = section_meta.groupby("ordinance_doc_id", dropna=False)
    for ordinance_doc_id, group in section_group:
        idx = group["embedding_row_index"].to_numpy(dtype=int)
        ordinance_vectors.append(section_matrix[idx].mean(axis=0))
        base = ordinances.loc[ordinances["ordinance_doc_id"].astype(str) == str(ordinance_doc_id)].head(1)
        row = base.iloc[0].to_dict() if not base.empty else {}
        row.update({"item_id": clean_optional_str(ordinance_doc_id), "ordinance_doc_id": clean_optional_str(ordinance_doc_id)})
        ordinance_rows.append({k: ("" if pd.isna(v) else v) for k, v in row.items()})
    ordinance_matrix = np.asarray(ordinance_vectors, dtype=np.float32) if ordinance_vectors else np.zeros((0, 0), dtype=np.float32)
    ordinance_meta = pd.DataFrame.from_records(ordinance_rows)
    ordinance_meta["embedding_row_index"] = np.arange(len(ordinance_meta))
    np.save(ord_dir / "embedding_matrix.npy", ordinance_matrix)
    np.save(ord_dir / "umap_5d.npy", _safe_umap(ordinance_matrix, 5))
    write_parquet(ordinance_meta, ord_dir / "metadata.parquet")
    ord_umap_2d = _safe_umap(ordinance_matrix, 2)
    write_parquet(pd.DataFrame({"item_id": ordinance_meta["item_id"], "umap_x": ord_umap_2d[:, 0], "umap_y": ord_umap_2d[:, 1]}), ord_dir / "umap_2d.parquet")
    ord_pca = _safe_pca(ordinance_matrix, 10)
    ord_pca_df = pd.DataFrame({"item_id": ordinance_meta["item_id"]})
    for idx in range(ord_pca.shape[1]):
        ord_pca_df[f"pc{idx+1}"] = ord_pca[:, idx]
    write_parquet(ord_pca_df, ord_dir / "pca_coords.parquet")
    write_json(ord_dir / "summary.json", {"item_count": int(len(ordinance_meta)), "embedding_model": f"mean({SECTION_EMBEDDING_MODEL})"})

    clause_dir = run_root / "embeddings" / "clause"
    clause_item_count = 0
    operative_clause_path = run_root / "semantics" / "operative_clause_semantics.parquet"
    enriched_sample_path = run_root / "semantics" / f"enriched_clauses__{sample}.parquet"
    clause_surface_available = operative_clause_path.is_file() and (
        sample == "pilot" or enriched_sample_path.is_file()
    )
    if clause_surface_available:
        clause_dir.mkdir(parents=True, exist_ok=True)
        clauses = pd.read_parquet(operative_clause_path)
        if sample == "pilot":
            sample_doc_ids = set(_sample_docs(run_root, sample)["ordinance_doc_id"].astype(str))
            clauses = clauses[clauses["ordinance_doc_id"].astype(str).isin(sample_doc_ids)].copy()
        clause_texts = clauses["requirement"].astype(str).tolist()
        clause_matrix, clause_meta = _dense_embedding_matrix(clause_texts)
        clause_umap_2d = _safe_umap(clause_matrix, 2)
        clause_umap_5d = _safe_umap(clause_matrix, 5)
        clause_pca = _safe_pca(clause_matrix, 10)
        np.save(clause_dir / "embedding_matrix.npy", clause_matrix)
        np.save(clause_dir / "umap_5d.npy", clause_umap_5d)
        clause_meta_df = clauses.copy()
        clause_meta_df["item_id"] = clause_meta_df["clause_id"].astype(str)
        clause_meta_df["embedding_row_index"] = np.arange(len(clause_meta_df))
        clause_meta_df["clause_embedding_source"] = "llm_enriched_requirement_tfidf_svd"
        write_parquet(clause_meta_df, clause_dir / "metadata.parquet")
        write_parquet(pd.DataFrame({"item_id": clause_meta_df["item_id"], "umap_x": clause_umap_2d[:, 0], "umap_y": clause_umap_2d[:, 1]}), clause_dir / "umap_2d.parquet")
        clause_pca_df = pd.DataFrame({"item_id": clause_meta_df["item_id"]})
        for idx in range(clause_pca.shape[1]):
            clause_pca_df[f"pc{idx+1}"] = clause_pca[:, idx]
        write_parquet(clause_pca_df, clause_dir / "pca_coords.parquet")
        write_json(clause_dir / "summary.json", {"item_count": int(len(clause_meta_df)), "embedding_model": "tfidf_svd_on_enriched_requirements", **clause_meta})
        clause_item_count = int(len(clause_meta_df))

    comparison = {
        "section_openai_item_count": int(len(section_meta)),
        "section_local_item_count": int(len(local_meta_df)),
        "section_embedding_model": SECTION_EMBEDDING_MODEL,
        "section_local_vectorizer": local_meta.get("vectorizer", ""),
        "clause_item_count": int(clause_item_count),
        "ordinance_item_count": int(len(ordinance_meta)),
    }
    write_json(run_root / "embeddings" / "embedding_comparison_summary.json", comparison)
    return comparison


def _placeholder_cluster_labels(assign_df: pd.DataFrame, label_taxonomy: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for cluster_id, group in assign_df.groupby("cluster_id", dropna=False):
        rows.append(
            {
                "cluster_id": int(cluster_id),
                "cluster_size": int(len(group)),
                "cluster_label": f"cluster {int(cluster_id)}" if int(cluster_id) >= 0 else "noise",
                "cluster_label_slug": f"cluster-{int(cluster_id)}" if int(cluster_id) >= 0 else "noise",
                "cluster_description": "",
                "keywords": "",
                "label_taxonomy": label_taxonomy,
            }
        )
    return pd.DataFrame.from_records(rows).sort_values("cluster_id").reset_index(drop=True)


def _cluster_matrix_hdbscan(matrix: np.ndarray, min_cluster_size: int) -> np.ndarray:
    if matrix.shape[0] <= 3:
        return np.zeros((matrix.shape[0],), dtype=int)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=max(2, int(min_cluster_size)), min_samples=max(1, int(min_cluster_size // 3)))
    return clusterer.fit_predict(matrix).astype(int)


def _write_cluster_outputs(
    *,
    run_root: Path,
    experiment_name: str,
    representation_level: str,
    matrix: np.ndarray,
    meta_df: pd.DataFrame,
    feature_matrix: np.ndarray,
    label_taxonomy: str,
    clusterer: str,
    min_cluster_size: int,
) -> dict[str, Any]:
    out_dir = run_root / "clusters" / experiment_name
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = _cluster_matrix_hdbscan(feature_matrix, min_cluster_size)
    assign_df = meta_df.copy()
    assign_df["cluster_id"] = labels
    assign_df["row_index"] = np.arange(len(assign_df))
    cluster_df = _placeholder_cluster_labels(assign_df, label_taxonomy)
    assign_df = assign_df.merge(cluster_df, on="cluster_id", how="left")
    exemplar_df, outlier_df = _assign_exemplars(assign_df, matrix)
    write_parquet(assign_df, out_dir / "assignments.parquet")
    write_parquet(cluster_df, out_dir / "cluster_labels.parquet")
    write_parquet(exemplar_df, out_dir / "cluster_exemplars.parquet")
    write_parquet(outlier_df, out_dir / "cluster_outliers.parquet")
    sweep_df = pd.DataFrame.from_records(
        [
            {
                "parameter_name": "hdbscan_min_cluster_size",
                "parameter_value": int(min_cluster_size),
                "cluster_count": int(len([x for x in assign_df["cluster_id"].unique().tolist() if int(x) >= 0])),
                "noise_count": int((assign_df["cluster_id"].astype(int) < 0).sum()),
                "ari_vs_baseline": 1.0,
            }
        ]
    )
    write_parquet(sweep_df, out_dir / "stability_sweep.parquet")
    write_json(
        out_dir / "summary.json",
        {
            "experiment_name": experiment_name,
            "representation_level": representation_level,
            "labeler": label_taxonomy,
            "clusterer": clusterer,
            "item_count": int(len(assign_df)),
            "cluster_count": int(len([x for x in assign_df["cluster_id"].unique().tolist() if int(x) >= 0])),
            "noise_count": int((assign_df["cluster_id"].astype(int) < 0).sum()),
        },
    )
    return {
        "experiment_name": experiment_name,
        "representation_level": representation_level,
        "labeler": label_taxonomy,
        "clusterer": clusterer,
        "item_count": int(len(assign_df)),
        "cluster_count": int(len([x for x in assign_df["cluster_id"].unique().tolist() if int(x) >= 0])),
        "noise_count": int((assign_df["cluster_id"].astype(int) < 0).sum()),
    }


def run_llm_v2_clusters(*, run_root: Path, sample: Literal["pilot", "full"] | None = None) -> dict[str, Any]:
    summaries: list[dict[str, Any]] = []
    if (run_root / "embeddings" / "section" / "metadata.parquet").is_file():
        section_meta = pd.read_parquet(run_root / "embeddings" / "section" / "metadata.parquet")
        section_matrix = np.load(run_root / "embeddings" / "section" / "embedding_matrix.npy")
        section_umap = np.load(run_root / "embeddings" / "section" / "umap_5d.npy")
        summaries.append(
            _write_cluster_outputs(
                run_root=run_root,
                experiment_name=SECTION_CLUSTER_EXPERIMENT,
                representation_level="section",
                matrix=section_matrix,
                meta_df=section_meta,
                feature_matrix=section_umap,
                label_taxonomy="semantic",
                clusterer="hdbscan",
                min_cluster_size=SECTION_CLUSTER_MIN_SIZE,
            )
        )
    if (run_root / "embeddings" / "section_local_baseline" / "metadata.parquet").is_file():
        local_section_meta = pd.read_parquet(run_root / "embeddings" / "section_local_baseline" / "metadata.parquet")
        local_section_matrix = np.load(run_root / "embeddings" / "section_local_baseline" / "embedding_matrix.npy")
        local_section_umap = np.load(run_root / "embeddings" / "section_local_baseline" / "umap_5d.npy")
        summaries.append(
            _write_cluster_outputs(
                run_root=run_root,
                experiment_name=SECTION_LOCAL_CLUSTER_EXPERIMENT,
                representation_level="section_local_baseline",
                matrix=local_section_matrix,
                meta_df=local_section_meta,
                feature_matrix=local_section_umap,
                label_taxonomy="semantic",
                clusterer="hdbscan",
                min_cluster_size=SECTION_CLUSTER_MIN_SIZE,
            )
        )
    clause_surface_available = True
    if sample == "full":
        clause_surface_available = (run_root / "semantics" / "enriched_clauses__full.parquet").is_file()
    elif sample == "pilot":
        clause_surface_available = (run_root / "semantics" / "enriched_clauses__pilot.parquet").is_file()
    if clause_surface_available and (run_root / "embeddings" / "clause" / "metadata.parquet").is_file():
        clause_meta = pd.read_parquet(run_root / "embeddings" / "clause" / "metadata.parquet")
        clause_matrix = np.load(run_root / "embeddings" / "clause" / "embedding_matrix.npy")
        clause_umap = np.load(run_root / "embeddings" / "clause" / "umap_5d.npy")
        summaries.append(
            _write_cluster_outputs(
                run_root=run_root,
                experiment_name=CLAUSE_CLUSTER_EXPERIMENT,
                representation_level="clause",
                matrix=clause_matrix,
                meta_df=clause_meta,
                feature_matrix=clause_umap,
                label_taxonomy="instrument",
                clusterer="hdbscan",
                min_cluster_size=CLAUSE_CLUSTER_MIN_SIZE,
            )
        )
    if not summaries:
        raise SystemExit(f"No embedding artifacts found under {run_root / 'embeddings'} for clustering.")
    index_df = pd.DataFrame.from_records(summaries)
    write_parquet(index_df, run_root / "clusters" / "experiment_index.parquet")
    write_json(run_root / "clusters" / "clusters_summary.json", {"experiments": summaries})
    return {"experiments": summaries}


def _cluster_examples_payload(assignments: pd.DataFrame, *, text_field: str, extra_count_fields: list[str], experiment_name: str, representation_level: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for cluster_id, group in assignments.groupby("cluster_id", dropna=False):
        cluster_id_int = int(cluster_id)
        if cluster_id_int < 0 or len(group) < REPORTABLE_CLUSTER_MIN_SIZE:
            continue
        exemplars = group.sort_values(["cluster_size", "item_id"], ascending=[False, True]).head(8)
        texts = "\n\n<<<NEXT>>>\n\n".join(clean_optional_str(x)[:1800] for x in exemplars[text_field].tolist())
        base = {
            "cluster_id": cluster_id_int,
            "cluster_size": int(len(group)),
            "experiment_name": experiment_name,
            "representation_level": representation_level,
            "representative_texts": texts,
            "motive_counts": "|".join(f"{k}:{v}" for k, v in group.get("primary_motive", pd.Series(dtype=str)).astype(str).value_counts().head(8).items()),
        }
        if "title" in group.columns:
            base["section_title_counts"] = "|".join(f"{k}:{v}" for k, v in group["title"].astype(str).value_counts().head(8).items())
        if "instrument_type" in group.columns:
            base["instrument_type_counts"] = "|".join(f"{k}:{v}" for k, v in group["instrument_type"].astype(str).value_counts().head(8).items())
        if "modality" in group.columns:
            base["modality_counts"] = "|".join(f"{k}:{v}" for k, v in group["modality"].astype(str).value_counts().head(8).items())
        rows.append(base)
    return pd.DataFrame.from_records(rows)


def build_cluster_label_request_sources(*, run_root: Path) -> dict[str, Any]:
    summary = {"section_cluster_request_count": 0, "section_local_cluster_request_count": 0, "clause_cluster_request_count": 0}
    section_assign_path = run_root / "clusters" / SECTION_CLUSTER_EXPERIMENT / "assignments.parquet"
    section_local_assign_path = run_root / "clusters" / SECTION_LOCAL_CLUSTER_EXPERIMENT / "assignments.parquet"
    clause_assign_path = run_root / "clusters" / CLAUSE_CLUSTER_EXPERIMENT / "assignments.parquet"
    wrote_any = False
    if section_assign_path.is_file():
        section_assign = pd.read_parquet(section_assign_path)
        section_df = _cluster_examples_payload(
            section_assign,
            text_field="text",
            extra_count_fields=["title"],
            experiment_name=SECTION_CLUSTER_EXPERIMENT,
            representation_level="section",
        )
        write_parquet(section_df, run_root / "clusters" / SECTION_CLUSTER_EXPERIMENT / "cluster_label_request_source.parquet")
        summary["section_cluster_request_count"] = int(len(section_df))
        wrote_any = True
    elif (run_root / "clusters" / "experiment_index.parquet").is_file():
        write_parquet(
            pd.DataFrame(columns=["cluster_id", "cluster_size", "experiment_name", "representation_level", "motive_counts", "representative_texts", "section_title_counts"]),
            run_root / "clusters" / SECTION_CLUSTER_EXPERIMENT / "cluster_label_request_source.parquet",
        )
    if section_local_assign_path.is_file():
        section_local_assign = pd.read_parquet(section_local_assign_path)
        section_local_df = _cluster_examples_payload(
            section_local_assign,
            text_field="text",
            extra_count_fields=["title"],
            experiment_name=SECTION_LOCAL_CLUSTER_EXPERIMENT,
            representation_level="section",
        )
        write_parquet(section_local_df, run_root / "clusters" / SECTION_LOCAL_CLUSTER_EXPERIMENT / "cluster_label_request_source.parquet")
        summary["section_local_cluster_request_count"] = int(len(section_local_df))
        wrote_any = True
    elif (run_root / "clusters" / "experiment_index.parquet").is_file():
        write_parquet(
            pd.DataFrame(columns=["cluster_id", "cluster_size", "experiment_name", "representation_level", "motive_counts", "representative_texts", "section_title_counts"]),
            run_root / "clusters" / SECTION_LOCAL_CLUSTER_EXPERIMENT / "cluster_label_request_source.parquet",
        )
    if clause_assign_path.is_file():
        clause_assign = pd.read_parquet(clause_assign_path)
        clause_df = _cluster_examples_payload(
            clause_assign,
            text_field="requirement",
            extra_count_fields=["instrument_type", "modality"],
            experiment_name=CLAUSE_CLUSTER_EXPERIMENT,
            representation_level="clause",
        )
        write_parquet(clause_df, run_root / "clusters" / CLAUSE_CLUSTER_EXPERIMENT / "cluster_label_request_source.parquet")
        summary["clause_cluster_request_count"] = int(len(clause_df))
        wrote_any = True
    elif (run_root / "clusters" / "experiment_index.parquet").is_file():
        write_parquet(
            pd.DataFrame(columns=["cluster_id", "cluster_size", "experiment_name", "representation_level", "motive_counts", "representative_texts", "instrument_type_counts", "modality_counts"]),
            run_root / "clusters" / CLAUSE_CLUSTER_EXPERIMENT / "cluster_label_request_source.parquet",
        )
    if not wrote_any and not (run_root / "clusters" / "experiment_index.parquet").is_file():
        raise SystemExit(f"No cluster assignments found under {run_root / 'clusters'} for request-source generation.")
    return summary


def _cluster_label_source_df(run_root: Path, experiment_name: str) -> pd.DataFrame:
    path = run_root / "clusters" / experiment_name / "cluster_label_request_source.parquet"
    if not path.is_file():
        raise SystemExit(f"Missing cluster label request source: {path}. Run build_cluster_label_request_sources first.")
    return pd.read_parquet(path)


def apply_llm_cluster_labels(*, run_root: Path) -> dict[str, Any]:
    updated: dict[str, int] = {}
    for experiment_name in (SECTION_CLUSTER_EXPERIMENT, SECTION_LOCAL_CLUSTER_EXPERIMENT, CLAUSE_CLUSTER_EXPERIMENT):
        cluster_dir = run_root / "clusters" / experiment_name
        llm_path = cluster_dir / "cluster_labels_llm.parquet"
        base_path = cluster_dir / "cluster_labels.parquet"
        assign_path = cluster_dir / "assignments.parquet"
        if not llm_path.is_file():
            continue
        llm_df = pd.read_parquet(llm_path)
        llm_keep_cols = [
            col
            for col in (
                "cluster_id",
                "cluster_label",
                "cluster_label_slug",
                "cluster_description",
                "keywords",
                "label_taxonomy",
            )
            if col in llm_df.columns
        ]
        if "cluster_id" not in llm_keep_cols:
            continue
        llm_df = llm_df[llm_keep_cols].drop_duplicates(subset=["cluster_id"], keep="first")
        base_df = pd.read_parquet(base_path)
        assign_df = pd.read_parquet(assign_path)
        keep_cols = [c for c in base_df.columns if c not in {"cluster_label", "cluster_label_slug", "cluster_description", "keywords", "label_taxonomy"}]
        merged = base_df[keep_cols].merge(llm_df, on="cluster_id", how="left")
        for col in ("cluster_label", "cluster_label_slug", "cluster_description", "keywords", "label_taxonomy"):
            if col not in merged.columns:
                merged[col] = ""
        merged["cluster_label"] = merged["cluster_label"].fillna(base_df["cluster_label"])
        merged["cluster_label_slug"] = merged["cluster_label_slug"].fillna(base_df["cluster_label_slug"])
        merged["cluster_description"] = merged["cluster_description"].fillna("")
        merged["keywords"] = merged["keywords"].fillna("")
        merged["label_taxonomy"] = merged["label_taxonomy"].fillna(base_df["label_taxonomy"])
        write_parquet(merged, base_path)
        assign_df = assign_df.drop(columns=[c for c in ("cluster_label", "cluster_label_slug", "cluster_description", "keywords", "label_taxonomy") if c in assign_df.columns]).merge(
            merged[["cluster_id", "cluster_label", "cluster_label_slug", "cluster_description", "keywords", "label_taxonomy"]],
            on="cluster_id",
            how="left",
        )
        write_parquet(assign_df, assign_path)
        updated[experiment_name] = int(len(llm_df))
    write_json(run_root / "clusters" / "llm_cluster_labels_applied.json", updated)
    return updated


def build_llm_v2_review_packets(*, run_root: Path, sample: Literal["pilot", "full"]) -> dict[str, Any]:
    review_dir = run_root / "review"
    review_dir.mkdir(parents=True, exist_ok=True)
    rnd = random.Random(0)

    sections = pd.read_parquet(run_root / "structure" / f"sections__{sample}.parquet")
    section_motives = pd.read_parquet(run_root / "semantics" / f"section_motives__{sample}.parquet")
    enriched = pd.read_parquet(run_root / "semantics" / f"enriched_clauses__{sample}.parquet")
    docs = _sample_docs(run_root, sample)
    section_assign = pd.read_parquet(run_root / "clusters" / SECTION_CLUSTER_EXPERIMENT / "assignments.parquet")
    section_local_assign = pd.read_parquet(run_root / "clusters" / SECTION_LOCAL_CLUSTER_EXPERIMENT / "assignments.parquet")
    policy_assign = pd.read_parquet(run_root / "clusters" / "policy_signature_kmeans_probe" / "assignments.parquet")

    def _sample_rows(df: pd.DataFrame, n: int, group_col: str | None = None) -> pd.DataFrame:
        if df.empty:
            return df.copy()
        if not group_col or group_col not in df.columns:
            return df.sample(min(len(df), n), random_state=0).reset_index(drop=True)
        groups = []
        nunique = max(1, df[group_col].nunique(dropna=False))
        per_group = max(1, n // nunique)
        for _, group in df.groupby(group_col, dropna=False):
            groups.append(group.sample(min(len(group), per_group), random_state=0))
        out = pd.concat(groups, ignore_index=True)
        if len(out) < min(len(df), n):
            remaining = df.loc[~df.index.isin(out.index)].copy()
            if not remaining.empty:
                out = pd.concat([out, remaining.sample(min(len(remaining), n - len(out)), random_state=1)], ignore_index=True)
        return out.head(n).reset_index(drop=True)

    # Section segmentation review.
    seg_rows = _sample_rows(docs, REVIEW_SAMPLE_SIZES["section_segmentation"], "jurisdiction_region")
    seg_items: list[dict[str, Any]] = []
    for row in seg_rows.to_dict(orient="records"):
        doc_id = clean_optional_str(row["ordinance_doc_id"])
        sec = sections[sections["ordinance_doc_id"].astype(str) == doc_id].sort_values("section_number")
        seg_items.append(
            {
                "doc_id": doc_id,
                "issue_id": clean_optional_str(row["issue_id"]),
                "jurisdiction_region": clean_optional_str(row["jurisdiction_region"]),
                "jurisdiction_level": clean_optional_str(row["jurisdiction_level"]),
                "sections": sec[["section_id", "section_number", "title", "start_line", "end_line", "word_count"]].to_dict(orient="records"),
                "text_excerpt": clean_optional_str(row["text"])[:5000],
            }
        )
    _write_review_packet(review_dir / "section_segmentation_review", seg_items, ["doc_id", "boundary_ok", "coverage_ok", "notes"])

    motive_df = sections.merge(section_motives, on="section_id", how="inner", suffixes=("", "__motive"))
    if "ordinance_doc_id" not in motive_df.columns and "ordinance_doc_id__motive" in motive_df.columns:
        motive_df["ordinance_doc_id"] = motive_df["ordinance_doc_id__motive"]
    motive_rows = _sample_rows(motive_df, REVIEW_SAMPLE_SIZES["section_motive"], "primary_motive")
    motive_items = motive_rows[["section_id", "ordinance_doc_id", "title", "primary_motive", "confidence", "evidence", "text"]].to_dict(orient="records")
    _write_review_packet(review_dir / "section_motive_review", motive_items, ["section_id", "motive_ok", "evidence_ok", "notes"])

    clause_rows = _sample_rows(enriched, REVIEW_SAMPLE_SIZES["clause"], "instrument_type")
    clause_items = clause_rows[
        [
            "clause_id",
            "ordinance_doc_id",
            "section_id",
            "section_title",
            "clause_text",
            "requirement",
            "motive",
            "modality",
            "instrument_type",
            "actor",
            "action",
            "regulated_object",
            "condition",
            "exception",
            "threshold_or_quantity",
            "spatial_scope",
            "operative_flag",
            "evidence_quote",
        ]
    ].to_dict(orient="records")
    _write_review_packet(review_dir / "clause_review", clause_items, ["clause_id", "atomic_ok", "enrichment_ok", "operative_ok", "notes"])

    # Section embedding comparison packets.
    openai_clusters = section_assign[section_assign["cluster_id"].astype(int) >= 0].copy()
    local_clusters = section_local_assign[section_local_assign["cluster_id"].astype(int) >= 0].copy()
    cluster_ids = sorted(openai_clusters["cluster_id"].astype(int).unique().tolist())
    rnd.shuffle(cluster_ids)
    comparison_items: list[dict[str, Any]] = []
    for cluster_id in cluster_ids[: REVIEW_SAMPLE_SIZES["section_embedding_comparison"]]:
        openai_subset = openai_clusters[openai_clusters["cluster_id"].astype(int) == cluster_id].head(5)
        local_cluster_id = _best_local_cluster_match(openai_subset["section_id"].astype(str).tolist(), local_clusters)
        local_subset = local_clusters[local_clusters["cluster_id"].astype(int) == local_cluster_id].head(5) if local_cluster_id >= 0 else pd.DataFrame()
        comparison_items.append(
            {
                "comparison_id": f"section_cluster_{cluster_id:04d}",
                "openai_cluster_id": int(cluster_id),
                "openai_examples": openai_subset[["section_id", "title", "text", "cluster_label"]].to_dict(orient="records"),
                "local_cluster_id": int(local_cluster_id),
                "local_examples": local_subset[["section_id", "title", "text", "cluster_label"]].to_dict(orient="records") if not local_subset.empty else [],
            }
        )
    _write_review_packet(review_dir / "section_embedding_comparison_review", comparison_items, ["comparison_id", "openai_better", "coherent_cluster", "notes"])

    # Policy signature review.
    policy_rows = _sample_rows(policy_assign, REVIEW_SAMPLE_SIZES["policy_signature"], "policy_signature_cluster_id")
    policy_items = []
    for row in policy_rows.to_dict(orient="records"):
        doc_id = clean_optional_str(row["ordinance_doc_id"])
        doc = docs[docs["ordinance_doc_id"].astype(str) == doc_id].head(1)
        text = clean_optional_str(doc["text"].iloc[0])[:5000] if not doc.empty else ""
        policy_items.append(
            {
                "ordinance_doc_id": doc_id,
                "policy_signature_cluster_id": _safe_int(row.get("policy_signature_cluster_id")),
                "cluster_label": clean_optional_str(row.get("cluster_label")),
                "top_features": clean_optional_str(row.get("top_features")),
                "jurisdiction_region": clean_optional_str(row.get("jurisdiction_region")),
                "jurisdiction_level": clean_optional_str(row.get("jurisdiction_level")),
                "text_excerpt": text,
            }
        )
    _write_review_packet(review_dir / "policy_signature_review", policy_items, ["ordinance_doc_id", "cluster_ok", "label_ok", "notes"])

    summary = {
        "sample": sample,
        "section_segmentation_review_count": int(len(seg_items)),
        "section_motive_review_count": int(len(motive_items)),
        "clause_review_count": int(len(clause_items)),
        "section_embedding_comparison_review_count": int(len(comparison_items)),
        "policy_signature_review_count": int(len(policy_items)),
    }
    write_json(review_dir / "review_summary.json", summary)
    return summary


def _best_local_cluster_match(section_ids: list[str], local_clusters: pd.DataFrame) -> int:
    if local_clusters.empty:
        return -1
    scores: dict[int, int] = {}
    id_set = set(section_ids)
    for cluster_id, group in local_clusters.groupby("cluster_id", dropna=False):
        overlap = len(id_set & set(group["section_id"].astype(str).tolist()))
        scores[int(cluster_id)] = overlap
    best = max(scores.items(), key=lambda kv: (kv[1], -abs(kv[0])))
    return int(best[0]) if best[1] > 0 else -1


def _write_review_packet(packet_dir: Path, items: list[dict[str, Any]], review_columns: list[str]) -> None:
    packet_dir.mkdir(parents=True, exist_ok=True)
    write_json(packet_dir / "metadata.json", {"item_count": len(items), "review_columns": review_columns})
    write_jsonl(packet_dir / "index.jsonl", items)
    pd.DataFrame.from_records(items).to_json(packet_dir / "items.json", orient="records", force_ascii=False, indent=2)
    with (packet_dir / "review_snapshot.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=review_columns)
        writer.writeheader()
    (packet_dir / "review_events.jsonl").write_text("", encoding="utf-8")


def evaluate_llm_v2_review_quality(*, run_root: Path) -> dict[str, Any]:
    review_dir = run_root / "review"
    gates: dict[str, Any] = {"status": "pending_review"}
    def _rate(packet_name: str, positive_col: str) -> float | None:
        path = review_dir / packet_name / "review_snapshot.csv"
        if not path.is_file():
            return None
        df = pd.read_csv(path, low_memory=False)
        if positive_col not in df.columns or df.empty:
            return None
        valid = df[positive_col].astype(str).str.strip().str.lower()
        valid = valid[valid.isin({"1", "true", "yes", "0", "false", "no"})]
        if valid.empty:
            return None
        return float(valid.isin({"1", "true", "yes"}).mean())

    rates = {
        "section_segmentation_rate": _rate("section_segmentation_review", "boundary_ok"),
        "section_motive_rate": _rate("section_motive_review", "motive_ok"),
        "clause_atomic_rate": _rate("clause_review", "atomic_ok"),
        "clause_enrichment_rate": _rate("clause_review", "enrichment_ok"),
        "section_embedding_openai_win_rate": _rate("section_embedding_comparison_review", "openai_better"),
        "policy_signature_rate": _rate("policy_signature_review", "cluster_ok"),
    }
    gates.update(rates)
    if all(value is not None for value in rates.values()):
        gates["status"] = "pass" if (
            float(rates["section_segmentation_rate"] or 0.0) >= 0.85
            and float(rates["section_motive_rate"] or 0.0) >= 0.80
            and float(rates["clause_atomic_rate"] or 0.0) >= 0.80
            and float(rates["clause_enrichment_rate"] or 0.0) >= 0.90
            and float(rates["section_embedding_openai_win_rate"] or 0.0) >= 0.70
        ) else "fail"
    write_json(review_dir / "quality_gates.json", gates)
    return gates


def build_llm_v2_story_and_report(*, run_root: Path, sample: Literal["pilot", "full"] | None = None) -> dict[str, Any]:
    target_sample: Literal["pilot", "full"] = sample or _story_report_sample(run_root)
    research_summary = build_llm_v2_research_outputs(run_root=run_root, sample=target_sample)
    if target_sample == "full" and not bool(research_summary.get("enriched_clause_doc_count", 0)):
        raise SystemExit(
            "Full-sample story/report generation is disabled until a real full enriched-clause surface exists. "
            "Use the integrated full-sample architecture writeup builder instead."
        )
    baseline_root = _baseline_run_root(run_root)
    story_summary = build_story_outputs(run_root=run_root)
    report_summary = build_report_bundle(run_root=run_root, sample=target_sample)
    baseline_doc = pd.read_parquet(baseline_root / "semantics" / "document_semantics.parquet")
    v2_doc = pd.read_parquet(run_root / "semantics" / "document_semantics.parquet")
    compare = baseline_doc[["ordinance_doc_id", "primary_motive", "dominant_instrument", "operative_dominant_instrument"]].merge(
        v2_doc[["ordinance_doc_id", "primary_motive", "dominant_instrument", "operative_dominant_instrument"]].rename(
            columns={
                "primary_motive": "v2_primary_motive",
                "dominant_instrument": "v2_dominant_instrument",
                "operative_dominant_instrument": "v2_operative_dominant_instrument",
            }
        ),
        on="ordinance_doc_id",
        how="inner",
    )
    comparison_summary = {
        "shared_document_count": int(len(compare)),
        "primary_motive_agreement_rate": float((compare["primary_motive"].astype(str) == compare["v2_primary_motive"].astype(str)).mean()) if not compare.empty else 0.0,
        "dominant_instrument_agreement_rate": float((compare["dominant_instrument"].astype(str) == compare["v2_dominant_instrument"].astype(str)).mean()) if not compare.empty else 0.0,
        "operative_dominant_instrument_agreement_rate": float((compare["operative_dominant_instrument"].astype(str) == compare["v2_operative_dominant_instrument"].astype(str)).mean()) if not compare.empty else 0.0,
    }
    write_json(run_root / "reports" / "baseline_local_comparison_summary.json", comparison_summary)
    compare.to_csv(run_root / "reports" / "tables" / "baseline_local_vs_v2_document_comparison.csv", index=False)
    appendix_lines = [
        "# Baseline Local vs Frontier LLM v2",
        "",
        f"- shared_document_count: {comparison_summary['shared_document_count']}",
        f"- primary_motive_agreement_rate: {comparison_summary['primary_motive_agreement_rate']:.4f}",
        f"- dominant_instrument_agreement_rate: {comparison_summary['dominant_instrument_agreement_rate']:.4f}",
        f"- operative_dominant_instrument_agreement_rate: {comparison_summary['operative_dominant_instrument_agreement_rate']:.4f}",
        "",
        "See `tables/baseline_local_vs_v2_document_comparison.csv` for the document-level comparison table.",
    ]
    (run_root / "reports" / "baseline_local_comparison_appendix.md").write_text("\n".join(appendix_lines), encoding="utf-8")
    return {
        "research": research_summary,
        "story": story_summary,
        "report": report_summary,
        "comparison": comparison_summary,
        "sample": target_sample,
    }


def run_frontier_llm_v2_live_pilot(
    *,
    run_root: Path,
    gateway_model: str = "openai:gpt-5-mini",
    max_concurrency: int = 4,
    timeout: float = 600.0,
    source_env_file: Path | None = None,
    gateway_port: int = 8000,
) -> dict[str, Any]:
    """
    Execute pilot Responses stages live via direct OpenAI calls and section embeddings live via direct OpenAI calls.

    This intentionally runs only the pilot sample. Full runs should use exported Batch requests.
    """
    repo_root = REPO_ROOT
    env = dict(os.environ)
    if source_env_file is not None and source_env_file.is_file():
        for raw in source_env_file.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip("'").strip('"')
            if key and key not in env:
                env[key] = value

    responses_stages = ["section_segmentation", "document_purpose", "section_motive", "clause_extraction", "clause_enrichment"]
    summaries: dict[str, Any] = {}
    for stage in responses_stages:
        export_llm_v2_requests(run_root=run_root, sample="pilot", stage=stage)
        subprocess.run(
            [
                "python",
                str(repo_root / "scripts" / "run_openai_responses_requests_sync.py"),
                "--request-dir",
                str(_requests_stage_dir(run_root, "pilot", stage)),
                "--output-dir",
                str(_results_stage_dir(run_root, "pilot", stage)),
                "--env-file",
                str(source_env_file or (repo_root / ".env")),
                "--max-concurrency",
                str(max_concurrency),
                "--timeout",
                str(timeout),
            ],
            check=True,
            cwd=str(repo_root),
            env=env,
        )
        summaries[stage] = rehydrate_llm_v2_results(run_root=run_root, sample="pilot", stage=stage, allow_errors=False)
    export_llm_v2_requests(run_root=run_root, sample="pilot", stage="section_embedding")
    subprocess.run(
        [
            "python",
            str(repo_root / "scripts" / "run_openai_embedding_requests_sync.py"),
            "--request-dir",
            str(_requests_stage_dir(run_root, "pilot", "section_embedding")),
            "--output-dir",
            str(_results_stage_dir(run_root, "pilot", "section_embedding")),
        ],
        check=True,
        cwd=str(repo_root),
        env=env,
    )
    summaries["section_embedding"] = rehydrate_llm_v2_results(run_root=run_root, sample="pilot", stage="section_embedding", allow_errors=False)
    return summaries
