from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd

from .io_utils import collapse_text, clean_optional_str, slugify, write_json, write_parquet


ARTICLE_RE = re.compile(r"^(article|chapter)\s+([ivx0-9a-z\-]+)\b", re.IGNORECASE)
SECTION_RE = re.compile(r"^(section|sec\.?)\s+([0-9ivxA-Za-z.\-]+)\b", re.IGNORECASE)
ENUM_RE = re.compile(r"^\(?([0-9]+|[A-Za-z]|[ivxIVX]+)[\).:-]\s+")
ALL_CAPS_RE = re.compile(r"^[A-Z0-9 ,.&'\"/\-():;]{4,140}$")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.;:])\s+(?=(?:[A-Z(]|\"))")
CLAUSE_MODAL_RE = re.compile(
    r"\b(shall not|must not|prohibited|unlawful|shall|must|required|may|permitted|means|defined as|is defined as)\b",
    re.IGNORECASE,
)


def _is_heading(line: str, prev_blank: bool, next_blank: bool) -> bool:
    s = clean_optional_str(line)
    if not s:
        return False
    if ARTICLE_RE.match(s) or SECTION_RE.match(s):
        return True
    if len(s) <= 140 and ALL_CAPS_RE.match(s) and (prev_blank or next_blank):
        return True
    if prev_blank and next_blank and len(s.split()) <= 12 and s == s.title():
        return True
    return False


def _heading_level(heading: str) -> str:
    s = clean_optional_str(heading)
    if ARTICLE_RE.match(s):
        return "article"
    if SECTION_RE.match(s):
        return "section"
    if s:
        return "heading"
    return "preamble"


def classify_section_kind(heading: str, text: str) -> str:
    sample = f"{heading}\n{text}".lower()
    if "purpose" in sample or "intent" in sample or "general welfare" in sample:
        return "purpose"
    if "definition" in sample or "means" in sample or "shall mean" in sample:
        return "definitions"
    if "district" in sample and "map" in sample:
        return "district_map"
    if "district" in sample:
        return "districts"
    if "permitted use" in sample or "uses permitted" in sample or "use regulations" in sample:
        return "uses"
    if any(key in sample for key in ("setback", "yard", "lot area", "lot width", "height", "parking", "loading")):
        return "dimensional"
    if any(key in sample for key in ("variance", "appeal", "board of adjustment", "board of appeals")):
        return "appeals_variances"
    if any(key in sample for key in ("permit", "application", "site plan", "conditional use", "special exception")):
        return "administration"
    if any(key in sample for key in ("penalty", "violation", "enforcement", "fine", "unlawful")):
        return "enforcement"
    if "subdivision" in sample or "plat" in sample:
        return "subdivision"
    if "severability" in sample or "effective date" in sample:
        return "closing"
    return "other"


def split_sections(text: str) -> list[dict[str, Any]]:
    lines = collapse_text(text).splitlines()
    if not lines:
        return []
    line_starts: list[int] = []
    cursor = 0
    for line in lines:
        line_starts.append(cursor)
        cursor += len(line) + 1

    heading_positions: list[int] = []
    for idx, line in enumerate(lines):
        prev_blank = idx == 0 or not clean_optional_str(lines[idx - 1])
        next_blank = idx == len(lines) - 1 or not clean_optional_str(lines[idx + 1])
        if _is_heading(line, prev_blank, next_blank):
            heading_positions.append(idx)

    if 0 not in heading_positions:
        heading_positions = [0] + heading_positions
    heading_positions = sorted(set(heading_positions))

    sections: list[dict[str, Any]] = []
    for section_index, start_idx in enumerate(heading_positions, start=1):
        end_idx = heading_positions[section_index] if section_index < len(heading_positions) else len(lines)
        chunk_lines = lines[start_idx:end_idx]
        heading = clean_optional_str(chunk_lines[0]) if chunk_lines else ""
        body_lines = chunk_lines[1:] if _is_heading(chunk_lines[0], True, False) else chunk_lines
        text_block = collapse_text("\n".join(body_lines).strip())
        if not text_block:
            text_block = collapse_text("\n".join(chunk_lines).strip())
        start_line = start_idx + 1
        end_line = end_idx
        start_char = line_starts[start_idx]
        end_char = line_starts[end_idx - 1] + len(lines[end_idx - 1]) if end_idx > start_idx else start_char
        kind = classify_section_kind(heading, text_block)
        sections.append(
            {
                "section_index": section_index,
                "heading": heading[:240],
                "heading_slug": slugify(heading) if heading else f"section-{section_index}",
                "heading_level": _heading_level(heading),
                "section_kind": kind,
                "start_line": start_line,
                "end_line": end_line,
                "start_char": start_char,
                "end_char": end_char,
                "text": text_block,
            }
        )
    return sections


def _normalize_clause_text(text: str) -> str:
    s = collapse_text(text)
    s = ENUM_RE.sub("", s)
    return s.strip(" -:;")


def _modality_for_clause(text: str) -> str:
    sample = text.lower()
    if re.search(r"\b(shall not|must not|prohibited|unlawful|no\s+[a-z].+shall)\b", sample):
        return "must_not"
    if re.search(r"\b(means|shall mean|is defined as|defined as)\b", sample):
        return "definition"
    if re.search(r"\b(may|authorized|permitted to)\b", sample):
        return "may"
    if re.search(r"\b(shall|must|required)\b", sample):
        return "must"
    return "other"


def _instrument_hint(text: str, section_kind: str) -> str:
    sample = text.lower()
    mapping = [
        ("minimum lot area", "Minimum lot area"),
        ("lot width", "Minimum lot width"),
        ("front yard", "Front setback"),
        ("side yard", "Side setback"),
        ("rear yard", "Rear setback"),
        ("height", "Height limit"),
        ("parking", "Parking requirement"),
        ("loading", "Loading requirement"),
        ("conditional use", "Conditional use"),
        ("special exception", "Special exception"),
        ("variance", "Variance procedure"),
        ("appeal", "Appeal procedure"),
        ("zoning map", "Zoning map change"),
        ("district", "District classification"),
        ("nonconforming", "Nonconforming use"),
        ("subdivision", "Subdivision standard"),
        ("plat", "Plat approval"),
        ("penalty", "Penalty provision"),
        ("enforcement", "Enforcement rule"),
        ("permit", "Permit requirement"),
    ]
    for needle, label in mapping:
        if needle in sample:
            return label
    if section_kind == "definitions":
        return "Definition"
    if section_kind == "purpose":
        return "Purpose statement"
    return "General ordinance clause"


def _candidate_clauses(section_text: str) -> list[str]:
    lines = [line for line in section_text.splitlines() if clean_optional_str(line)]
    grouped: list[str] = []
    current: list[str] = []
    for line in lines:
        if ENUM_RE.match(line) and current:
            grouped.append(" ".join(current))
            current = [line]
        else:
            current.append(line)
    if current:
        grouped.append(" ".join(current))
    if len(grouped) <= 1:
        grouped = SENTENCE_SPLIT_RE.split(collapse_text(section_text))
    out = []
    seen: set[str] = set()
    for chunk in grouped:
        candidate = _normalize_clause_text(chunk)
        if len(candidate) < 25:
            continue
        if candidate in seen:
            continue
        if not CLAUSE_MODAL_RE.search(candidate) and len(candidate.split()) < 10:
            continue
        seen.add(candidate)
        out.append(candidate)
    return out


def build_structure_outputs(*, run_root: Path) -> dict[str, Any]:
    corpus_dir = run_root / "corpus"
    structure_dir = run_root / "structure"
    ordinances = pd.read_parquet(corpus_dir / "ordinances.parquet")
    if "jurisdiction_resolution_status" in ordinances.columns:
        ordinances = ordinances[ordinances["jurisdiction_resolution_status"].astype(str) == "resolved_with_census"].copy()

    section_rows: list[dict[str, Any]] = []
    clause_rows: list[dict[str, Any]] = []
    for row in ordinances.itertuples(index=False):
        ordinance_doc_id = clean_optional_str(getattr(row, "ordinance_doc_id", ""))
        sections = split_sections(clean_optional_str(getattr(row, "clean_text", "")))
        if not sections and clean_optional_str(getattr(row, "clean_text", "")):
            sections = [
                {
                    "section_index": 1,
                    "heading": clean_optional_str(getattr(row, "title_line", "")),
                    "heading_slug": "full-text",
                    "heading_level": "preamble",
                    "section_kind": "other",
                    "start_line": 1,
                    "end_line": len(clean_optional_str(getattr(row, "clean_text", "")).splitlines()),
                    "start_char": 0,
                    "end_char": len(clean_optional_str(getattr(row, "clean_text", ""))),
                    "text": clean_optional_str(getattr(row, "clean_text", "")),
                }
            ]
        for section in sections:
            section_id = f"{ordinance_doc_id}::section::{section['section_index']:03d}"
            section_row = {
                "section_id": section_id,
                "ordinance_doc_id": ordinance_doc_id,
                "issue_id": clean_optional_str(getattr(row, "issue_id", "")),
                "issue_date": clean_optional_str(getattr(row, "issue_date", "")),
                "publication_key": clean_optional_str(getattr(row, "publication_key", "")),
                "jurisdiction_key": clean_optional_str(getattr(row, "jurisdiction_key", "")),
                **section,
                "word_count": len(section["text"].split()),
                "char_count": len(section["text"]),
            }
            section_rows.append(section_row)

            for clause_index, clause_text in enumerate(_candidate_clauses(section["text"]), start=1):
                start_char_rel = section["text"].find(clause_text)
                start_char = section["start_char"] + max(start_char_rel, 0)
                end_char = start_char + len(clause_text)
                clause_rows.append(
                    {
                        "clause_id": f"{section_id}::clause::{clause_index:03d}",
                        "section_id": section_id,
                        "ordinance_doc_id": ordinance_doc_id,
                        "issue_id": clean_optional_str(getattr(row, "issue_id", "")),
                        "issue_date": clean_optional_str(getattr(row, "issue_date", "")),
                        "publication_key": clean_optional_str(getattr(row, "publication_key", "")),
                        "jurisdiction_key": clean_optional_str(getattr(row, "jurisdiction_key", "")),
                        "section_index": section["section_index"],
                        "section_kind": section["section_kind"],
                        "section_heading": section["heading"],
                        "clause_index": clause_index,
                        "clause_text": clause_text,
                        "requirement_text": clause_text,
                        "modality": _modality_for_clause(clause_text),
                        "instrument_hint": _instrument_hint(clause_text, section["section_kind"]),
                        "source_start_line": section["start_line"],
                        "source_end_line": section["end_line"],
                        "source_start_char": start_char,
                        "source_end_char": end_char,
                        "word_count": len(clause_text.split()),
                        "char_count": len(clause_text),
                    }
                )

    sections_df = pd.DataFrame.from_records(section_rows).sort_values(["ordinance_doc_id", "section_index"]).reset_index(drop=True)
    clauses_df = pd.DataFrame.from_records(clause_rows).sort_values(["ordinance_doc_id", "section_index", "clause_index"]).reset_index(drop=True)

    write_parquet(sections_df, structure_dir / "sections.parquet")
    write_parquet(clauses_df, structure_dir / "clauses.parquet")
    write_json(
        structure_dir / "structure_summary.json",
        {
            "ordinance_count": int(ordinances["ordinance_doc_id"].nunique()),
            "section_count": int(len(sections_df)),
            "clause_count": int(len(clauses_df)),
            "sections_per_ordinance_mean": float(sections_df.groupby("ordinance_doc_id").size().mean()) if not sections_df.empty else 0.0,
            "clauses_per_ordinance_mean": float(clauses_df.groupby("ordinance_doc_id").size().mean()) if not clauses_df.empty else 0.0,
        },
    )
    return {
        "section_count": int(len(sections_df)),
        "clause_count": int(len(clauses_df)),
    }
