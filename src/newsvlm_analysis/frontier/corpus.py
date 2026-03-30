from __future__ import annotations

import csv
import json
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import pandas as pd

from .contracts import FrontierPaths
from .io_utils import (
    clean_optional_str,
    collapse_text,
    iter_jsonl,
    make_city_key,
    norm_str,
    normalize_city_name,
    normalize_for_fingerprint,
    region_for_state,
    relative_to_or_empty,
    safe_int,
    sha256_text,
    simhash64,
    slugify,
    write_json,
    write_parquet,
)
from .jurisdiction import JurisdictionResolver, jurisdiction_resolution_summary


@dataclass(slots=True)
class TorchManifestRow:
    issue_id: str
    transcript_path: str
    page_count: int
    page_ids: list[str]


def _stage_remote_file_if_needed(path_str: str, torch_host: str, tmp_dir: Path) -> Path | None:
    local = Path(path_str).expanduser()
    if local.exists():
        return local.resolve()
    if not path_str.startswith("/"):
        return None
    staged = tmp_dir / local.name
    cmd = ["rsync", "-a", f"{torch_host}:{path_str}", str(staged)]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except Exception:
        return None
    return staged if staged.exists() else None


def _load_torch_manifest(paths: FrontierPaths) -> dict[str, TorchManifestRow]:
    out: dict[str, TorchManifestRow] = {}
    with tempfile.TemporaryDirectory(prefix="frontier_fullcode_torch_manifest_") as tmp:
        staged = _stage_remote_file_if_needed(paths.torch_manifest_jsonl, paths.torch_host, Path(tmp))
        if staged is None or not staged.is_file():
            return out
        for row in iter_jsonl(staged):
            issue_id = clean_optional_str(row.get("issue_id"))
            if not issue_id:
                continue
            pages = row.get("pages") if isinstance(row.get("pages"), list) else []
            page_ids = [clean_optional_str(p.get("page_id")) for p in pages if isinstance(p, dict) and clean_optional_str(p.get("page_id"))]
            out[issue_id] = TorchManifestRow(
                issue_id=issue_id,
                transcript_path=clean_optional_str(row.get("issue_transcript_path")),
                page_count=len(page_ids),
                page_ids=page_ids,
            )
    return out


def _load_publication_indexes(paths: FrontierPaths) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    archive_issue_index: dict[str, dict[str, Any]] = {}
    archive_slug_index: dict[str, dict[str, Any]] = {}
    locations_slug_index: dict[str, dict[str, Any]] = {}

    archive_path = Path(paths.archive_metadata_csv).expanduser()
    if archive_path.is_file():
        archive_df = pd.read_csv(archive_path, low_memory=False)
        for row in archive_df.itertuples(index=False):
            issue_id = clean_optional_str(getattr(row, "issue_id", ""))
            slug = clean_optional_str(getattr(row, "newspaper_slug", ""))
            record = {
                "publication_city_name": clean_optional_str(getattr(row, "city_name", "")),
                "publication_state_abbr": clean_optional_str(getattr(row, "state_abbr", "")).upper(),
                "publication_city_key": clean_optional_str(getattr(row, "city_key", "")),
                "publication_cbsa_name": clean_optional_str(getattr(row, "cbsa23_name_primary", "")),
                "publication_census_join_method": clean_optional_str(getattr(row, "census_join_method", "")),
                "publication_geo_source": "archive_v2",
            }
            if issue_id and record["publication_city_name"]:
                archive_issue_index[issue_id] = record
            if slug and record["publication_city_name"] and slug not in archive_slug_index:
                archive_slug_index[slug] = record

    locations_path = Path(paths.publication_locations_jsonl).expanduser()
    if locations_path.is_file():
        for row in iter_jsonl(locations_path):
            slug = clean_optional_str(row.get("pub_slug"))
            city = clean_optional_str(row.get("city_name"))
            state = clean_optional_str(row.get("state_abbr")).upper()
            if not slug or not city:
                continue
            locations_slug_index[slug] = {
                "publication_city_name": city,
                "publication_state_abbr": state,
                "publication_city_key": make_city_key(city, state),
                "publication_cbsa_name": "",
                "publication_census_join_method": "locations_headful_slug",
                "publication_geo_source": "locations_headful",
            }
    return archive_issue_index, archive_slug_index, locations_slug_index


def _select_publication_geo(
    *,
    issue_id: str,
    newspaper_slug: str,
    archive_issue_index: dict[str, dict[str, Any]],
    archive_slug_index: dict[str, dict[str, Any]],
    locations_slug_index: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if issue_id in archive_issue_index:
        return archive_issue_index[issue_id]
    if newspaper_slug in archive_slug_index:
        return archive_slug_index[newspaper_slug]
    if newspaper_slug in locations_slug_index:
        return locations_slug_index[newspaper_slug]
    return {
        "publication_city_name": "",
        "publication_state_abbr": "",
        "publication_city_key": "",
        "publication_cbsa_name": "",
        "publication_census_join_method": "",
        "publication_geo_source": "missing",
    }


def _load_bundle_rows(bundle_dir: Path) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    meta_path = bundle_dir / "metadata.csv"
    if not meta_path.is_file():
        raise SystemExit(f"Missing bundle metadata.csv: {meta_path}")
    metadata = pd.read_csv(meta_path, low_memory=False)
    issue_jsons: dict[str, dict[str, Any]] = {}
    for row in metadata.itertuples(index=False):
        issue_id = clean_optional_str(getattr(row, "issue_id", ""))
        issue_json_relpath = clean_optional_str(getattr(row, "issue_json_relpath", ""))
        if not issue_id or not issue_json_relpath:
            continue
        issue_json_path = bundle_dir / issue_json_relpath
        if not issue_json_path.is_file():
            raise SystemExit(f"Missing issue.json for issue_id={issue_id}: {issue_json_path}")
        issue_jsons[issue_id] = json.loads(issue_json_path.read_text(encoding="utf-8"))
    return metadata, issue_jsons


def _canonical_ordinance_path(bundle_dir: Path, row: Any) -> tuple[Path | None, str]:
    status = clean_optional_str(getattr(row, "ordinance_status", ""))
    ordinance_relpath = clean_optional_str(getattr(row, "ordinance_relpath", ""))
    raw_relpath = clean_optional_str(getattr(row, "ordinance_raw_relpath", ""))
    if status in {"ok", "repaired_from_raw"} and ordinance_relpath:
        return bundle_dir / ordinance_relpath, "ordinance_txt"
    if status == "raw_only" and raw_relpath:
        return bundle_dir / raw_relpath, "ordinance_raw"
    if ordinance_relpath:
        return bundle_dir / ordinance_relpath, "ordinance_txt"
    return None, "missing"


def _coerce_list_json(text: str) -> list[str]:
    raw = clean_optional_str(text)
    if not raw:
        return []
    try:
        obj = json.loads(raw)
    except Exception:
        return [raw]
    if isinstance(obj, list):
        return [clean_optional_str(x) for x in obj if clean_optional_str(x)]
    return [raw]


def _extract_title_line(text: str) -> str:
    for line in collapse_text(text).splitlines():
        candidate = clean_optional_str(line)
        if candidate:
            return candidate[:240]
    return ""


def _simhash_to_int(value: Any) -> int:
    text = clean_optional_str(value)
    if not text:
        return 0
    try:
        return int(text, 16)
    except Exception:
        return 0


def _deduplicate_ordinances(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    parent = {idx: idx for idx in range(len(df))}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra = find(a)
        rb = find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    df["_simhash_int"] = df["simhash64"].map(_simhash_to_int)
    df["_title_key"] = df["title_line"].map(normalize_for_fingerprint)

    exact_groups = df.groupby(["jurisdiction_key", "normalized_sha256"]).indices
    for idxs in exact_groups.values():
        idx_list = list(idxs)
        for idx in idx_list[1:]:
            union(idx_list[0], idx)

    for _, group in df.groupby("jurisdiction_key", dropna=False):
        ordered = group.sort_values(["word_count", "ordinance_doc_id"]).index.tolist()
        for i, left_idx in enumerate(ordered):
            left = df.loc[left_idx]
            left_word_count = int(left["word_count"])
            left_title = clean_optional_str(left["_title_key"])
            left_simhash = int(left["_simhash_int"])
            for right_idx in ordered[i + 1 :]:
                right = df.loc[right_idx]
                right_word_count = int(right["word_count"])
                allowed_word_gap = max(20, int(0.03 * max(right_word_count, 1)))
                if (right_word_count - left_word_count) > allowed_word_gap:
                    break
                title_match = bool(left_title) and left_title == clean_optional_str(right["_title_key"])
                hamming_distance = (left_simhash ^ int(right["_simhash_int"])).bit_count()
                if not title_match and hamming_distance > 8:
                    continue
                sim = SequenceMatcher(None, left["normalized_text"], right["normalized_text"]).ratio()
                if sim >= 0.985:
                    union(left_idx, right_idx)

    root_to_group: dict[int, str] = {}
    group_rows: list[dict[str, Any]] = []
    dedup_group_ids: list[str] = []
    strategies: list[str] = []
    for idx, row in df.iterrows():
        root = find(idx)
        if root not in root_to_group:
            representative = df.loc[root, "ordinance_doc_id"]
            root_to_group[root] = f"dedup::{representative}"
        group_id = root_to_group[root]
        dedup_group_ids.append(group_id)
        strategies.append("grouped" if root != idx else "representative")

    df["dedup_group_id"] = dedup_group_ids
    df["dedup_role"] = strategies
    df = df.drop(columns=["_simhash_int", "_title_key"])
    sizes = df.groupby("dedup_group_id")["ordinance_doc_id"].count().to_dict()
    reps = df.sort_values(["issue_date", "ordinance_doc_id"]).groupby("dedup_group_id", as_index=False).first()
    rep_map = dict(zip(reps["dedup_group_id"], reps["ordinance_doc_id"]))
    for group_id, size in sizes.items():
        members = df[df["dedup_group_id"] == group_id].sort_values(["issue_date", "ordinance_doc_id"])
        group_rows.append(
            {
                "dedup_group_id": group_id,
                "group_size": int(size),
                "representative_ordinance_doc_id": rep_map[group_id],
                "jurisdiction_key": clean_optional_str(members.iloc[0]["jurisdiction_key"]),
                "dedup_strategy": "single" if size == 1 else "exact_or_near_duplicate",
                "member_ordinance_doc_ids": "|".join(members["ordinance_doc_id"].astype(str).tolist()),
            }
        )
    return df, pd.DataFrame.from_records(group_rows).sort_values("dedup_group_id").reset_index(drop=True)


def build_fullcode_corpus(
    *,
    run_root: Path,
    paths: FrontierPaths,
    max_issues: int | None = None,
) -> dict[str, Any]:
    bundle_dir = Path(paths.bundle_dir).expanduser().resolve()
    metadata, issue_jsons = _load_bundle_rows(bundle_dir)
    if max_issues is not None and max_issues > 0:
        metadata = metadata.head(int(max_issues)).copy()

    archive_issue_index, archive_slug_index, locations_slug_index = _load_publication_indexes(paths)
    torch_manifest = _load_torch_manifest(paths)
    jurisdiction_resolver = JurisdictionResolver(paths)

    publication_rows: list[dict[str, Any]] = []
    ordinance_rows: list[dict[str, Any]] = []
    page_rows: list[dict[str, Any]] = []
    geo_rows: list[dict[str, Any]] = []
    jurisdiction_rows: list[dict[str, Any]] = []

    for row in metadata.itertuples(index=False):
        issue_id = clean_optional_str(getattr(row, "issue_id", ""))
        if not issue_id:
            continue
        issue_json = issue_jsons[issue_id]
        newspaper_slug = clean_optional_str(getattr(row, "newspaper_slug", ""))
        publication_geo = _select_publication_geo(
            issue_id=issue_id,
            newspaper_slug=newspaper_slug,
            archive_issue_index=archive_issue_index,
            archive_slug_index=archive_slug_index,
            locations_slug_index=locations_slug_index,
        )
        publication_city_key = clean_optional_str(publication_geo.get("publication_city_key")) or make_city_key(
            clean_optional_str(publication_geo.get("publication_city_name")),
            clean_optional_str(publication_geo.get("publication_state_abbr")),
        )

        jurisdiction = jurisdiction_resolver.resolve_issue(
            issue_id=issue_id,
            issue_json=issue_json,
            metadata_row=row,
            publication_geo=publication_geo,
        )
        jurisdiction_rows.append(jurisdiction)

        jurisdiction_city_name = clean_optional_str(jurisdiction.get("jurisdiction_city_name"))
        jurisdiction_state_abbr = clean_optional_str(jurisdiction.get("jurisdiction_state_abbr")).upper()
        jurisdiction_key = clean_optional_str(jurisdiction.get("jurisdiction_key")) or make_city_key(jurisdiction_city_name, jurisdiction_state_abbr)
        ordinance_path, canonical_source = _canonical_ordinance_path(bundle_dir, row)
        text = ordinance_path.read_text(encoding="utf-8", errors="replace") if ordinance_path and ordinance_path.is_file() else ""
        clean_text = collapse_text(text)
        normalized_text = normalize_for_fingerprint(clean_text)
        title_line = _extract_title_line(clean_text)
        ordinance_doc_id = issue_id
        pages = issue_json.get("pages") if isinstance(issue_json.get("pages"), list) else []
        torch_meta = torch_manifest.get(issue_id)

        publication_row = {
            "issue_id": issue_id,
            "ordinance_doc_id": ordinance_doc_id,
            "publication_key": newspaper_slug,
            "newspaper_slug": newspaper_slug,
            "issue_date": clean_optional_str(getattr(row, "issue_date", "")),
            "document_class": clean_optional_str(getattr(row, "document_class", "")),
            "document_operativity": clean_optional_str(getattr(row, "document_operativity", "")),
            "document_scope": clean_optional_str(getattr(row, "document_scope", "")),
            "publication_city_name": clean_optional_str(publication_geo.get("publication_city_name")),
            "publication_state_abbr": clean_optional_str(publication_geo.get("publication_state_abbr")).upper(),
            "publication_city_key": publication_city_key,
            "publication_region": region_for_state(clean_optional_str(publication_geo.get("publication_state_abbr"))),
            "publication_cbsa_name": clean_optional_str(publication_geo.get("publication_cbsa_name")),
            "publication_geo_source": clean_optional_str(publication_geo.get("publication_geo_source")),
            "publication_census_join_method": clean_optional_str(publication_geo.get("publication_census_join_method")),
            "jurisdiction_city_name": jurisdiction_city_name,
            "jurisdiction_city_name_normalized": normalize_city_name(jurisdiction_city_name),
            "jurisdiction_state_abbr": jurisdiction_state_abbr,
            "jurisdiction_level": clean_optional_str(jurisdiction.get("jurisdiction_level")),
            "jurisdiction_key": jurisdiction_key,
            "jurisdiction_region": region_for_state(jurisdiction_state_abbr),
            "jurisdiction_display_name": clean_optional_str(jurisdiction.get("jurisdiction_display_name")),
            "jurisdiction_input_city_name": clean_optional_str(jurisdiction.get("jurisdiction_input_city_name")),
            "jurisdiction_input_city_name_normalized": clean_optional_str(jurisdiction.get("jurisdiction_input_city_name_normalized")),
            "jurisdiction_input_state_abbr": clean_optional_str(jurisdiction.get("jurisdiction_input_state_abbr")).upper(),
            "jurisdiction_input_level": clean_optional_str(jurisdiction.get("jurisdiction_input_level")),
            "jurisdiction_input_confidence_0_to_1": jurisdiction.get("jurisdiction_input_confidence_0_to_1", 0.0),
            "jurisdiction_input_source": clean_optional_str(jurisdiction.get("jurisdiction_input_source")),
            "jurisdiction_resolution_status": clean_optional_str(jurisdiction.get("jurisdiction_resolution_status")),
            "jurisdiction_resolution_method": clean_optional_str(jurisdiction.get("jurisdiction_resolution_method")),
            "jurisdiction_resolution_score": safe_int(jurisdiction.get("jurisdiction_resolution_score")),
            "jurisdiction_candidate_count": safe_int(jurisdiction.get("jurisdiction_candidate_count")),
            "jurisdiction_candidate_pid6s": clean_optional_str(jurisdiction.get("jurisdiction_candidate_pid6s")),
            "jurisdiction_state_source": clean_optional_str(jurisdiction.get("jurisdiction_state_source")),
            "jurisdiction_state_backfill_used": safe_int(jurisdiction.get("jurisdiction_state_backfill_used")),
            "jurisdiction_evidence_quote": clean_optional_str(jurisdiction.get("jurisdiction_evidence_quote")),
            "place20_geoid": clean_optional_str(jurisdiction.get("place20_geoid")),
            "county20_fips_primary": clean_optional_str(jurisdiction.get("county20_fips_primary")),
            "puma20_geoid_primary": clean_optional_str(jurisdiction.get("puma20_geoid_primary")),
            "cbsa23_geoid_primary": clean_optional_str(jurisdiction.get("cbsa23_geoid_primary")),
            "cbsa23_name_primary": clean_optional_str(jurisdiction.get("cbsa23_name_primary")),
            "census_id_pid6": clean_optional_str(jurisdiction.get("census_id_pid6")),
            "census_id_gidid": clean_optional_str(jurisdiction.get("census_id_gidid")),
            "census_unit_name": clean_optional_str(jurisdiction.get("census_unit_name")),
            "census_unit_type": clean_optional_str(jurisdiction.get("census_unit_type")),
            "census_join_method": clean_optional_str(jurisdiction.get("census_join_method")),
            "page_count": safe_int(getattr(row, "page_count", 0)),
            "quality_flags_json": clean_optional_str(getattr(row, "quality_flags_json", "")),
            "analysis_geo_mode": "jurisdiction",
            "bundle_dir": str(bundle_dir),
            "issue_json_relpath": clean_optional_str(getattr(row, "issue_json_relpath", "")),
            "issue_transcript_relpath": clean_optional_str(getattr(row, "issue_transcript_relpath", "")),
            "torch_manifest_path": paths.torch_manifest_jsonl,
            "torch_run_root": paths.torch_run_root,
            "torch_manifest_match": int(torch_meta is not None),
            "torch_manifest_transcript_path": torch_meta.transcript_path if torch_meta else "",
            "torch_manifest_page_count": torch_meta.page_count if torch_meta else 0,
            "torch_manifest_page_ids": "|".join(torch_meta.page_ids) if torch_meta else "",
        }
        publication_rows.append(publication_row)

        ordinance_rows.append(
            {
                **publication_row,
                "canonical_text_status": clean_optional_str(getattr(row, "ordinance_status", "")),
                "canonical_text_source": canonical_source,
                "canonical_text_path": str(ordinance_path) if ordinance_path else "",
                "raw_text_path": str(bundle_dir / clean_optional_str(getattr(row, "ordinance_raw_relpath", "")))
                if clean_optional_str(getattr(row, "ordinance_raw_relpath", ""))
                else "",
                "notes_text_path": str(bundle_dir / clean_optional_str(getattr(row, "ordinance_notes_relpath", "")))
                if clean_optional_str(getattr(row, "ordinance_notes_relpath", ""))
                else "",
                "ordinance_contract_ok": safe_int(getattr(row, "ordinance_contract_ok", 0)),
                "ordinance_repair_used": safe_int(getattr(row, "ordinance_repair_used", 0)),
                "ordinance_model": clean_optional_str(getattr(row, "ordinance_model", "")),
                "ordinance_prompt": clean_optional_str(getattr(row, "ordinance_prompt", "")),
                "ordinance_source_line": clean_optional_str(getattr(row, "ordinance_source_line", "")),
                "text": text,
                "clean_text": clean_text,
                "normalized_text": normalized_text,
                "title_line": title_line,
                "text_sha256": sha256_text(text),
                "clean_text_sha256": sha256_text(clean_text),
                "normalized_sha256": sha256_text(normalized_text),
                "simhash64": simhash64(normalized_text),
                "char_count": len(clean_text),
                "word_count": len(normalized_text.split()),
                "line_count": len(clean_text.splitlines()),
            }
        )

        geo_rows.append(
            {
                "issue_id": issue_id,
                "ordinance_doc_id": ordinance_doc_id,
                "publication_key": newspaper_slug,
                "publication_city_name": publication_row["publication_city_name"],
                "publication_state_abbr": publication_row["publication_state_abbr"],
                "publication_city_key": publication_city_key,
                "publication_region": publication_row["publication_region"],
                "publication_geo_source": publication_row["publication_geo_source"],
                "jurisdiction_city_name": jurisdiction_city_name,
                "jurisdiction_state_abbr": jurisdiction_state_abbr,
                "jurisdiction_level": publication_row["jurisdiction_level"],
                "jurisdiction_key": jurisdiction_key,
                "jurisdiction_region": publication_row["jurisdiction_region"],
                "jurisdiction_resolution_status": publication_row["jurisdiction_resolution_status"],
                "jurisdiction_resolution_method": publication_row["jurisdiction_resolution_method"],
                "jurisdiction_state_backfill_used": publication_row["jurisdiction_state_backfill_used"],
                "place20_geoid": publication_row["place20_geoid"],
                "census_id_pid6": publication_row["census_id_pid6"],
                "analysis_geo_mode": "jurisdiction",
            }
        )

        for page in pages:
            if not isinstance(page, dict):
                continue
            page_rows.append(
                {
                    "issue_id": issue_id,
                    "ordinance_doc_id": ordinance_doc_id,
                    "page_id": clean_optional_str(page.get("page_id")),
                    "page_num": safe_int(page.get("page_num")),
                    "page_text_relpath": clean_optional_str(page.get("page_text_relpath")),
                    "image_relpath": clean_optional_str(page.get("image_relpath")),
                    "text_source": clean_optional_str(page.get("text_source")),
                    "page_text_chars": safe_int(page.get("page_text_chars")),
                    "ocr_input_mode": clean_optional_str(page.get("ocr_input_mode")),
                    "ocr_input_png_external_path": clean_optional_str(page.get("ocr_input_png_external_path")),
                    "ocr_input_provenance_external_path": clean_optional_str(page.get("ocr_input_provenance_external_path")),
                    "ocr_input_provenance_type": clean_optional_str(page.get("ocr_input_provenance_type")),
                    "source_text_origin_external_path": clean_optional_str(page.get("source_text_origin_external_path")),
                }
            )

    publication_df = pd.DataFrame.from_records(publication_rows).sort_values(["issue_date", "issue_id"]).reset_index(drop=True)
    ordinance_df = pd.DataFrame.from_records(ordinance_rows).sort_values(["issue_date", "issue_id"]).reset_index(drop=True)
    geo_df = pd.DataFrame.from_records(geo_rows).sort_values(["issue_id"]).reset_index(drop=True)
    pages_df = pd.DataFrame.from_records(page_rows).sort_values(["issue_id", "page_num", "page_id"]).reset_index(drop=True)
    jurisdiction_df = pd.DataFrame.from_records(jurisdiction_rows).sort_values(["issue_id"]).reset_index(drop=True)

    ordinance_df, dedup_groups_df = _deduplicate_ordinances(ordinance_df)
    dedup_map = ordinance_df[["issue_id", "dedup_group_id", "dedup_role"]]
    publication_df = publication_df.merge(dedup_map, on="issue_id", how="left")
    geo_df = geo_df.merge(dedup_map, on="issue_id", how="left")

    timeline_df = (
        ordinance_df.sort_values(["jurisdiction_key", "issue_date", "ordinance_doc_id"])
        .assign(jurisdiction_sequence_index=lambda x: x.groupby("jurisdiction_key").cumcount() + 1)
        [["ordinance_doc_id", "issue_id", "jurisdiction_key", "issue_date", "jurisdiction_sequence_index", "dedup_group_id"]]
        .reset_index(drop=True)
    )

    corpus_dir = run_root / "corpus"
    write_parquet(publication_df, corpus_dir / "publication_issues.parquet")
    write_parquet(ordinance_df, corpus_dir / "ordinances.parquet")
    write_parquet(geo_df, corpus_dir / "geo_links.parquet")
    write_parquet(pages_df, corpus_dir / "pages.parquet")
    write_parquet(dedup_groups_df, corpus_dir / "dedup_groups.parquet")
    write_parquet(timeline_df, corpus_dir / "timeline_scaffold.parquet")
    write_parquet(jurisdiction_df, corpus_dir / "jurisdiction_resolution.parquet")
    manual_override_df = jurisdiction_df[
        jurisdiction_df["jurisdiction_resolution_method"].astype(str).eq("manual_issue_override")
    ].reset_index(drop=True)
    write_parquet(manual_override_df, corpus_dir / "manual_jurisdiction_overrides_applied.parquet")

    resolution_summary = jurisdiction_resolution_summary(jurisdiction_df)
    write_json(corpus_dir / "jurisdiction_resolution_summary.json", resolution_summary)

    summary = {
        "issue_count": int(len(publication_df)),
        "ordinance_count": int(len(ordinance_df)),
        "page_count": int(len(pages_df)),
        "dedup_group_count": int(len(dedup_groups_df)),
        "duplicate_issue_count": int((dedup_groups_df["group_size"] > 1).sum()) if not dedup_groups_df.empty else 0,
        "publication_geo_sources": publication_df["publication_geo_source"].value_counts(dropna=False).to_dict(),
        "canonical_status_counts": ordinance_df["canonical_text_status"].value_counts(dropna=False).to_dict() if "canonical_text_status" in ordinance_df else {},
        "jurisdiction_status_counts": resolution_summary["status_counts"],
        "jurisdiction_method_counts": resolution_summary["method_counts"],
        "resolver_pid6_coverage": resolution_summary["resolver_pid6_coverage"],
        "bundle_pid6_coverage": resolution_summary["bundle_pid6_coverage"],
    }
    write_json(corpus_dir / "corpus_summary.json", summary)
    return summary
