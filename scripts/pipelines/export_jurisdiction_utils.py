#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.newsvlm_analysis.frontier.contracts import FrontierPaths
from src.newsvlm_analysis.frontier.io_utils import normalize_city_name
from src.newsvlm_analysis.frontier.jurisdiction import JurisdictionResolver


def build_jurisdiction_resolver(
    *,
    archive_metadata_csv: Path,
    census_of_gov_xlsx: Path,
) -> JurisdictionResolver:
    paths = FrontierPaths(
        archive_metadata_csv=str(archive_metadata_csv),
        census_of_gov_xlsx=str(census_of_gov_xlsx),
    )
    return JurisdictionResolver(paths)


def metadata_namespace(fields: dict[str, Any]) -> Any:
    return SimpleNamespace(**fields)


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none", "null"} else text


def normalize_state(value: Any) -> str:
    return clean_text(value).upper()


def derive_geo_target_fields(
    *,
    publication_geo: dict[str, Any],
    resolution: dict[str, Any],
) -> dict[str, str]:
    publication_city = clean_text(publication_geo.get("publication_city_name"))
    publication_state = normalize_state(publication_geo.get("publication_state_abbr"))

    jurisdiction_city = clean_text(resolution.get("jurisdiction_city_name"))
    jurisdiction_state = normalize_state(resolution.get("jurisdiction_state_abbr"))
    jurisdiction_level = clean_text(resolution.get("jurisdiction_level"))
    input_source = clean_text(resolution.get("jurisdiction_input_source"))
    state_source = clean_text(resolution.get("jurisdiction_state_source"))
    state_backfill_used = "1" if str(resolution.get("jurisdiction_state_backfill_used", "")).strip() in {"1", "True", "true"} else "0"
    resolution_status = clean_text(resolution.get("jurisdiction_resolution_status"))
    resolution_method = clean_text(resolution.get("jurisdiction_resolution_method"))

    use_jurisdiction = bool(jurisdiction_city) and jurisdiction_level != "publication" and input_source != "publication_location_fallback"
    geo_target_city = jurisdiction_city if use_jurisdiction else publication_city
    geo_target_state = jurisdiction_state if jurisdiction_state else publication_state
    geo_target_level = jurisdiction_level if use_jurisdiction else "publication"

    if not use_jurisdiction:
        basis = "publication_location"
    elif resolution_method == "manual_issue_override":
        basis = "extracted_jurisdiction_manual_override"
    elif state_source == "publication_backfill" or state_backfill_used == "1":
        basis = "extracted_jurisdiction_publication_state_backfill"
    elif jurisdiction_state:
        basis = "extracted_jurisdiction"
    else:
        basis = "extracted_jurisdiction_missing_state"

    overrode = "1" if (
        use_jurisdiction
        and publication_city
        and (
            jurisdiction_city.casefold() != publication_city.casefold()
            or jurisdiction_state != publication_state
        )
    ) else "0"

    return {
        "publication_city_name": publication_city,
        "publication_state_abbr": publication_state,
        "geo_target_city_name": geo_target_city,
        "geo_target_state_abbr": geo_target_state,
        "geo_target_level": geo_target_level,
        "geo_target_basis": basis,
        "geo_target_state_backfill_used": state_backfill_used,
        "geo_target_overrode_publication_location": overrode,
        "geo_target_match_status": resolution_status or clean_text(resolution.get("census_join_method")),
    }


def stringify_resolution_fields(resolution: dict[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    for key, value in resolution.items():
        if key == "issue_id":
            continue
        out[key] = clean_text(value)
    return out


def resolution_has_census_fields(resolution: dict[str, Any]) -> bool:
    return any(
        clean_text(resolution.get(field))
        for field in ("census_id_pid6", "census_id_gidid", "census_unit_name", "census_unit_type")
    )


def resolve_issue_with_publication_fallback(
    *,
    resolver: JurisdictionResolver,
    issue_id: str,
    issue_json: dict[str, Any],
    metadata_row: Any,
    publication_geo: dict[str, Any],
) -> dict[str, Any]:
    resolution = resolver.resolve_issue(
        issue_id=issue_id,
        issue_json=issue_json,
        metadata_row=metadata_row,
        publication_geo=publication_geo,
    )
    if clean_text(resolution.get("jurisdiction_resolution_status")) != "missing_extraction":
        return resolution

    publication_city = clean_text(publication_geo.get("publication_city_name"))
    publication_state = normalize_state(publication_geo.get("publication_state_abbr"))
    if not publication_city or not publication_state:
        return resolution

    publication_resolution: dict[str, Any] | None = None
    for fallback_level in ("city", "town"):
        publication_issue_json = {
            "classification": {
                "ordinance_jurisdiction": {
                    "city": publication_city,
                    "state_abbr": publication_state,
                    "jurisdiction_level": fallback_level,
                    "confidence_0_to_1": 0.0,
                    "evidence_quote": "",
                }
            }
        }
        candidate_resolution = resolver.resolve_issue(
            issue_id=issue_id,
            issue_json=publication_issue_json,
            metadata_row=metadata_row,
            publication_geo=publication_geo,
        )
        if clean_text(candidate_resolution.get("jurisdiction_resolution_status")) == "resolved_with_census":
            publication_resolution = candidate_resolution
            break

    if publication_resolution is None:
        return resolution

    publication_resolution = dict(publication_resolution)
    publication_resolution["jurisdiction_input_source"] = "publication_location_fallback"
    publication_resolution["jurisdiction_input_city_name"] = publication_city
    publication_resolution["jurisdiction_input_city_name_normalized"] = normalize_city_name(
        publication_city
    )
    publication_resolution["jurisdiction_input_state_abbr"] = publication_state
    publication_resolution["jurisdiction_input_level"] = "publication"
    publication_resolution["jurisdiction_input_confidence_0_to_1"] = 0.0
    publication_resolution["jurisdiction_state_source"] = "publication_location"
    publication_resolution["jurisdiction_state_backfill_used"] = 0
    publication_resolution["jurisdiction_level"] = "publication"

    original_resolution_method = clean_text(
        publication_resolution.get("jurisdiction_resolution_method")
    )
    original_census_join_method = clean_text(
        publication_resolution.get("census_join_method")
    )
    if original_resolution_method:
        publication_resolution["jurisdiction_resolution_method"] = (
            f"publication_location_fallback::{original_resolution_method}"
        )
    else:
        publication_resolution["jurisdiction_resolution_method"] = (
            "publication_location_fallback"
        )
    if original_census_join_method:
        publication_resolution["census_join_method"] = (
            f"publication::{original_census_join_method}"
        )
    else:
        publication_resolution["census_join_method"] = "publication"
    return publication_resolution


def resolve_issue_with_preferred_public_fallback(
    *,
    resolver: JurisdictionResolver,
    issue_id: str,
    issue_json: dict[str, Any],
    metadata_row: Any,
    publication_geo: dict[str, Any],
) -> dict[str, Any]:
    extracted_resolution = resolver.resolve_issue(
        issue_id=issue_id,
        issue_json=issue_json,
        metadata_row=metadata_row,
        publication_geo=publication_geo,
    )
    if resolution_has_census_fields(extracted_resolution):
        return extracted_resolution

    publication_resolution = resolve_issue_with_publication_fallback(
        resolver=resolver,
        issue_id=f"{issue_id}::publication_fallback_probe",
        issue_json={"classification": {}},
        metadata_row=metadata_namespace({}),
        publication_geo=publication_geo,
    )
    if resolution_has_census_fields(publication_resolution):
        publication_resolution = dict(publication_resolution)
        publication_resolution["issue_id"] = issue_id
        return publication_resolution

    return extracted_resolution
