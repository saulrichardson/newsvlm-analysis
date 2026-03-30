from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import pandas as pd

from .contracts import FrontierPaths
from .io_utils import clean_optional_str, make_city_key, normalize_city_name, safe_float
from .manual_jurisdiction_overrides import MANUAL_JURISDICTION_OVERRIDES

WORKBOOK_COLUMNS = [
    "CENSUS_ID_PID6",
    "CENSUS_ID_GIDID",
    "UNIT_NAME",
    "UNIT_TYPE",
    "CITY",
    "STATE",
    "FIPS_STATE",
    "FIPS_PLACE",
]

MUNICIPAL_LEVELS = {"city", "village", "borough"}
COUNTY_LEVELS = {"county", "parish"}
TOWNSHIP_LEVELS = {"township"}
TOWN_LEVELS = {"town"}


def _norm_text(value: Any) -> str:
    return clean_optional_str(value)


def _norm_upper(value: Any) -> str:
    return clean_optional_str(value).upper()


def _norm_lower(value: Any) -> str:
    return clean_optional_str(value).lower()


def _norm_id(value: Any, *, width: int | None = None) -> str:
    out = clean_optional_str(value)
    if not out:
        return ""
    if out.endswith(".0"):
        out = out[:-2]
    if width and out.isdigit():
        out = out.zfill(width)
    return out


def _title_case(value: str) -> str:
    parts = clean_optional_str(value).split()
    return " ".join(part.capitalize() for part in parts)


def _family_from_level(level: str, raw_name: str) -> list[str]:
    level_norm = _norm_lower(level)
    name_norm = _norm_lower(raw_name)
    if level_norm in COUNTY_LEVELS or " county" in name_norm or name_norm.startswith("county of "):
        return ["county"]
    if level_norm in TOWNSHIP_LEVELS or " township" in name_norm or name_norm.startswith("township of "):
        return ["township"]
    if level_norm in TOWN_LEVELS:
        return ["municipal", "township"]
    if level_norm in MUNICIPAL_LEVELS:
        return ["municipal"]
    return ["municipal", "township", "county"]


def _unit_family(unit_type: str) -> str:
    value = _norm_lower(unit_type)
    if value.startswith("1 - county"):
        return "county"
    if value.startswith("2 - municipal"):
        return "municipal"
    if value.startswith("3 - township"):
        return "township"
    return ""


def _unique_preserve_order(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return out


def _candidate_key(pid6: str, gidid: str, place20_geoid: str, unit_name: str) -> str:
    for value in [pid6, gidid, place20_geoid, unit_name]:
        if clean_optional_str(value):
            return clean_optional_str(value)
    return ""


def _pick_resolved_core_name(candidate: dict[str, str], input_name_norm: str) -> str:
    family = clean_optional_str(candidate.get("unit_family"))
    unit_core = normalize_city_name(clean_optional_str(candidate.get("unit_name")))
    city_core = normalize_city_name(clean_optional_str(candidate.get("city_name")))
    if family in {"county", "township"}:
        return unit_core or city_core
    if not input_name_norm:
        return city_core or unit_core
    unit_score = SequenceMatcher(None, input_name_norm, unit_core.lower()).ratio() if unit_core else 0.0
    city_score = SequenceMatcher(None, input_name_norm, city_core.lower()).ratio() if city_core else 0.0
    return unit_core if unit_score > city_score else city_core or unit_core


@dataclass(slots=True)
class JurisdictionReferenceTables:
    census_by_pid6: dict[str, dict[str, str]]
    census_by_place20: dict[str, list[dict[str, str]]]
    census_by_unit_name_state_family: dict[tuple[str, str, str], list[dict[str, str]]]
    census_by_name_state_family: dict[tuple[str, str, str], list[dict[str, str]]]
    census_by_unit_name_family: dict[tuple[str, str], list[dict[str, str]]]
    census_by_name_family: dict[tuple[str, str], list[dict[str, str]]]


class JurisdictionResolver:
    def __init__(self, paths: FrontierPaths) -> None:
        self.paths = paths
        self.refs = self._load_reference_tables(paths)

    @staticmethod
    def _load_reference_tables(paths: FrontierPaths) -> JurisdictionReferenceTables:
        census_df = JurisdictionResolver._load_census_workbook(Path(paths.census_of_gov_xlsx))

        census_by_pid6: dict[str, dict[str, str]] = {}
        census_by_place20: dict[str, list[dict[str, str]]] = {}
        census_by_unit_name_state_family: dict[tuple[str, str, str], list[dict[str, str]]] = {}
        census_by_name_state_family: dict[tuple[str, str, str], list[dict[str, str]]] = {}
        census_by_unit_name_family: dict[tuple[str, str], list[dict[str, str]]] = {}
        census_by_name_family: dict[tuple[str, str], list[dict[str, str]]] = {}

        for record in census_df.to_dict(orient="records"):
            pid6 = clean_optional_str(record["census_id_pid6"])
            if pid6:
                census_by_pid6[pid6] = record
            place20 = clean_optional_str(record["place20_geoid"])
            if place20:
                census_by_place20.setdefault(place20, []).append(record)

            family = clean_optional_str(record["unit_family"])
            state = clean_optional_str(record["state_abbr"]).lower()
            unit_name_key = clean_optional_str(record["unit_name_norm"])
            if unit_name_key:
                census_by_unit_name_state_family.setdefault((unit_name_key, state, family), []).append(record)
                census_by_unit_name_family.setdefault((unit_name_key, family), []).append(record)
            for name_key in _unique_preserve_order(
                [
                    clean_optional_str(record["name_norm"]),
                    clean_optional_str(record["city_norm"]),
                    clean_optional_str(record["unit_name_norm"]),
                ]
            ):
                census_by_name_state_family.setdefault((name_key, state, family), []).append(record)
                census_by_name_family.setdefault((name_key, family), []).append(record)

        return JurisdictionReferenceTables(
            census_by_pid6=census_by_pid6,
            census_by_place20=census_by_place20,
            census_by_unit_name_state_family=census_by_unit_name_state_family,
            census_by_name_state_family=census_by_name_state_family,
            census_by_unit_name_family=census_by_unit_name_family,
            census_by_name_family=census_by_name_family,
        )

    @staticmethod
    def _load_census_workbook(path: Path) -> pd.DataFrame:
        if not path.is_file():
            raise SystemExit(f"Census of Governments workbook not found: {path}")
        workbook = pd.read_excel(path, sheet_name="General Purpose", engine="openpyxl", usecols=WORKBOOK_COLUMNS)
        missing = sorted(column for column in WORKBOOK_COLUMNS if column not in workbook.columns)
        if missing:
            raise SystemExit(f"Census workbook missing required columns: {missing}")

        df = workbook.copy()
        df["state_abbr"] = df["STATE"].map(_norm_upper)
        df["city_name"] = df["CITY"].map(_title_case)
        df["unit_name"] = df["UNIT_NAME"].map(_norm_text)
        df["city_norm"] = df["city_name"].map(lambda value: normalize_city_name(value).lower())
        df["unit_name_norm"] = df["unit_name"].map(lambda value: normalize_city_name(value).lower())
        df["name_norm"] = df["unit_name_norm"].where(df["unit_name_norm"].astype(str).str.len().gt(0), df["city_norm"])
        df["unit_family"] = df["UNIT_TYPE"].map(_unit_family)
        df["place20_geoid"] = df.apply(
            lambda row: f"{_norm_id(row['FIPS_STATE'], width=2)}{_norm_id(row['FIPS_PLACE'], width=5)}"
            if _norm_id(row["FIPS_STATE"], width=2) and _norm_id(row["FIPS_PLACE"], width=5)
            else "",
            axis=1,
        )
        df["census_id_pid6"] = df["CENSUS_ID_PID6"].map(_norm_id)
        df["census_id_gidid"] = df["CENSUS_ID_GIDID"].map(_norm_id)
        df["jurisdiction_key"] = df.apply(
            lambda row: make_city_key(clean_optional_str(row["city_name"]), clean_optional_str(row["state_abbr"])),
            axis=1,
        )
        return df[
            [
                "census_id_pid6",
                "census_id_gidid",
                "unit_name",
                "UNIT_TYPE",
                "city_name",
                "state_abbr",
                "city_norm",
                "unit_name_norm",
                "name_norm",
                "unit_family",
                "place20_geoid",
                "jurisdiction_key",
            ]
        ].rename(columns={"UNIT_TYPE": "census_unit_type"})

    def _score_candidate(
        self,
        *,
        candidate: dict[str, str],
        family: str,
        state_source: str,
        input_name_norm: str,
        raw_level: str,
    ) -> int:
        score = 0
        if clean_optional_str(candidate.get("unit_family")) == family:
            score += 60
        if state_source == "extracted":
            score += 20
        elif state_source == "publication_backfill":
            score += 10
        elif state_source == "national_unique_name_family":
            score += 5
        if clean_optional_str(candidate.get("unit_name_norm")) == input_name_norm:
            score += 12
        if clean_optional_str(candidate.get("city_norm")) == input_name_norm:
            score += 8
        level_norm = _norm_lower(raw_level)
        unit_type_norm = _norm_lower(candidate.get("census_unit_type"))
        if level_norm and level_norm in unit_type_norm:
            score += 6
        return score

    def resolve_issue(
        self,
        *,
        issue_id: str,
        issue_json: dict[str, Any],
        metadata_row: Any,
        publication_geo: dict[str, Any],
    ) -> dict[str, Any]:
        classification = issue_json.get("classification") if isinstance(issue_json.get("classification"), dict) else {}
        raw_jur = classification.get("ordinance_jurisdiction") if isinstance(classification.get("ordinance_jurisdiction"), dict) else {}
        raw_city = clean_optional_str(raw_jur.get("city")) or clean_optional_str(getattr(metadata_row, "jurisdiction_city_name", ""))
        raw_state = _norm_upper(raw_jur.get("state_abbr")) or _norm_upper(getattr(metadata_row, "jurisdiction_state_abbr", ""))
        raw_level = _norm_lower(raw_jur.get("jurisdiction_level")) or _norm_lower(getattr(metadata_row, "jurisdiction_level", ""))
        evidence_quote = clean_optional_str(raw_jur.get("evidence_quote")) or clean_optional_str(getattr(metadata_row, "jurisdiction_evidence_quote", ""))
        input_source = "classification.ordinance_jurisdiction" if raw_jur else "metadata_row"

        manual_override = MANUAL_JURISDICTION_OVERRIDES.get(issue_id)
        if manual_override:
            return self._resolve_manual_override(
                issue_id=issue_id,
                raw_jur=raw_jur,
                metadata_row=metadata_row,
                manual_override=manual_override,
                input_source=input_source,
                evidence_quote=evidence_quote,
            )

        publication_state = _norm_upper(publication_geo.get("publication_state_abbr"))
        name_norm = normalize_city_name(raw_city).lower()

        family_preferences = _family_from_level(raw_level, raw_city)
        state_attempts: list[tuple[str, str]] = []
        if raw_state:
            state_attempts.append((raw_state, "extracted"))
        if not raw_state and publication_state:
            state_attempts.append((publication_state, "publication_backfill"))

        candidate_entries: list[dict[str, Any]] = []
        state_backfill_used = 0

        def collect_candidates(
            *,
            use_unit_name_index: bool,
            national: bool,
        ) -> list[dict[str, Any]]:
            entries: list[dict[str, Any]] = []
            seen_candidate_keys: set[str] = set()
            if national:
                index = self.refs.census_by_unit_name_family if use_unit_name_index else self.refs.census_by_name_family
                for family in family_preferences:
                    national_candidates = index.get((name_norm, family), [])
                    if len(national_candidates) != 1:
                        continue
                    candidate = national_candidates[0]
                    key = _candidate_key(
                        clean_optional_str(candidate.get("census_id_pid6")),
                        clean_optional_str(candidate.get("census_id_gidid")),
                        clean_optional_str(candidate.get("place20_geoid")),
                        clean_optional_str(candidate.get("unit_name")),
                    )
                    if not key or key in seen_candidate_keys:
                        continue
                    seen_candidate_keys.add(key)
                    entries.append(
                        {
                            **candidate,
                            "_candidate_key": key,
                            "_family": family,
                            "_state_source": "national_unique_name_family",
                            "_score": self._score_candidate(
                                candidate=candidate,
                                family=family,
                                state_source="national_unique_name_family",
                                input_name_norm=name_norm,
                                raw_level=raw_level,
                            ),
                        }
                    )
                    break
                return entries

            index = (
                self.refs.census_by_unit_name_state_family
                if use_unit_name_index
                else self.refs.census_by_name_state_family
            )
            for state_abbr, state_source in state_attempts:
                if state_source == "publication_backfill":
                    state_backfill_used = 1
                for family in family_preferences:
                    candidates = index.get((name_norm, state_abbr.lower(), family), [])
                    for candidate in candidates:
                        key = _candidate_key(
                            clean_optional_str(candidate.get("census_id_pid6")),
                            clean_optional_str(candidate.get("census_id_gidid")),
                            clean_optional_str(candidate.get("place20_geoid")),
                            clean_optional_str(candidate.get("unit_name")),
                        )
                        if not key or key in seen_candidate_keys:
                            continue
                        seen_candidate_keys.add(key)
                        entries.append(
                            {
                                **candidate,
                                "_candidate_key": key,
                                "_family": family,
                                "_state_source": state_source,
                                "_score": self._score_candidate(
                                    candidate=candidate,
                                    family=family,
                                    state_source=state_source,
                                    input_name_norm=name_norm,
                                    raw_level=raw_level,
                                ),
                            }
                        )
            return entries

        candidate_entries = collect_candidates(use_unit_name_index=True, national=False)

        if not candidate_entries:
            candidate_entries = collect_candidates(use_unit_name_index=False, national=False)

        if not candidate_entries and name_norm:
            candidate_entries = collect_candidates(use_unit_name_index=True, national=True)

        if not candidate_entries and name_norm:
            candidate_entries = collect_candidates(use_unit_name_index=False, national=True)

        legacy_city_key = clean_optional_str(getattr(metadata_row, "city_key", ""))
        legacy_pid6 = _norm_id(getattr(metadata_row, "census_id_pid6", ""))
        legacy_place20 = _norm_id(getattr(metadata_row, "place20_geoid", ""), width=7)

        if not raw_city:
            return {
                "issue_id": issue_id,
                "jurisdiction_input_source": input_source,
                "jurisdiction_input_city_name": "",
                "jurisdiction_input_city_name_normalized": "",
                "jurisdiction_input_state_abbr": raw_state,
                "jurisdiction_input_level": raw_level,
                "jurisdiction_input_confidence_0_to_1": safe_float(raw_jur.get("confidence_0_to_1"), 0.0),
                "jurisdiction_evidence_quote": evidence_quote,
                "jurisdiction_resolution_status": "missing_extraction",
                "jurisdiction_resolution_method": "no_input_city",
                "jurisdiction_resolution_score": 0,
                "jurisdiction_candidate_count": 0,
                "jurisdiction_candidate_pid6s": "",
                "jurisdiction_state_source": "missing",
                "jurisdiction_state_backfill_used": 0,
                "jurisdiction_city_name": "",
                "jurisdiction_city_name_normalized": "",
                "jurisdiction_state_abbr": "",
                "jurisdiction_level": raw_level,
                "jurisdiction_key": "",
                "jurisdiction_display_name": "",
                "place20_geoid": "",
                "county20_fips_primary": "",
                "puma20_geoid_primary": "",
                "cbsa23_geoid_primary": "",
                "cbsa23_name_primary": "",
                "census_id_pid6": "",
                "census_id_gidid": "",
                "census_unit_name": "",
                "census_unit_type": "",
                "census_join_method": "missing_jurisdiction",
                "legacy_bundle_city_key": legacy_city_key,
                "legacy_bundle_place20_geoid": legacy_place20,
                "legacy_bundle_census_id_pid6": legacy_pid6,
            }

        if not candidate_entries:
            state_source = state_attempts[0][1] if state_attempts else "missing"
            resolved_state = state_attempts[0][0] if state_attempts else ""
            unresolved_key = make_city_key(raw_city, resolved_state)
            return {
                "issue_id": issue_id,
                "jurisdiction_input_source": input_source,
                "jurisdiction_input_city_name": raw_city,
                "jurisdiction_input_city_name_normalized": normalize_city_name(raw_city),
                "jurisdiction_input_state_abbr": raw_state,
                "jurisdiction_input_level": raw_level,
                "jurisdiction_input_confidence_0_to_1": safe_float(raw_jur.get("confidence_0_to_1"), 0.0),
                "jurisdiction_evidence_quote": evidence_quote,
                "jurisdiction_resolution_status": "extracted_unresolved",
                "jurisdiction_resolution_method": "no_census_candidate",
                "jurisdiction_resolution_score": 0,
                "jurisdiction_candidate_count": 0,
                "jurisdiction_candidate_pid6s": "",
                "jurisdiction_state_source": state_source,
                "jurisdiction_state_backfill_used": state_backfill_used,
                "jurisdiction_city_name": normalize_city_name(raw_city) or raw_city,
                "jurisdiction_city_name_normalized": normalize_city_name(raw_city),
                "jurisdiction_state_abbr": resolved_state,
                "jurisdiction_level": raw_level,
                "jurisdiction_key": unresolved_key,
                "jurisdiction_display_name": normalize_city_name(raw_city) or raw_city,
                "place20_geoid": "",
                "county20_fips_primary": "",
                "puma20_geoid_primary": "",
                "cbsa23_geoid_primary": "",
                "cbsa23_name_primary": "",
                "census_id_pid6": "",
                "census_id_gidid": "",
                "census_unit_name": "",
                "census_unit_type": "",
                "census_join_method": "unmatched_city_key" if unresolved_key else "missing_jurisdiction",
                "legacy_bundle_city_key": legacy_city_key,
                "legacy_bundle_place20_geoid": legacy_place20,
                "legacy_bundle_census_id_pid6": legacy_pid6,
            }

        candidate_entries = sorted(candidate_entries, key=lambda row: (-int(row["_score"]), clean_optional_str(row.get("census_id_pid6")), clean_optional_str(row.get("unit_name"))))
        best = candidate_entries[0]
        top_score = int(best["_score"])
        tied = [candidate for candidate in candidate_entries if int(candidate["_score"]) == top_score]
        if len({clean_optional_str(candidate.get("_candidate_key")) for candidate in tied}) > 1:
            return {
                "issue_id": issue_id,
                "jurisdiction_input_source": input_source,
                "jurisdiction_input_city_name": raw_city,
                "jurisdiction_input_city_name_normalized": normalize_city_name(raw_city),
                "jurisdiction_input_state_abbr": raw_state,
                "jurisdiction_input_level": raw_level,
                "jurisdiction_input_confidence_0_to_1": safe_float(raw_jur.get("confidence_0_to_1"), 0.0),
                "jurisdiction_evidence_quote": evidence_quote,
                "jurisdiction_resolution_status": "ambiguous_multiple_candidates",
                "jurisdiction_resolution_method": "ambiguous_top_score",
                "jurisdiction_resolution_score": top_score,
                "jurisdiction_candidate_count": len(candidate_entries),
                "jurisdiction_candidate_pid6s": "|".join(_unique_preserve_order([clean_optional_str(candidate.get("census_id_pid6")) for candidate in candidate_entries])),
                "jurisdiction_state_source": clean_optional_str(best["_state_source"]),
                "jurisdiction_state_backfill_used": state_backfill_used,
                "jurisdiction_city_name": normalize_city_name(raw_city) or raw_city,
                "jurisdiction_city_name_normalized": normalize_city_name(raw_city),
                "jurisdiction_state_abbr": _norm_upper(best.get("state_abbr")),
                "jurisdiction_level": raw_level,
                "jurisdiction_key": "",
                "jurisdiction_display_name": "",
                "place20_geoid": "",
                "county20_fips_primary": "",
                "puma20_geoid_primary": "",
                "cbsa23_geoid_primary": "",
                "cbsa23_name_primary": "",
                "census_id_pid6": "",
                "census_id_gidid": "",
                "census_unit_name": "",
                "census_unit_type": "",
                "census_join_method": "ambiguous",
                "legacy_bundle_city_key": legacy_city_key,
                "legacy_bundle_place20_geoid": legacy_place20,
                "legacy_bundle_census_id_pid6": legacy_pid6,
            }

        resolved_state = _norm_upper(best.get("state_abbr"))
        resolved_city = _pick_resolved_core_name(best, name_norm) or normalize_city_name(raw_city) or clean_optional_str(best.get("city_name"))
        resolved_level = raw_level or clean_optional_str(best.get("unit_family"))
        resolved_key = make_city_key(resolved_city, resolved_state)
        resolved_method = clean_optional_str(best["_state_source"])
        if clean_optional_str(best["_state_source"]) == "extracted":
            resolved_method = "census_city_state_exact"
        elif clean_optional_str(best["_state_source"]) == "publication_backfill":
            resolved_method = "census_city_plus_publication_state"
        elif clean_optional_str(best["_state_source"]) == "national_unique_name_family":
            resolved_method = "national_unique_name_family"

        return {
            "issue_id": issue_id,
            "jurisdiction_input_source": input_source,
            "jurisdiction_input_city_name": raw_city,
            "jurisdiction_input_city_name_normalized": normalize_city_name(raw_city),
            "jurisdiction_input_state_abbr": raw_state,
            "jurisdiction_input_level": raw_level,
            "jurisdiction_input_confidence_0_to_1": safe_float(raw_jur.get("confidence_0_to_1"), 0.0),
            "jurisdiction_evidence_quote": evidence_quote,
            "jurisdiction_resolution_status": "resolved_with_census",
            "jurisdiction_resolution_method": resolved_method,
            "jurisdiction_resolution_score": top_score,
            "jurisdiction_candidate_count": len(candidate_entries),
            "jurisdiction_candidate_pid6s": "|".join(_unique_preserve_order([clean_optional_str(candidate.get("census_id_pid6")) for candidate in candidate_entries])),
            "jurisdiction_state_source": clean_optional_str(best["_state_source"]),
            "jurisdiction_state_backfill_used": state_backfill_used,
            "jurisdiction_city_name": resolved_city,
            "jurisdiction_city_name_normalized": normalize_city_name(resolved_city),
            "jurisdiction_state_abbr": resolved_state,
            "jurisdiction_level": resolved_level,
            "jurisdiction_key": resolved_key,
            "jurisdiction_display_name": clean_optional_str(best.get("unit_name")) or resolved_city,
            "place20_geoid": clean_optional_str(best.get("place20_geoid")),
            "county20_fips_primary": "",
            "puma20_geoid_primary": "",
            "cbsa23_geoid_primary": "",
            "cbsa23_name_primary": "",
            "census_id_pid6": clean_optional_str(best.get("census_id_pid6")),
            "census_id_gidid": clean_optional_str(best.get("census_id_gidid")),
            "census_unit_name": clean_optional_str(best.get("unit_name")),
            "census_unit_type": clean_optional_str(best.get("census_unit_type")),
            "census_join_method": resolved_method,
            "legacy_bundle_city_key": legacy_city_key,
            "legacy_bundle_place20_geoid": legacy_place20,
            "legacy_bundle_census_id_pid6": legacy_pid6,
        }

    def _resolve_manual_override(
        self,
        *,
        issue_id: str,
        raw_jur: dict[str, Any],
        metadata_row: Any,
        manual_override: dict[str, Any],
        input_source: str,
        evidence_quote: str,
    ) -> dict[str, Any]:
        target_pid6 = clean_optional_str(manual_override.get("target_pid6"))
        candidate = self.refs.census_by_pid6.get(target_pid6)
        if candidate is None:
            raise SystemExit(f"Manual jurisdiction override for issue_id={issue_id} references missing pid6={target_pid6}")

        resolved_city = clean_optional_str(manual_override.get("city_name")) or _pick_resolved_core_name(candidate, normalize_city_name(clean_optional_str(manual_override.get("city_name"))).lower())
        resolved_state = _norm_upper(manual_override.get("state_abbr")) or _norm_upper(candidate.get("state_abbr"))
        resolved_level = _norm_lower(manual_override.get("jurisdiction_level")) or clean_optional_str(candidate.get("unit_family"))
        resolved_key = make_city_key(resolved_city, resolved_state)

        raw_city = clean_optional_str(raw_jur.get("city")) or clean_optional_str(getattr(metadata_row, "jurisdiction_city_name", ""))
        raw_state = _norm_upper(raw_jur.get("state_abbr")) or _norm_upper(getattr(metadata_row, "jurisdiction_state_abbr", ""))
        raw_level = _norm_lower(raw_jur.get("jurisdiction_level")) or _norm_lower(getattr(metadata_row, "jurisdiction_level", ""))
        legacy_city_key = clean_optional_str(getattr(metadata_row, "city_key", ""))
        legacy_pid6 = _norm_id(getattr(metadata_row, "census_id_pid6", ""))
        legacy_place20 = _norm_id(getattr(metadata_row, "place20_geoid", ""), width=7)

        return {
            "issue_id": issue_id,
            "jurisdiction_input_source": input_source,
            "jurisdiction_input_city_name": raw_city,
            "jurisdiction_input_city_name_normalized": normalize_city_name(raw_city),
            "jurisdiction_input_state_abbr": raw_state,
            "jurisdiction_input_level": raw_level,
            "jurisdiction_input_confidence_0_to_1": safe_float(raw_jur.get("confidence_0_to_1"), 0.0),
            "jurisdiction_evidence_quote": evidence_quote,
            "jurisdiction_resolution_status": "resolved_with_census",
            "jurisdiction_resolution_method": "manual_issue_override",
            "jurisdiction_resolution_score": 999,
            "jurisdiction_candidate_count": 1,
            "jurisdiction_candidate_pid6s": target_pid6,
            "jurisdiction_state_source": "manual_override",
            "jurisdiction_state_backfill_used": 0,
            "jurisdiction_city_name": resolved_city,
            "jurisdiction_city_name_normalized": normalize_city_name(resolved_city),
            "jurisdiction_state_abbr": resolved_state,
            "jurisdiction_level": resolved_level,
            "jurisdiction_key": resolved_key,
            "jurisdiction_display_name": clean_optional_str(candidate.get("unit_name")),
            "place20_geoid": clean_optional_str(candidate.get("place20_geoid")),
            "county20_fips_primary": "",
            "puma20_geoid_primary": "",
            "cbsa23_geoid_primary": "",
            "cbsa23_name_primary": "",
            "census_id_pid6": target_pid6,
            "census_id_gidid": clean_optional_str(candidate.get("census_id_gidid")),
            "census_unit_name": clean_optional_str(candidate.get("unit_name")),
            "census_unit_type": clean_optional_str(candidate.get("census_unit_type")),
            "census_join_method": "manual_issue_override",
            "manual_override_note": clean_optional_str(manual_override.get("note")),
            "legacy_bundle_city_key": legacy_city_key,
            "legacy_bundle_place20_geoid": legacy_place20,
            "legacy_bundle_census_id_pid6": legacy_pid6,
        }


def jurisdiction_resolution_summary(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {
            "issue_count": 0,
            "status_counts": {},
            "method_counts": {},
            "resolved_with_census_count": 0,
            "state_backfill_count": 0,
            "bundle_city_key_coverage": 0,
            "resolver_city_key_coverage": 0,
            "bundle_pid6_coverage": 0,
            "resolver_pid6_coverage": 0,
            "city_key_agreement_count": 0,
            "pid6_agreement_count": 0,
            "new_city_key_count": 0,
            "new_pid6_count": 0,
        }

    bundle_city_key = df["legacy_bundle_city_key"].astype(str).str.strip()
    bundle_pid6 = df["legacy_bundle_census_id_pid6"].astype(str).str.strip()
    resolver_city_key = df["jurisdiction_key"].astype(str).str.strip()
    resolver_pid6 = df["census_id_pid6"].astype(str).str.strip()

    bundle_city_mask = bundle_city_key.ne("")
    bundle_pid6_mask = bundle_pid6.ne("")
    resolver_city_mask = resolver_city_key.ne("")
    resolver_pid6_mask = resolver_pid6.ne("")

    city_key_agreement = int((bundle_city_mask & resolver_city_mask & bundle_city_key.eq(resolver_city_key)).sum())
    pid6_agreement = int((bundle_pid6_mask & resolver_pid6_mask & bundle_pid6.eq(resolver_pid6)).sum())

    return {
        "issue_count": int(len(df)),
        "status_counts": {str(k): int(v) for k, v in df["jurisdiction_resolution_status"].value_counts(dropna=False).to_dict().items()},
        "method_counts": {str(k): int(v) for k, v in df["jurisdiction_resolution_method"].value_counts(dropna=False).to_dict().items()},
        "resolved_with_census_count": int((df["jurisdiction_resolution_status"] == "resolved_with_census").sum()),
        "state_backfill_count": int(df["jurisdiction_state_backfill_used"].fillna(0).astype(int).sum()),
        "bundle_city_key_coverage": int(bundle_city_mask.sum()),
        "resolver_city_key_coverage": int(resolver_city_mask.sum()),
        "bundle_pid6_coverage": int(bundle_pid6_mask.sum()),
        "resolver_pid6_coverage": int(resolver_pid6_mask.sum()),
        "city_key_agreement_count": city_key_agreement,
        "pid6_agreement_count": pid6_agreement,
        "new_city_key_count": int((~bundle_city_mask & resolver_city_mask).sum()),
        "new_pid6_count": int((~bundle_pid6_mask & resolver_pid6_mask).sum()),
    }
