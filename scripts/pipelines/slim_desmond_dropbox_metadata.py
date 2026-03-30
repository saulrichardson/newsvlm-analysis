#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from export_jurisdiction_utils import (
    build_jurisdiction_resolver,
    metadata_namespace,
    resolution_has_census_fields,
    resolve_issue_with_publication_fallback,
    resolve_issue_with_preferred_public_fallback,
)


RAW_DATA_ROOT = Path(
    "/Users/saulrichardson/Dropbox/Inclusionary Zoning/Historical Analysis/raw_data"
)
LOCAL_ARCHIVE_ROOT = Path(
    "/Users/saulrichardson/projects/newspapers/newspaper-analysis/artifacts/scratch/"
    "desmond_newspaper_ordinances_archive_20260311"
)
RUN_ARCHIVE_ROOT = LOCAL_ARCHIVE_ROOT / "metadata_cleanup_pass9_20260313"
ARCHIVE_METADATA_CSV = Path(
    "/Users/saulrichardson/Dropbox/Inclusionary Zoning/Historical Analysis/newspapers/archive/v2/metadata.csv"
)
CENSUS_OF_GOV_XLSX = Path(
    "/Users/saulrichardson/Dropbox/Inclusionary Zoning/Historical Analysis/raw_data/census_data/census_of_gov_22.xlsx"
)


@dataclass(frozen=True)
class CorpusConfig:
    folder_name: str
    fields: tuple[str, ...]
    readme: str
    rich_metadata_path: Path
    normalize_document_class: str


FULL_FIELDS = (
    "issue_id",
    "issue_date",
    "newspaper_slug",
    "document_class",
    "document_operativity",
    "document_legal_action",
    "extracted_jurisdiction_city_name",
    "extracted_jurisdiction_state_abbr",
    "newspaperarchive_publication_city_name",
    "newspaperarchive_publication_state_abbr",
    "census_id_pid6",
    "census_id_gidid",
    "census_unit_name",
    "census_unit_type",
    "census_match_source",
    "census_match_method",
    "page_count",
    "parsed_relpath",
    "raw_dir_relpath",
    "raw_image_count",
    "transcript_relpath",
    "full_ordinance_origin",
)


AMENDMENT_FIELDS = (
    "issue_id",
    "issue_date",
    "newspaper_slug",
    "document_class",
    "document_operativity",
    "document_legal_action",
    "extracted_jurisdiction_city_name",
    "extracted_jurisdiction_state_abbr",
    "newspaperarchive_publication_city_name",
    "newspaperarchive_publication_state_abbr",
    "census_id_pid6",
    "census_id_gidid",
    "census_unit_name",
    "census_unit_type",
    "census_match_source",
    "census_match_method",
    "page_count",
    "parsed_relpath",
    "raw_dir_relpath",
    "raw_image_count",
    "transcript_relpath",
)


CORPORA = (
    CorpusConfig(
        folder_name="newspaper_ordinances",
        fields=FULL_FIELDS,
        normalize_document_class="code_publication_full_issue",
        rich_metadata_path=LOCAL_ARCHIVE_ROOT
        / "dropbox_extras/metadata_before_stripping_dedupe_columns_20260311.csv",
        readme="""# Newspaper Ordinances

This folder is the canonical cleaned full-ordinance corpus for project use.

Layout:
- `metadata.csv`: the working metadata file for the corpus
- `parsed/<issue_id>.md`: clean ordinance text per issue
- `raw/<issue_id>/*.png`: source page images for that issue
- `transcript/<issue_id>.txt`: source newspaper OCR transcript for that issue

The metadata file keeps only the public fields needed to work with the corpus:
- issue identity and dates
- final ordinance classification fields
- extracted jurisdiction city/state from the LLM classification pipeline, when present
- NewspaperArchive publication city/state
- final Census of Governments identifiers and unit labels
- human-readable Census match provenance
- relative file paths into this folder

Key metadata notes:
- `document_class` is normalized to `code_publication_full_issue` for every row in this corpus.
- `document_operativity` excludes proposal-state rows but may still be `unclear` for some rows.
- `census_id_*` and `census_unit_*` store the final public Census match for the issue. The builder always prefers a resolved ordinance-target jurisdiction. If that target is missing, ambiguous, or otherwise unresolved, the public Census fields fall back to the newspaper publication municipality when that resolves cleanly.
- `census_match_source` says whether the public Census match came from extracted jurisdiction, publication fallback, manual override, or no usable jurisdiction.
- `census_match_method` says how the Census match was obtained, for example exact city/state match or unique national name-family match.
- `extracted_jurisdiction_*` can therefore differ from the final public `census_id_*` target on publication-fallback rows.

This corpus already excludes proposal-state ordinances and reviewed duplicate / reprint instances.
""",
    ),
    CorpusConfig(
        folder_name="newspaper_ordinances_amendment_substantial",
        fields=AMENDMENT_FIELDS,
        normalize_document_class="amendment_substantial_text",
        rich_metadata_path=LOCAL_ARCHIVE_ROOT
        / "amendment_corpus_cleanup_20260312/newspaper_ordinances_amendment_substantial_remainder/metadata.csv",
        readme="""# Amendment Substantial Ordinances

This folder contains substantial zoning amendments with usable clean transcriptions.

Layout:
- `metadata.csv`: the working metadata file for the corpus
- `parsed/<issue_id>.md`: clean ordinance text
- `raw/<issue_id>/*.png`: source page images for that issue
- `transcript/<issue_id>.txt`: source newspaper OCR transcript for that issue

The metadata file keeps only the public fields needed to work with the corpus:
- issue identity and dates
- amendment classification fields
- extracted jurisdiction city/state from the LLM classification pipeline, when present
- NewspaperArchive publication city/state
- final Census of Governments identifiers and unit labels
- human-readable Census match provenance
- relative file paths into this folder

Key metadata notes:
- `document_class` is normalized to `amendment_substantial_text` for every row in this corpus.
- `document_operativity` excludes proposal-state rows but may still be `unclear` for some rows.
- `census_id_*` and `census_unit_*` store the final public Census match for the issue. The builder always prefers a resolved ordinance-target jurisdiction. If that target is missing, ambiguous, or otherwise unresolved, the public Census fields fall back to the newspaper publication municipality when that resolves cleanly.
- `census_match_source` says whether the public Census match came from extracted jurisdiction, publication fallback, manual override, or no usable jurisdiction.
- `census_match_method` says how the Census match was obtained, for example exact city/state match or unique national name-family match.
- `extracted_jurisdiction_*` can therefore differ from the final public `census_id_*` target on publication-fallback rows.

This corpus excludes rows the LLM marked as proposal-state and excludes rows promoted into the full-ordinance corpus.
""",
    ),
)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, fieldnames: tuple[str, ...], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def backup_file(src: Path, backup_dir: Path) -> None:
    backup_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, backup_dir / src.name)


def load_rich_rows(path: Path) -> dict[str, dict[str, str]]:
    rows = read_rows(path)
    return {row["issue_id"]: row for row in rows}


def clean(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def derive_match_fields(resolution: dict[str, str]) -> tuple[str, str]:
    raw_method = clean(resolution.get("census_join_method"))
    resolution_method = clean(resolution.get("jurisdiction_resolution_method"))
    input_source = clean(resolution.get("jurisdiction_input_source"))

    if raw_method in {"missing_jurisdiction", "Missing jurisdiction"}:
        return ("Missing jurisdiction", "Missing jurisdiction")
    if raw_method in {"manual_issue_override", "Manual override"} or resolution_method == "manual_issue_override":
        return ("Manual override", "Manual override")

    core_method = raw_method.removeprefix("publication::")

    if core_method in {"archive_geoid_plus_extracted", "census_city_state_exact"}:
        method = "Exact city/state match"
    elif core_method in {"archive_geoid_plus_publication_backfill", "census_city_plus_publication_state"}:
        method = "City match with publication-state backfill"
    elif core_method == "national_unique_name_family":
        method = "Unique national name-family match"
    elif core_method == "ambiguous":
        method = "Ambiguous municipality match"
    elif core_method == "unmatched_city_key":
        method = "Unresolved city/state match"
    elif core_method:
        raise SystemExit(
            f"Unrecognized census join method '{raw_method}' for issue_id={resolution.get('issue_id', '')}"
        )
    else:
        raise SystemExit(f"Blank census join method for issue_id={resolution.get('issue_id', '')}")

    if raw_method.startswith("publication::") or input_source == "publication_location_fallback":
        source = "Publication fallback"
    else:
        source = "Extracted jurisdiction"

    return (source, method)


def public_extracted_city(public_row: dict[str, str], rich_row: dict[str, str]) -> str:
    return clean(rich_row.get("jurisdiction_city_name")) or clean(
        public_row.get("extracted_jurisdiction_city_name")
    )


def public_extracted_state(public_row: dict[str, str], rich_row: dict[str, str]) -> str:
    return clean(rich_row.get("jurisdiction_state_abbr")) or clean(
        public_row.get("extracted_jurisdiction_state_abbr")
    )


def public_publication_city(public_row: dict[str, str], rich_row: dict[str, str]) -> str:
    return clean(rich_row.get("publication_city_name")) or clean(
        public_row.get("newspaperarchive_publication_city_name")
    )


def public_publication_state(public_row: dict[str, str], rich_row: dict[str, str]) -> str:
    return clean(rich_row.get("publication_state_abbr")) or clean(
        public_row.get("newspaperarchive_publication_state_abbr")
    )


def resolve_publication_fallback_census(
    public_row: dict[str, str],
    rich_row: dict[str, str],
    resolver: object,
) -> dict[str, str]:
    publication_geo = {
        "publication_city_name": public_publication_city(public_row, rich_row),
        "publication_state_abbr": public_publication_state(public_row, rich_row),
    }
    if not publication_geo["publication_city_name"] or not publication_geo["publication_state_abbr"]:
        return {}
    resolution = resolve_issue_with_publication_fallback(
        resolver=resolver,
        issue_id=f"{public_row['issue_id']}::publication_fallback_probe",
        issue_json={"classification": {}},
        metadata_row=metadata_namespace(
            {
                "jurisdiction_city_name": "",
                "jurisdiction_state_abbr": "",
                "jurisdiction_level": "",
                "jurisdiction_evidence_quote": "",
            }
        ),
        publication_geo=publication_geo,
    )
    cleaned = {key: clean(value) for key, value in resolution.items()}
    if not resolution_has_census_fields(cleaned):
        return {}
    cleaned["issue_id"] = public_row["issue_id"]
    return cleaned


def rebuild_rows(
    public_rows: list[dict[str, str]],
    rich_rows_by_issue: dict[str, dict[str, str]],
    config: CorpusConfig,
    resolver: object,
) -> list[dict[str, str]]:
    rebuilt: list[dict[str, str]] = []
    for row in public_rows:
        issue_id = row["issue_id"]
        if config.folder_name == "newspaper_ordinances" and issue_id == "joplin-globe__1949-02-16":
            continue
        rich = rich_rows_by_issue.get(issue_id)
        if rich is None:
            raise SystemExit(f"Missing rich metadata row for issue_id={issue_id}")

        resolution = {
            key: clean(value)
            for key, value in resolve_issue_with_preferred_public_fallback(
                resolver=resolver,
                issue_id=issue_id,
                issue_json={"classification": {}},
                metadata_row=metadata_namespace(
                    {
                        **dict(rich),
                        "jurisdiction_city_name": public_extracted_city(row, rich),
                        "jurisdiction_state_abbr": public_extracted_state(row, rich),
                    }
                ),
                publication_geo={
                    "publication_city_name": public_publication_city(row, rich),
                    "publication_state_abbr": public_publication_state(row, rich),
                },
            ).items()
        }
        match_source, match_method = derive_match_fields(resolution)

        out = {field: "" for field in config.fields}
        out["issue_id"] = issue_id
        out["issue_date"] = clean(row.get("issue_date"))
        out["newspaper_slug"] = clean(row.get("newspaper_slug"))
        out["document_class"] = config.normalize_document_class
        out["document_operativity"] = clean(row.get("document_operativity"))
        out["document_legal_action"] = clean(row.get("document_legal_action"))
        out["extracted_jurisdiction_city_name"] = public_extracted_city(row, rich)
        out["extracted_jurisdiction_state_abbr"] = public_extracted_state(row, rich)
        out["newspaperarchive_publication_city_name"] = public_publication_city(row, rich)
        out["newspaperarchive_publication_state_abbr"] = public_publication_state(row, rich)
        out["census_id_pid6"] = clean(resolution.get("census_id_pid6"))
        out["census_id_gidid"] = clean(resolution.get("census_id_gidid"))
        out["census_unit_name"] = clean(resolution.get("census_unit_name"))
        out["census_unit_type"] = clean(resolution.get("census_unit_type"))
        out["census_match_source"] = match_source
        out["census_match_method"] = match_method
        out["page_count"] = clean(row.get("page_count"))
        out["parsed_relpath"] = clean(row.get("parsed_relpath"))
        out["raw_dir_relpath"] = clean(row.get("raw_dir_relpath"))
        out["raw_image_count"] = clean(row.get("raw_image_count"))
        out["transcript_relpath"] = clean(row.get("transcript_relpath"))
        if "full_ordinance_origin" in config.fields:
            out["full_ordinance_origin"] = clean(row.get("full_ordinance_origin"))

        rebuilt.append(out)
    return rebuilt


def validate_census_consistency(rows: list[dict[str, str]]) -> None:
    violations: list[str] = []
    for row in rows:
        issue_id = row["issue_id"]
        source = clean(row.get("census_match_source"))
        method = clean(row.get("census_match_method"))
        pid6 = clean(row.get("census_id_pid6"))
        gidid = clean(row.get("census_id_gidid"))
        unit_name = clean(row.get("census_unit_name"))
        unit_type = clean(row.get("census_unit_type"))
        publication_city = clean(row.get("newspaperarchive_publication_city_name"))
        publication_state = clean(row.get("newspaperarchive_publication_state_abbr"))
        has_census = bool(pid6 or gidid or unit_name or unit_type)

        if source == "Missing jurisdiction" and has_census:
            violations.append(f"{issue_id}: Missing jurisdiction source cannot keep public Census identifiers")
        if method == "Missing jurisdiction" and has_census:
            violations.append(f"{issue_id}: Missing jurisdiction method cannot keep public Census identifiers")
        if (source == "Missing jurisdiction") != (method == "Missing jurisdiction"):
            violations.append(f"{issue_id}: Missing jurisdiction source/method must agree")
        if source == "Publication fallback" and (not publication_city or not publication_state):
            violations.append(f"{issue_id}: Publication fallback requires publication city/state")
        if has_census and method == "Missing jurisdiction":
            violations.append(f"{issue_id}: Census identifiers require a non-missing public match method")
        if not has_census and source == "Publication fallback":
            violations.append(f"{issue_id}: Publication fallback rows must keep public Census identifiers")

    if violations:
        preview = "\n".join(violations[:20])
        raise SystemExit(f"Census consistency validation failed:\n{preview}")


def validate_blank_pid_policy(
    public_rows: list[dict[str, str]],
    rich_rows_by_issue: dict[str, dict[str, str]],
    resolver: object,
) -> None:
    violations: list[str] = []
    for row in public_rows:
        if clean(row.get("census_id_pid6")):
            continue
        issue_id = row["issue_id"]
        rich = rich_rows_by_issue[issue_id]
        publication_probe = resolve_publication_fallback_census(row, rich, resolver)
        if publication_probe:
            violations.append(
                f"{issue_id}: blank public PID is not allowed because publication fallback resolves to "
                f"{publication_probe.get('census_id_pid6', '')}"
            )

    if violations:
        preview = "\n".join(violations[:20])
        raise SystemExit(f"Blank PID fallback validation failed:\n{preview}")


def summarize_rows(rows: list[dict[str, str]]) -> dict[str, object]:
    source_counts = Counter(clean(row.get("census_match_source")) for row in rows)
    method_counts = Counter(clean(row.get("census_match_method")) for row in rows)
    blank_pid_rows = [row for row in rows if not clean(row.get("census_id_pid6"))]
    blank_pid_counts = Counter(
        (clean(row.get("census_match_source")), clean(row.get("census_match_method")))
        for row in blank_pid_rows
    )
    return {
        "blank_pid_rows": len(blank_pid_rows),
        "census_match_source_counts": dict(source_counts),
        "census_match_method_counts": dict(method_counts),
        "blank_pid_breakdown": {
            f"{source} | {method}": count
            for (source, method), count in blank_pid_counts.items()
        },
    }


def validate_paths(root: Path, rows: list[dict[str, str]]) -> dict[str, int]:
    missing = {"parsed": 0, "transcript": 0, "raw": 0}
    for row in rows:
        parsed = row.get("parsed_relpath", "")
        transcript = row.get("transcript_relpath", "")
        raw_dir = row.get("raw_dir_relpath", "")
        if parsed and not (root / parsed).is_file():
            missing["parsed"] += 1
        if transcript and not (root / transcript).is_file():
            missing["transcript"] += 1
        if raw_dir and not (root / raw_dir).is_dir():
            missing["raw"] += 1
    return missing


def remove_ds_store(root: Path) -> None:
    ds_store = root / ".DS_Store"
    if ds_store.exists():
        ds_store.unlink()


def move_joplin_to_archive(corpus_root: Path, archive_dir: Path) -> dict[str, bool]:
    issue_id = "joplin-globe__1949-02-16"
    moved: dict[str, bool] = {}
    for rel in (
        f"parsed/{issue_id}.md",
        f"transcript/{issue_id}.txt",
        f"raw/{issue_id}",
    ):
        src = corpus_root / rel
        dst = archive_dir / rel
        if src.is_dir():
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists():
                shutil.rmtree(dst)
            shutil.move(str(src), str(dst))
            moved[rel] = True
        elif src.is_file():
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists():
                dst.unlink()
            shutil.move(str(src), str(dst))
            moved[rel] = True
        else:
            moved[rel] = False
    return moved


def main() -> None:
    RUN_ARCHIVE_ROOT.mkdir(parents=True, exist_ok=True)
    summary: dict[str, object] = {"corpora": {}}
    if not ARCHIVE_METADATA_CSV.is_file():
        raise SystemExit(f"Missing archive metadata source: {ARCHIVE_METADATA_CSV}")
    if not CENSUS_OF_GOV_XLSX.is_file():
        raise SystemExit(f"Missing Census workbook: {CENSUS_OF_GOV_XLSX}")
    resolver = build_jurisdiction_resolver(
        archive_metadata_csv=ARCHIVE_METADATA_CSV,
        census_of_gov_xlsx=CENSUS_OF_GOV_XLSX,
    )

    for config in CORPORA:
        corpus_root = RAW_DATA_ROOT / config.folder_name
        metadata_path = corpus_root / "metadata.csv"
        readme_path = corpus_root / "README.md"
        if not metadata_path.is_file():
            raise SystemExit(f"Missing metadata file: {metadata_path}")
        if not readme_path.is_file():
            raise SystemExit(f"Missing README file: {readme_path}")
        if not config.rich_metadata_path.is_file():
            raise SystemExit(f"Missing rich metadata source: {config.rich_metadata_path}")

        backup_dir = RUN_ARCHIVE_ROOT / config.folder_name
        backup_file(metadata_path, backup_dir)
        backup_file(readme_path, backup_dir)

        if config.folder_name == "newspaper_ordinances":
            moved = move_joplin_to_archive(corpus_root, backup_dir / "removed_issue_files")
            summary["joplin_removed"] = moved

        rows = read_rows(metadata_path)
        rich_rows = load_rich_rows(config.rich_metadata_path)
        rebuilt = rebuild_rows(rows, rich_rows, config, resolver)
        validate_census_consistency(rebuilt)
        validate_blank_pid_policy(rebuilt, rich_rows, resolver)
        write_rows(metadata_path, config.fields, rebuilt)
        readme_path.write_text(config.readme, encoding="utf-8")

        remove_ds_store(corpus_root)
        missing = validate_paths(corpus_root, rebuilt)
        summary["corpora"][config.folder_name] = {
            "rows": len(rebuilt),
            "fields": list(config.fields),
            "missing": missing,
            "census_summary": summarize_rows(rebuilt),
        }

    remove_ds_store(RAW_DATA_ROOT)
    (RUN_ARCHIVE_ROOT / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
