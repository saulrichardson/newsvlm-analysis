from __future__ import annotations

import csv
import json
from pathlib import Path

import pandas as pd

from newsvlm_analysis.frontier import (
    FrontierPaths,
    build_embedding_outputs,
    build_fullcode_corpus,
    build_geo_outputs,
    build_report_bundle,
    build_review_packets,
    build_semantics_outputs,
    build_story_outputs,
    build_structure_outputs,
    default_experiments,
    ensure_run_layout,
    run_cluster_experiments,
)


def _write_issue_bundle(
    bundle_dir: Path,
    *,
    issue_id: str,
    newspaper_slug: str,
    issue_date: str,
    jurisdiction_city: str,
    jurisdiction_state: str,
    jurisdiction_level: str,
    ordinance_text: str,
    ordinance_status: str = "ok",
) -> dict[str, str]:
    issue_dir = bundle_dir / "issues" / issue_id
    ordinance_dir = issue_dir / "ordinance"
    pages_dir = issue_dir / "pages"
    images_dir = issue_dir / "images"
    issue_dir.mkdir(parents=True, exist_ok=True)
    ordinance_dir.mkdir(parents=True, exist_ok=True)
    pages_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    issue_transcript_relpath = f"issues/{issue_id}/issue_transcript.txt"
    issue_json_relpath = f"issues/{issue_id}/issue.json"
    ordinance_relpath = f"issues/{issue_id}/ordinance/ordinance.txt"
    ordinance_raw_relpath = f"issues/{issue_id}/ordinance/ordinance_raw.txt"
    ordinance_notes_relpath = f"issues/{issue_id}/ordinance/ordinance_notes.txt"
    page_text_relpath = f"issues/{issue_id}/pages/0001__{issue_id}-p-1.txt"
    image_relpath = f"issues/{issue_id}/images/0001__{issue_id}-p-1.png"

    (bundle_dir / issue_transcript_relpath).write_text(ordinance_text, encoding="utf-8")
    (bundle_dir / ordinance_relpath).write_text(ordinance_text, encoding="utf-8")
    (bundle_dir / ordinance_raw_relpath).write_text(f"SOURCE: test\n=== ORDINANCE TEXT ===\n{ordinance_text}\n--- NOTES ---\n", encoding="utf-8")
    (bundle_dir / ordinance_notes_relpath).write_text("", encoding="utf-8")
    (bundle_dir / page_text_relpath).write_text(ordinance_text[:500], encoding="utf-8")
    (bundle_dir / image_relpath).write_bytes(b"")

    issue_json = {
        "schema_version": "test",
        "issue_id": issue_id,
        "issue_date": issue_date,
        "newspaper_slug": newspaper_slug,
        "classification": {
            "status": "ok",
            "model": "test-classifier",
            "ordinance_jurisdiction": {
                "city": jurisdiction_city,
                "state_abbr": jurisdiction_state,
                "jurisdiction_level": jurisdiction_level,
                "confidence_0_to_1": 0.95,
                "evidence_quote": f"{jurisdiction_level.title()} of {jurisdiction_city}",
            },
        },
        "metadata": {
            "issue_id": issue_id,
            "issue_date": issue_date,
            "newspaper_slug": newspaper_slug,
            "document_class": "code_publication_full_issue",
            "document_operativity": "operative",
            "document_scope": "citywide",
            "jurisdiction_city_name": jurisdiction_city,
            "jurisdiction_state_abbr": jurisdiction_state,
            "jurisdiction_level": jurisdiction_level,
            "city_key": f"{jurisdiction_city.lower().replace(' ', '-')}__{jurisdiction_state.lower()}",
            "issue_json_relpath": issue_json_relpath,
            "issue_transcript_relpath": issue_transcript_relpath,
            "page_count": 1,
            "ordinance_status": ordinance_status,
            "ordinance_relpath": ordinance_relpath,
            "ordinance_raw_relpath": ordinance_raw_relpath,
            "ordinance_notes_relpath": ordinance_notes_relpath,
            "ordinance_contract_ok": "1",
            "ordinance_repair_used": "0",
            "ordinance_chars": str(len(ordinance_text)),
            "ordinance_model": "test-model",
            "ordinance_prompt": "test-prompt",
            "ordinance_source_line": "page_id=test-p-1",
        },
        "issue_transcript_relpath": issue_transcript_relpath,
        "pages": [
            {
                "page_id": f"{issue_id}-p-1",
                "page_num": 1,
                "page_text_relpath": page_text_relpath,
                "image_relpath": image_relpath,
                "text_source": "test",
                "page_text_chars": len(ordinance_text[:500]),
                "ocr_input_mode": "test",
                "ocr_input_png_external_path": "",
                "ocr_input_provenance_external_path": "",
                "ocr_input_provenance_type": "",
                "source_text_origin_external_path": "",
            }
        ],
        "ordinance_reconstruction": {
            "status": ordinance_status,
            "contract_ok": 1,
            "repair_used": False,
            "ordinance_relpath": ordinance_relpath,
            "ordinance_raw_relpath": ordinance_raw_relpath,
        },
    }
    (bundle_dir / issue_json_relpath).write_text(json.dumps(issue_json, indent=2) + "\n", encoding="utf-8")

    return {
        "issue_id": issue_id,
        "issue_date": issue_date,
        "newspaper_slug": newspaper_slug,
        "document_class": "code_publication_full_issue",
        "document_operativity": "operative",
        "document_scope": "citywide",
        "jurisdiction_city_name": jurisdiction_city,
        "jurisdiction_state_abbr": jurisdiction_state,
        "jurisdiction_level": jurisdiction_level,
        "jurisdiction_evidence_quote": f"City of {jurisdiction_city}",
        "city_key": f"{jurisdiction_city.lower().replace(' ', '-')}__{jurisdiction_state.lower()}",
        "page_count": 1,
        "quality_flags_json": "[]",
        "issue_json_relpath": issue_json_relpath,
        "issue_transcript_relpath": issue_transcript_relpath,
        "ordinance_status": ordinance_status,
        "ordinance_relpath": ordinance_relpath,
        "ordinance_raw_relpath": ordinance_raw_relpath,
        "ordinance_notes_relpath": ordinance_notes_relpath,
        "ordinance_contract_ok": "1",
        "ordinance_repair_used": "0",
        "ordinance_chars": len(ordinance_text),
        "ordinance_model": "test-model",
        "ordinance_prompt": "test-prompt",
        "ordinance_source_line": "page_id=test-p-1",
    }


def _build_fixture(tmp_path: Path) -> tuple[Path, FrontierPaths]:
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir(parents=True, exist_ok=True)

    rows = [
        _write_issue_bundle(
            bundle_dir,
            issue_id="springfield-register__1950-01-01",
            newspaper_slug="springfield-register",
            issue_date="1950-01-01",
            jurisdiction_city="Springfield",
            jurisdiction_state="IL",
            jurisdiction_level="city",
            ordinance_text=(
                "ORDINANCE NO. 1\n\nARTICLE I PURPOSE\n"
                "The purpose of this ordinance is to promote health, safety, and the general welfare.\n\n"
                "SECTION 1 DEFINITIONS\nDwelling means a building designed for residential occupancy.\n\n"
                "SECTION 2 LOT AND YARD\nMinimum lot area shall be 6000 square feet. Front yard shall be 25 feet. "
                "Off-street parking shall provide 2 spaces per dwelling.\n"
            ),
        ),
        _write_issue_bundle(
            bundle_dir,
            issue_id="shelbyville-times__1950-01-02",
            newspaper_slug="shelbyville-times",
            issue_date="1950-01-02",
            jurisdiction_city="Springfield",
            jurisdiction_state="IL",
            jurisdiction_level="city",
            ordinance_text=(
                "ORDINANCE NO. 2\n\nARTICLE I PURPOSE\n"
                "This ordinance is intended to assure orderly growth and adequate streets, water, and sewer.\n\n"
                "SECTION 3 DISTRICTS\nThe R-1 district is established.\n\n"
                "SECTION 4 USES\nSingle-family dwellings shall be permitted. Industrial uses shall not be permitted.\n"
            ),
        ),
        _write_issue_bundle(
            bundle_dir,
            issue_id="capital-gazette__1950-01-03",
            newspaper_slug="capital-gazette",
            issue_date="1950-01-03",
            jurisdiction_city="Village of River Forest",
            jurisdiction_state="IL",
            jurisdiction_level="village",
            ordinance_text=(
                "ORDINANCE NO. 2\n\nARTICLE I PURPOSE\n"
                "This ordinance is intended to assure orderly growth and adequate streets, water, and sewer.\n\n"
                "SECTION 3 DISTRICTS\nThe R-1 district is established.\n\n"
                "SECTION 4 USES\nSingle-family dwellings shall be permitted. Industrial uses shall not be permitted.\n"
            ),
        ),
        _write_issue_bundle(
            bundle_dir,
            issue_id="ogdenville-herald__1951-02-01",
            newspaper_slug="ogdenville-herald",
            issue_date="1951-02-01",
            jurisdiction_city="Ogdenville",
            jurisdiction_state="",
            jurisdiction_level="city",
            ordinance_text=(
                "ORDINANCE NO. 4\n\nCHAPTER 1 PURPOSE\n"
                "The intent of this ordinance is to conserve the value of buildings and coordinate public services.\n\n"
                "SECTION 6 ADMINISTRATION\nThe planning commission may review site plans. Variances may be heard by the board of appeals.\n\n"
                "SECTION 7 ENFORCEMENT\nViolations shall be unlawful and penalties may include a fine of 100 dollars.\n"
            ),
        ),
    ]
    with (bundle_dir / "metadata.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    archive_csv = tmp_path / "archive_v2_metadata.csv"
    pd.DataFrame(
        [
            {
                "issue_id": "springfield-register__1950-01-01",
                "newspaper_slug": "springfield-register",
                "city_name": "Springfield",
                "state_abbr": "il",
                "city_key": "springfield__il",
                "place20_geoid": "1716700",
                "county20_fips_primary": "17167",
                "puma20_geoid_primary": "17001",
                "cbsa23_geoid_primary": "44123",
                "cbsa23_name_primary": "Springfield CBSA",
                "census_id_pid6": "200001",
                "census_id_gidid": "1101716700000",
                "census_unit_name": "CITY OF SPRINGFIELD",
                "census_unit_type": "2 - MUNICIPAL",
                "census_join_method": "place20_geoid_to_municipal",
            },
            {
                "issue_id": "shelbyville-times__1950-01-02",
                "newspaper_slug": "shelbyville-times",
                "city_name": "Shelbyville",
                "state_abbr": "il",
                "city_key": "shelbyville__il",
                "place20_geoid": "1772500",
                "county20_fips_primary": "17173",
                "puma20_geoid_primary": "17002",
                "cbsa23_geoid_primary": "44124",
                "cbsa23_name_primary": "Shelbyville CBSA",
                "census_id_pid6": "200002",
                "census_id_gidid": "1101772500000",
                "census_unit_name": "CITY OF SHELBYVILLE",
                "census_unit_type": "2 - MUNICIPAL",
                "census_join_method": "place20_geoid_to_municipal",
            },
            {
                "issue_id": "capital-gazette__1950-01-03",
                "newspaper_slug": "capital-gazette",
                "city_name": "Capital City",
                "state_abbr": "il",
                "city_key": "capital-city__il",
                "place20_geoid": "",
                "county20_fips_primary": "",
                "puma20_geoid_primary": "",
                "cbsa23_geoid_primary": "",
                "cbsa23_name_primary": "",
                "census_id_pid6": "",
                "census_id_gidid": "",
                "census_unit_name": "",
                "census_unit_type": "",
                "census_join_method": "test",
            },
            {
                "issue_id": "ogdenville-herald__1951-02-01",
                "newspaper_slug": "ogdenville-herald",
                "city_name": "Ogdenville",
                "state_abbr": "ks",
                "city_key": "ogdenville__ks",
                "place20_geoid": "2049500",
                "county20_fips_primary": "20103",
                "puma20_geoid_primary": "20001",
                "cbsa23_geoid_primary": "44125",
                "cbsa23_name_primary": "Ogdenville CBSA",
                "census_id_pid6": "200004",
                "census_id_gidid": "1102049500000",
                "census_unit_name": "CITY OF OGDENVILLE",
                "census_unit_type": "2 - MUNICIPAL",
                "census_join_method": "place20_geoid_to_municipal",
            },
        ]
    ).to_csv(archive_csv, index=False)

    census_xlsx = tmp_path / "census_of_gov_22.xlsx"
    pd.DataFrame(
        [
            {
                "CENSUS_ID_PID6": "200001",
                "CENSUS_ID_GIDID": "1101716700000",
                "UNIT_NAME": "CITY OF SPRINGFIELD",
                "UNIT_TYPE": "2 - MUNICIPAL",
                "CITY": "SPRINGFIELD",
                "STATE": "IL",
                "FIPS_STATE": 17,
                "FIPS_PLACE": 16700,
            },
            {
                "CENSUS_ID_PID6": "200003",
                "CENSUS_ID_GIDID": "1101764304000",
                "UNIT_NAME": "VILLAGE OF RIVER FOREST",
                "UNIT_TYPE": "2 - MUNICIPAL",
                "CITY": "RIVER FOREST",
                "STATE": "IL",
                "FIPS_STATE": 17,
                "FIPS_PLACE": 64304,
            },
            {
                "CENSUS_ID_PID6": "300003",
                "CENSUS_ID_GIDID": "1101764317000",
                "UNIT_NAME": "TOWNSHIP OF RIVER FOREST",
                "UNIT_TYPE": "3 - TOWNSHIP",
                "CITY": "RIVER FOREST",
                "STATE": "IL",
                "FIPS_STATE": 17,
                "FIPS_PLACE": 64317,
            },
            {
                "CENSUS_ID_PID6": "200004",
                "CENSUS_ID_GIDID": "1102049500000",
                "UNIT_NAME": "CITY OF OGDENVILLE",
                "UNIT_TYPE": "2 - MUNICIPAL",
                "CITY": "OGDENVILLE",
                "STATE": "KS",
                "FIPS_STATE": 20,
                "FIPS_PLACE": 49500,
            },
        ]
    ).to_excel(census_xlsx, sheet_name="General Purpose", index=False)

    locations_jsonl = tmp_path / "locations_headful.jsonl"
    with locations_jsonl.open("w", encoding="utf-8") as handle:
        for row in [
            {"pub_slug": "springfield-register", "city_name": "Springfield", "state_abbr": "il"},
            {"pub_slug": "shelbyville-times", "city_name": "Shelbyville", "state_abbr": "il"},
            {"pub_slug": "capital-gazette", "city_name": "Capital City", "state_abbr": "il"},
            {"pub_slug": "ogdenville-herald", "city_name": "Ogdenville", "state_abbr": "ks"},
        ]:
            handle.write(json.dumps(row) + "\n")

    return bundle_dir, FrontierPaths(
        bundle_dir=str(bundle_dir),
        archive_metadata_csv=str(archive_csv),
        publication_locations_jsonl=str(locations_jsonl),
        torch_manifest_jsonl=str(tmp_path / "missing_torch_manifest.jsonl"),
        torch_run_root="/tmp/torch_run",
        torch_host="torch",
        census_of_gov_xlsx=str(census_xlsx),
    )


def test_frontier_corpus_and_geo(tmp_path: Path) -> None:
    bundle_dir, paths = _build_fixture(tmp_path)
    layout = ensure_run_layout(base_dir=tmp_path / "runs", run_id="test_run")
    corpus_summary = build_fullcode_corpus(run_root=layout["run_root"], paths=paths)
    geo_summary = build_geo_outputs(run_root=layout["run_root"])

    ordinances = pd.read_parquet(layout["run_root"] / "corpus" / "ordinances.parquet")
    audit = pd.read_parquet(layout["run_root"] / "geo" / "jurisdiction_audit.parquet")
    dedup = pd.read_parquet(layout["run_root"] / "corpus" / "dedup_groups.parquet")
    jurisdiction = pd.read_parquet(layout["run_root"] / "corpus" / "jurisdiction_resolution.parquet")

    assert corpus_summary["issue_count"] == 4
    assert geo_summary["match_count"] == 2
    assert geo_summary["mismatch_count"] == 2
    assert len(dedup) == 4
    assert set(audit["jurisdiction_match_status"]) == {"match", "mismatch"}
    assert ordinances["dedup_group_id"].nunique() == 4
    assert geo_summary["resolved_with_census_count"] == 4
    assert jurisdiction["census_id_pid6"].astype(str).str.len().gt(0).all()
    assert "river-forest__il" in set(jurisdiction["jurisdiction_key"])
    assert int(jurisdiction["jurisdiction_state_backfill_used"].sum()) == 1


def test_frontier_structure_and_semantics(tmp_path: Path) -> None:
    _, paths = _build_fixture(tmp_path)
    layout = ensure_run_layout(base_dir=tmp_path / "runs", run_id="test_run")
    build_fullcode_corpus(run_root=layout["run_root"], paths=paths)
    build_geo_outputs(run_root=layout["run_root"])
    structure_summary = build_structure_outputs(run_root=layout["run_root"])
    semantics_summary = build_semantics_outputs(run_root=layout["run_root"])

    sections = pd.read_parquet(layout["run_root"] / "structure" / "sections.parquet")
    clauses = pd.read_parquet(layout["run_root"] / "structure" / "clauses.parquet")
    doc_sem = pd.read_parquet(layout["run_root"] / "semantics" / "document_semantics.parquet")
    features = pd.read_parquet(layout["run_root"] / "semantics" / "hard_feature_observations.parquet")

    assert structure_summary["section_count"] >= 8
    assert structure_summary["clause_count"] >= 6
    assert semantics_summary["document_rows"] == 4
    assert "public_health_safety" in set(doc_sem["primary_motive"])
    assert {"minimum_lot_area", "parking_spaces"}.issubset(set(features["feature_name"]))
    assert (clauses["modality"] == "must").any()
    assert (sections["section_kind"] == "purpose").any()


def test_frontier_embeddings_and_clusters(tmp_path: Path) -> None:
    _, paths = _build_fixture(tmp_path)
    layout = ensure_run_layout(base_dir=tmp_path / "runs", run_id="test_run")
    build_fullcode_corpus(run_root=layout["run_root"], paths=paths)
    build_geo_outputs(run_root=layout["run_root"])
    build_structure_outputs(run_root=layout["run_root"])
    build_semantics_outputs(run_root=layout["run_root"])
    emb_summary = build_embedding_outputs(run_root=layout["run_root"])
    cluster_summary = run_cluster_experiments(run_root=layout["run_root"], experiments=default_experiments())

    exp_index = pd.read_parquet(layout["run_root"] / "clusters" / "experiment_index.parquet")
    assert emb_summary["ordinance"]["item_count"] == 4
    assert emb_summary["section"]["item_count"] > 4
    assert cluster_summary["experiment_count"] >= 4
    assert not exp_index.empty
    assert (layout["run_root"] / "clusters" / "clause_instrument_hdbscan" / "cluster_labels.parquet").is_file()


def test_frontier_review_story_report_bundle(tmp_path: Path) -> None:
    _, paths = _build_fixture(tmp_path)
    layout = ensure_run_layout(base_dir=tmp_path / "runs", run_id="test_run")
    build_fullcode_corpus(run_root=layout["run_root"], paths=paths)
    build_geo_outputs(run_root=layout["run_root"])
    build_structure_outputs(run_root=layout["run_root"])
    build_semantics_outputs(run_root=layout["run_root"])
    build_embedding_outputs(run_root=layout["run_root"])
    run_cluster_experiments(run_root=layout["run_root"], experiments=default_experiments())
    review_summary = build_review_packets(run_root=layout["run_root"])
    story_summary = build_story_outputs(run_root=layout["run_root"])
    report_summary = build_report_bundle(run_root=layout["run_root"])

    assert review_summary["jurisdiction_review_count"] > 0
    assert story_summary["high_mismatch_publications"] > 0
    assert Path(report_summary["report_path"]).is_file()
    assert (layout["run_root"] / "story" / "packets").is_dir()
    assert (layout["run_root"] / "review" / "cluster_packets").is_dir()
