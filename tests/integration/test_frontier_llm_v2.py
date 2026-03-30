from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from newsvlm_analysis.frontier.io_utils import write_json, write_parquet
from newsvlm_analysis.frontier.llm_v2 import (
    build_cluster_label_request_sources,
    build_llm_v2_embeddings,
    build_llm_v2_semantics,
    ensure_llm_v2_run_layout,
    export_llm_v2_requests,
    initialize_llm_v2_manifest,
    prepare_llm_v2_documents,
    rehydrate_llm_v2_results,
    run_llm_v2_clusters,
)


def _baseline_doc(region: str, idx: int) -> dict[str, object]:
    state_by_region = {
        "northeast": "NY",
        "midwest": "IL",
        "south": "TX",
        "west": "CA",
    }
    level_by_region = {
        "northeast": "village",
        "midwest": "city",
        "south": "county",
        "west": "township",
    }
    state = state_by_region[region]
    level = level_by_region[region]
    city = f"{region}-place-{idx}"
    issue_id = f"{region}-paper-{idx}__1950-01-{idx + 1:02d}"
    ordinance_doc_id = f"{region}-ordinance-{idx}"
    text = "\n".join(
        [
            f"ORDINANCE {idx}",
            "ARTICLE I PURPOSE",
            "The purpose of this ordinance is to promote health, safety, and general welfare.",
            "SECTION 1 LOT AREA",
            "Minimum lot area shall be 6000 square feet.",
            "SECTION 2 PARKING",
            "Off-street parking shall provide two spaces for each dwelling.",
            "SECTION 3 ADMINISTRATION",
            "The board of appeals may hear appeals and variances.",
        ]
    )
    return {
        "issue_id": issue_id,
        "ordinance_doc_id": ordinance_doc_id,
        "issue_date": f"1950-01-{idx + 1:02d}",
        "census_id_pid6": f"{100000 + idx}",
        "publication_key": f"{region}-paper-{idx}",
        "publication_city_name": f"{region.title()} News City {idx}",
        "publication_state_abbr": state,
        "publication_region": region,
        "jurisdiction_key": f"{city}__{state.lower()}",
        "jurisdiction_city_name": city,
        "jurisdiction_state_abbr": state,
        "jurisdiction_region": region,
        "jurisdiction_level": level,
        "dedup_group_id": f"dedup-{ordinance_doc_id}",
        "jurisdiction_resolution_status": "resolved_with_census",
        "clean_text": text,
        "title_line": f"{level.title()} of {city}",
    }


def _build_baseline_run(tmp_path: Path) -> Path:
    baseline_root = tmp_path / "baseline_run"
    for subdir in ("corpus", "geo", "clusters/policy_signature_kmeans_probe", "semantics"):
        (baseline_root / subdir).mkdir(parents=True, exist_ok=True)

    docs: list[dict[str, object]] = []
    idx = 0
    for region in ("northeast", "midwest", "south", "west"):
        for _ in range(2):
            docs.append(_baseline_doc(region, idx))
            idx += 1
    ordinances = pd.DataFrame.from_records(docs)
    publication_issues = ordinances[["issue_id", "issue_date", "publication_key", "publication_city_name", "publication_state_abbr"]].copy()
    pages = ordinances[["issue_id", "ordinance_doc_id"]].assign(page_id=lambda df: df["issue_id"] + "::p1", page_num=1)
    geo_links = ordinances[
        [
            "issue_id",
            "ordinance_doc_id",
            "publication_key",
            "jurisdiction_key",
            "census_id_pid6",
            "publication_region",
            "jurisdiction_region",
        ]
    ].copy()
    dedup_groups = ordinances[["dedup_group_id", "ordinance_doc_id"]].rename(columns={"ordinance_doc_id": "representative_ordinance_doc_id"})
    timeline_scaffold = ordinances[["ordinance_doc_id", "jurisdiction_key", "issue_date"]].copy()
    jurisdiction_audit = ordinances[["issue_id", "publication_key", "jurisdiction_key", "jurisdiction_region"]].assign(jurisdiction_match_status="match")
    by_jurisdiction_geo = ordinances.groupby(["jurisdiction_region", "jurisdiction_level"], dropna=False)["ordinance_doc_id"].count().reset_index(name="ordinance_count")
    by_publication_geo = ordinances.groupby(["publication_region"], dropna=False)["ordinance_doc_id"].count().reset_index(name="ordinance_count")
    high_mismatch = pd.DataFrame(columns=["publication_key", "issue_count", "mismatch_count", "mismatch_share"])
    policy_assign = ordinances[["ordinance_doc_id", "jurisdiction_region"]].copy()
    region_cluster = {"northeast": 0, "midwest": 1, "south": 2, "west": 3}
    policy_assign["policy_signature_cluster_id"] = policy_assign["jurisdiction_region"].map(region_cluster).astype(int)
    policy_assign["cluster_label"] = policy_assign["jurisdiction_region"].astype(str) + " baseline family"

    write_parquet(publication_issues, baseline_root / "corpus" / "publication_issues.parquet")
    write_parquet(ordinances, baseline_root / "corpus" / "ordinances.parquet")
    write_parquet(pages, baseline_root / "corpus" / "pages.parquet")
    write_parquet(geo_links, baseline_root / "corpus" / "geo_links.parquet")
    write_parquet(dedup_groups, baseline_root / "corpus" / "dedup_groups.parquet")
    write_parquet(timeline_scaffold, baseline_root / "corpus" / "timeline_scaffold.parquet")
    write_parquet(jurisdiction_audit, baseline_root / "geo" / "jurisdiction_audit.parquet")
    write_parquet(by_jurisdiction_geo, baseline_root / "geo" / "by_jurisdiction_geo.parquet")
    write_parquet(by_publication_geo, baseline_root / "geo" / "by_publication_geo.parquet")
    write_parquet(high_mismatch, baseline_root / "geo" / "high_mismatch_publications.parquet")
    write_parquet(policy_assign, baseline_root / "clusters" / "policy_signature_kmeans_probe" / "assignments.parquet")
    write_json(baseline_root / "semantics" / "document_semantics.parquet.placeholder.json", {"note": "baseline semantics not needed for prep tests"})
    return baseline_root


def _make_run(tmp_path: Path) -> Path:
    baseline_root = _build_baseline_run(tmp_path)
    layout = ensure_llm_v2_run_layout(base_dir=tmp_path / "runs", run_id="llm_v2_test")
    initialize_llm_v2_manifest(layout=layout, baseline_run_root=baseline_root)
    return layout["run_root"]


def _write_result_lines(output_path: Path, rows: list[dict[str, object]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _response_line(custom_id: str, payload: dict[str, object]) -> dict[str, object]:
    return {
        "custom_id": custom_id,
        "response": {
            "status_code": 200,
            "body": {
                "output": [
                    {
                        "content": [
                            {
                                "type": "output_text",
                                "text": json.dumps(payload),
                            }
                        ]
                    }
                ]
            },
        },
        "error": None,
    }


def _embedding_line(custom_id: str, embedding: list[float]) -> dict[str, object]:
    return {
        "custom_id": custom_id,
        "response": {
            "status_code": 200,
            "body": {
                "data": [
                    {
                        "embedding": embedding,
                        "index": 0,
                        "object": "embedding",
                    }
                ],
                "model": "text-embedding-3-large",
                "object": "list",
            },
        },
        "error": None,
    }


def _mapping_rows(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_llm_v2_prepare_and_export_requests(tmp_path: Path) -> None:
    run_root = _make_run(tmp_path)
    summary = prepare_llm_v2_documents(
        run_root=run_root,
        pilot_size=4,
        pilot_region_quotas={"northeast": 1, "midwest": 1, "south": 1, "west": 1},
    )
    assert summary["full_doc_count"] == 8
    assert summary["pilot_doc_count"] == 4

    seg_export = export_llm_v2_requests(run_root=run_root, sample="pilot", stage="section_segmentation")
    purpose_export = export_llm_v2_requests(run_root=run_root, sample="pilot", stage="document_purpose")
    assert seg_export["exported_requests"] == 4
    assert purpose_export["exported_requests"] == 4
    assert (run_root / "requests" / "pilot" / "section_segmentation" / "mapping_shard000.jsonl").is_file()


def test_llm_v2_rehydrate_semantics_embeddings_and_clusters(tmp_path: Path) -> None:
    run_root = _make_run(tmp_path)
    prepare_llm_v2_documents(
        run_root=run_root,
        pilot_size=4,
        pilot_region_quotas={"northeast": 1, "midwest": 1, "south": 1, "west": 1},
    )

    export_llm_v2_requests(run_root=run_root, sample="pilot", stage="section_segmentation")
    seg_mapping = _mapping_rows(run_root / "requests" / "pilot" / "section_segmentation" / "mapping_shard000.jsonl")
    seg_results = []
    for row in seg_mapping:
        doc_id = str(row["doc_id"])
        seg_results.append(
            _response_line(
                str(row["custom_id"]),
                {
                    "doc_id": doc_id,
                    "sections": [
                        {"section_number": 1, "title": "Purpose", "start_line": 1, "end_line": 3},
                        {"section_number": 2, "title": "Lot Area", "start_line": 4, "end_line": 5},
                        {"section_number": 3, "title": "Parking and Appeals", "start_line": 6, "end_line": 9},
                    ],
                    "notes": "",
                },
            )
        )
    _write_result_lines(run_root / "results" / "pilot" / "section_segmentation" / "openai_results_shard000.jsonl", seg_results)
    rehydrate_llm_v2_results(run_root=run_root, sample="pilot", stage="section_segmentation", allow_errors=False)

    export_llm_v2_requests(run_root=run_root, sample="pilot", stage="document_purpose")
    purpose_mapping = _mapping_rows(run_root / "requests" / "pilot" / "document_purpose" / "mapping_shard000.jsonl")
    purpose_results = [
        _response_line(
            str(row["custom_id"]),
            {
                "doc_id": str(row["doc_id"]),
                "purposes": [
                    {
                        "motive": "public_health_safety",
                        "description": "Promote health and safety through zoning controls.",
                        "evidence": "promote health, safety, and general welfare",
                        "confidence": 0.92,
                    }
                ],
                "notes": "",
            },
        )
        for row in purpose_mapping
    ]
    _write_result_lines(run_root / "results" / "pilot" / "document_purpose" / "openai_results_shard000.jsonl", purpose_results)
    rehydrate_llm_v2_results(run_root=run_root, sample="pilot", stage="document_purpose", allow_errors=False)

    export_llm_v2_requests(run_root=run_root, sample="pilot", stage="section_motive")
    motive_mapping = _mapping_rows(run_root / "requests" / "pilot" / "section_motive" / "mapping_shard000.jsonl")
    motive_results = []
    for row in motive_mapping:
        title = str(pd.read_parquet(run_root / "structure" / "sections__pilot.parquet").set_index("section_id").loc[str(row["section_id"]), "title"])
        primary = "public_health_safety" if "purpose" in title.lower() else "infrastructure_coordination"
        motive_results.append(
            _response_line(
                str(row["custom_id"]),
                {
                    "section_id": str(row["section_id"]),
                    "primary_motive": primary,
                    "secondary_motives": [],
                    "evidence": title,
                    "confidence": 0.88,
                    "notes": "",
                },
            )
        )
    _write_result_lines(run_root / "results" / "pilot" / "section_motive" / "openai_results_shard000.jsonl", motive_results)
    rehydrate_llm_v2_results(run_root=run_root, sample="pilot", stage="section_motive", allow_errors=False)

    export_llm_v2_requests(run_root=run_root, sample="pilot", stage="clause_extraction")
    clause_mapping = _mapping_rows(run_root / "requests" / "pilot" / "clause_extraction" / "mapping_shard000.jsonl")
    section_lookup = pd.read_parquet(run_root / "structure" / "sections__pilot.parquet").set_index("section_id")
    clause_results = []
    for row in clause_mapping:
        section_id = str(row["section_id"])
        title = str(section_lookup.loc[section_id, "title"])
        clauses = []
        if "purpose" in title.lower():
            clauses.append(
                {
                    "clause_number": 1,
                    "clause_text": "The purpose of this ordinance is to promote health, safety, and general welfare.",
                    "requirement": "This clause states the ordinance purpose.",
                    "instrument_hint": "purpose",
                    "motive": "public_health_safety",
                    "modality": "other",
                    "confidence": 0.9,
                }
            )
        elif "lot area" in title.lower():
            clauses.append(
                {
                    "clause_number": 1,
                    "clause_text": "Minimum lot area shall be 6000 square feet.",
                    "requirement": "This clause imposes a zoning standard.",
                    "instrument_hint": "lot_area_standard",
                    "motive": "infrastructure_coordination",
                    "modality": "must",
                    "confidence": 0.93,
                }
            )
            clauses.append(
                {
                    "clause_number": 2,
                    "clause_text": "Front setback shall be 25 feet.",
                    "requirement": "This clause imposes a zoning standard.",
                    "instrument_hint": "setback_standard",
                    "motive": "infrastructure_coordination",
                    "modality": "must",
                    "confidence": 0.91,
                }
            )
        else:
            clauses.append(
                {
                    "clause_number": 1,
                    "clause_text": "Off-street parking shall provide two spaces for each dwelling.",
                    "requirement": "This clause imposes a zoning standard.",
                    "instrument_hint": "parking_standard",
                    "motive": "infrastructure_coordination",
                    "modality": "must",
                    "confidence": 0.94,
                }
            )
            clauses.append(
                {
                    "clause_number": 2,
                    "clause_text": "The board of appeals may hear appeals and variances.",
                    "requirement": "This clause imposes a zoning standard.",
                    "instrument_hint": "appeal",
                    "motive": "infrastructure_coordination",
                    "modality": "may",
                    "confidence": 0.9,
                }
            )
        clause_results.append(
            _response_line(
                str(row["custom_id"]),
                {
                    "section_id": section_id,
                    "clauses": clauses,
                    "notes": "",
                },
            )
        )
    _write_result_lines(run_root / "results" / "pilot" / "clause_extraction" / "openai_results_shard000.jsonl", clause_results)
    rehydrate_llm_v2_results(run_root=run_root, sample="pilot", stage="clause_extraction", allow_errors=False)

    export_llm_v2_requests(run_root=run_root, sample="pilot", stage="clause_enrichment")
    enrich_mapping = _mapping_rows(run_root / "requests" / "pilot" / "clause_enrichment" / "mapping_shard000.jsonl")
    clauses_df = pd.read_parquet(run_root / "semantics" / "clauses__pilot.parquet").set_index("clause_id")
    enrich_results = []
    for row in enrich_mapping:
        clause_id = str(row["clause_id"])
        clause = clauses_df.loc[clause_id]
        hint = str(clause["instrument_hint"])
        enrich_results.append(
            _response_line(
                str(row["custom_id"]),
                {
                    "clause_id": clause_id,
                    "actor": "property owner" if hint != "appeal" else "board of appeals",
                    "action": "comply",
                    "regulated_object": "lot" if "lot" in hint else "parking",
                    "condition": None,
                    "exception": None,
                    "threshold_or_quantity": "6000 square feet" if hint == "lot_area_standard" else "two spaces" if hint == "parking_standard" else None,
                    "spatial_scope": None,
                    "instrument_type": "purpose" if hint == "purpose" else hint,
                    "operative_flag": 0 if hint == "purpose" else 1,
                    "evidence_quote": str(clause["clause_text"])[:300],
                    "confidence": 0.93,
                    "notes": "",
                },
            )
        )
    _write_result_lines(run_root / "results" / "pilot" / "clause_enrichment" / "openai_results_shard000.jsonl", enrich_results)
    rehydrate_llm_v2_results(run_root=run_root, sample="pilot", stage="clause_enrichment", allow_errors=False)

    semantics_summary = build_llm_v2_semantics(run_root=run_root, sample="pilot")
    doc_sem = pd.read_parquet(run_root / "semantics" / "document_semantics.parquet")
    operative_clauses = pd.read_parquet(run_root / "semantics" / "operative_clause_semantics.parquet")
    assert semantics_summary["document_rows"] == 4
    assert len(doc_sem) == 4
    assert set(operative_clauses["instrument_type"]) >= {"lot_area_standard", "parking_standard", "appeal", "setback_standard"}

    export_llm_v2_requests(run_root=run_root, sample="pilot", stage="section_embedding")
    emb_mapping = _mapping_rows(run_root / "requests" / "pilot" / "section_embedding" / "mapping_shard000.jsonl")
    emb_results = []
    for i, row in enumerate(emb_mapping):
        _ = i
        emb_results.append(_embedding_line(str(row["custom_id"]), [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]))
    _write_result_lines(run_root / "results" / "pilot" / "section_embedding" / "openai_results_shard000.jsonl", emb_results)
    rehydrate_llm_v2_results(run_root=run_root, sample="pilot", stage="section_embedding", allow_errors=False)

    embedding_summary = build_llm_v2_embeddings(run_root=run_root, sample="pilot")
    cluster_summary = run_llm_v2_clusters(run_root=run_root)
    label_source_summary = build_cluster_label_request_sources(run_root=run_root)
    experiment_index = pd.read_parquet(run_root / "clusters" / "experiment_index.parquet")

    assert embedding_summary["section_openai_item_count"] == 12
    assert cluster_summary["experiments"]
    assert {"labeler", "clusterer"}.issubset(set(experiment_index.columns))
    assert {
        "section_cluster_request_count",
        "section_local_cluster_request_count",
        "clause_cluster_request_count",
    } == set(label_source_summary)
    assert (run_root / "clusters" / "section_semantic_hdbscan" / "cluster_label_request_source.parquet").is_file()
    assert (run_root / "clusters" / "clause_instrument_hdbscan" / "cluster_label_request_source.parquet").is_file()
