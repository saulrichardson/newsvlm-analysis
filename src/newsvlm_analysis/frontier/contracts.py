from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RUNS_BASE = REPO_ROOT / "artifacts" / "runs" / "frontier_fullcode_lab_v1"
DEFAULT_BUNDLE_DIR = Path(
    "/Users/saulrichardson/Dropbox/Inclusionary Zoning/Historical Analysis/newspapers/code_publication_full_issue_bundle_20260306"
)
DEFAULT_ARCHIVE_METADATA_CSV = Path(
    "/Users/saulrichardson/Dropbox/Inclusionary Zoning/Historical Analysis/newspapers/archive/v2/metadata.csv"
)
DEFAULT_LOCATIONS_JSONL = Path(
    "/Users/saulrichardson/projects/newspapers/newspaper-scrapping/data/locations_headful.jsonl"
)
DEFAULT_TORCH_MANIFEST_JSONL = Path(
    "/scratch/sxr203/newspaper-downloads/dedupe-webp/issue_classifier_v7_gpt_full_dataset_20260306/code_publication_full_issue_dataset_manifest/issues.jsonl"
)
DEFAULT_TORCH_RUN_ROOT = Path(
    "/scratch/sxr203/newspaper-downloads/dedupe-webp/postgate_fullcode_openai_prepared_20260306_733workers"
)
DEFAULT_TORCH_HOST = "torch"
DEFAULT_CENSUS_OF_GOV_XLSX = Path(
    "/Users/saulrichardson/Dropbox/Inclusionary Zoning/Historical Analysis/raw_data/census_data/census_of_gov_22.xlsx"
)

RUN_SUBDIRS = (
    "corpus",
    "geo",
    "structure",
    "semantics",
    "embeddings",
    "clusters",
    "review",
    "story",
    "reports",
    "manifests",
)


@dataclass(slots=True)
class FrontierPaths:
    bundle_dir: str = str(DEFAULT_BUNDLE_DIR)
    archive_metadata_csv: str = str(DEFAULT_ARCHIVE_METADATA_CSV)
    publication_locations_jsonl: str = str(DEFAULT_LOCATIONS_JSONL)
    torch_manifest_jsonl: str = str(DEFAULT_TORCH_MANIFEST_JSONL)
    torch_run_root: str = str(DEFAULT_TORCH_RUN_ROOT)
    torch_host: str = DEFAULT_TORCH_HOST
    census_of_gov_xlsx: str = str(DEFAULT_CENSUS_OF_GOV_XLSX)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class FrontierExperiment:
    name: str
    input_filter: str
    representation_level: str
    reducer: str
    clusterer: str
    labeler: str
    output_targets: list[str] = field(default_factory=list)
    enabled: bool = True
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class FrontierManifest:
    schema_version: str
    run_id: str
    created_at_utc: str
    run_root: str
    scope_label: str
    paper_framing: str
    paths: FrontierPaths
    experiments: list[FrontierExperiment]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["paths"] = self.paths.to_dict()
        payload["experiments"] = [exp.to_dict() for exp in self.experiments]
        return payload


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def default_experiments() -> list[FrontierExperiment]:
    return [
        FrontierExperiment(
            name="ordinance_semantic_hdbscan",
            input_filter="all",
            representation_level="ordinance",
            reducer="umap",
            clusterer="hdbscan",
            labeler="semantic",
            output_targets=["clusters", "story", "report"],
            params={"umap_components": 5, "hdbscan_min_cluster_size": 8},
        ),
        FrontierExperiment(
            name="section_semantic_hdbscan",
            input_filter="all",
            representation_level="section",
            reducer="umap",
            clusterer="hdbscan",
            labeler="semantic",
            output_targets=["clusters", "story", "report"],
            params={"umap_components": 5, "hdbscan_min_cluster_size": 10},
        ),
        FrontierExperiment(
            name="section_graph_mechanics",
            input_filter="all",
            representation_level="section",
            reducer="pca",
            clusterer="graph",
            labeler="mechanics",
            output_targets=["clusters", "story", "report"],
            params={"graph_k": 12, "graph_similarity_threshold": 0.42},
        ),
        FrontierExperiment(
            name="clause_instrument_hdbscan",
            input_filter="all",
            representation_level="clause",
            reducer="umap",
            clusterer="hdbscan",
            labeler="instrument",
            output_targets=["clusters", "story", "report"],
            params={"umap_components": 5, "hdbscan_min_cluster_size": 12},
        ),
        FrontierExperiment(
            name="clause_hierarchical_instrument",
            input_filter="all",
            representation_level="clause",
            reducer="pca",
            clusterer="hierarchical",
            labeler="instrument",
            output_targets=["clusters", "story"],
            params={"distance_threshold": 0.85},
        ),
    ]
