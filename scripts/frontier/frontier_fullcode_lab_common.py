#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import frontier_bootstrap  # noqa: F401

from newsvlm_analysis.frontier import FrontierPaths, ensure_run_layout, initialize_run_manifest, layout_from_run_root


def add_run_args(parser: argparse.ArgumentParser, *, include_sources: bool) -> None:
    parser.add_argument("--run-root", default="", help="Existing run root. If omitted, a new run is created.")
    parser.add_argument("--run-id", default="", help="Optional run id when creating a new run.")
    parser.add_argument("--base-dir", default="", help="Optional base dir for new runs.")
    if include_sources:
        parser.add_argument("--bundle-dir", default="", help="Override full-code bundle dir.")
        parser.add_argument("--archive-metadata-csv", default="", help="Override archive v2 metadata CSV.")
        parser.add_argument("--publication-locations-jsonl", default="", help="Override publication locations JSONL.")
        parser.add_argument("--torch-manifest-jsonl", default="", help="Override Torch issue manifest JSONL.")
        parser.add_argument("--torch-run-root", default="", help="Override Torch run root.")
        parser.add_argument("--torch-host", default="", help="Override Torch SSH host.")
        parser.add_argument("--census-of-gov-xlsx", default="", help="Override Census of Governments workbook.")


def paths_from_args(args: argparse.Namespace) -> FrontierPaths:
    defaults = FrontierPaths()
    return FrontierPaths(
        bundle_dir=str(args.bundle_dir or defaults.bundle_dir),
        archive_metadata_csv=str(args.archive_metadata_csv or defaults.archive_metadata_csv),
        publication_locations_jsonl=str(args.publication_locations_jsonl or defaults.publication_locations_jsonl),
        torch_manifest_jsonl=str(args.torch_manifest_jsonl or defaults.torch_manifest_jsonl),
        torch_run_root=str(args.torch_run_root or defaults.torch_run_root),
        torch_host=str(args.torch_host or defaults.torch_host),
        census_of_gov_xlsx=str(args.census_of_gov_xlsx or defaults.census_of_gov_xlsx),
    )


def resolve_layout_and_manifest(args: argparse.Namespace, *, include_sources: bool) -> tuple[dict[str, Path], FrontierPaths]:
    if str(getattr(args, "run_root", "")).strip():
        layout = layout_from_run_root(args.run_root)
    else:
        layout = ensure_run_layout(base_dir=args.base_dir or None, run_id=args.run_id or None)

    manifest_path = layout["manifests"] / "run_manifest.json"
    if manifest_path.is_file():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        raw_paths = payload.get("paths") if isinstance(payload.get("paths"), dict) else {}
        paths = FrontierPaths(
            bundle_dir=str(raw_paths.get("bundle_dir", FrontierPaths().bundle_dir)),
            archive_metadata_csv=str(raw_paths.get("archive_metadata_csv", FrontierPaths().archive_metadata_csv)),
            publication_locations_jsonl=str(raw_paths.get("publication_locations_jsonl", FrontierPaths().publication_locations_jsonl)),
            torch_manifest_jsonl=str(raw_paths.get("torch_manifest_jsonl", FrontierPaths().torch_manifest_jsonl)),
            torch_run_root=str(raw_paths.get("torch_run_root", FrontierPaths().torch_run_root)),
            torch_host=str(raw_paths.get("torch_host", FrontierPaths().torch_host)),
            census_of_gov_xlsx=str(raw_paths.get("census_of_gov_xlsx", FrontierPaths().census_of_gov_xlsx)),
        )
        return layout, paths

    paths = paths_from_args(args) if include_sources else FrontierPaths()
    initialize_run_manifest(layout=layout, paths=paths)
    return layout, paths
