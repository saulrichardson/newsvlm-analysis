#!/usr/bin/env python3
"""
Build portable metadata manifests for an exported page artifact packet.

Outputs are written under: <run_dir>/summary/metadata/

Files:
- newspaper_index.csv
- issue_index_relative.csv
- page_index_relative.csv
- newspaper_location_mapping.csv
- unmatched_newspaper_slugs.csv
- README.md
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd


DEFAULT_RUN_DIR = Path(
    "/Users/saulrichardson/Dropbox/Inclusionary Zoning/saul/newspapers/"
    "page_artifact_packets_v5_full_20260225"
)
DEFAULT_LOCATIONS_CSV = Path(
    "/Users/saulrichardson/projects/newspapers/old-ocr/tmp/locations_headful.csv"
)


@dataclass(frozen=True)
class AliasRule:
    newspaper_slug: str
    pub_slug: str
    note: str


# Conservative alias: high-confidence lexical match and matching year range.
DEFAULT_ALIAS_RULES = [
    AliasRule(
        newspaper_slug="gastonia-gazette",
        pub_slug="the-gastonia-gazette",
        note="Alias override (lexical match, same city, overlapping years).",
    ),
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--locations-csv", type=Path, default=DEFAULT_LOCATIONS_CSV)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to <run_dir>/summary/metadata",
    )
    parser.add_argument(
        "--label-root",
        default="classified_pages",
        help="Top-level folder name that contains class-labeled issue folders.",
    )
    return parser.parse_args()


def _norm(x: object) -> str:
    return str(x or "").strip()


def _require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    if not path.is_file():
        raise ValueError(f"{label} is not a file: {path}")


def _load_inputs(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    issues_csv = run_dir / "summary" / "issues.csv"
    page_csv = run_dir / "summary" / "page_index.csv"
    _require_file(issues_csv, "issues.csv")
    _require_file(page_csv, "page_index.csv")

    issues = pd.read_csv(issues_csv)
    pages = pd.read_csv(page_csv)
    required_issue_cols = {
        "issue_id",
        "issue_date",
        "newspaper_slug",
        "primary_label",
        "label_set",
        "page_count",
    }
    required_page_cols = {
        "issue_id",
        "issue_slug",
        "page_id",
        "primary_label",
        "label",
    }
    missing_issue = required_issue_cols - set(issues.columns)
    missing_page = required_page_cols - set(pages.columns)
    if missing_issue:
        raise ValueError(f"issues.csv missing required columns: {sorted(missing_issue)}")
    if missing_page:
        raise ValueError(f"page_index.csv missing required columns: {sorted(missing_page)}")

    issues = issues.copy()
    pages = pages.copy()
    issues["newspaper_slug"] = issues["newspaper_slug"].astype(str).str.strip().str.lower()
    pages["issue_id"] = pages["issue_id"].astype(str).str.strip()
    pages["issue_slug"] = pages["issue_slug"].astype(str).str.strip()
    pages["page_id"] = pages["page_id"].astype(str).str.strip()
    pages["primary_label"] = pages["primary_label"].astype(str).str.strip()
    pages["label"] = pages["label"].astype(str).str.strip()
    return issues, pages


def _build_location_mapping(
    run_issues: pd.DataFrame, locations_csv: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    _require_file(locations_csv, "locations-csv")
    loc = pd.read_csv(locations_csv)
    required = {
        "pub_slug",
        "pub_title",
        "country_id",
        "country_abbr",
        "state_id",
        "state_abbr",
        "state_name",
        "city_id",
        "city_name",
        "pub_id",
        "pub_min_year",
        "pub_max_year",
    }
    missing = required - set(loc.columns)
    if missing:
        raise ValueError(
            f"locations-csv missing required columns: {sorted(missing)} in {locations_csv}"
        )

    loc = loc.copy()
    loc["pub_slug"] = loc["pub_slug"].astype(str).str.strip().str.lower()
    loc_unique = (
        loc.sort_values(["pub_slug", "pub_min_year", "pub_max_year"])
        .drop_duplicates("pub_slug", keep="first")
        .reset_index(drop=True)
    )

    newspaper_slugs = (
        run_issues[["newspaper_slug"]].drop_duplicates().reset_index(drop=True).copy()
    )
    mapping = newspaper_slugs.merge(
        loc_unique,
        left_on="newspaper_slug",
        right_on="pub_slug",
        how="left",
        suffixes=("", "_loc"),
    )
    mapping["match_method"] = "exact_slug"
    mapping["match_note"] = ""

    # Conservative alias backfill for known slug variants.
    for rule in DEFAULT_ALIAS_RULES:
        mask = (
            mapping["newspaper_slug"].eq(rule.newspaper_slug)
            & mapping["pub_slug"].isna()
        )
        if not mask.any():
            continue
        alias_row = loc_unique[loc_unique["pub_slug"].eq(rule.pub_slug)]
        if alias_row.empty:
            continue
        alias = alias_row.iloc[0]
        for col in loc_unique.columns:
            mapping.loc[mask, col] = alias[col]
        mapping.loc[mask, "match_method"] = "alias_override"
        mapping.loc[mask, "match_note"] = rule.note

    unmatched = mapping[mapping["pub_slug"].isna()][["newspaper_slug"]].copy()
    mapping.loc[mapping["pub_slug"].isna(), "match_method"] = "unmatched"
    mapping.loc[mapping["pub_slug"].isna(), "match_note"] = (
        "No slug match in locations metadata; needs manual mapping."
    )

    keep_cols = [
        "newspaper_slug",
        "pub_slug",
        "pub_title",
        "country_id",
        "country_abbr",
        "state_id",
        "state_abbr",
        "state_name",
        "city_id",
        "city_name",
        "pub_id",
        "pub_min_year",
        "pub_max_year",
        "match_method",
        "match_note",
    ]
    mapping = mapping[keep_cols].sort_values("newspaper_slug").reset_index(drop=True)
    unmatched = unmatched.sort_values("newspaper_slug").reset_index(drop=True)
    return mapping, unmatched


def _build_issue_index_relative(
    run_issues: pd.DataFrame, location_map: pd.DataFrame, label_root: str
) -> pd.DataFrame:
    issues = run_issues.copy()
    issues["issue_id"] = issues["issue_id"].astype(str).str.strip()
    issues["issue_rel_dir"] = (
        f"{label_root}/"
        + issues["primary_label"].astype(str)
        + "/"
        + issues["issue_id"].astype(str)
    )
    issues["issue_readme_rel_path"] = issues["issue_rel_dir"] + "/README.md"
    issues = issues.merge(location_map, on="newspaper_slug", how="left")
    order_cols = [
        "issue_id",
        "issue_date",
        "newspaper_slug",
        "pub_title",
        "city_name",
        "state_abbr",
        "state_name",
        "country_abbr",
        "primary_label",
        "label_set",
        "page_count",
        "issue_rel_dir",
        "issue_readme_rel_path",
        "match_method",
        "match_note",
        "pub_slug",
        "pub_id",
        "city_id",
        "state_id",
        "country_id",
        "pub_min_year",
        "pub_max_year",
    ]
    return issues[order_cols].sort_values(["newspaper_slug", "issue_date", "issue_id"])


def _build_page_index_relative(
    pages: pd.DataFrame, run_issues: pd.DataFrame, location_map: pd.DataFrame, label_root: str
) -> pd.DataFrame:
    page_idx = pages.copy()
    page_idx = page_idx.merge(
        run_issues[["issue_id", "issue_date", "newspaper_slug"]], on="issue_id", how="left"
    )
    page_idx["page_rel_dir"] = (
        f"{label_root}/"
        + page_idx["primary_label"].astype(str)
        + "/"
        + page_idx["issue_id"].astype(str)
        + "/pages/"
        + page_idx["page_id"].astype(str)
    )
    page_idx["issue_rel_dir"] = (
        f"{label_root}/"
        + page_idx["primary_label"].astype(str)
        + "/"
        + page_idx["issue_id"].astype(str)
    )
    page_idx["transcript_rel_path"] = page_idx["page_rel_dir"] + "/transcript.txt"
    page_idx["png_rel_path"] = page_idx["page_rel_dir"] + "/original.png"
    page_idx["layout_rel_path"] = page_idx["page_rel_dir"] + "/layout.json"
    page_idx["classification_rel_path"] = page_idx["page_rel_dir"] + "/classification.json"
    page_idx = page_idx.merge(location_map, on="newspaper_slug", how="left")

    order_cols = [
        "issue_id",
        "issue_date",
        "newspaper_slug",
        "pub_title",
        "city_name",
        "state_abbr",
        "state_name",
        "country_abbr",
        "primary_label",
        "label",
        "page_id",
        "issue_rel_dir",
        "page_rel_dir",
        "transcript_rel_path",
        "png_rel_path",
        "layout_rel_path",
        "classification_rel_path",
        "match_method",
        "match_note",
        "pub_slug",
        "pub_id",
        "city_id",
        "state_id",
        "country_id",
        "pub_min_year",
        "pub_max_year",
    ]
    page_idx = page_idx[order_cols].sort_values(
        ["newspaper_slug", "issue_date", "issue_id", "page_id"]
    )
    return page_idx


def _build_newspaper_index(
    issues_rel: pd.DataFrame, pages_rel: pd.DataFrame, location_map: pd.DataFrame
) -> pd.DataFrame:
    issue_counts = (
        issues_rel.groupby("newspaper_slug", as_index=False)
        .agg(
            issue_count=("issue_id", "nunique"),
            page_count_from_issues=("page_count", "sum"),
            first_issue_date=("issue_date", "min"),
            last_issue_date=("issue_date", "max"),
        )
        .reset_index(drop=True)
    )
    page_counts = (
        pages_rel.groupby("newspaper_slug", as_index=False)
        .agg(
            page_count=("page_id", "nunique"),
            label_cardinality=("label", "nunique"),
            primary_label_cardinality=("primary_label", "nunique"),
        )
        .reset_index(drop=True)
    )
    primary_label_modes = (
        issues_rel.groupby(["newspaper_slug", "primary_label"], as_index=False)["issue_id"]
        .nunique()
        .rename(columns={"issue_id": "issue_count_in_primary_label"})
    )
    primary_label_modes = primary_label_modes.sort_values(
        ["newspaper_slug", "issue_count_in_primary_label", "primary_label"],
        ascending=[True, False, True],
    )
    primary_mode = primary_label_modes.drop_duplicates("newspaper_slug", keep="first")[
        ["newspaper_slug", "primary_label"]
    ].rename(columns={"primary_label": "most_common_primary_label"})

    newspaper = issue_counts.merge(page_counts, on="newspaper_slug", how="left")
    newspaper = newspaper.merge(primary_mode, on="newspaper_slug", how="left")
    newspaper = newspaper.merge(location_map, on="newspaper_slug", how="left")
    order_cols = [
        "newspaper_slug",
        "pub_title",
        "city_name",
        "state_abbr",
        "state_name",
        "country_abbr",
        "issue_count",
        "page_count",
        "page_count_from_issues",
        "first_issue_date",
        "last_issue_date",
        "most_common_primary_label",
        "label_cardinality",
        "primary_label_cardinality",
        "match_method",
        "match_note",
        "pub_slug",
        "pub_id",
        "city_id",
        "state_id",
        "country_id",
        "pub_min_year",
        "pub_max_year",
    ]
    newspaper = newspaper[order_cols].sort_values(["state_abbr", "city_name", "newspaper_slug"])
    return newspaper


def _write_readme(
    out_dir: Path,
    run_dir: Path,
    issues_rel: pd.DataFrame,
    pages_rel: pd.DataFrame,
    newspaper_idx: pd.DataFrame,
    unmatched: pd.DataFrame,
    label_root: str,
) -> None:
    total_issues = int(issues_rel["issue_id"].nunique())
    total_pages = int(pages_rel["page_id"].nunique())
    total_newspapers = int(newspaper_idx["newspaper_slug"].nunique())
    matched = int((newspaper_idx["match_method"] != "unmatched").sum())
    unmatched_n = int((newspaper_idx["match_method"] == "unmatched").sum())
    unmatched_list = ", ".join(unmatched["newspaper_slug"].tolist()[:20]) or "(none)"
    readme = f"""# Metadata Index

This directory contains portable metadata for:

`{run_dir}`

All file-system references in these tables are **run-relative paths** (not absolute paths).

## Files

- `newspaper_index.csv`: one row per newspaper slug with city/state enrichment and run coverage stats.
- `issue_index_relative.csv`: one row per issue with canonical relative issue paths.
- `page_index_relative.csv`: one row per page with canonical relative page/file paths.
- `newspaper_location_mapping.csv`: slug-level mapping to location metadata and match method.
- `unmatched_newspaper_slugs.csv`: slugs not matched to location metadata.

## Coverage

- newspapers: {total_newspapers}
- matched to location metadata: {matched}
- unmatched location slugs: {unmatched_n}
- issues: {total_issues}
- pages: {total_pages}

Unmatched slug sample: {unmatched_list}

## Path conventions

- Issue directory: `{label_root}/<primary_label>/<issue_id>/`
- Page directory: `{label_root}/<primary_label>/<issue_id>/pages/<page_id>/`
- Page files:
  - transcript: `<page_dir>/transcript.txt`
  - png: `<page_dir>/original.png`
  - layout: `<page_dir>/layout.json`
  - classification: `<page_dir>/classification.json`
"""
    (out_dir / "README.md").write_text(readme)


def main() -> None:
    args = _parse_args()
    run_dir = args.run_dir.resolve()
    out_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else (run_dir / "summary" / "metadata")
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    issues, pages = _load_inputs(run_dir)
    location_map, unmatched = _build_location_mapping(issues, args.locations_csv.resolve())
    label_root = _norm(args.label_root) or "classified_pages"
    issues_rel = _build_issue_index_relative(issues, location_map, label_root)
    pages_rel = _build_page_index_relative(pages, issues, location_map, label_root)
    newspaper_idx = _build_newspaper_index(issues_rel, pages_rel, location_map)

    issues_rel.to_csv(out_dir / "issue_index_relative.csv", index=False)
    pages_rel.to_csv(out_dir / "page_index_relative.csv", index=False)
    newspaper_idx.to_csv(out_dir / "newspaper_index.csv", index=False)
    location_map.to_csv(out_dir / "newspaper_location_mapping.csv", index=False)
    unmatched.to_csv(out_dir / "unmatched_newspaper_slugs.csv", index=False)

    summary = {
        "run_dir": str(run_dir),
        "locations_csv": str(args.locations_csv.resolve()),
        "newspaper_count": int(newspaper_idx["newspaper_slug"].nunique()),
        "issue_count": int(issues_rel["issue_id"].nunique()),
        "page_count": int(pages_rel["page_id"].nunique()),
        "location_matched_newspapers": int((newspaper_idx["match_method"] != "unmatched").sum()),
        "location_unmatched_newspapers": int(
            (newspaper_idx["match_method"] == "unmatched").sum()
        ),
    }
    (out_dir / "metadata_build_summary.json").write_text(json.dumps(summary, indent=2))
    _write_readme(out_dir, run_dir, issues_rel, pages_rel, newspaper_idx, unmatched, label_root)

    print(json.dumps(summary, indent=2))
    if not unmatched.empty:
        print("Unmatched slugs:", ", ".join(unmatched["newspaper_slug"].tolist()))


if __name__ == "__main__":
    main()
