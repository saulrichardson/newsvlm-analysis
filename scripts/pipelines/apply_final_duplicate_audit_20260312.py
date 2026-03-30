#!/usr/bin/env python3
from __future__ import annotations

import csv
import shutil
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(
    "/Users/saulrichardson/Dropbox/Inclusionary Zoning/Historical Analysis/raw_data/newspaper_ordinances"
)
ARCHIVE_ROOT = Path(
    "/Users/saulrichardson/projects/newspapers/newspaper-analysis/artifacts/scratch/desmond_newspaper_ordinances_archive_20260311"
)


@dataclass(frozen=True)
class ReviewGroup:
    canonical_issue_id: str
    issue_ids: tuple[str, ...]
    notes: str


APPROVED_GROUPS: tuple[ReviewGroup, ...] = (
    ReviewGroup(
        canonical_issue_id="belton-journal-and-bell-county-democrat__1945-06-21",
        issue_ids=(
            "belton-journal-and-bell-county-democrat__1945-06-21",
            "belton-journal-and-bell-county-democrat__1945-07-12",
        ),
        notes="Same Belton zoning ordinance body reprinted three weeks later in the same paper; keep the slightly fuller June 21 printing.",
    ),
    ReviewGroup(
        canonical_issue_id="western-hills-press__1967-03-23",
        issue_ids=(
            "western-hills-press__1966-12-15",
            "western-hills-press__1967-03-23",
        ),
        notes="Same City of Cheviot Ordinance No. 714, 1966 reprinted later with a fuller body; keep the March 23 printing.",
    ),
    ReviewGroup(
        canonical_issue_id="cumberland-news__1944-12-13",
        issue_ids=(
            "cumberland-evening-times__1944-12-12",
            "cumberland-news__1944-12-13",
        ),
        notes="Same Cumberland Ordinance No. 1776 printed across companion papers on adjacent days; keep the slightly fuller Cumberland News version.",
    ),
    ReviewGroup(
        canonical_issue_id="kingsville-record__1951-02-14",
        issue_ids=(
            "kingsville-record__1951-02-14",
            "kingsville-record__1951-02-21",
        ),
        notes="Same Kingsville zoning ordinance summary/legal notice reprinted one week later; keep the earlier slightly fuller printing.",
    ),
    ReviewGroup(
        canonical_issue_id="racine-journal-times__1969-12-23",
        issue_ids=(
            "racine-journal-times__1969-12-10",
            "racine-journal-times__1969-12-23",
        ),
        notes="Same Racine County Ordinance No. 69-137 comprehensive revision reprinted later in the month; keep the fuller December 23 printing.",
    ),
    ReviewGroup(
        canonical_issue_id="racine-journal-times__1976-02-02",
        issue_ids=(
            "racine-journal-times__1976-02-02",
            "racine-journal-times__1976-02-25",
        ),
        notes="Same Racine Title 16 zoning ordinance recreation reprinted later in the month; keep the much fuller February 2 printing.",
    ),
    ReviewGroup(
        canonical_issue_id="santa-fe-new-mexican__1980-12-08",
        issue_ids=(
            "santa-fe-new-mexican__1980-10-21",
            "santa-fe-new-mexican__1980-12-08",
        ),
        notes="Same Santa Fe County Land Development Code ordinance event; keep the later December printing because it resolves the ordinance number and adoption date.",
    ),
    ReviewGroup(
        canonical_issue_id="steubenville-herald-star__1953-11-25",
        issue_ids=(
            "steubenville-herald-star__1953-11-18",
            "steubenville-herald-star__1953-11-25",
        ),
        notes="Same Steubenville Ordinance No. 8250 printed one week later; keep the slightly fuller November 25 printing.",
    ),
)


def clean(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or []), list(reader)


def write_rows(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def move_path(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        if src.is_dir():
            shutil.rmtree(src)
        else:
            src.unlink()
        return
    shutil.move(str(src), str(dst))


def main() -> None:
    metadata_path = ROOT / "metadata.csv"
    archive_metadata_path = ARCHIVE_ROOT / "metadata_archived_duplicates.csv"
    review_path = ARCHIVE_ROOT / "pid_dedupe_review_decisions_20260312_final.csv"

    main_fieldnames, main_rows = read_rows(metadata_path)
    archive_fieldnames, archive_rows = read_rows(archive_metadata_path)

    main_by_issue = {row["issue_id"]: row for row in main_rows}
    review_rows: list[dict[str, str]] = []
    newly_archived_rows: list[dict[str, str]] = []

    archive_parsed_root = ARCHIVE_ROOT / "archived_duplicates" / "parsed"
    archive_transcript_root = ARCHIVE_ROOT / "archived_duplicates" / "transcript"
    archive_raw_root = ARCHIVE_ROOT / "archived_duplicates" / "raw"

    archive_field_set = set(archive_fieldnames)
    extra_archive_fields = [
        "dedupe_review_status",
        "dedupe_active_for_analysis",
        "dedupe_review_notes",
        "canonical_issue_id",
        "archive_source",
    ]
    for field in extra_archive_fields:
        if field not in archive_field_set:
            archive_fieldnames.append(field)
            archive_field_set.add(field)

    archive_issue_ids = {row["issue_id"] for row in archive_rows}

    for group in APPROVED_GROUPS:
        for issue_id in group.issue_ids:
            if issue_id not in main_by_issue:
                raise SystemExit(f"Missing issue in main metadata: {issue_id}")
        canonical_id = group.canonical_issue_id

        for issue_id in group.issue_ids:
            review_rows.append(
                {
                    "canonical_issue_id": canonical_id,
                    "issue_id": issue_id,
                    "action": "keep" if issue_id == canonical_id else "archive_duplicate",
                    "notes": group.notes,
                }
            )
            if issue_id == canonical_id:
                continue

            row = dict(main_by_issue[issue_id])

            parsed_relpath = clean(row.get("parsed_relpath"))
            if parsed_relpath:
                move_path(ROOT / parsed_relpath, archive_parsed_root / Path(parsed_relpath).name)
                row["parsed_relpath"] = f"archived_duplicates/parsed/{Path(parsed_relpath).name}"

            transcript_relpath = clean(row.get("transcript_relpath"))
            if transcript_relpath:
                move_path(ROOT / transcript_relpath, archive_transcript_root / Path(transcript_relpath).name)
                row["transcript_relpath"] = (
                    f"archived_duplicates/transcript/{Path(transcript_relpath).name}"
                )

            raw_dir_relpath = clean(row.get("raw_dir_relpath"))
            if raw_dir_relpath:
                issue_raw_dir = Path(raw_dir_relpath).name
                move_path(ROOT / raw_dir_relpath, archive_raw_root / issue_raw_dir)
                row["raw_dir_relpath"] = f"archived_duplicates/raw/{issue_raw_dir}"

            raw_image_relpaths_json = clean(row.get("raw_image_relpaths_json"))
            if raw_image_relpaths_json:
                import json

                rels = json.loads(raw_image_relpaths_json)
                row["raw_image_relpaths_json"] = json.dumps(
                    [f"archived_duplicates/raw/{Path(rel).parent.name}/{Path(rel).name}" for rel in rels]
                )

            row["dedupe_review_status"] = "approved_duplicate"
            row["dedupe_active_for_analysis"] = "0"
            row["dedupe_review_notes"] = group.notes
            row["canonical_issue_id"] = canonical_id
            row["archive_source"] = "final_same_pid_residual_review_20260312"

            if issue_id not in archive_issue_ids:
                archive_rows.append({field: row.get(field, "") for field in archive_fieldnames})
                archive_issue_ids.add(issue_id)
                newly_archived_rows.append(row)

    archived_ids = {row["issue_id"] for row in newly_archived_rows}
    remaining_main_rows = [row for row in main_rows if row["issue_id"] not in archived_ids]

    write_rows(metadata_path, main_fieldnames, remaining_main_rows)
    write_rows(archive_metadata_path, archive_fieldnames, archive_rows)
    write_rows(review_path, ["canonical_issue_id", "issue_id", "action", "notes"], review_rows)

    print(
        {
            "active_rows_after": len(remaining_main_rows),
            "newly_archived_duplicates": len(newly_archived_rows),
            "review_rows_written": len(review_rows),
        }
    )


if __name__ == "__main__":
    main()
