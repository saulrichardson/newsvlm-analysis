#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
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
        canonical_issue_id="abilene-reporter-news__1946-04-29",
        issue_ids=(
            "abilene-reporter-news__1946-04-29",
            "abilene-reporter-news__1946-05-01",
        ),
        notes="Same 1946 Abilene zoning ordinance reprinted two days later in the same paper; keep the slightly fuller April 29 printing.",
    ),
    ReviewGroup(
        canonical_issue_id="st-joseph-herald-press__1969-09-19",
        issue_ids=(
            "benton-harbor-news-palladium__1969-09-19",
            "st-joseph-herald-press__1969-09-19",
        ),
        notes="Same-day companion-title duplicate of the Sodus Township revised zoning ordinance; keep the slightly fuller St. Joseph Herald-Press version.",
    ),
    ReviewGroup(
        canonical_issue_id="biloxi-daily-herald__1940-08-28",
        issue_ids=(
            "biloxi-daily-herald__1940-07-31",
            "biloxi-daily-herald__1940-08-28",
        ),
        notes="Same Gulfport Ordinance No. 611 reprinted later with a slightly fuller body; keep the August 28 printing.",
    ),
    ReviewGroup(
        canonical_issue_id="brewster-standard__1961-01-12",
        issue_ids=(
            "brewster-standard__1961-01-05",
            "brewster-standard__1961-01-12",
        ),
        notes="Same adopted Town of Southeast zoning ordinance reprinted one week later; keep the slightly fuller January 12 printing.",
    ),
    ReviewGroup(
        canonical_issue_id="cedar-city-iron-county-record__1952-12-04",
        issue_ids=(
            "cedar-city-iron-county-record__1952-12-04",
            "cedar-city-iron-county-record__1952-12-25",
        ),
        notes="Same Cedar City zoning amendment ordinance reprinted later in the month; keep the earlier printing with the fuller opening and enacting clause.",
    ),
    ReviewGroup(
        canonical_issue_id="dunkirk-evening-observer__1969-08-06",
        issue_ids=(
            "dunkirk-evening-observer__1969-08-06",
            "dunkirk-evening-observer__1969-09-12",
        ),
        notes="Same Fredonia amended zoning ordinance reprinted later with only minor wording variation; keep the slightly fuller August 6 printing.",
    ),
    ReviewGroup(
        canonical_issue_id="east-hampton-news__1940-10-04",
        issue_ids=(
            "east-hampton-news__1940-10-04",
            "east-hampton-news__1940-10-18",
        ),
        notes="Same partial East Hampton legal-notice continuation reprinted two weeks later; keep a single representative printing.",
    ),
    ReviewGroup(
        canonical_issue_id="fort-recovery-journal__1972-11-23",
        issue_ids=(
            "fort-recovery-journal__1972-11-23",
            "fort-recovery-journal__1972-11-30",
        ),
        notes="Same Ft. Recovery Ordinance No. 444 reprinted one week later; keep the slightly fuller November 23 printing.",
    ),
    ReviewGroup(
        canonical_issue_id="logan-daily-news__1964-08-13",
        issue_ids=(
            "logan-daily-news__1964-08-13",
            "logan-daily-news__1964-08-20",
        ),
        notes="Same Logan Ordinance No. 24, 1964 reprinted one week later; keep the slightly fuller August 13 printing.",
    ),
    ReviewGroup(
        canonical_issue_id="rhinelander-new-north__1933-07-13",
        issue_ids=(
            "rhinelander-new-north__1933-06-29",
            "rhinelander-new-north__1933-07-13",
        ),
        notes="Same Oneida County zoning ordinance reprinted with the official map heading; keep the July 13 printing as the more informative representative.",
    ),
    ReviewGroup(
        canonical_issue_id="salem-news__1973-05-25",
        issue_ids=(
            "salem-news__1973-05-18",
            "salem-news__1973-05-25",
        ),
        notes="Same Salem Ordinance No. 711103-112 reprinted one week later; keep the fuller May 25 printing.",
    ),
)


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
    review_path = ARCHIVE_ROOT / "pid_dedupe_review_decisions_20260312_round2.csv"

    main_fieldnames, main_rows = read_rows(metadata_path)
    archive_fieldnames, archive_rows = read_rows(archive_metadata_path)
    main_by_issue = {row["issue_id"]: row for row in main_rows}
    archive_issue_ids = {row["issue_id"] for row in archive_rows}

    archive_parsed_root = ARCHIVE_ROOT / "archived_duplicates" / "parsed"
    archive_transcript_root = ARCHIVE_ROOT / "archived_duplicates" / "transcript"
    archive_raw_root = ARCHIVE_ROOT / "archived_duplicates" / "raw"

    for field in [
        "dedupe_review_status",
        "dedupe_active_for_analysis",
        "dedupe_review_notes",
        "canonical_issue_id",
        "archive_source",
    ]:
        if field not in archive_fieldnames:
            archive_fieldnames.append(field)

    review_rows: list[dict[str, str]] = []
    newly_archived_ids: list[str] = []

    for group in APPROVED_GROUPS:
        for issue_id in group.issue_ids:
            if issue_id not in main_by_issue:
                raise SystemExit(f"Missing issue in main metadata: {issue_id}")

        for issue_id in group.issue_ids:
            review_rows.append(
                {
                    "canonical_issue_id": group.canonical_issue_id,
                    "issue_id": issue_id,
                    "action": "keep" if issue_id == group.canonical_issue_id else "archive_duplicate",
                    "notes": group.notes,
                }
            )
            if issue_id == group.canonical_issue_id:
                continue

            row = dict(main_by_issue[issue_id])
            parsed_relpath = row.get("parsed_relpath", "").strip()
            transcript_relpath = row.get("transcript_relpath", "").strip()
            raw_dir_relpath = row.get("raw_dir_relpath", "").strip()

            if parsed_relpath:
                move_path(ROOT / parsed_relpath, archive_parsed_root / Path(parsed_relpath).name)
                row["parsed_relpath"] = f"archived_duplicates/parsed/{Path(parsed_relpath).name}"

            if transcript_relpath:
                move_path(
                    ROOT / transcript_relpath,
                    archive_transcript_root / Path(transcript_relpath).name,
                )
                row["transcript_relpath"] = (
                    f"archived_duplicates/transcript/{Path(transcript_relpath).name}"
                )

            if raw_dir_relpath:
                issue_raw_dir = Path(raw_dir_relpath).name
                move_path(ROOT / raw_dir_relpath, archive_raw_root / issue_raw_dir)
                row["raw_dir_relpath"] = f"archived_duplicates/raw/{issue_raw_dir}"

            raw_image_relpaths_json = row.get("raw_image_relpaths_json", "").strip()
            if raw_image_relpaths_json:
                rels = json.loads(raw_image_relpaths_json)
                row["raw_image_relpaths_json"] = json.dumps(
                    [f"archived_duplicates/raw/{Path(rel).parent.name}/{Path(rel).name}" for rel in rels]
                )

            row["dedupe_review_status"] = "approved_duplicate"
            row["dedupe_active_for_analysis"] = "0"
            row["dedupe_review_notes"] = group.notes
            row["canonical_issue_id"] = group.canonical_issue_id
            row["archive_source"] = "final_same_pid_residual_review_round2_20260312"

            if issue_id not in archive_issue_ids:
                archive_rows.append({field: row.get(field, "") for field in archive_fieldnames})
                archive_issue_ids.add(issue_id)
                newly_archived_ids.append(issue_id)

    remaining_main_rows = [row for row in main_rows if row["issue_id"] not in set(newly_archived_ids)]
    write_rows(metadata_path, main_fieldnames, remaining_main_rows)
    write_rows(archive_metadata_path, archive_fieldnames, archive_rows)
    write_rows(review_path, ["canonical_issue_id", "issue_id", "action", "notes"], review_rows)

    print(
        {
            "active_rows_after": len(remaining_main_rows),
            "newly_archived_duplicates": len(newly_archived_ids),
            "review_rows_written": len(review_rows),
        }
    )


if __name__ == "__main__":
    main()
