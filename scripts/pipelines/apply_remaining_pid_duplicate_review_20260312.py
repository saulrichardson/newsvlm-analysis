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
            "abilene-reporter-news__1946-04-30",
            "abilene-reporter-news__1946-05-02",
        ),
        notes="Same Abilene 1946 zoning ordinance reprinted within three days in the same paper; keep the fullest 1946-04-29 text.",
    ),
    ReviewGroup(
        canonical_issue_id="ironwood-daily-globe__1975-05-23",
        issue_ids=(
            "daily-globe__1975-05-23",
            "ironwood-daily-globe__1975-05-23",
        ),
        notes="Same Bergland Township proposed zoning ordinance printed the same day under companion titles; keep the fuller Ironwood version.",
    ),
    ReviewGroup(
        canonical_issue_id="ironwood-daily-globe__2008-04-28",
        issue_ids=(
            "daily-globe__2008-04-28",
            "ironwood-daily-globe__2008-04-28",
        ),
        notes="Same Bergland Township 2008 zoning ordinance printed the same day under companion titles; keep the fuller Ironwood version.",
    ),
    ReviewGroup(
        canonical_issue_id="belton-journal-and-bell-county-democrat__1945-06-21",
        issue_ids=(
            "belton-journal-and-bell-county-democrat__1945-06-21",
            "belton-journal__1945-06-21",
        ),
        notes="Same-day sister-title duplicate of the Belton zoning ordinance; keep the fuller Journal and Bell County Democrat text.",
    ),
    ReviewGroup(
        canonical_issue_id="belton-journal-and-bell-county-democrat__1945-07-12",
        issue_ids=(
            "belton-journal-and-bell-county-democrat__1945-07-05",
            "belton-journal-and-bell-county-democrat__1945-07-12",
            "belton-journal__1945-07-12",
        ),
        notes="Same Belton zoning ordinance reprinted one week later and across sister titles; keep the fullest 1945-07-12 Journal and Bell County Democrat printing.",
    ),
    ReviewGroup(
        canonical_issue_id="brewster-standard__1970-11-12",
        issue_ids=(
            "brewster-standard__1970-11-12",
            "brewster-standard__1970-11-19",
            "brewster-standard__1970-11-26",
        ),
        notes="Same enacted Village of Brewster Zoning Ordinance 1970 reprinted over three consecutive weeks; keep the earliest full printing.",
    ),
    ReviewGroup(
        canonical_issue_id="cumberland-news__1944-12-13",
        issue_ids=(
            "cumberland-evening-times__1944-12-13",
            "cumberland-news__1944-12-13",
        ),
        notes="Same-day companion-title duplicate of Cumberland Ordinance No. 1776; keep the slightly fuller Cumberland News version.",
    ),
    ReviewGroup(
        canonical_issue_id="delphi-journal__1960-02-04",
        issue_ids=(
            "delphi-citizen__1960-02-04",
            "delphi-journal__1960-02-04",
        ),
        notes="Same-day companion-title duplicate of Delphi General Ordinance No. 7-1959; keep the fuller Delphi Journal printing.",
    ),
    ReviewGroup(
        canonical_issue_id="ironwood-daily-globe__1974-05-13",
        issue_ids=(
            "daily-globe__1974-05-13",
            "ironwood-daily-globe__1974-05-13",
        ),
        notes="Same-day companion-title duplicate of Erwin Township Zoning Ordinance No. 7; keep the fuller Ironwood Daily Globe printing.",
    ),
    ReviewGroup(
        canonical_issue_id="la-crosse-tribune__1953-09-18",
        issue_ids=(
            "la-crosse-tribune__1953-09-18",
            "la-crosse-tribune__1953-09-25",
        ),
        notes="Same La Crosse County zoning ordinance reprinted one week later; keep the earlier fuller 1953-09-18 printing.",
    ),
    ReviewGroup(
        canonical_issue_id="montclair-tribune__1974-12-12",
        issue_ids=(
            "la-verne-leader__1974-12-12",
            "montclair-tribune__1974-12-12",
        ),
        notes="Same-day duplicate of La Verne Ordinance No. 475 across papers; keep the slightly fuller Montclair Tribune printing.",
    ),
    ReviewGroup(
        canonical_issue_id="lawrence-daily-journal-world__1949-11-30",
        issue_ids=(
            "lawrence-daily-journal-world__1949-11-30",
            "lawrence-journal-world__1949-11-30",
        ),
        notes="Same-day Lawrence zoning ordinance duplicate across title variants; keep the version that preserves Ordinance No. 1987.",
    ),
    ReviewGroup(
        canonical_issue_id="hanover-evening-sun__1976-12-04",
        issue_ids=(
            "hanover-evening-sun__1976-12-03",
            "hanover-evening-sun__1976-12-04",
        ),
        notes="Same Town of Manchester Zoning Ordinance No. 1276 reprinted the next day; keep the fuller 1976-12-04 printing.",
    ),
    ReviewGroup(
        canonical_issue_id="orrville-courier-crescent__1960-04-07",
        issue_ids=(
            "orrville-courier-crescent__1960-04-07",
            "orrville-courier-crescent__1960-04-14",
        ),
        notes="Same Orrville Ordinance No. 4-60 reprinted one week later; keep the earlier fuller 1960-04-07 printing.",
    ),
    ReviewGroup(
        canonical_issue_id="orrville-courier-crescent__1969-12-23",
        issue_ids=(
            "orrville-courier-crescent__1969-12-23",
            "orrville-courier-crescent__1969-12-31",
        ),
        notes="Same Orrville Ordinance 38-69 reprinted one week later; keep the earlier fuller 1969-12-23 printing.",
    ),
    ReviewGroup(
        canonical_issue_id="arlington-heights-daily-herald-suburban-chicago__1927-07-08",
        issue_ids=(
            "arlington-heights-daily-herald-suburban-chicago__1927-07-08",
            "arlington-heights-herald__1927-07-08",
        ),
        notes="Same-day Palatine Ordinance No. 555 duplicate across title variants; keep the fuller Daily Herald suburban printing.",
    ),
    ReviewGroup(
        canonical_issue_id="pampa-daily-news__1937-11-16",
        issue_ids=(
            "pampa-daily-news__1937-11-16",
            "pampa-daily-news__1937-11-21",
        ),
        notes="Same Pampa Ordinance No. 184 reprinted five days later; keep the earlier fuller 1937-11-16 printing.",
    ),
    ReviewGroup(
        canonical_issue_id="richwood-gazette__1956-08-02",
        issue_ids=(
            "richwood-gazette__1956-07-26",
            "richwood-gazette__1956-08-02",
        ),
        notes="Same Richwood Ordinance No. 543 reprinted one week later; keep the slightly fuller 1956-08-02 printing.",
    ),
    ReviewGroup(
        canonical_issue_id="daily-telegram__1962-04-20",
        issue_ids=(
            "daily-telegram__1962-04-20",
            "eau-claire-leader__1962-04-21",
        ),
        notes="Same Town of Washington revised zoning ordinance printed across companion titles on adjacent days; keep the fuller 1962-04-20 printing.",
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


def normalize_rel(path: Path, base: Path) -> str:
    return path.relative_to(base).as_posix()


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
    main_fieldnames, main_rows = read_rows(metadata_path)
    archive_fieldnames, archive_rows = read_rows(archive_metadata_path)

    main_by_issue = {row["issue_id"]: row for row in main_rows}
    review_rows: list[dict[str, str]] = []
    newly_archived_rows: list[dict[str, str]] = []

    archive_parsed_root = ARCHIVE_ROOT / "archived_duplicates" / "parsed"
    archive_transcript_root = ARCHIVE_ROOT / "archived_duplicates" / "transcript"
    archive_raw_root = ARCHIVE_ROOT / "archived_duplicates" / "raw"

    for group in APPROVED_GROUPS:
        for issue_id in group.issue_ids:
            if issue_id not in main_by_issue:
                raise SystemExit(f"Missing issue in main metadata: {issue_id}")

        canonical_id = group.canonical_issue_id
        if canonical_id not in group.issue_ids:
            raise SystemExit(f"Canonical {canonical_id} not in issue_ids for group")

        for issue_id in group.issue_ids:
            row = main_by_issue[issue_id]
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

            parsed_relpath = clean(row.get("parsed_relpath"))
            if parsed_relpath:
                src = ROOT / parsed_relpath
                dst = archive_parsed_root / f"{issue_id}.md"
                move_path(src, dst)
                row["parsed_relpath"] = normalize_rel(dst, ARCHIVE_ROOT)
                row["ordinance_relpath"] = row["parsed_relpath"]

            transcript_relpath = clean(row.get("transcript_relpath"))
            if transcript_relpath:
                src = ROOT / transcript_relpath
                dst = archive_transcript_root / f"{issue_id}.txt"
                move_path(src, dst)
                row["transcript_relpath"] = normalize_rel(dst, ARCHIVE_ROOT)
                row["issue_transcript_relpath"] = row["transcript_relpath"]

            raw_dir_relpath = clean(row.get("raw_dir_relpath"))
            if raw_dir_relpath:
                src = ROOT / raw_dir_relpath
                dst = archive_raw_root / issue_id
                move_path(src, dst)
                row["raw_dir_relpath"] = normalize_rel(dst, ARCHIVE_ROOT)
                raw_paths = json.loads(row.get("raw_image_relpaths_json") or "[]")
                row["raw_image_relpaths_json"] = json.dumps(
                    [normalize_rel(dst / Path(path).name, ARCHIVE_ROOT) for path in raw_paths],
                    ensure_ascii=True,
                )

            row["archive_source"] = "pid_near_time_duplicate_round2"
            row["canonical_issue_id"] = canonical_id
            row["dedupe_review_status"] = "approved_duplicate_pid_round2"
            row["dedupe_active_for_analysis"] = "0"
            row["dedupe_review_notes"] = group.notes
            newly_archived_rows.append(dict(row))

    duplicate_ids = {row["issue_id"] for row in newly_archived_rows}
    active_rows = [row for row in main_rows if row["issue_id"] not in duplicate_ids]

    write_rows(metadata_path, main_fieldnames, active_rows)

    archive_fieldnames_set = list(archive_fieldnames)
    for field in ("archive_source", "canonical_issue_id", "dedupe_review_status", "dedupe_active_for_analysis", "dedupe_review_notes"):
        if field not in archive_fieldnames_set:
            archive_fieldnames_set.append(field)
    write_rows(archive_metadata_path, archive_fieldnames_set, archive_rows + newly_archived_rows)

    review_path = ARCHIVE_ROOT / "pid_dedupe_review_decisions_20260312.csv"
    write_rows(
        review_path,
        ["canonical_issue_id", "issue_id", "action", "notes"],
        review_rows,
    )

    print(
        json.dumps(
            {
                "active_rows_after": len(active_rows),
                "newly_archived_duplicates": len(newly_archived_rows),
                "review_csv": str(review_path),
                "archive_metadata_path": str(archive_metadata_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
