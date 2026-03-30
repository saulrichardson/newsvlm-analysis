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
        canonical_issue_id="santa-ana-register__1927-03-05",
        issue_ids=(
            "santa-ana-daily-evening-register__1927-03-05",
            "santa-ana-register__1927-03-05",
            "santa-ana-daily-evening-register__1927-03-07",
            "santa-ana-daily-evening-register__1927-03-08",
            "santa-ana-register__1927-03-08",
        ),
        notes="Same Santa Ana zoning ordinance reprinted across companion issues within three days; ordinance number drift appears OCR-related.",
    ),
    ReviewGroup(
        canonical_issue_id="ironwood-daily-globe__1996-03-09",
        issue_ids=(
            "daily-globe__1996-03-09",
            "ironwood-daily-globe__1996-03-09",
        ),
        notes="Same-day duplicate under generic and specific newspaper slugs.",
    ),
    ReviewGroup(
        canonical_issue_id="cambridge-daily-jeffersonian__1961-03-07",
        issue_ids=(
            "cambridge-daily-jeffersonian__1961-02-28",
            "cambridge-daily-jeffersonian__1961-03-07",
        ),
        notes="Near-identical Byesville zoning ordinance text one week apart; ordinance-number difference is best read as OCR noise rather than a distinct enactment.",
    ),
    ReviewGroup(
        canonical_issue_id="belton-journal-and-bell-county-democrat__1945-06-21",
        issue_ids=(
            "belton-journal-and-bell-county-democrat__1945-06-21",
            "belton-journal-and-bell-county-democrat__1945-06-28",
            "belton-journal__1945-06-28",
        ),
        notes="Belton zoning ordinance reprinted across sister titles and the following week.",
    ),
    ReviewGroup(
        canonical_issue_id="blue-island-sun-standard__1971-07-04",
        issue_ids=(
            "blue-island-sun-standard__1971-07-01",
            "blue-island-sun-standard__1971-07-04",
        ),
        notes="Same Blue Island ordinance No. 2151 reprinted three days later.",
    ),
    ReviewGroup(
        canonical_issue_id="delphi-journal__1960-02-04",
        issue_ids=(
            "delphi-citizen__1960-01-28",
            "delphi-journal__1960-01-28",
            "delphi-journal__1960-02-04",
        ),
        notes="Same Delphi ordinance No. 7-1959 published across companion titles and the following week.",
    ),
    ReviewGroup(
        canonical_issue_id="pella-chronicle__1957-03-07",
        issue_ids=(
            "pella-chronicle__1957-02-28",
            "pella-chronicle__1957-03-07",
        ),
        notes="Same Pella ordinance No. 154 reprinted one week later.",
    ),
    ReviewGroup(
        canonical_issue_id="biddeford-saco-journal__1968-07-16",
        issue_ids=(
            "biddeford-journal__1968-07-16",
            "biddeford-saco-journal__1968-07-16",
        ),
        notes="Same-day duplicate across Biddeford/Biddeford-Saco title variants.",
    ),
    ReviewGroup(
        canonical_issue_id="fergus-falls-daily-journal__1975-11-21",
        issue_ids=(
            "daily-journal__1975-11-21",
            "fergus-falls-daily-journal__1975-11-21",
        ),
        notes="Same-day duplicate across generic and specific Fergus Falls title variants.",
    ),
    ReviewGroup(
        canonical_issue_id="richwood-gazette__1966-08-11",
        issue_ids=(
            "richwood-gazette__1966-08-11",
            "richwood-gazette__1966-08-18",
        ),
        notes="Same Richwood ordinance No. 101 reprinted one week later.",
    ),
    ReviewGroup(
        canonical_issue_id="pampa-daily-news__1937-11-16",
        issue_ids=(
            "pampa-daily-news__1937-11-16",
            "pampa-daily-news__1937-11-24",
            "pampa-daily-news__1937-11-25",
        ),
        notes="Same Pampa ordinance No. 184 published repeatedly within nine days.",
    ),
    ReviewGroup(
        canonical_issue_id="big-spring-daily-herald__1961-06-02",
        issue_ids=(
            "big-spring-daily-herald__1961-05-26",
            "big-spring-daily-herald__1961-06-02",
        ),
        notes="Same Big Spring zoning ordinance text reprinted one week apart with identical effective-date framing.",
    ),
    ReviewGroup(
        canonical_issue_id="racine-journal-times__1976-02-02",
        issue_ids=(
            "racine-journal-times__1976-02-02",
            "racine-journal-times__1976-02-09",
        ),
        notes="Same Racine Title 16 recreation text reprinted in near-identical form one week apart.",
    ),
    ReviewGroup(
        canonical_issue_id="mckinney-daily-courier-gazette__1946-10-21",
        issue_ids=(
            "mckinney-daily-courier-gazette__1946-10-19",
            "mckinney-daily-courier-gazette__1946-10-21",
        ),
        notes="Same McKinney zoning ordinance reprinted two days later.",
    ),
    ReviewGroup(
        canonical_issue_id="cumberland-news__1944-12-13",
        issue_ids=(
            "cumberland-evening-times__1944-12-11",
            "cumberland-news__1944-12-12",
            "cumberland-news__1944-12-13",
            "cumberland-news__1944-12-14",
        ),
        notes="Same Cumberland ordinance No. 1776 printed across paper-title variants over four days.",
    ),
    ReviewGroup(
        canonical_issue_id="amarillo-globe__1931-08-10",
        issue_ids=(
            "amarillo-globe__1931-08-03",
            "amarillo-globe__1931-08-10",
        ),
        notes="Same Amarillo ordinance No. 1407 reprinted one week later.",
    ),
    ReviewGroup(
        canonical_issue_id="eau-claire-leader__1968-03-21",
        issue_ids=(
            "daily-telegram__1968-03-21",
            "eau-claire-leader__1968-03-21",
        ),
        notes="Same-day duplicate across Eau Claire title variants.",
    ),
    ReviewGroup(
        canonical_issue_id="santa-fe-new-mexican__1980-10-21",
        issue_ids=(
            "santa-fe-new-mexican__1980-10-13",
            "santa-fe-new-mexican__1980-10-21",
        ),
        notes="Same Santa Fe ordinance series text reprinted eight days later.",
    ),
)


def clean(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def load_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or []), list(reader)


def write_rows(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def ensure_fields(fieldnames: list[str], extra_fields: list[str]) -> list[str]:
    out = list(fieldnames)
    for field in extra_fields:
        if field not in out:
            out.append(field)
    return out


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


def archive_existing_extras(root: Path, archive_root: Path) -> list[str]:
    moved: list[str] = []
    backups_dir = archive_root / "dropbox_extras"
    backups_dir.mkdir(parents=True, exist_ok=True)
    for name in (
        "metadata_before_manual_dedupe_20260310.csv",
        "metadata_before_promoted_merge_20260311.csv",
        "metadata_pre_repair_20260311.csv",
        "metadata.csv.bak",
    ):
        src = root / name
        if src.exists():
            dst = backups_dir / src.name
            move_path(src, dst)
            moved.append(str(dst))
    duplicates_dir = root / "duplicates"
    if duplicates_dir.exists():
        dst = archive_root / "previous_duplicate_archive"
        if dst.exists():
            shutil.rmtree(dst)
        move_path(duplicates_dir, dst)
        moved.append(str(dst))
    return moved


def normalize_rel(path: Path, base: Path) -> str:
    return path.relative_to(base).as_posix()


def finalize(root: Path, archive_root: Path) -> dict[str, int | str]:
    archive_root.mkdir(parents=True, exist_ok=True)
    moved_extras = archive_existing_extras(root, archive_root)

    metadata_path = root / "metadata.csv"
    fieldnames, rows = load_rows(metadata_path)
    fieldnames = ensure_fields(
        fieldnames,
        [
            "dedupe_review_status",
            "dedupe_active_for_analysis",
            "dedupe_review_notes",
            "canonical_issue_id",
        ],
    )
    row_by_issue = {row["issue_id"]: row for row in rows}

    old_archive_rows: list[dict[str, str]] = []
    old_archive_path = archive_root / "previous_duplicate_archive" / "metadata_duplicates.csv"
    if old_archive_path.exists():
        _, old_archive_rows = load_rows(old_archive_path)

    review_rows: list[dict[str, str]] = []
    newly_archived_rows: list[dict[str, str]] = []
    archived_duplicates_root = archive_root / "archived_duplicates"

    for group in APPROVED_GROUPS:
        for issue_id in group.issue_ids:
            if issue_id not in row_by_issue:
                raise SystemExit(f"Missing issue_id in metadata: {issue_id}")

        canonical_id = group.canonical_issue_id
        for issue_id in group.issue_ids:
            row = row_by_issue[issue_id]
            row["canonical_issue_id"] = canonical_id
            row["dedupe_review_notes"] = group.notes
            if issue_id == canonical_id:
                row["dedupe_review_status"] = "approved_canonical_pid"
                row["dedupe_active_for_analysis"] = "1"
                review_rows.append(
                    {
                        "canonical_issue_id": canonical_id,
                        "issue_id": issue_id,
                        "action": "keep",
                        "notes": group.notes,
                    }
                )
                continue

            row["dedupe_review_status"] = "approved_duplicate_pid"
            row["dedupe_active_for_analysis"] = "0"
            review_rows.append(
                {
                    "canonical_issue_id": canonical_id,
                    "issue_id": issue_id,
                    "action": "archive_duplicate",
                    "notes": group.notes,
                }
            )

            parsed_rel = clean(row.get("parsed_relpath"))
            if parsed_rel:
                src = root / parsed_rel
                dst = archived_duplicates_root / "parsed" / f"{issue_id}.md"
                move_path(src, dst)
                row["parsed_relpath"] = normalize_rel(dst, archive_root)
                row["ordinance_relpath"] = row["parsed_relpath"]

            transcript_rel = clean(row.get("transcript_relpath"))
            if transcript_rel:
                src = root / transcript_rel
                dst = archived_duplicates_root / "transcript" / f"{issue_id}.txt"
                move_path(src, dst)
                row["transcript_relpath"] = normalize_rel(dst, archive_root)
                row["issue_transcript_relpath"] = row["transcript_relpath"]

            raw_dir_rel = clean(row.get("raw_dir_relpath"))
            if raw_dir_rel:
                src = root / raw_dir_rel
                dst = archived_duplicates_root / "raw" / issue_id
                move_path(src, dst)
                row["raw_dir_relpath"] = normalize_rel(dst, archive_root)
                raw_paths = json.loads(row.get("raw_image_relpaths_json") or "[]")
                row["raw_image_relpaths_json"] = json.dumps(
                    [normalize_rel(dst / Path(path).name, archive_root) for path in raw_paths],
                    ensure_ascii=True,
                )

            newly_archived_rows.append(dict(row))

    active_rows = [row for row in rows if clean(row.get("dedupe_active_for_analysis")) != "0"]
    write_rows(metadata_path, fieldnames, active_rows)

    archive_fieldnames = ensure_fields(
        fieldnames,
        ["archive_source"],
    )
    for row in old_archive_rows:
        row.setdefault("archive_source", "preexisting_dropbox_duplicate")
    for row in newly_archived_rows:
        row["archive_source"] = "pid_near_time_duplicate"
    combined_archived = old_archive_rows + newly_archived_rows
    write_rows(archive_root / "metadata_archived_duplicates.csv", archive_fieldnames, combined_archived)

    write_rows(
        archive_root / "pid_dedupe_review_decisions_20260311.csv",
        ["canonical_issue_id", "issue_id", "action", "notes"],
        review_rows,
    )

    return {
        "moved_extra_items": len(moved_extras),
        "newly_archived_duplicates": len(newly_archived_rows),
        "preexisting_archived_duplicates": len(old_archive_rows),
        "active_rows": len(active_rows),
        "archived_rows_total": len(combined_archived),
        "archive_root": str(archive_root),
    }


if __name__ == "__main__":
    result = finalize(ROOT, ARCHIVE_ROOT)
    print(json.dumps(result, indent=2, ensure_ascii=True))
