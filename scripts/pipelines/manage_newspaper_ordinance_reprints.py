#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path


def clean(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def load_metadata_rows(metadata_path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with metadata_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        return list(reader.fieldnames or []), rows


def write_metadata_rows(
    metadata_path: Path, fieldnames: list[str], rows: list[dict[str, str]]
) -> None:
    with metadata_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def ensure_fieldnames(fieldnames: list[str], extra_fields: list[str]) -> list[str]:
    result = list(fieldnames)
    for field in extra_fields:
        if field not in result:
            result.append(field)
    return result


def grouped_reprints(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    groups: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        group_id = clean(row.get("reprint_group_id"))
        if not group_id:
            continue
        groups.setdefault(group_id, []).append(row)
    return groups


def write_review_template(metadata_path: Path, out_csv: Path) -> None:
    _, rows = load_metadata_rows(metadata_path)
    groups = grouped_reprints(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "reprint_group_id",
        "issue_count",
        "issue_ids_json",
        "auto_canonical_issue_id",
        "approved",
        "canonical_issue_id",
        "notes",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for group_id, members in sorted(groups.items()):
            members_sorted = sorted(members, key=lambda row: row["issue_date"])
            auto_canonical = next(
                (
                    row["issue_id"]
                    for row in members_sorted
                    if clean(row.get("reprint_status")) == "canonical"
                ),
                members_sorted[0]["issue_id"],
            )
            writer.writerow(
                {
                    "reprint_group_id": group_id,
                    "issue_count": str(len(members_sorted)),
                    "issue_ids_json": json.dumps(
                        [row["issue_id"] for row in members_sorted], ensure_ascii=True
                    ),
                    "auto_canonical_issue_id": auto_canonical,
                    "approved": "",
                    "canonical_issue_id": auto_canonical,
                    "notes": "",
                }
            )


def normalize_relpath(path: str) -> str:
    return path.replace("\\", "/")


def move_if_needed(root: Path, current_relpath: str, target_relpath: str) -> str:
    current_relpath = clean(current_relpath)
    target_relpath = clean(target_relpath)
    if not current_relpath:
        return ""
    src = root / current_relpath
    dst = root / target_relpath
    if current_relpath == target_relpath:
        return target_relpath
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        if src.exists() and src.resolve() != dst.resolve():
            if src.is_dir():
                shutil.rmtree(src)
            else:
                src.unlink()
        return target_relpath
    if src.exists():
        shutil.move(str(src), str(dst))
        return target_relpath
    return target_relpath


def apply_review(root: Path, metadata_path: Path, review_csv: Path) -> dict[str, int]:
    fieldnames, rows = load_metadata_rows(metadata_path)
    fieldnames = ensure_fieldnames(
        fieldnames,
        [
            "dedupe_review_status",
            "dedupe_active_for_analysis",
            "dedupe_review_notes",
        ],
    )
    groups = grouped_reprints(rows)

    decisions: dict[str, dict[str, str]] = {}
    with review_csv.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            group_id = clean(row.get("reprint_group_id"))
            if group_id:
                decisions[group_id] = {k: clean(v) for k, v in row.items()}

    moved_parsed = 0
    moved_transcripts = 0
    moved_raw_dirs = 0
    approved_groups = 0
    rejected_groups = 0

    for row in rows:
        row.setdefault("dedupe_review_status", "")
        row.setdefault("dedupe_active_for_analysis", "")
        row.setdefault("dedupe_review_notes", "")

    for group_id, members in groups.items():
        decision = decisions.get(group_id)
        if not decision:
            for row in members:
                row["dedupe_review_status"] = "unreviewed_group"
                row["dedupe_active_for_analysis"] = "1"
            continue

        approved = clean(decision.get("approved")).lower() in {"1", "true", "yes", "y"}
        canonical_issue_id = clean(decision.get("canonical_issue_id"))
        notes = clean(decision.get("notes"))
        member_ids = {row["issue_id"] for row in members}
        if canonical_issue_id and canonical_issue_id not in member_ids:
            raise SystemExit(
                f"Review CSV canonical_issue_id {canonical_issue_id} is not in group {group_id}"
            )

        if not approved:
            rejected_groups += 1
            for row in members:
                row["dedupe_review_status"] = "rejected_group"
                row["dedupe_active_for_analysis"] = "1"
                row["dedupe_review_notes"] = notes
                row["reprint_status"] = "unique"
                row["reprint_is_canonical"] = "1"
                row["reprint_canonical_issue_id"] = row["issue_id"]
                row["reprint_canonical_parsed_relpath"] = clean(
                    row.get("parsed_relpath")
                )
                row["reprint_days_from_canonical"] = "0"
                row["reprint_similarity_to_canonical"] = "1.0"
                row["reprint_detection_method"] = "manual_review_rejected"
            continue

        approved_groups += 1
        if not canonical_issue_id:
            canonical_issue_id = next(
                (
                    row["issue_id"]
                    for row in members
                    if clean(row.get("reprint_status")) == "canonical"
                ),
                sorted(members, key=lambda row: row["issue_date"])[0]["issue_id"],
            )

        canonical_row = next(
            row for row in members if row["issue_id"] == canonical_issue_id
        )
        canonical_parsed_relpath = clean(canonical_row.get("parsed_relpath"))

        for row in members:
            row["reprint_canonical_issue_id"] = canonical_issue_id
            row["reprint_canonical_parsed_relpath"] = canonical_parsed_relpath
            row["dedupe_review_notes"] = notes
            if row["issue_id"] == canonical_issue_id:
                row["dedupe_review_status"] = "approved_canonical"
                row["dedupe_active_for_analysis"] = "1"
                row["reprint_status"] = "canonical"
                row["reprint_is_canonical"] = "1"
                continue

            issue_id = row["issue_id"]
            row["dedupe_review_status"] = "approved_duplicate"
            row["dedupe_active_for_analysis"] = "0"
            row["reprint_status"] = "duplicate"
            row["reprint_is_canonical"] = "0"

            parsed_relpath = clean(row.get("parsed_relpath"))
            if parsed_relpath:
                target_relpath = normalize_relpath(f"duplicates/parsed/{issue_id}.md")
                row["parsed_relpath"] = move_if_needed(root, parsed_relpath, target_relpath)
                row["ordinance_relpath"] = row["parsed_relpath"]
                moved_parsed += 1

            transcript_relpath = clean(row.get("transcript_relpath"))
            if transcript_relpath:
                target_relpath = normalize_relpath(f"duplicates/transcript/{issue_id}.txt")
                row["transcript_relpath"] = move_if_needed(
                    root, transcript_relpath, target_relpath
                )
                row["issue_transcript_relpath"] = row["transcript_relpath"]
                moved_transcripts += 1

            raw_dir_relpath = clean(row.get("raw_dir_relpath"))
            if raw_dir_relpath:
                target_relpath = normalize_relpath(f"duplicates/raw/{issue_id}")
                row["raw_dir_relpath"] = move_if_needed(root, raw_dir_relpath, target_relpath)
                raw_paths = json.loads(row.get("raw_image_relpaths_json") or "[]")
                row["raw_image_relpaths_json"] = json.dumps(
                    [
                        normalize_relpath(f"duplicates/raw/{issue_id}/{Path(path).name}")
                        for path in raw_paths
                    ],
                    ensure_ascii=True,
                )
                moved_raw_dirs += 1

    for row in rows:
        if clean(row.get("reprint_group_id")):
            continue
        row["dedupe_review_status"] = "unique"
        row["dedupe_active_for_analysis"] = "1"
        row["dedupe_review_notes"] = ""

    active_rows = [row for row in rows if clean(row.get("dedupe_active_for_analysis")) != "0"]
    duplicate_rows = [row for row in rows if clean(row.get("dedupe_active_for_analysis")) == "0"]

    duplicates_root = root / "duplicates"
    duplicates_root.mkdir(parents=True, exist_ok=True)
    duplicates_metadata_path = duplicates_root / "metadata_duplicates.csv"

    write_metadata_rows(metadata_path, fieldnames, active_rows)
    write_metadata_rows(duplicates_metadata_path, fieldnames, duplicate_rows)
    return {
        "approved_groups": approved_groups,
        "rejected_groups": rejected_groups,
        "moved_parsed": moved_parsed,
        "moved_transcripts": moved_transcripts,
        "moved_raw_dirs": moved_raw_dirs,
        "active_rows_written": len(active_rows),
        "duplicate_rows_written": len(duplicate_rows),
        "duplicates_metadata_path": str(duplicates_metadata_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Manage manual reprint curation for Desmond's newspaper_ordinances export. "
            "Can write a review template from current reprint groups and/or apply a reviewed CSV "
            "by moving approved duplicate files into duplicates/ and updating metadata.csv."
        )
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--write-review-template", type=Path)
    parser.add_argument("--apply-review-csv", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    metadata_path = root / "metadata.csv"
    if not metadata_path.is_file():
        raise SystemExit(f"Missing metadata.csv under root: {metadata_path}")

    if not args.write_review_template and not args.apply_review_csv:
        raise SystemExit("Specify at least one of --write-review-template or --apply-review-csv")

    if args.write_review_template:
        write_review_template(metadata_path, args.write_review_template.resolve())

    summary = None
    if args.apply_review_csv:
        summary = apply_review(root, metadata_path, args.apply_review_csv.resolve())

    print(
        json.dumps(
            {
                "root": str(root),
                "review_template_written": str(args.write_review_template.resolve())
                if args.write_review_template
                else "",
                "review_applied": str(args.apply_review_csv.resolve())
                if args.apply_review_csv
                else "",
                "summary": summary or {},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
