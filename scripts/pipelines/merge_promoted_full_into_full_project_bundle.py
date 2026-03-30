#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path


NEW_PROMOTED_REPRINT_GROUPS = {
    "reprint_promoted_oak_park_1921": [
        "oak-park-oak-leaves__1921-05-21",
        "oak-park-oak-leaves__1921-06-25",
    ],
    "reprint_promoted_manlius_1968_dec": [
        "fayetteville-eagle-bulletin-and-dewitt-news-times__1968-12-19",
        "fayetteville-eagle-bulletin-and-dewitt-news-times__1968-12-24",
    ],
    "reprint_promoted_laverne_1974_crosspaper": [
        "san-dimas-press__1974-12-12",
        "montclair-tribune__1974-12-12",
    ],
    "reprint_promoted_pompey_1968_crosspaper": [
        "fayetteville-eagle-bulletin-and-dewitt-news-times__1968-06-20",
        "fayetteville-eagle-bulletin__1968-06-20",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge the promoted hidden full-ordinance corpus into the main "
            "newspaper_ordinances Dropbox corpus."
        )
    )
    parser.add_argument("--full-root", type=Path, required=True)
    parser.add_argument("--promoted-root", type=Path, required=True)
    parser.add_argument("--backup-metadata", action="store_true")
    return parser.parse_args()


def load_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        if reader.fieldnames is None:
            raise SystemExit(f"No CSV header found in {path}")
        return list(reader.fieldnames), rows


def ensure_fields(fieldnames: list[str], extra: list[str]) -> list[str]:
    out = list(fieldnames)
    for field in extra:
        if field not in out:
            out.append(field)
    return out


def copy_if_missing(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)


def seed_new_reprint_groups(rows: list[dict[str, str]]) -> None:
    by_issue = {row["issue_id"]: row for row in rows}
    for group_id, issue_ids in NEW_PROMOTED_REPRINT_GROUPS.items():
        for issue_id in issue_ids:
            row = by_issue.get(issue_id)
            if row is None:
                raise SystemExit(f"Missing issue for seeded reprint group {group_id}: {issue_id}")
            row["reprint_group_id"] = group_id
            row["reprint_group_size"] = str(len(issue_ids))
            row["reprint_group_issue_ids_json"] = json.dumps(issue_ids, ensure_ascii=True)


def main() -> None:
    args = parse_args()
    full_root = args.full_root.resolve()
    promoted_root = args.promoted_root.resolve()

    full_meta = full_root / "metadata.csv"
    prom_meta = promoted_root / "metadata.csv"
    if not full_meta.is_file():
        raise SystemExit(f"Missing full metadata: {full_meta}")
    if not prom_meta.is_file():
        raise SystemExit(f"Missing promoted metadata: {prom_meta}")

    if args.backup_metadata:
        backup_path = full_root / "metadata_before_promoted_merge_20260311.csv"
        shutil.copy2(full_meta, backup_path)

    full_fields, full_rows = load_rows(full_meta)
    prom_fields, prom_rows = load_rows(prom_meta)

    full_ids = {row["issue_id"] for row in full_rows}
    prom_ids = {row["issue_id"] for row in prom_rows}
    overlap = sorted(full_ids & prom_ids)
    if overlap:
        raise SystemExit(f"Found overlapping issue_ids between full and promoted corpora: {overlap[:10]}")

    all_fields = ensure_fields(full_fields, list(prom_fields))
    all_fields = ensure_fields(
        all_fields,
        [
            "full_ordinance_origin",
            "source_label",
            "promotion_second_stage_class",
            "promotion_instrument_family",
            "promotion_operativity",
            "promotion_confidence_0_to_1",
            "promotion_evidence_quote",
            "promotion_rationale",
            "promotion_quality_flags_json",
            "promotion_response_json_ok",
            "transcription_contract_ok",
            "transcription_ordinance_chars",
            "transcription_source_line",
            "dedupe_review_status",
            "dedupe_active_for_analysis",
            "dedupe_review_notes",
        ],
    )

    normalized_full: list[dict[str, str]] = []
    for row in full_rows:
        merged = {field: row.get(field, "") for field in all_fields}
        merged["full_ordinance_origin"] = "original_full"
        if not merged.get("source_label"):
            merged["source_label"] = "code_publication_full_issue"
        normalized_full.append(merged)

    normalized_prom: list[dict[str, str]] = []
    for row in prom_rows:
        merged = {field: row.get(field, "") for field in all_fields}
        merged["full_ordinance_origin"] = "promoted_hidden_full"
        if not merged.get("ordinance_relpath") and merged.get("parsed_relpath"):
            merged["ordinance_relpath"] = merged["parsed_relpath"]
        if not merged.get("ordinance_status") and merged.get("parsed_relpath"):
            merged["ordinance_status"] = "ok"
        if not merged.get("ordinance_contract_ok"):
            merged["ordinance_contract_ok"] = merged.get("transcription_contract_ok", "")
        if not merged.get("ordinance_chars"):
            merged["ordinance_chars"] = merged.get("transcription_ordinance_chars", "")
        if not merged.get("ordinance_source_line"):
            merged["ordinance_source_line"] = merged.get("transcription_source_line", "")
        normalized_prom.append(merged)

    # Copy promoted files into the main corpus layout.
    for row in normalized_prom:
        issue_id = row["issue_id"]
        parsed_rel = row.get("parsed_relpath", "")
        transcript_rel = row.get("transcript_relpath", "")
        raw_rel = row.get("raw_dir_relpath", "")
        if parsed_rel:
            copy_if_missing(promoted_root / parsed_rel, full_root / parsed_rel)
        if transcript_rel:
            copy_if_missing(promoted_root / transcript_rel, full_root / transcript_rel)
        if raw_rel:
            copy_if_missing(promoted_root / raw_rel, full_root / raw_rel)
        # Normalize root-specific relpaths to the merged full root.
        row["parsed_relpath"] = f"parsed/{issue_id}.md"
        row["ordinance_relpath"] = row.get("ordinance_relpath") or row["parsed_relpath"]
        row["transcript_relpath"] = f"transcript/{issue_id}.txt"
        row["raw_dir_relpath"] = f"raw/{issue_id}"

    merged_rows = normalized_full + normalized_prom
    seed_new_reprint_groups(merged_rows)

    with full_meta.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=all_fields)
        writer.writeheader()
        writer.writerows(merged_rows)

    print(
        {
            "full_root": str(full_root),
            "full_rows_before": len(full_rows),
            "promoted_rows_added": len(prom_rows),
            "merged_rows": len(merged_rows),
        }
    )


if __name__ == "__main__":
    main()
