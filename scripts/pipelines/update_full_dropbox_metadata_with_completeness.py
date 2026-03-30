#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path


DROPBOX_METADATA_CSV = Path(
    "/Users/saulrichardson/Dropbox/Inclusionary Zoning/Historical Analysis/raw_data/"
    "newspaper_ordinances/metadata.csv"
)
DROPBOX_README = Path(
    "/Users/saulrichardson/Dropbox/Inclusionary Zoning/Historical Analysis/raw_data/"
    "newspaper_ordinances/README.md"
)
COMPLETENESS_PREDICTIONS_CSV = Path(
    "/Users/saulrichardson/projects/newspapers/newspaper-analysis/artifacts/scratch/"
    "full_ordinance_completeness_eval_20260318_full_gpt54_v10/all_predictions.csv"
)
REPORT_DIR = Path(
    "/Users/saulrichardson/projects/newspapers/newspaper-analysis/artifacts/reports/"
    "full_ordinance_metadata_completeness_update_20260321"
)

COMPLETENESS_FIELDS = (
    "ordinance_artifact_completeness_label",
    "ordinance_artifact_estimated_missing_share_0_to_1",
    "ordinance_artifact_missing_share_band",
    "ordinance_artifact_completeness_confidence_0_to_1",
)

ALLOWED_LABELS = {
    "complete_or_nearly_complete",
    "uncertain",
    "likely_incomplete",
    "definitely_incomplete",
}
ALLOWED_BANDS = {
    "none_or_trivial_lt_5",
    "small_5_15",
    "moderate_15_35",
    "large_35_70",
    "very_large_gt_70",
    "unknown",
}

README_OLD_BULLET = "- human-readable Census match provenance"
README_NEW_BULLET = (
    "- human-readable Census match provenance\n"
    "- ordinance artifact completeness fields from the full-ordinance completeness review"
)

README_OLD_NOTE = (
    "- `extracted_jurisdiction_*` can therefore differ from the final public "
    "`census_id_*` target on publication-fallback rows."
)
README_NEW_NOTE = (
    "- `extracted_jurisdiction_*` can therefore differ from the final public "
    "`census_id_*` target on publication-fallback rows.\n"
    "- `ordinance_artifact_completeness_label` stores the categorical completeness "
    "judgment for the extracted ordinance artifact in this corpus.\n"
    "- `ordinance_artifact_estimated_missing_share_0_to_1` stores the model's "
    "numeric estimate of how much ordinance text appears to be missing. Blank means "
    "the model assigned a label but did not make a defensible numeric estimate.\n"
    "- `ordinance_artifact_missing_share_band` stores the bucketed missing-share "
    "estimate, including `unknown` for rows without a numeric estimate.\n"
    "- `ordinance_artifact_completeness_confidence_0_to_1` stores the model's "
    "confidence in the completeness judgment.\n"
    "- These `ordinance_artifact_*` fields describe the apparent completeness of the "
    "extracted ordinance artifact in this corpus, not a legal certification that the "
    "underlying enacted ordinance was published in full."
)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def clean(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def load_completeness_lookup(path: Path) -> tuple[dict[str, dict[str, str]], dict[str, object]]:
    rows = read_rows(path)
    if not rows:
        raise SystemExit(f"Completeness predictions file is empty: {path}")

    prompt_names = sorted({clean(row.get("prompt_name")) for row in rows if clean(row.get("prompt_name"))})
    duplicates: list[str] = []
    lookup: dict[str, dict[str, str]] = {}

    for row in rows:
        issue_id = clean(row.get("issue_id"))
        if not issue_id:
            raise SystemExit("Completeness predictions contain a blank issue_id")
        if issue_id in lookup:
            duplicates.append(issue_id)
            continue

        label = clean(row.get("completeness_label"))
        band = clean(row.get("missing_share_band"))
        estimate = clean(row.get("estimated_missing_share_0_to_1"))
        confidence = clean(row.get("confidence_0_to_1"))

        if label not in ALLOWED_LABELS:
            raise SystemExit(f"Unexpected completeness_label '{label}' for issue_id={issue_id}")
        if band not in ALLOWED_BANDS:
            raise SystemExit(f"Unexpected missing_share_band '{band}' for issue_id={issue_id}")
        if estimate:
            try:
                value = float(estimate)
            except ValueError as exc:
                raise SystemExit(
                    f"Invalid estimated_missing_share_0_to_1 '{estimate}' for issue_id={issue_id}"
                ) from exc
            if not (0.0 <= value <= 1.0):
                raise SystemExit(
                    f"Out-of-range estimated_missing_share_0_to_1 '{estimate}' for issue_id={issue_id}"
                )
        if confidence:
            try:
                confidence_value = float(confidence)
            except ValueError as exc:
                raise SystemExit(
                    f"Invalid confidence_0_to_1 '{confidence}' for issue_id={issue_id}"
                ) from exc
            if not (0.0 <= confidence_value <= 1.0):
                raise SystemExit(
                    f"Out-of-range confidence_0_to_1 '{confidence}' for issue_id={issue_id}"
                )

        lookup[issue_id] = {
            "ordinance_artifact_completeness_label": label,
            "ordinance_artifact_estimated_missing_share_0_to_1": estimate,
            "ordinance_artifact_missing_share_band": band,
            "ordinance_artifact_completeness_confidence_0_to_1": confidence,
        }

    if duplicates:
        preview = ", ".join(sorted(duplicates)[:10])
        raise SystemExit(f"Duplicate issue_id values in completeness predictions: {preview}")

    summary = {
        "prediction_rows": len(rows),
        "prediction_unique_issue_ids": len(lookup),
        "prompt_names": prompt_names,
    }
    return lookup, summary


def update_readme(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    if "ordinance_artifact_completeness_label" in text:
        return
    if README_OLD_BULLET not in text:
        raise SystemExit(f"Could not find README bullet anchor in {path}")
    if README_OLD_NOTE not in text:
        raise SystemExit(f"Could not find README note anchor in {path}")
    text = text.replace(README_OLD_BULLET, README_NEW_BULLET, 1)
    text = text.replace(README_OLD_NOTE, README_NEW_NOTE, 1)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    if not DROPBOX_METADATA_CSV.is_file():
        raise SystemExit(f"Missing Dropbox metadata file: {DROPBOX_METADATA_CSV}")
    if not DROPBOX_README.is_file():
        raise SystemExit(f"Missing Dropbox README file: {DROPBOX_README}")
    if not COMPLETENESS_PREDICTIONS_CSV.is_file():
        raise SystemExit(f"Missing completeness predictions file: {COMPLETENESS_PREDICTIONS_CSV}")

    metadata_rows = read_rows(DROPBOX_METADATA_CSV)
    if not metadata_rows:
        raise SystemExit(f"Dropbox metadata is empty: {DROPBOX_METADATA_CSV}")

    metadata_fieldnames = list(metadata_rows[0].keys())
    if "issue_id" not in metadata_fieldnames:
        raise SystemExit("Dropbox metadata is missing required column 'issue_id'")

    metadata_issue_ids = [clean(row.get("issue_id")) for row in metadata_rows]
    if len(metadata_issue_ids) != len(set(metadata_issue_ids)):
        raise SystemExit("Dropbox metadata contains duplicate issue_id values")

    completeness_lookup, completeness_summary = load_completeness_lookup(COMPLETENESS_PREDICTIONS_CSV)
    metadata_issue_id_set = set(metadata_issue_ids)
    completeness_issue_id_set = set(completeness_lookup)

    metadata_minus_completeness = sorted(metadata_issue_id_set - completeness_issue_id_set)
    completeness_minus_metadata = sorted(completeness_issue_id_set - metadata_issue_id_set)
    if metadata_minus_completeness or completeness_minus_metadata:
        raise SystemExit(
            "Dropbox metadata and completeness predictions do not have identical issue_id sets: "
            f"metadata_minus_completeness={metadata_minus_completeness[:10]} "
            f"completeness_minus_metadata={completeness_minus_metadata[:10]}"
        )

    final_fieldnames = [field for field in metadata_fieldnames if field not in COMPLETENESS_FIELDS]
    final_fieldnames.extend(COMPLETENESS_FIELDS)

    updated_rows: list[dict[str, str]] = []
    for row in metadata_rows:
        issue_id = clean(row["issue_id"])
        merged = {field: clean(row.get(field)) for field in final_fieldnames}
        merged.update(completeness_lookup[issue_id])
        updated_rows.append(merged)

    backup_metadata = REPORT_DIR / "metadata_before_completeness_update.csv"
    backup_readme = REPORT_DIR / "README_before_completeness_update.md"
    normalized_lookup = REPORT_DIR / "full_ordinance_completeness_public_lookup.csv"
    summary_path = REPORT_DIR / "summary.json"

    shutil.copy2(DROPBOX_METADATA_CSV, backup_metadata)
    shutil.copy2(DROPBOX_README, backup_readme)

    lookup_rows = [
        {"issue_id": issue_id, **completeness_lookup[issue_id]}
        for issue_id in sorted(completeness_lookup)
    ]
    write_rows(normalized_lookup, ["issue_id", *COMPLETENESS_FIELDS], lookup_rows)

    write_rows(DROPBOX_METADATA_CSV, final_fieldnames, updated_rows)
    update_readme(DROPBOX_README)

    label_counts: dict[str, int] = {}
    for row in updated_rows:
        label = row["ordinance_artifact_completeness_label"]
        label_counts[label] = label_counts.get(label, 0) + 1

    summary = {
        "dropbox_metadata_csv": str(DROPBOX_METADATA_CSV),
        "dropbox_readme": str(DROPBOX_README),
        "backup_metadata": str(backup_metadata),
        "backup_readme": str(backup_readme),
        "normalized_lookup": str(normalized_lookup),
        "metadata_rows": len(updated_rows),
        "metadata_fieldnames": final_fieldnames,
        "completeness_summary": completeness_summary,
        "label_counts": label_counts,
        "null_estimated_missing_share_count": sum(
            1
            for row in updated_rows
            if not row["ordinance_artifact_estimated_missing_share_0_to_1"]
        ),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
