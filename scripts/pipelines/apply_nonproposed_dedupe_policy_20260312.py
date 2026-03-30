from __future__ import annotations

import csv
import shutil
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


ROOT = Path(
    "/Users/saulrichardson/Dropbox/Inclusionary Zoning/Historical Analysis/raw_data/newspaper_ordinances"
)
ARCHIVE_ROOT = Path(
    "/Users/saulrichardson/projects/newspapers/newspaper-analysis/artifacts/scratch/desmond_newspaper_ordinances_archive_20260311"
)
METADATA_PATH = ROOT / "metadata.csv"


METADATA_PROPOSED_REMOVE = {
    "andrews-county-news__1958-06-29",
    "appleton-post-crescent__1922-10-24",
    "bar-harbor-times__1940-02-29",
    "bennington-banner__1968-02-01",
    "bennington-banner__1970-02-17",
    "bennington-banner__1971-01-28",
    "bennington-banner__1972-02-29",
    "brewster-standard__1960-11-17",
    "daily-independent-journal__1951-03-28",
    "doylestown-intelligencer__1969-09-15",
    "dunkirk-evening-observer__1946-08-02",
    "fitchburg-sentinel__1965-05-10",
    "hammond-lake-county-times__1931-07-08",
    "holland-evening-sentinel__1960-07-01",
    "hurst-mid-cities-news-texan__1969-11-02",
    "lebanon-daily-news__1974-04-26",
    "lowell-sun__1966-06-17",
    "madison-wisconsin-state-journal__1949-12-27",
    "north-adams-transcript__1967-02-04",
    "north-hills-news-record__1961-12-28",
    "perth-amboy-evening-news__1923-06-28",
    "potsdam-courier-and-freeman__1961-09-07",
    "provo-daily-herald__1959-08-23",
    "racine-journal-times__1961-06-23",
    "red-bank-register__1957-03-14",
    "sarasota-herald-tribune__1966-03-25",
    "sturgeon-bay-door-county-advocate__1940-01-12",
    "the-berkshire-eagle__1967-01-11",
    "the-times-herald-record__1967-06-07",
}


# These were not flagged in metadata but the text itself presents the ordinance
# as proposed or as a hearing/proposal publication. This set is intentionally
# conservative and excludes enacted ordinances that merely mention proposed text
# in their recitals.
MANUAL_PROPOSAL_REMOVE = {
    "doylestown-daily-intelligencer__1975-12-08",
    "edwardsville-intelligencer__1963-03-19",
    "fond-du-lac-reporter__1972-02-25",
    "hattiesburg-american__1941-07-14",
    "monessen-daily-independent__1928-06-12",
    "naugatuck-daily-news__1958-05-26",
    "newport-daily-news__1972-01-14",
    "spirit-lake-beacon__1971-06-24",
    "the-crescent-news__1956-07-10",
}


DUPLICATE_TO_CANONICAL = {
    "south-amboy-citizen__1968-02-29": "south-amboy-citizen__1968-03-28",
    "madison-capital-times__1922-12-21": "madison-capital-times__1922-12-14",
    "la-crosse-tribune-and-leader-press__1938-08-19": "la-crosse-tribune-and-leader-press__1938-08-27",
    "kokomo-tribune__1948-12-27": "kokomo-tribune__1948-12-20",
    "biloxi-daily-herald__1940-08-28": "biloxi-daily-herald__1940-09-04",
    "lawrence-journal-world__1926-10-28": "lawrence-daily-journal-world__1926-10-28",
    "pampa-daily-news__1937-11-16": "pampa-daily-news__1937-11-23",
    "east-liverpool-review__1967-07-07": "east-liverpool-review__1967-06-30",
    "new-philadelphia-daily-times__1951-06-07": "new-philadelphia-daily-times__1951-05-31",
    "marysville-appeal-democrat__1996-04-19": "marysville-yuba-city-appeal-democrat__1996-04-19",
    "coshocton-tribune__1962-03-20": "coshocton-tribune__1962-03-27",
    "austin-daily-herald__1976-02-03": "austin-daily-herald__1976-02-10",
}


@dataclass(frozen=True)
class RemovalDecision:
    issue_id: str
    category: str
    reason: str
    canonical_issue_id: str = ""


def move_relpath(relpath: str, src_root: Path, dst_root: Path) -> str:
    if not relpath or pd.isna(relpath):
        return ""
    rel = Path(str(relpath))
    src = src_root / rel
    dst = dst_root / rel
    if not src.exists():
        return str(rel).replace("\\", "/")
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        raise FileExistsError(f"Refusing to overwrite archive path: {dst}")
    shutil.move(str(src), str(dst))
    return str(rel).replace("\\", "/")


def main() -> None:
    df = pd.read_csv(METADATA_PATH)
    by_issue = {row.issue_id: row for row in df.itertuples(index=False)}

    decisions: dict[str, RemovalDecision] = {}
    for issue_id in sorted(METADATA_PROPOSED_REMOVE):
        decisions[issue_id] = RemovalDecision(
            issue_id=issue_id,
            category="proposal",
            reason="Removed because the ordinance is explicitly marked proposed in metadata.",
        )
    for issue_id in sorted(MANUAL_PROPOSAL_REMOVE):
        decisions[issue_id] = RemovalDecision(
            issue_id=issue_id,
            category="proposal",
            reason="Removed because the parsed ordinance text presents the ordinance as proposed or as a hearing/proposal publication.",
        )
    for dup_issue, canonical_issue in DUPLICATE_TO_CANONICAL.items():
        decisions[dup_issue] = RemovalDecision(
            issue_id=dup_issue,
            category="duplicate",
            reason="Removed as a near-time same-municipality duplicate after manual review; kept the cleaner or fuller canonical printing.",
            canonical_issue_id=canonical_issue,
        )

    missing = sorted(issue_id for issue_id in decisions if issue_id not in by_issue)
    if missing:
        raise RuntimeError(f"These reviewed issue_ids were not found in active metadata: {missing}")

    archive_run = ARCHIVE_ROOT / "policy_pass_20260312"
    archive_run.mkdir(parents=True, exist_ok=True)

    metadata_backup = archive_run / "metadata_before_policy_pass_20260312.csv"
    if not metadata_backup.exists():
        shutil.copy2(METADATA_PATH, metadata_backup)

    removed_rows = []
    for issue_id, decision in decisions.items():
        row = by_issue[issue_id]
        row_dict = row._asdict()
        row_dict["removal_category"] = decision.category
        row_dict["removal_reason"] = decision.reason
        row_dict["kept_canonical_issue_id"] = decision.canonical_issue_id

        row_dict["archived_parsed_relpath"] = move_relpath(
            row_dict.get("parsed_relpath", ""),
            ROOT,
            archive_run / "parsed",
        )
        row_dict["archived_transcript_relpath"] = move_relpath(
            row_dict.get("transcript_relpath", ""),
            ROOT,
            archive_run / "transcript",
        )
        row_dict["archived_raw_dir_relpath"] = move_relpath(
            row_dict.get("raw_dir_relpath", ""),
            ROOT,
            archive_run / "raw",
        )
        removed_rows.append(row_dict)

    active_df = df[~df["issue_id"].isin(decisions)].copy()
    active_df.to_csv(METADATA_PATH, index=False)

    removed_path = archive_run / "removed_rows_policy_pass_20260312.csv"
    pd.DataFrame(removed_rows).to_csv(removed_path, index=False)

    summary_path = archive_run / "summary_policy_pass_20260312.txt"
    proposal_count = sum(1 for d in decisions.values() if d.category == "proposal")
    duplicate_count = sum(1 for d in decisions.values() if d.category == "duplicate")
    with summary_path.open("w", encoding="utf-8") as fh:
        fh.write(f"active_rows_after={len(active_df)}\n")
        fh.write(f"removed_total={len(decisions)}\n")
        fh.write(f"removed_proposal={proposal_count}\n")
        fh.write(f"removed_duplicate={duplicate_count}\n")


if __name__ == "__main__":
    main()
