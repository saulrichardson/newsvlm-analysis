#!/usr/bin/env python3
"""
Relabel an existing issue-artifact packet using strict V5 issue-level predictions.

This script updates:
1) Folder placement under classified_pages/<v5_page_class>/<issue_id>
2) metadata_core/issues.csv
3) metadata_core/pages.csv
4) per-issue issue_manifest.json and per-page classification.json (V5-only labels)
5) v2 parsed markdown filenames and metadata CSVs

Semantic source-of-truth is the provided issue-level prediction CSV with:
  - issue_id
  - page_class
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime
from pathlib import Path

import pandas as pd


VALID_V5_PAGE_CLASSES = {
    "zoning_ordinance_comprehensive",
    "zoning_ordinance_noncomprehensive",
    "zoning_amendment_or_rezoning",
    "zoning_legal_notice",
    "building_code_or_other_law",
    "zoning_narrative_nonverbatim",
    "non_zoning",
    "uncertain",
}


def _norm(x: object) -> str:
    return str(x or "").strip()


def _read_predictions(pred_csv: Path) -> dict[str, str]:
    df = pd.read_csv(pred_csv)
    required = {"issue_id", "page_class"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"prediction csv missing required columns: {sorted(missing)}")

    df = df[["issue_id", "page_class"]].copy()
    df["issue_id"] = df["issue_id"].astype(str).str.strip()
    df["page_class"] = df["page_class"].astype(str).str.strip()
    df = df[df["issue_id"] != ""].copy()

    bad = sorted(
        set(df.loc[~df["page_class"].isin(VALID_V5_PAGE_CLASSES), "page_class"].tolist())
    )
    if bad:
        raise ValueError(f"prediction csv includes unknown v5 page_class values: {bad}")

    # Fail on contradictory duplicate predictions for same issue.
    dup = (
        df.groupby("issue_id", dropna=False)["page_class"]
        .nunique(dropna=False)
        .reset_index(name="n_page_class")
    )
    contradictory = dup[dup["n_page_class"] > 1]
    if not contradictory.empty:
        sample = contradictory.head(20)["issue_id"].tolist()
        raise ValueError(
            f"contradictory predictions for {len(contradictory)} issue_ids; sample={sample}"
        )

    out = (
        df.drop_duplicates(subset=["issue_id"], keep="first")
        .set_index("issue_id")["page_class"]
        .to_dict()
    )
    return {str(k): str(v) for k, v in out.items()}


def _collect_issue_dirs(classified_root: Path) -> list[tuple[str, str, Path]]:
    rows: list[tuple[str, str, Path]] = []
    for label_dir in sorted([p for p in classified_root.iterdir() if p.is_dir()]):
        label = label_dir.name
        for issue_dir in sorted([p for p in label_dir.iterdir() if p.is_dir()]):
            rows.append((label, issue_dir.name, issue_dir))
    return rows


def _move_issue_dirs(
    classified_root: Path,
    pred_map: dict[str, str],
    dry_run: bool,
    missing_policy: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = _collect_issue_dirs(classified_root)
    missing_issue_ids = [issue_id for _, issue_id, _ in rows if issue_id not in pred_map]
    if missing_issue_ids and missing_policy == "error":
        sample = missing_issue_ids[:30]
        raise RuntimeError(
            f"missing predictions for {len(missing_issue_ids)} issue ids; "
            f"sample={sample}. "
            "Use --missing-policy uncertain to place unresolved issues in uncertain."
        )

    moves: list[dict[str, object]] = []
    for old_label, issue_id, issue_dir in rows:
        new_label = pred_map.get(issue_id, "uncertain")
        src = issue_dir
        dst = classified_root / new_label / issue_id
        needs_move = old_label != new_label
        status = "unchanged"
        if needs_move:
            status = "moved"
            if not dry_run:
                dst.parent.mkdir(parents=True, exist_ok=True)
                if dst.exists():
                    raise RuntimeError(f"destination already exists: {dst}")
                src.rename(dst)
        moves.append(
            {
                "issue_id": issue_id,
                "old_label": old_label,
                "new_label": new_label,
                "moved": 1 if needs_move else 0,
                "status": status,
                "old_path": str(src),
                "new_path": str(dst if needs_move else src),
                "prediction_missing_fallback_uncertain": 1 if issue_id not in pred_map else 0,
            }
        )

    move_df = pd.DataFrame(moves)
    if move_df.empty:
        count_df = pd.DataFrame(columns=["label", "issue_count"])
    else:
        count_df = (
            move_df.groupby("new_label", dropna=False)["issue_id"]
            .count()
            .reset_index(name="issue_count")
            .rename(columns={"new_label": "label"})
            .sort_values(["issue_count", "label"], ascending=[False, True])
            .reset_index(drop=True)
        )
    return move_df, count_df


def _update_metadata_core(packet_dir: Path, move_df: pd.DataFrame) -> None:
    metadata_dir = packet_dir / "metadata_core"
    issues_csv = metadata_dir / "issues.csv"
    pages_csv = metadata_dir / "pages.csv"
    if not issues_csv.is_file() or not pages_csv.is_file():
        raise FileNotFoundError("metadata_core/issues.csv or metadata_core/pages.csv missing")

    issues = pd.read_csv(issues_csv)
    pages = pd.read_csv(pages_csv)
    req_issue = {"issue_id", "primary_label", "issue_rel_dir"}
    req_page = {
        "issue_id",
        "page_id",
        "primary_label",
        "label",
        "page_rel_dir",
        "transcript_rel_path",
        "png_rel_path",
        "layout_rel_path",
        "classification_rel_path",
    }
    miss_i = req_issue - set(issues.columns)
    miss_p = req_page - set(pages.columns)
    if miss_i:
        raise ValueError(f"issues.csv missing columns: {sorted(miss_i)}")
    if miss_p:
        raise ValueError(f"pages.csv missing columns: {sorted(miss_p)}")

    label_map = move_df.set_index("issue_id")["new_label"].to_dict()

    issues["issue_id"] = issues["issue_id"].astype(str).str.strip()
    issues["primary_label"] = issues["issue_id"].map(label_map).fillna("uncertain")
    if "label_set" in issues.columns:
        issues["label_set"] = issues["primary_label"]
    issues["issue_rel_dir"] = (
        "classified_pages/"
        + issues["primary_label"].astype(str)
        + "/"
        + issues["issue_id"].astype(str)
    )

    pages["issue_id"] = pages["issue_id"].astype(str).str.strip()
    pages["page_id"] = pages["page_id"].astype(str).str.strip()
    pages["primary_label"] = pages["issue_id"].map(label_map).fillna("uncertain")
    pages["label"] = pages["primary_label"]
    base = (
        "classified_pages/"
        + pages["primary_label"].astype(str)
        + "/"
        + pages["issue_id"].astype(str)
        + "/pages/"
        + pages["page_id"].astype(str)
    )
    pages["page_rel_dir"] = base
    pages["transcript_rel_path"] = base + "/transcript.txt"
    pages["png_rel_path"] = base + "/original.png"
    pages["layout_rel_path"] = base + "/layout.json"
    pages["classification_rel_path"] = base + "/classification.json"

    issues.to_csv(issues_csv, index=False, quoting=csv.QUOTE_MINIMAL)
    pages.to_csv(pages_csv, index=False, quoting=csv.QUOTE_MINIMAL)


def _refresh_issue_internal_files(packet_dir: Path) -> None:
    classified_root = packet_dir / "classified_pages"
    for label_dir in sorted([p for p in classified_root.iterdir() if p.is_dir()]):
        v5_label = _norm(label_dir.name)
        if v5_label not in VALID_V5_PAGE_CLASSES:
            issue_dirs = [p for p in label_dir.iterdir() if p.is_dir()]
            if issue_dirs:
                raise RuntimeError(
                    f"unexpected non-v5 label folder with issue dirs found: {label_dir}"
                )
            # Keep idempotent cleanup local and explicit.
            for child in label_dir.iterdir():
                if child.is_file() or child.is_symlink():
                    child.unlink()
                elif child.is_dir():
                    raise RuntimeError(
                        f"unexpected nested directory in non-v5 label folder: {child}"
                    )
            label_dir.rmdir()
            continue
        for issue_dir in sorted([p for p in label_dir.iterdir() if p.is_dir()]):
            pages_csv = issue_dir / "pages.csv"
            page_classes: list[str] = []
            if pages_csv.is_file():
                dfp = pd.read_csv(pages_csv)
                if "page_class" not in dfp.columns:
                    raise RuntimeError(f"pages.csv missing page_class: {pages_csv}")
                dfp["page_class"] = (
                    dfp["page_class"]
                    .astype(str)
                    .str.strip()
                    .where(
                        lambda s: s.isin(VALID_V5_PAGE_CLASSES),
                        "uncertain",
                    )
                )
                if "label" in dfp.columns:
                    dfp["label"] = dfp["page_class"]
                page_classes = dfp["page_class"].tolist()
                dfp.to_csv(pages_csv, index=False, quoting=csv.QUOTE_MINIMAL)

            issue_manifest_path = issue_dir / "issue_manifest.json"
            manifest: dict[str, object] = {}
            if issue_manifest_path.is_file():
                try:
                    manifest = json.loads(issue_manifest_path.read_text(encoding="utf-8"))
                    if not isinstance(manifest, dict):
                        manifest = {}
                except Exception:
                    manifest = {}

            if not page_classes:
                page_classes = [v5_label]
            counts = (
                pd.Series(page_classes, dtype="string")
                .value_counts(dropna=False)
                .sort_index()
                .to_dict()
            )

            manifest["issue_id"] = _norm(manifest.get("issue_id")) or issue_dir.name
            manifest["primary_label"] = v5_label
            manifest["label_set"] = sorted(set(page_classes))
            manifest["label_page_counts"] = {str(k): int(v) for k, v in counts.items()}
            manifest["page_count"] = int(sum(int(v) for v in counts.values()))
            manifest["label_schema"] = "v5_page_class"
            issue_manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

            pages_root = issue_dir / "pages"
            for page_dir in sorted([p for p in pages_root.iterdir() if p.is_dir()]) if pages_root.is_dir() else []:
                cls_path = page_dir / "classification.json"
                if not cls_path.is_file():
                    continue
                try:
                    cls_obj = json.loads(cls_path.read_text(encoding="utf-8"))
                except Exception:
                    continue
                if not isinstance(cls_obj, dict):
                    continue
                page_class = _norm(cls_obj.get("page_class")).lower()
                if page_class not in VALID_V5_PAGE_CLASSES:
                    page_class = "uncertain"
                cls_obj["page_class"] = page_class
                cls_obj["label_schema"] = "v5_page_class"
                cls_obj.pop("legal_object_type", None)
                cls_path.write_text(
                    json.dumps(cls_obj, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )


ISSUE_MD_RE = re.compile(r"^(?P<label>.+?)__(?P<issue_id>.+)\.md$")


def _build_parsed_md_index(v2_parsed_dir: Path) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = {}
    for p in v2_parsed_dir.iterdir():
        if not p.is_file() or p.suffix.lower() != ".md":
            continue
        m = ISSUE_MD_RE.match(p.name)
        if not m:
            continue
        issue_id = _norm(m.group("issue_id"))
        if not issue_id:
            continue
        index.setdefault(issue_id, []).append(p)
    return index


def _rename_or_create_parsed_markdown(
    v2_parsed_dir: Path,
    issue_id: str,
    new_label: str,
    md_index: dict[str, list[Path]],
) -> Path:
    target = v2_parsed_dir / f"{new_label}__{issue_id}.md"
    if target.exists():
        md_index[issue_id] = [target]
        return target

    matches = md_index.get(issue_id, [])
    if len(matches) == 1:
        src = matches[0]
        src.rename(target)
        md_index[issue_id] = [target]
        return target
    if len(matches) > 1:
        raise RuntimeError(f"multiple parsed markdown files for issue_id={issue_id}: {[str(m) for m in matches]}")

    target.write_text("", encoding="utf-8")
    md_index[issue_id] = [target]
    return target


def _rewrite_parsed_markdown_header(
    parsed_md: Path,
    *,
    label: str,
    raw_newspaper_path: str,
) -> None:
    text = parsed_md.read_text(encoding="utf-8") if parsed_md.exists() else ""
    if not text:
        parsed_md.write_text(
            "\n".join(
                [
                    "- label: " + label,
                    "- raw_newspaper_path: " + raw_newspaper_path,
                    "",
                ]
            ),
            encoding="utf-8",
        )
        return

    out = text
    if re.search(r"(?m)^- label:\s*.*$", out):
        out = re.sub(r"(?m)^- label:\s*.*$", f"- label: {label}", out, count=1)
    else:
        out = f"- label: {label}\n" + out

    if re.search(r"(?m)^- raw_newspaper_path:\s*.*$", out):
        out = re.sub(
            r"(?m)^- raw_newspaper_path:\s*.*$",
            f"- raw_newspaper_path: {raw_newspaper_path}",
            out,
            count=1,
        )
    else:
        out = f"- raw_newspaper_path: {raw_newspaper_path}\n" + out

    if out != text:
        parsed_md.write_text(out, encoding="utf-8")


def _refresh_v2_metadata(packet_dir: Path) -> None:
    metadata_dir = packet_dir / "metadata_core"
    v2_dir = packet_dir / "v2"
    issues_csv = metadata_dir / "issues.csv"
    papers_csv = metadata_dir / "newspapers.csv"
    if not issues_csv.is_file() or not papers_csv.is_file():
        raise FileNotFoundError("metadata_core/issues.csv or metadata_core/newspapers.csv missing")
    if not v2_dir.is_dir():
        return

    parsed_dir = v2_dir / "parsed_data"
    raw_dir = v2_dir / "raw_issue_text"
    parsed_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    md_index = _build_parsed_md_index(parsed_dir)

    issues = pd.read_csv(issues_csv).copy()
    papers = pd.read_csv(papers_csv).copy()
    req_i = {"issue_id", "newspaper_slug", "issue_date", "primary_label", "issue_rel_dir"}
    req_p = {"newspaper_slug", "city_id"}
    miss_i = req_i - set(issues.columns)
    miss_p = req_p - set(papers.columns)
    if miss_i:
        raise ValueError(f"issues.csv missing columns for v2 refresh: {sorted(miss_i)}")
    if miss_p:
        raise ValueError(f"newspapers.csv missing columns for v2 refresh: {sorted(miss_p)}")

    issues["issue_id"] = issues["issue_id"].astype(str).str.strip()
    issues["newspaper_slug"] = issues["newspaper_slug"].astype(str).str.strip()
    issues["issue_date"] = issues["issue_date"].astype(str).str.strip()
    issues["primary_label"] = issues["primary_label"].astype(str).str.strip()
    issues["issue_rel_dir"] = issues["issue_rel_dir"].astype(str).str.strip()

    papers["newspaper_slug"] = papers["newspaper_slug"].astype(str).str.strip()
    city_map = papers.set_index("newspaper_slug")["city_id"].to_dict()

    rows: list[dict[str, object]] = []
    rows_with_date: list[dict[str, object]] = []
    for r in issues.itertuples(index=False):
        issue_id = _norm(getattr(r, "issue_id"))
        label = _norm(getattr(r, "primary_label"))
        issue_date = _norm(getattr(r, "issue_date"))
        slug = _norm(getattr(r, "newspaper_slug"))
        issue_rel_dir = _norm(getattr(r, "issue_rel_dir"))

        parsed_md = _rename_or_create_parsed_markdown(parsed_dir, issue_id, label, md_index)
        raw_txt = raw_dir / f"{issue_id}.txt"
        _rewrite_parsed_markdown_header(
            parsed_md,
            label=label,
            raw_newspaper_path=str(raw_txt.resolve()),
        )

        rows.append(
            {
                "town_id": city_map.get(slug, ""),
                "date": issue_date,
                "parsed_markdown_path": str(parsed_md.resolve()),
                "raw_newspaper_path": str(raw_txt.resolve()),
            }
        )
        rows_with_date.append(
            {
                "town_id": city_map.get(slug, ""),
                "date": issue_date,
                "parsed_markdown_path": str(parsed_md.resolve()),
                "raw_newspaper_path": str(raw_txt.resolve()),
            }
        )

    pd.DataFrame(rows).to_csv(v2_dir / "metadata.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    pd.DataFrame(rows_with_date).to_csv(
        v2_dir / "metadata_town_date_paths.csv", index=False, quoting=csv.QUOTE_MINIMAL
    )


def _refresh_newspaper_level_metadata(packet_dir: Path) -> None:
    metadata_dir = packet_dir / "metadata_core"
    issues_csv = metadata_dir / "issues.csv"
    pages_csv = metadata_dir / "pages.csv"
    papers_csv = metadata_dir / "newspapers.csv"
    papers_enriched_csv = metadata_dir / "newspapers_enriched.csv"
    if not issues_csv.is_file() or not pages_csv.is_file() or not papers_csv.is_file():
        return

    issues = pd.read_csv(issues_csv).copy()
    pages = pd.read_csv(pages_csv).copy()
    papers = pd.read_csv(papers_csv).copy()
    for c in ("newspaper_slug", "issue_date", "primary_label"):
        if c not in issues.columns:
            raise ValueError(f"issues.csv missing column for newspaper refresh: {c}")
    if "newspaper_slug" not in papers.columns:
        raise ValueError("newspapers.csv missing newspaper_slug")

    issues["newspaper_slug"] = issues["newspaper_slug"].astype(str).str.strip()
    issues["issue_date"] = issues["issue_date"].astype(str).str.strip()
    issues["primary_label"] = (
        issues["primary_label"]
        .astype(str)
        .str.strip()
        .where(lambda s: s.isin(VALID_V5_PAGE_CLASSES), "uncertain")
    )
    if "newspaper_slug" not in pages.columns:
        if "issue_id" not in pages.columns:
            raise ValueError(
                "pages.csv must include either newspaper_slug or issue_id for newspaper refresh"
            )
        pages["issue_id"] = pages["issue_id"].astype(str).str.strip()
        pages = pages.merge(
            issues[["issue_id", "newspaper_slug"]].drop_duplicates(),
            on="issue_id",
            how="left",
        )
    pages["newspaper_slug"] = pages["newspaper_slug"].astype(str).str.strip()
    papers["newspaper_slug"] = papers["newspaper_slug"].astype(str).str.strip()

    issue_agg = (
        issues.groupby("newspaper_slug", dropna=False)
        .agg(
            issue_count=("issue_id", "count"),
            first_issue_date=("issue_date", "min"),
            last_issue_date=("issue_date", "max"),
        )
        .reset_index()
    )
    label_mode = (
        issues.groupby(["newspaper_slug", "primary_label"], dropna=False)["issue_id"]
        .count()
        .reset_index(name="n")
        .sort_values(
            ["newspaper_slug", "n", "primary_label"],
            ascending=[True, False, True],
            kind="stable",
        )
        .groupby("newspaper_slug", as_index=False)
        .first()[["newspaper_slug", "primary_label"]]
        .rename(columns={"primary_label": "most_common_primary_label"})
    )
    page_agg = (
        pages.groupby("newspaper_slug", dropna=False)["page_id"]
        .count()
        .reset_index(name="page_count")
    )
    merged = (
        papers.drop(columns=[c for c in ["issue_count", "page_count", "first_issue_date", "last_issue_date", "most_common_primary_label"] if c in papers.columns])
        .merge(issue_agg, on="newspaper_slug", how="left")
        .merge(page_agg, on="newspaper_slug", how="left")
        .merge(label_mode, on="newspaper_slug", how="left")
    )
    merged["issue_count"] = merged["issue_count"].fillna(0).astype(int)
    merged["page_count"] = merged["page_count"].fillna(0).astype(int)
    merged["most_common_primary_label"] = merged["most_common_primary_label"].fillna("uncertain")
    merged.to_csv(papers_csv, index=False, quoting=csv.QUOTE_MINIMAL)

    if papers_enriched_csv.is_file():
        enriched = pd.read_csv(papers_enriched_csv).copy()
        if "newspaper_slug" in enriched.columns:
            enriched["newspaper_slug"] = enriched["newspaper_slug"].astype(str).str.strip()
            enriched = enriched.drop(
                columns=[
                    c
                    for c in [
                        "issue_count",
                        "page_count",
                        "first_issue_date",
                        "last_issue_date",
                        "most_common_primary_label",
                    ]
                    if c in enriched.columns
                ]
            ).merge(
                merged[
                    [
                        "newspaper_slug",
                        "issue_count",
                        "page_count",
                        "first_issue_date",
                        "last_issue_date",
                        "most_common_primary_label",
                    ]
                ],
                on="newspaper_slug",
                how="left",
            )
            enriched["issue_count"] = enriched["issue_count"].fillna(0).astype(int)
            enriched["page_count"] = enriched["page_count"].fillna(0).astype(int)
            enriched["most_common_primary_label"] = (
                enriched["most_common_primary_label"].fillna("uncertain")
            )
            enriched.to_csv(papers_enriched_csv, index=False, quoting=csv.QUOTE_MINIMAL)


def main() -> None:
    ap = argparse.ArgumentParser(description="Relabel Dropbox issue packet from latest issue-level predictions.")
    ap.add_argument("--packet-dir", required=True, help="Path to packet root containing classified_pages and metadata_core.")
    ap.add_argument(
        "--predictions-csv",
        required=True,
        help="Issue-level predictions CSV with columns: issue_id,page_class.",
    )
    ap.add_argument("--dry-run", action="store_true", help="Plan only; no file moves or metadata writes.")
    ap.add_argument(
        "--missing-policy",
        choices=["error", "uncertain"],
        default="error",
        help="How to handle issue ids present in packet but missing in predictions.",
    )
    args = ap.parse_args()

    packet_dir = Path(args.packet_dir).expanduser().resolve()
    pred_csv = Path(args.predictions_csv).expanduser().resolve()
    if not packet_dir.is_dir():
        raise SystemExit(f"packet dir not found: {packet_dir}")
    if not pred_csv.is_file():
        raise SystemExit(f"predictions csv not found: {pred_csv}")

    classified_root = packet_dir / "classified_pages"
    if not classified_root.is_dir():
        raise SystemExit(f"classified_pages dir not found: {classified_root}")

    pred_map = _read_predictions(pred_csv)
    move_df, count_df = _move_issue_dirs(
        classified_root,
        pred_map,
        dry_run=args.dry_run,
        missing_policy=str(args.missing_policy),
    )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_dir = packet_dir / "metadata_core"
    summary_dir.mkdir(parents=True, exist_ok=True)
    move_csv = summary_dir / f"relabel_moves_{stamp}.csv"
    count_csv = summary_dir / f"relabel_counts_{stamp}.csv"
    move_df.to_csv(move_csv, index=False, quoting=csv.QUOTE_MINIMAL)
    count_df.to_csv(count_csv, index=False, quoting=csv.QUOTE_MINIMAL)

    if args.dry_run:
        print(f"[dry-run] wrote move plan: {move_csv}")
        print(f"[dry-run] wrote counts: {count_csv}")
        print(f"[dry-run] total issues: {len(move_df)} moved: {int(move_df['moved'].sum())}")
        return

    _update_metadata_core(packet_dir, move_df)
    _refresh_issue_internal_files(packet_dir)
    _refresh_newspaper_level_metadata(packet_dir)
    _refresh_v2_metadata(packet_dir)

    print(f"Relabel complete. total_issues={len(move_df)} moved={int(move_df['moved'].sum())}")
    print(f"Move log: {move_csv}")
    print(f"Count log: {count_csv}")


if __name__ == "__main__":
    main()
