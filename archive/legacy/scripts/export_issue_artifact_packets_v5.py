#!/usr/bin/env python3
"""
Build issue-centric artifact packets from page-level v5 classifications.

For each issue, writes one folder with per-page artifacts:
  - transcript.txt (VLM transcript if available, else layout raw_text fallback)
  - original.png (symlink when local PNG exists)
  - original_png_source.txt (records source path even if PNG is missing locally)
  - layout.json (symlink to layout artifact)
  - classification.json (page classification snapshot)

Output hierarchy:
  - classified_pages/<primary_issue_v5_page_class>/<issue_id>/
Each issue folder contains all pages for that issue.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


def _norm(x: Any) -> str:
    return str(x or "").strip()


def _slug(s: str) -> str:
    t = _norm(s).lower()
    t = re.sub(r"[^a-z0-9._-]+", "-", t)
    t = re.sub(r"-{2,}", "-", t).strip("-")
    return t or "na"


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(float(x))
    except Exception:
        return default


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Export issue-level page packets (transcript + png + layout) from v5 page labels.")
    ap.add_argument(
        "--page-csv",
        default="/Users/saulrichardson/projects/newspapers/newspaper-analysis/reports/tmp/latest_v5_survey_20260225/combined_page_classification_outputs_dedup_pageid.csv",
        help="Input page-level classification CSV.",
    )
    ap.add_argument(
        "--output-dir",
        default="/Users/saulrichardson/projects/newspapers/newspaper-analysis/reports/runs/page_artifact_packets_v5_full",
        help="Output directory.",
    )
    ap.add_argument(
        "--layout-root",
        default="/Users/saulrichardson/projects/newspapers/old-ocr/newspaper-parsing-local/data/unique_outputs_dedup",
        help="Directory containing per-page layout JSON files named <page_id>.json.",
    )
    ap.add_argument(
        "--png-root",
        default="/Users/saulrichardson/projects/newspapers/old-ocr/newspaper-parsing-local/data/unique_png",
        help="Directory containing per-page PNG files named <page_id>.png.",
    )
    ap.add_argument(
        "--include-labels",
        default="",
        help="Optional comma-separated V5 page_class labels to include. Empty means all labels.",
    )
    ap.add_argument(
        "--max-pages",
        type=int,
        default=0,
        help="Optional limit for smoke tests (0 = all pages).",
    )
    return ap.parse_args()


def _resolve_layout_path(page_id: str, layout_root: Path) -> Path:
    return layout_root / f"{page_id}.json"


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _extract_transcript_from_vlm(vlm_obj: dict[str, Any] | None) -> str:
    if not isinstance(vlm_obj, dict):
        return ""
    boxes = vlm_obj.get("boxes")
    if not isinstance(boxes, list):
        return ""
    parts: list[tuple[int, str]] = []
    for b in boxes:
        if not isinstance(b, dict):
            continue
        box_id = _safe_int(b.get("id"), default=0)
        txt = _norm(b.get("transcript"))
        if txt:
            parts.append((box_id, txt))
    if not parts:
        return ""
    parts.sort(key=lambda x: x[0])
    return "\n\n".join(t for _, t in parts).strip()


def _extract_transcript_from_layout(layout_obj: dict[str, Any] | None) -> str:
    if not isinstance(layout_obj, dict):
        return ""
    bboxes = layout_obj.get("bboxes")
    if not isinstance(bboxes, list):
        return ""
    parts: list[tuple[int, str]] = []
    for b in bboxes:
        if not isinstance(b, dict):
            continue
        box_id = _safe_int(b.get("id"), default=0)
        txt = _norm(b.get("raw_text"))
        if txt:
            parts.append((box_id, txt))
    if not parts:
        return ""
    parts.sort(key=lambda x: x[0])
    return "\n\n".join(t for _, t in parts).strip()


def _resolve_png_local_path(page_id: str, png_root: Path) -> Path | None:
    p = png_root / f"{page_id}.png"
    return p if p.is_file() else None


def _resolve_png_source_hint(vlm_obj: dict[str, Any] | None, layout_obj: dict[str, Any] | None) -> str:
    if isinstance(vlm_obj, dict):
        p = _norm(vlm_obj.get("png_path"))
        if p:
            return p
    if isinstance(layout_obj, dict):
        p = _norm(layout_obj.get("scan_url"))
        if p:
            return p
    return ""


def _safe_symlink(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src)


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

PRIMARY_LABEL_PRIORITY = [
    "zoning_ordinance_comprehensive",
    "zoning_ordinance_noncomprehensive",
    "zoning_amendment_or_rezoning",
    "zoning_legal_notice",
    "building_code_or_other_law",
    "zoning_narrative_nonverbatim",
    "uncertain",
    "non_zoning",
]


def _primary_issue_label(label_counts: dict[str, int]) -> str:
    if not label_counts:
        return "unknown"
    pri = {lab: i for i, lab in enumerate(PRIMARY_LABEL_PRIORITY)}
    ranked = sorted(
        label_counts.items(),
        key=lambda kv: (-kv[1], pri.get(kv[0], 999), kv[0]),
    )
    return ranked[0][0]


def main() -> None:
    args = _parse_args()
    page_csv = Path(args.page_csv).expanduser().resolve()
    out_root = Path(args.output_dir).expanduser().resolve()
    layout_root = Path(args.layout_root).expanduser().resolve()
    png_root = Path(args.png_root).expanduser().resolve()

    if not page_csv.is_file():
        raise SystemExit(f"missing page csv: {page_csv}")

    out_root.mkdir(parents=True, exist_ok=True)
    by_label_root = out_root / "classified_pages"
    summary_root = out_root / "summary"
    for p in [by_label_root, summary_root]:
        p.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(page_csv)
    required = {"page_id", "issue_id", "issue_date", "newspaper_slug", "page_class", "vlm_path"}
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise SystemExit(f"input missing required columns: {missing_cols}")

    include_labels = {_norm(x) for x in _norm(args.include_labels).split(",") if _norm(x)}
    if include_labels:
        df = df[df["page_class"].astype(str).isin(include_labels)].copy()
    bad_page_class = sorted(
        set(
            df.loc[
                ~df["page_class"].astype(str).isin(VALID_V5_PAGE_CLASSES),
                "page_class",
            ].astype(str).tolist()
        )
    )
    if bad_page_class:
        raise SystemExit(f"input contains non-v5 page_class values: {bad_page_class}")
    if args.max_pages > 0:
        df = df.head(args.max_pages).copy()
    df = df.reset_index(drop=True)

    vlm_cache: dict[str, dict[str, Any] | None] = {}
    layout_cache: dict[str, dict[str, Any] | None] = {}

    issue_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    export_index_rows: list[dict[str, Any]] = []
    issue_summary_rows: list[dict[str, Any]] = []

    for _, r in df.iterrows():
        page_id = _norm(r.get("page_id"))
        issue_id = _norm(r.get("issue_id"))
        if not page_id or not issue_id:
            continue
        issue_rows[issue_id].append(dict(r))

    issue_count = 0
    page_count = 0
    transcript_from_vlm = 0
    transcript_from_layout = 0
    transcript_empty = 0
    png_local_ok = 0
    png_missing = 0
    layout_ok = 0
    layout_missing = 0

    for issue_id, rows in sorted(issue_rows.items(), key=lambda kv: kv[0]):
        issue_count += 1
        rows_sorted = sorted(rows, key=lambda x: (_norm(x.get("issue_date")), _safe_int(x.get("page_num")), _norm(x.get("page_id"))))
        label_counts: dict[str, int] = defaultdict(int)
        for x in rows_sorted:
            lab = _norm(x.get("page_class")) or "uncertain"
            label_counts[lab] += 1
        labels = sorted(label_counts.keys())
        primary_label = _primary_issue_label(dict(label_counts))

        issue_slug = _slug(issue_id)
        issue_dir = by_label_root / _slug(primary_label) / issue_slug
        pages_dir = issue_dir / "pages"
        pages_dir.mkdir(parents=True, exist_ok=True)

        issue_manifest = {
            "issue_id": issue_id,
            "issue_slug": issue_slug,
            "issue_date": _norm(rows_sorted[0].get("issue_date")),
            "newspaper_slug": _norm(rows_sorted[0].get("newspaper_slug")),
            "primary_label": primary_label,
            "label_set": labels,
            "label_page_counts": dict(sorted(label_counts.items(), key=lambda kv: kv[0])),
            "page_count": len(rows_sorted),
        }
        (issue_dir / "issue_manifest.json").write_text(json.dumps(issue_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        issue_summary_rows.append(
            {
                "issue_id": issue_id,
                "issue_date": _norm(rows_sorted[0].get("issue_date")),
                "newspaper_slug": _norm(rows_sorted[0].get("newspaper_slug")),
                "primary_label": primary_label,
                "label_set": "|".join(labels),
                "page_count": len(rows_sorted),
                "issue_dir": str(issue_dir),
            }
        )

        page_csv_rows: list[dict[str, Any]] = []
        for rec in rows_sorted:
            page_id = _norm(rec.get("page_id"))
            if not page_id:
                continue
            page_count += 1

            page_dir = pages_dir / _slug(page_id)
            page_dir.mkdir(parents=True, exist_ok=True)

            vlm_path_str = _norm(rec.get("vlm_path"))
            vlm_path = Path(vlm_path_str) if vlm_path_str else Path("")
            if vlm_path_str not in vlm_cache:
                vlm_cache[vlm_path_str] = _load_json(vlm_path) if vlm_path_str else None
            vlm_obj = vlm_cache[vlm_path_str]

            layout_path = _resolve_layout_path(page_id, layout_root)
            layout_path_str = str(layout_path)
            if layout_path_str not in layout_cache:
                layout_cache[layout_path_str] = _load_json(layout_path)
            layout_obj = layout_cache[layout_path_str]

            transcript = _extract_transcript_from_vlm(vlm_obj)
            transcript_source = "vlm"
            if not transcript:
                transcript = _extract_transcript_from_layout(layout_obj)
                transcript_source = "layout_raw_text" if transcript else "missing"

            if transcript_source == "vlm":
                transcript_from_vlm += 1
            elif transcript_source == "layout_raw_text":
                transcript_from_layout += 1
            else:
                transcript_empty += 1

            (page_dir / "transcript.txt").write_text(transcript + ("\n" if transcript else ""), encoding="utf-8")

            # classification snapshot for this page
            keep_fields = [
                "page_id",
                "issue_id",
                "issue_date",
                "newspaper_slug",
                "page_num",
                "page_class",
                "zoning_legal_document_form",
                "ordinance_scope_signal",
                "enactment_status_signal",
                "is_fragment",
                "contains_verbatim_zoning_law_language",
                "is_verbatim_legal_text",
                "confidence_0_to_1",
                "notes",
                "evidence_quotes",
                "source_run",
                "vlm_path",
            ]
            cls_obj = {k: rec.get(k) for k in keep_fields if k in rec}
            (page_dir / "classification.json").write_text(json.dumps(cls_obj, indent=2, default=str, sort_keys=True) + "\n", encoding="utf-8")

            # layout artifact
            if layout_path.is_file():
                _safe_symlink(layout_path, page_dir / "layout.json")
                layout_ok += 1
            else:
                layout_missing += 1
                (page_dir / "layout_missing.txt").write_text(f"missing layout: {layout_path}\n", encoding="utf-8")

            # png artifact
            local_png = _resolve_png_local_path(page_id, png_root)
            if local_png is not None:
                _safe_symlink(local_png, page_dir / "original.png")
                png_local_ok += 1
            else:
                png_missing += 1
            src_hint = _resolve_png_source_hint(vlm_obj, layout_obj)
            (page_dir / "original_png_source.txt").write_text((src_hint + "\n") if src_hint else "unknown\n", encoding="utf-8")

            page_csv_rows.append(
                {
                    "page_id": page_id,
                    "label": _norm(rec.get("page_class")),
                    "page_class": _norm(rec.get("page_class")),
                    "enactment_status_signal": _norm(rec.get("enactment_status_signal")),
                    "issue_date": _norm(rec.get("issue_date")),
                    "newspaper_slug": _norm(rec.get("newspaper_slug")),
                    "page_dir": str(page_dir),
                    "transcript_source": transcript_source,
                    "layout_exists": int(layout_path.is_file()),
                    "png_local_exists": int(local_png is not None),
                    "vlm_path": vlm_path_str,
                    "layout_path": str(layout_path),
                    "png_source_hint": src_hint,
                }
            )

            export_index_rows.append(
                {
                    "issue_id": issue_id,
                    "issue_slug": issue_slug,
                    "page_id": page_id,
                    "primary_label": primary_label,
                    "label": _norm(rec.get("page_class")),
                    "page_dir": str(page_dir),
                }
            )

        # issue-level page inventory
        with (issue_dir / "pages.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "page_id",
                    "label",
                    "page_class",
                    "enactment_status_signal",
                    "issue_date",
                    "newspaper_slug",
                    "page_dir",
                    "transcript_source",
                    "layout_exists",
                    "png_local_exists",
                    "vlm_path",
                    "layout_path",
                    "png_source_hint",
                ],
            )
            w.writeheader()
            w.writerows(page_csv_rows)

    # root-level indexes
    idx_df = pd.DataFrame.from_records(export_index_rows)
    issue_summary_df = pd.DataFrame.from_records(issue_summary_rows)
    if not idx_df.empty:
        idx_df.to_csv(summary_root / "page_index.csv", index=False)
        issue_label_counts = (
            issue_summary_df.groupby("primary_label", dropna=False)["issue_id"]
            .count()
            .reset_index(name="issue_count")
            .sort_values(["issue_count", "primary_label"], ascending=[False, True])
        )
        issue_label_counts.to_csv(summary_root / "issue_counts_by_primary_label.csv", index=False)
        page_label_counts = (
            idx_df.groupby("label", dropna=False)["page_id"]
            .count()
            .reset_index(name="page_count")
            .sort_values("page_count", ascending=False)
        )
        page_label_counts.to_csv(summary_root / "page_counts_by_label.csv", index=False)
        issue_summary_df.to_csv(summary_root / "issues.csv", index=False)

    summary = {
        "input_page_csv": str(page_csv),
        "output_dir": str(out_root),
        "issue_count": issue_count,
        "page_count": page_count,
        "transcript_from_vlm": transcript_from_vlm,
        "transcript_from_layout_raw_text": transcript_from_layout,
        "transcript_missing": transcript_empty,
        "layout_ok": layout_ok,
        "layout_missing": layout_missing,
        "png_local_ok": png_local_ok,
        "png_missing_local": png_missing,
        "layout_root": str(layout_root),
        "png_root": str(png_root),
    }
    (summary_root / "export_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# Issue Artifact Packets (strict V5 page_class labels)",
        "",
        f"- issues exported: {issue_count:,}",
        f"- pages exported: {page_count:,}",
        "",
        "## Coverage",
        f"- transcript from VLM json: {transcript_from_vlm:,}",
        f"- transcript from layout raw_text fallback: {transcript_from_layout:,}",
        f"- transcript missing: {transcript_empty:,}",
        f"- layout present: {layout_ok:,}",
        f"- layout missing: {layout_missing:,}",
        f"- png present locally: {png_local_ok:,}",
        f"- png missing locally: {png_missing:,}",
        "",
        "## Structure",
        "- `classified_pages/<primary_label>/<issue_id>/pages/<page_id>/`",
        "- `summary/*`",
        "",
        "Primary issue label is selected deterministically from V5 page_class values in that issue:",
        "highest page-count class wins; ties broken by explicit V5 priority.",
    ]
    (out_root / "README.md").write_text("\n".join(lines).strip() + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"done_issue_packets output_dir={out_root}")


if __name__ == "__main__":
    main()
