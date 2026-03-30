#!/usr/bin/env python3
"""
Build side-by-side issue folders with original page PNGs and transcripts.

Output structure:
  <output-dir>/
    by_newspaper/<newspaper_slug>/<type_label>/<issue_id>/
      metadata.json
      issue_transcript_full.txt
      clean_zoning_text.txt
      page_001__<page_id>.txt
      page_001__<page_id>.png
      ...
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
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


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _type_label(issue_class: str, event_type: str) -> str:
    ic = _norm(issue_class)
    et = _norm(event_type)
    if et == "building_code":
        return "building_code"
    if et in {"full_ordinance", "amendment"}:
        return et
    if et == "proposal":
        return "proposal"
    if ic.startswith("full_code_"):
        if ic in {"full_code_proposed", "full_code_partial_proposed"}:
            return "proposal"
        if ic == "full_code_partial_enacted":
            return "partial_full_ordinance"
        return "full_ordinance"
    if ic.startswith("amendment_"):
        if ic == "amendment_proposed":
            return "proposal"
        return "amendment"
    if ic == "building_code_non_zoning":
        return "building_code"
    if ic == "zoning_notice_procedural":
        return "procedural_notice"
    if ic == "not_verbatim_law_text":
        return "narrative_or_nonverbatim"
    if ic == "non_zoning":
        return "non_zoning"
    return _slug(ic or "other")


def _try_copy_png(
    *,
    page_id: str,
    dest_png: Path,
    cache_dirs: list[Path],
    torch_host: str,
    try_torch: bool,
) -> tuple[bool, str]:
    filename = f"{page_id}.png"

    if dest_png.is_file():
        return True, "existing"

    for root in cache_dirs:
        cand = root / filename
        if cand.is_file():
            dest_png.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(cand, dest_png)
            return True, str(cand)

    # recursive fallback only for cache roots that may contain nested folders
    for root in cache_dirs[1:]:
        if not root.is_dir():
            continue
        try:
            for cand in root.rglob(filename):
                if cand.is_file():
                    dest_png.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(cand, dest_png)
                    return True, str(cand)
        except Exception:
            continue

    if not try_torch:
        return False, "missing"

    remote_unique = f"/scratch/sxr203/newspaper-downloads/dedupe-webp/unique_png/{filename}"
    try:
        rc = subprocess.run(
            ["scp", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", f"{torch_host}:{remote_unique}", str(dest_png)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=40,
        )
        if rc.returncode == 0 and dest_png.is_file():
            return True, remote_unique
    except Exception:
        pass

    try:
        find_cmd = f"find /scratch/sxr203/newspaper-downloads -type f -name {json.dumps(filename)} | head -n 1"
        rs = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", torch_host, find_cmd],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=40,
        )
        remote_path = rs.stdout.strip().splitlines()[0] if rs.returncode == 0 and rs.stdout.strip() else ""
        if remote_path:
            rc2 = subprocess.run(
                ["scp", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", f"{torch_host}:{remote_path}", str(dest_png)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=40,
            )
            if rc2.returncode == 0 and dest_png.is_file():
                return True, remote_path
    except Exception:
        pass

    return False, "missing"


def _read_issue_pages_from_panels(panel_root: Path) -> tuple[dict[str, list[str]], dict[str, str], dict[str, str], dict[str, str]]:
    issue_page_ids: dict[str, list[str]] = {}
    issue_text: dict[str, str] = {}
    issue_date: dict[str, str] = {}
    issue_city: dict[str, str] = {}

    if not panel_root.is_dir():
        return issue_page_ids, issue_text, issue_date, issue_city

    for city_dir in sorted(panel_root.iterdir()):
        if not city_dir.is_dir():
            continue
        p = city_dir / "issue_texts.jsonl"
        if not p.is_file():
            continue
        with p.open("r", encoding="utf-8") as f:
            for raw in f:
                s = raw.strip()
                if not s:
                    continue
                try:
                    obj = json.loads(s)
                except Exception:
                    continue
                iid = _norm(obj.get("issue_id"))
                if not iid:
                    continue
                pids = obj.get("page_ids") if isinstance(obj.get("page_ids"), list) else []
                issue_page_ids[iid] = [_norm(x) for x in pids if _norm(x)]
                issue_text[iid] = _norm(obj.get("text"))
                issue_date[iid] = _norm(obj.get("issue_date"))
                issue_city[iid] = _norm(obj.get("city_key"))

    return issue_page_ids, issue_text, issue_date, issue_city


def _read_page_text_from_page_tags(path: Path) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = defaultdict(dict)
    if not path.is_file():
        return out
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            s = raw.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception:
                continue
            iid = _norm(obj.get("issue_id"))
            if not iid:
                continue
            segs = obj.get("page_segments")
            if not isinstance(segs, list):
                continue
            for seg in segs:
                if not isinstance(seg, dict):
                    continue
                pid = _norm(seg.get("page_id"))
                txt = _norm(seg.get("text"))
                if pid:
                    out[iid][pid] = txt
    return out


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Export side-by-side OG PNG + transcript folders by newspaper and type.")
    ap.add_argument(
        "--run-root",
        default="/Users/saulrichardson/projects/newspapers/newspaper-analysis/reports/runs/pi_v1_run60_fullnewspaper/fullnewspaper_v3_run_20260220_021657",
        help="Fullnewspaper v3 run root.",
    )
    ap.add_argument(
        "--output-dir",
        default="",
        help="Output directory (default: <run-root>/organized_side_by_side).",
    )
    ap.add_argument(
        "--torch-host",
        default="torch",
        help="SSH host for PNG fallback fetch.",
    )
    ap.add_argument(
        "--no-torch-fetch",
        action="store_true",
        help="Disable SSH fallback fetch from torch.",
    )
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    run_root = Path(args.run_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve() if _norm(args.output_dir) else run_root / "organized_side_by_side"
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = _safe_read_csv(run_root / "classification" / "issue_labels_v3.csv")
    clean = _safe_read_csv(run_root / "cleaning" / "clean_zoning_text_flat.csv")
    assign = _safe_read_csv(run_root / "events" / "issue_event_assignments_v3.csv")
    page_tags_path = run_root / "page_tagging" / "page_tags.jsonl"
    panels_root = run_root.parent / "panels"

    if labels.empty:
        raise SystemExit("classification/issue_labels_v3.csv is missing or empty.")

    clean_map = {}
    if not clean.empty and "issue_id" in clean.columns:
        for _, r in clean.iterrows():
            iid = _norm(r.get("issue_id"))
            if iid:
                clean_map[iid] = {
                    "clean_text": _norm(r.get("clean_zoning_text")),
                    "clean_chars": int(float(r.get("clean_chars") or 0)),
                    "kept_anchor_count": int(float(r.get("kept_anchor_count") or 0)),
                }

    assign_map = {}
    if not assign.empty and "issue_id" in assign.columns:
        for _, r in assign.iterrows():
            iid = _norm(r.get("issue_id"))
            if not iid:
                continue
            assign_map[iid] = {
                "event_type": _norm(r.get("event_type")),
                "ordinance_event_id": _norm(r.get("ordinance_event_id")),
                "publication_role": _norm(r.get("publication_role")),
            }

    issue_page_ids, issue_text_map, issue_date_map, issue_city_map = _read_issue_pages_from_panels(panels_root)
    page_text_map = _read_page_text_from_page_tags(page_tags_path)

    # Local caches to avoid repeated network fetches.
    cache_dirs = [
        run_root / "reports" / "fullnewspaper_v3" / "figures" / "appendix_pages",
        run_root / "reports" / "fullnewspaper_v3" / "figures" / "evidence_pages",
        Path("/Users/saulrichardson/projects/newspapers/old-ocr/newspaper-parsing-local/data/unique_png"),
        Path("/Users/saulrichardson/projects/newspapers/old-ocr/newspaper-parsing-local/data/issue_zoning_samples_20260125_223032"),
        Path("/Users/saulrichardson/projects/newspapers/old-ocr/newspaper-parsing-local/data/issue_zoning_samples_diverse_20260126_031245"),
        Path("/Users/saulrichardson/projects/newspapers/old-ocr/newspaper-parsing-local/data/issue_zoning_samples_diverse_fullissue_min2_20260126_032321"),
    ]

    export_rows: list[dict[str, Any]] = []
    png_ok = 0
    png_missing = 0

    for _, r in labels.iterrows():
        issue_id = _norm(r.get("issue_id"))
        if not issue_id:
            continue
        newspaper_slug = _norm(r.get("newspaper_slug")) or issue_id.split("__", 1)[0]
        issue_class = _norm(r.get("issue_class"))
        issue_date = _norm(r.get("issue_date")) or issue_date_map.get(issue_id, "")
        city_key = _norm(r.get("city_key")) or issue_city_map.get(issue_id, "")
        a = assign_map.get(issue_id, {})
        event_type = _norm(a.get("event_type"))
        type_label = _type_label(issue_class, event_type)
        event_id = _norm(a.get("ordinance_event_id"))

        issue_dir = output_dir / "by_newspaper" / _slug(newspaper_slug) / _slug(type_label) / _slug(issue_id)
        issue_dir.mkdir(parents=True, exist_ok=True)

        clean_entry = clean_map.get(issue_id, {})
        issue_text = issue_text_map.get(issue_id, "")
        clean_text = _norm(clean_entry.get("clean_text"))
        page_ids = issue_page_ids.get(issue_id, [])

        metadata = {
            "newspaper_slug": newspaper_slug,
            "city_key": city_key,
            "issue_id": issue_id,
            "issue_date": issue_date,
            "type_label": type_label,
            "issue_class": issue_class,
            "event_type": event_type,
            "ordinance_event_id": event_id,
            "publication_role": _norm(a.get("publication_role")),
            "contains_verbatim_zoning_law": _norm(r.get("contains_verbatim_zoning_law")),
            "law_text_sufficiency": _norm(r.get("law_text_sufficiency")),
            "clean_chars": int(clean_entry.get("clean_chars") or 0),
            "kept_anchor_count": int(clean_entry.get("kept_anchor_count") or 0),
            "page_ids": page_ids,
        }
        (issue_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (issue_dir / "issue_transcript_full.txt").write_text(issue_text + ("\n" if issue_text else ""), encoding="utf-8")
        (issue_dir / "clean_zoning_text.txt").write_text(clean_text + ("\n" if clean_text else ""), encoding="utf-8")

        per_issue_png_ok = 0
        for idx, pid in enumerate(page_ids, start=1):
            stem = f"page_{idx:03d}__{_slug(pid)}"
            txt_path = issue_dir / f"{stem}.txt"
            png_path = issue_dir / f"{stem}.png"
            page_txt = _norm(page_text_map.get(issue_id, {}).get(pid))
            txt_path.write_text(page_txt + ("\n" if page_txt else ""), encoding="utf-8")
            ok, source = _try_copy_png(
                page_id=pid,
                dest_png=png_path,
                cache_dirs=cache_dirs,
                torch_host=args.torch_host,
                try_torch=not args.no_torch_fetch,
            )
            if ok:
                png_ok += 1
                per_issue_png_ok += 1
            else:
                png_missing += 1
            export_rows.append(
                {
                    "newspaper_slug": newspaper_slug,
                    "type_label": type_label,
                    "issue_id": issue_id,
                    "issue_date": issue_date,
                    "page_id": pid,
                    "txt_file": str(txt_path.relative_to(output_dir)),
                    "png_file": str(png_path.relative_to(output_dir)),
                    "png_status": "ok" if ok else "missing",
                    "png_source": source,
                }
            )

        # Write lightweight issue index for quick browse.
        lines = [
            f"# {newspaper_slug} | {type_label} | {issue_id}",
            "",
            f"- issue_date: {issue_date}",
            f"- issue_class: {issue_class}",
            f"- event_type: {event_type}",
            f"- ordinance_event_id: {event_id}",
            f"- pages: {len(page_ids)}",
            f"- png_found: {per_issue_png_ok}/{len(page_ids)}",
            "",
            "## Page Files",
            "",
            "| page_id | transcript | png |",
            "|---|---|---|",
        ]
        for idx, pid in enumerate(page_ids, start=1):
            stem = f"page_{idx:03d}__{_slug(pid)}"
            lines.append(f"| `{pid}` | `{stem}.txt` | `{stem}.png` |")
        (issue_dir / "README.md").write_text("\n".join(lines).strip() + "\n", encoding="utf-8")

    summary_df = pd.DataFrame.from_records(export_rows)
    summary_dir = output_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    if not summary_df.empty:
        summary_df.to_csv(summary_dir / "page_file_inventory.csv", index=False)
        by_type = (
            summary_df.groupby(["newspaper_slug", "type_label"], dropna=False)
            .agg(
                page_count=("page_id", "count"),
                png_ok=("png_status", lambda s: int((s == "ok").sum())),
                png_missing=("png_status", lambda s: int((s != "ok").sum())),
            )
            .reset_index()
            .sort_values(["newspaper_slug", "type_label"])
            .reset_index(drop=True)
        )
        by_type.to_csv(summary_dir / "inventory_by_newspaper_type.csv", index=False)

    summary = {
        "run_root": str(run_root),
        "output_dir": str(output_dir),
        "issue_count": int(labels["issue_id"].astype(str).nunique()) if "issue_id" in labels.columns else 0,
        "page_rows": int(len(summary_df)),
        "png_ok": int(png_ok),
        "png_missing": int(png_missing),
    }
    (summary_dir / "export_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"done_side_by_side_export output_dir={output_dir}")


if __name__ == "__main__":
    main()
