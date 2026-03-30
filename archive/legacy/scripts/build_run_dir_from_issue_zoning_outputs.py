#!/usr/bin/env python3
"""
Build a pipeline-ready run directory from issue_zoning output JSON files.

Input format expected:
- <issue-zoning-dir>/outputs/*.issue_zoning.json
  where each file contains:
    - issue_id
    - slug
    - issue_date
    - source_pages[] with page_id and page_path
    - classification.label (optional)

Output format produced:
- <output-run-dir>/panels/<city_key>/issue_texts.jsonl
- <output-run-dir>/panels/selected_panel_issues.csv
- <output-run-dir>/panels/selected_panels.csv
- <output-run-dir>/build_from_issue_zoning_outputs_provenance.json
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

def _norm(x: Any) -> str:
    return str(x or "").strip()


def _slugify_city_key(city_name: str, state_abbr: str) -> str:
    a = re.sub(r"[^a-z0-9]+", "_", _norm(city_name).lower()).strip("_")
    b = re.sub(r"[^a-z0-9]+", "", _norm(state_abbr).lower())
    return f"{a}__{b}" if a and b else a or "unknown__xx"


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build run-dir panels from issue_zoning outputs.")
    ap.add_argument(
        "--issue-zoning-dir",
        required=True,
        help="Directory containing outputs/*.issue_zoning.json",
    )
    ap.add_argument(
        "--output-run-dir",
        required=True,
        help="Output run directory to write panels/*/issue_texts.jsonl",
    )
    ap.add_argument(
        "--vlm-roots",
        default="",
        help="Comma-separated directories to search for <page_id>.vlm.json files.",
    )
    ap.add_argument(
        "--slug-city-map-csv",
        default="",
        help=(
            "Optional CSV with columns: slug,city_key,city_name,state_abbr,region,urbanicity_proxy. "
            "If omitted, --source-selected-issues is used."
        ),
    )
    ap.add_argument(
        "--source-selected-issues",
        default="/Users/saulrichardson/projects/newspapers/newspaper-analysis/reports/runs/pi_v1_run60_fullnewspaper/panels/selected_panel_issues.csv",
        help="Fallback metadata source for slug->city mapping.",
    )
    ap.add_argument(
        "--newspapers",
        default="",
        help="Optional comma-separated slug filter.",
    )
    ap.add_argument(
        "--max-issues-per-newspaper",
        type=int,
        default=0,
        help="Optional cap per slug (0 = no cap).",
    )
    ap.add_argument(
        "--allow-missing-page-text",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If false, rows with no recovered page text are dropped.",
    )
    return ap.parse_args()


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def _parse_csv_paths(raw: str) -> list[Path]:
    out: list[Path] = []
    for tok in _norm(raw).split(","):
        t = tok.strip()
        if t:
            out.append(Path(t).expanduser().resolve())
    return out


def _box_sort_key(box: dict[str, Any]) -> tuple[int, str]:
    bid = _norm(box.get("id"))
    m = re.search(r"(\d+)$", bid)
    if m:
        return (int(m.group(1)), bid)
    return (10**9, bid)


def _extract_text_from_vlm_page(path: Path) -> str:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return ""
    boxes = obj.get("boxes")
    if not isinstance(boxes, list):
        return ""
    parts: list[str] = []
    for b in sorted([x for x in boxes if isinstance(x, dict)], key=_box_sort_key):
        status = _norm(b.get("status")).lower()
        txt = _norm(b.get("transcript"))
        if not txt:
            continue
        if status in {"unreadable", "error"}:
            continue
        parts.append(txt)
    text = "\n\n".join(parts).strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def _candidate_vlm_paths(page_id: str, page_path: str, vlm_roots: list[Path]) -> list[Path]:
    out: list[Path] = []
    p = Path(page_path).expanduser()
    out.append(p)
    # Torch paths sometimes embed /vast while data is mounted under /scratch.
    sp = _norm(page_path)
    if sp.startswith("/vast/sxr203/"):
        out.append(Path("/scratch/sxr203/" + sp[len("/vast/sxr203/") :]))
    fn = f"{_norm(page_id)}.vlm.json"
    for r in vlm_roots:
        out.append(r / fn)
    # Dedupe while preserving order.
    dedup: list[Path] = []
    seen: set[str] = set()
    for c in out:
        k = str(c)
        if k in seen:
            continue
        seen.add(k)
        dedup.append(c)
    return dedup


def _resolve_vlm_path(page_id: str, page_path: str, vlm_roots: list[Path]) -> Path | None:
    for c in _candidate_vlm_paths(page_id, page_path, vlm_roots):
        if c.is_file():
            return c
    return None


def _load_slug_city_map_from_selected_issues(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        return {}
    out: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        cols = set(r.fieldnames or [])
        required = {"issue_id", "city_key", "city_name", "state_abbr"}
        if not required.issubset(cols):
            return {}
        for row in r:
            issue_id = _norm(row.get("issue_id"))
            slug = issue_id.rsplit("__", 1)[0].lower() if "__" in issue_id else ""
            if not slug:
                continue
            if slug in out:
                continue
            out[slug] = {
                "slug": slug,
                "city_key": _norm(row.get("city_key")),
                "city_name": _norm(row.get("city_name")),
                "state_abbr": _norm(row.get("state_abbr")).lower(),
                "region": _norm(row.get("region")),
                "urbanicity_proxy": _norm(row.get("urbanicity_proxy")) or "unknown",
            }
    return out


def _load_slug_city_map(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        raise SystemExit(f"Missing slug-city-map-csv: {path}")
    out: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        cols = set(r.fieldnames or [])
        req = {"slug", "city_key", "city_name", "state_abbr"}
        miss = sorted(req - cols)
        if miss:
            raise SystemExit(f"slug-city-map-csv missing columns: {miss}")
        for row in r:
            slug = _norm(row.get("slug")).lower()
            if not slug:
                continue
            out[slug] = {
                "slug": slug,
                "city_key": _norm(row.get("city_key")),
                "city_name": _norm(row.get("city_name")),
                "state_abbr": _norm(row.get("state_abbr")).lower(),
                "region": _norm(row.get("region")),
                "urbanicity_proxy": _norm(row.get("urbanicity_proxy")) or "unknown",
            }
    return out


def _infer_date_from_issue_id(issue_id: str) -> str:
    # expected slug__YYYY-MM-DD
    if "__" in issue_id:
        tail = issue_id.split("__", 1)[1]
        if re.match(r"^\d{4}-\d{2}-\d{2}$", tail):
            return tail
    return ""


def main() -> None:
    args = _parse_args()
    issue_zoning_dir = Path(args.issue_zoning_dir).expanduser().resolve()
    out_run = Path(args.output_run_dir).expanduser().resolve()
    vlm_roots = _parse_csv_paths(args.vlm_roots)
    out_panels = out_run / "panels"
    out_panels.mkdir(parents=True, exist_ok=True)

    outputs_dir = issue_zoning_dir / "outputs"
    if not outputs_dir.is_dir():
        raise SystemExit(f"Missing outputs directory: {outputs_dir}")

    wanted_newspapers = {x.strip().lower() for x in _norm(args.newspapers).split(",") if x.strip()}

    if _norm(args.slug_city_map_csv):
        slug_map = _load_slug_city_map(Path(args.slug_city_map_csv).expanduser().resolve())
    else:
        slug_map = _load_slug_city_map_from_selected_issues(Path(args.source_selected_issues).expanduser().resolve())

    by_city_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    selected_rows: list[dict[str, Any]] = []
    by_slug_count: dict[str, int] = defaultdict(int)

    files = sorted(outputs_dir.glob("*.issue_zoning.json"))
    if not files:
        raise SystemExit(f"No *.issue_zoning.json files in {outputs_dir}")

    for f in files:
        try:
            obj = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue

        issue_id = _norm(obj.get("issue_id"))
        slug = _norm(obj.get("slug")).lower()
        if not slug and "__" in issue_id:
            slug = issue_id.rsplit("__", 1)[0].lower()
        if not issue_id or not slug:
            continue
        if wanted_newspapers and slug not in wanted_newspapers:
            continue
        if int(args.max_issues_per_newspaper) > 0 and by_slug_count[slug] >= int(args.max_issues_per_newspaper):
            continue

        meta = slug_map.get(slug, {})
        city_name = _norm(meta.get("city_name"))
        state_abbr = _norm(meta.get("state_abbr")).lower()
        city_key = _norm(meta.get("city_key"))
        region = _norm(meta.get("region"))
        urbanicity = _norm(meta.get("urbanicity_proxy")) or "unknown"

        # If slug map is unavailable, still write a usable panel with fallback metadata.
        if not city_name:
            city_name = slug.replace("-", " ").title()
        if not state_abbr:
            state_abbr = "xx"
        if not city_key:
            city_key = _slugify_city_key(city_name, state_abbr)

        source_pages = obj.get("source_pages")
        if not isinstance(source_pages, list):
            source_pages = []
        page_ids: list[str] = []
        page_texts: list[str] = []
        page_rows: list[dict[str, Any]] = []
        for sp in source_pages:
            if not isinstance(sp, dict):
                continue
            page_id = _norm(sp.get("page_id"))
            page_path = _norm(sp.get("page_path"))
            if not page_id:
                continue
            page_ids.append(page_id)
            vp = _resolve_vlm_path(page_id, page_path, vlm_roots)
            txt = ""
            if vp is not None:
                txt = _extract_text_from_vlm_page(vp)
                if txt:
                    page_texts.append(txt)
            page_rows.append(
                {
                    "page_id": page_id,
                    "page_num": int(sp.get("page_num") or 0),
                    "page_path": page_path,
                    "resolved_vlm_path": str(vp) if vp is not None else "",
                    "text": txt,
                    "text_chars": int(len(txt)),
                }
            )

        text = "\n\n".join(page_texts).strip()
        if not bool(args.allow_missing_page_text) and not text:
            continue

        issue_date = _norm(obj.get("issue_date")) or _infer_date_from_issue_id(issue_id)
        label = _norm(((obj.get("classification") or {}) if isinstance(obj.get("classification"), dict) else {}).get("label"))

        row = {
            "issue_id": issue_id,
            "city_name": city_name,
            "state_abbr": state_abbr,
            "city_key": city_key,
            "region": region,
            "urbanicity_proxy": urbanicity,
            "issue_date": issue_date,
            "classification_label": label,
            "pub_slug": slug,
            "page_ids": page_ids,
            "page_rows": page_rows,
            "text": text,
            "text_chars": int(len(text)),
            "page_count": int(len(page_ids)),
        }
        by_city_rows[city_key].append(row)
        by_slug_count[slug] += 1

        selected_rows.append(
            {
                "city_key": city_key,
                "city_name": city_name,
                "state_abbr": state_abbr,
                "region": region,
                "urbanicity_proxy": urbanicity,
                "issue_id": issue_id,
                "issue_date": issue_date,
                "classification_label": label,
                "text_chars": int(len(text)),
                "page_count": int(len(page_ids)),
            }
        )

    if not selected_rows:
        raise SystemExit("No rows were written. Check slug map, newspapers filter, and VLM roots.")

    def _date_key(s: str) -> tuple[int, int, int, str]:
        m = re.match(r"^(\d{4})-(\d{2})-(\d{2})$", _norm(s))
        if not m:
            return (10**9, 99, 99, _norm(s))
        return (int(m.group(1)), int(m.group(2)), int(m.group(3)), _norm(s))

    selected_rows = sorted(
        selected_rows,
        key=lambda r: (_norm(r.get("city_key")), _date_key(_norm(r.get("issue_date"))), _norm(r.get("issue_id"))),
    )
    _write_csv(
        out_panels / "selected_panel_issues.csv",
        selected_rows,
        [
            "city_key",
            "city_name",
            "state_abbr",
            "region",
            "urbanicity_proxy",
            "issue_id",
            "issue_date",
            "classification_label",
            "text_chars",
            "page_count",
        ],
    )

    panel_rows: list[dict[str, Any]] = []
    for city_key, rows in sorted(by_city_rows.items()):
        rows = sorted(rows, key=lambda r: (_norm(r.get("issue_date")), _norm(r.get("issue_id"))))
        _write_jsonl(out_panels / city_key / "issue_texts.jsonl", rows)
        panel_rows.append(
            {
                "city_key": city_key,
                "city_name": _norm(rows[0].get("city_name")),
                "state_abbr": _norm(rows[0].get("state_abbr")),
                "region": _norm(rows[0].get("region")),
                "urbanicity_proxy": _norm(rows[0].get("urbanicity_proxy")),
                "selected_issue_count": int(len(rows)),
                "issue_date_min": _norm(rows[0].get("issue_date")),
                "issue_date_max": _norm(rows[-1].get("issue_date")),
            }
        )
    panel_rows = sorted(panel_rows, key=lambda r: _norm(r.get("city_key")))
    _write_csv(
        out_panels / "selected_panels.csv",
        panel_rows,
        [
            "city_key",
            "city_name",
            "state_abbr",
            "region",
            "urbanicity_proxy",
            "selected_issue_count",
            "issue_date_min",
            "issue_date_max",
        ],
    )

    city_count = len({str(r.get("city_key")) for r in selected_rows if _norm(r.get("city_key"))})
    newspaper_count = len(
        {
            _norm(r.get("issue_id")).split("__", 1)[0]
            for r in selected_rows
            if "__" in _norm(r.get("issue_id"))
        }
    )

    prov = {
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "issue_zoning_dir": str(issue_zoning_dir),
        "output_run_dir": str(out_run),
        "vlm_roots": [str(x) for x in vlm_roots],
        "source_selected_issues": str(Path(args.source_selected_issues).expanduser().resolve()),
        "slug_city_map_csv": _norm(args.slug_city_map_csv),
        "newspapers_filter": sorted(list(wanted_newspapers)),
        "rows_written": int(len(selected_rows)),
        "city_count_written": int(city_count),
        "newspaper_count_written": int(newspaper_count),
    }
    (out_run / "build_from_issue_zoning_outputs_provenance.json").write_text(
        json.dumps(prov, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(
        f"done_build_run_dir_from_issue_zoning_outputs rows={len(selected_rows)} "
        f"cities={city_count} output={out_run}"
    )


if __name__ == "__main__":
    main()
