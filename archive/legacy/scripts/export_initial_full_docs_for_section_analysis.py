#!/usr/bin/env python3
"""
Export initial full-ordinance documents for section-level purpose analysis.

Inputs:
  - <run_dir>/tables/initial_full_ordinance_by_city.csv
  - <run_dir>/panels/*/issue_texts.jsonl

Outputs:
  - <output_dir>/docs.jsonl
  - <output_dir>/docs_manifest.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import pandas as pd


def _iter_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                out.append(obj)
    return out


def _safe_slug(s: str) -> str:
    import re

    x = str(s or "").strip().lower()
    x = re.sub(r"[^a-z0-9]+", "_", x)
    x = re.sub(r"_+", "_", x).strip("_")
    return x[:120] or "x"


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Export initial full-ordinance docs for section-level purpose analysis.")
    ap.add_argument(
        "--run-dir",
        default="/Users/saulrichardson/projects/newspapers/newspaper-analysis/reports/runs/prototype_zoning_panel_analysis/prototype_zoning_panel_analysis_iter8_scale30",
        help="Prototype run directory with tables/ and panels/.",
    )
    ap.add_argument(
        "--output-dir",
        default="/Users/saulrichardson/projects/newspapers/newspaper-analysis/reports/runs/prototype_zoning_panel_analysis/prototype_zoning_panel_analysis_iter8_scale30/section_analysis",
        help="Directory to write docs.jsonl and manifest CSV.",
    )
    ap.add_argument(
        "--max-docs",
        type=int,
        default=0,
        help="Optional max number of exported docs (0 = all).",
    )
    ap.add_argument(
        "--use-earliest-per-city",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When true, export earliest reconstructed full ordinance per city-state.",
    )
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    first_city_path = run_dir / "tables" / "initial_full_ordinance_by_city.csv"
    if not first_city_path.is_file():
        raise SystemExit(f"Missing required file: {first_city_path}")
    first_city = pd.read_csv(first_city_path)
    first_city["state_abbr"] = first_city["state_abbr"].astype(str).str.lower()
    first_city["city_key"] = first_city.apply(
        lambda r: f"{_safe_slug(str(r['city_name']))}__{_safe_slug(str(r['state_abbr']))}",
        axis=1,
    )
    first_city = first_city.drop_duplicates(subset=["city_key"], keep="first")

    rows: list[dict[str, Any]] = []
    panel_dir = run_dir / "panels"
    if not panel_dir.is_dir():
        raise SystemExit(f"Missing panel dir: {panel_dir}")

    for city_dir in sorted(panel_dir.iterdir()):
        if not city_dir.is_dir():
            continue
        issue_path = city_dir / "issue_texts.jsonl"
        if not issue_path.is_file():
            continue
        city_key = city_dir.name
        for obj in _iter_jsonl(issue_path):
            if str(obj.get("classification_label") or "") != "full_ordinance":
                continue
            issue_id = str(obj.get("issue_id") or "").strip()
            issue_date = str(obj.get("issue_date") or "").strip()
            text = str(obj.get("text") or "").strip()
            if not issue_id or not issue_date or not text:
                continue
            rows.append(
                {
                    "city_key": city_key,
                    "issue_id": issue_id,
                    "issue_date": issue_date,
                    "text_chars": int(len(text)),
                    "text": text,
                }
            )

    if not rows:
        raise SystemExit("No reconstructed full-ordinance texts found in panels/*/issue_texts.jsonl")

    docs = pd.DataFrame.from_records(rows).drop_duplicates(subset=["issue_id"], keep="first")
    docs["issue_date"] = pd.to_datetime(docs["issue_date"], errors="coerce")
    docs = docs.dropna(subset=["issue_date"]).copy()
    docs = docs.sort_values(["city_key", "issue_date", "issue_id"]).reset_index(drop=True)

    if bool(args.use_earliest_per_city):
        docs = docs.groupby("city_key", as_index=False).first()

    docs = docs.merge(
        first_city[
            [
                "city_key",
                "city_name",
                "state_abbr",
                "region",
                "urbanicity_proxy",
                "first_full_ordinance_date",
                "first_full_ordinance_year",
            ]
        ],
        on="city_key",
        how="left",
    )

    docs["doc_id"] = docs["issue_id"]
    docs["issue_date"] = pd.to_datetime(docs["issue_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    docs = docs.sort_values(["issue_date", "city_key"]).reset_index(drop=True)
    if int(args.max_docs) > 0:
        docs = docs.head(int(args.max_docs)).copy()

    docs_jsonl = out_dir / "docs.jsonl"
    with docs_jsonl.open("w", encoding="utf-8") as f:
        for r in docs.itertuples(index=False):
            payload = {
                "doc_id": str(r.doc_id),
                "city_key": str(r.city_key),
                "city_name": str(r.city_name),
                "state_abbr": str(r.state_abbr),
                "region": str(r.region),
                "urbanicity_proxy": str(r.urbanicity_proxy),
                "issue_id": str(r.issue_id),
                "issue_date": str(r.issue_date),
                "text": str(r.text),
            }
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    manifest = docs[
        [
            "doc_id",
            "city_key",
            "city_name",
            "state_abbr",
            "region",
            "urbanicity_proxy",
            "issue_id",
            "issue_date",
            "text_chars",
            "first_full_ordinance_date",
            "first_full_ordinance_year",
        ]
    ].copy()
    manifest_path = out_dir / "docs_manifest.csv"
    manifest.to_csv(manifest_path, index=False, quoting=csv.QUOTE_MINIMAL)

    print(f"Done. docs={len(docs)} docs_jsonl={docs_jsonl} manifest={manifest_path}")


if __name__ == "__main__":
    main()
