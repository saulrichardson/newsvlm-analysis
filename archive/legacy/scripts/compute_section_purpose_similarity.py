#!/usr/bin/env python3
"""
Compute section-level purpose similarity diagnostics for initial full ordinances.

Inputs:
  - docs_manifest.csv (from export_initial_full_docs_for_section_analysis.py)
  - sections.jsonl (rehydrated section segmentation output)
  - doc_purposes.jsonl (rehydrated document purpose output)

Outputs:
  - purpose_section_similarity_pairs.csv
  - purpose_similarity_by_region_time_urbanicity.csv
  - purpose_section_examples.csv
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


_PURPOSE_TITLE_RE = re.compile(r"(?:purpose|intent|objective|findings|declaration)", flags=re.I)


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


def _js_similarity(p: np.ndarray, q: np.ndarray) -> float:
    # Jensen-Shannon similarity = 1 - JSD.
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p_sum = float(p.sum())
    q_sum = float(q.sum())
    if p_sum <= 0.0 or q_sum <= 0.0:
        return math.nan
    p = p / p_sum
    q = q / q_sum
    m = 0.5 * (p + q)

    def _kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = (a > 0) & (b > 0)
        if not np.any(mask):
            return 0.0
        return float(np.sum(a[mask] * np.log(a[mask] / b[mask])))

    jsd = 0.5 * _kl(p, m) + 0.5 * _kl(q, m)
    # normalize to [0,1] using ln(2).
    jsd_norm = float(jsd / math.log(2.0))
    jsd_norm = min(1.0, max(0.0, jsd_norm))
    return float(1.0 - jsd_norm)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Compute section-level purpose similarity metrics.")
    ap.add_argument(
        "--docs-manifest",
        default="/Users/saulrichardson/projects/newspapers/newspaper-analysis/reports/runs/prototype_zoning_panel_analysis/prototype_zoning_panel_analysis_iter8_scale30/section_analysis/docs_manifest.csv",
        help="Docs manifest CSV from export_initial_full_docs_for_section_analysis.py",
    )
    ap.add_argument(
        "--sections-jsonl",
        default="/Users/saulrichardson/projects/newspapers/old-ocr/tmp_hybrid_pi_real_1/sections_out/sections.jsonl",
        help="Section segmentation output JSONL.",
    )
    ap.add_argument(
        "--purposes-jsonl",
        default="/Users/saulrichardson/projects/newspapers/old-ocr/tmp_hybrid_pi_real_1/purpose_out/doc_purposes.jsonl",
        help="Document purpose output JSONL.",
    )
    ap.add_argument(
        "--output-dir",
        default="/Users/saulrichardson/projects/newspapers/newspaper-analysis/reports/runs/prototype_zoning_panel_analysis/prototype_zoning_panel_analysis_iter8_scale30/section_analysis",
        help="Output directory for similarity tables.",
    )
    ap.add_argument("--max-pairs", type=int, default=0, help="Optional cap on number of pair rows (0 = all).")
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    docs_manifest = Path(args.docs_manifest).expanduser().resolve()
    sections_jsonl = Path(args.sections_jsonl).expanduser().resolve()
    purposes_jsonl = Path(args.purposes_jsonl).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not docs_manifest.is_file():
        raise SystemExit(f"Missing docs manifest: {docs_manifest}")
    if not sections_jsonl.is_file():
        raise SystemExit(f"Missing sections JSONL: {sections_jsonl}")
    if not purposes_jsonl.is_file():
        raise SystemExit(f"Missing purposes JSONL: {purposes_jsonl}")

    docs = pd.read_csv(docs_manifest)
    docs["issue_date"] = pd.to_datetime(docs["issue_date"], errors="coerce")
    docs["issue_year"] = docs["issue_date"].dt.year
    docs["issue_decade"] = (docs["issue_year"] // 10) * 10

    section_rows = _iter_jsonl(sections_jsonl)
    sec_df = pd.DataFrame.from_records(section_rows)
    if sec_df.empty:
        raise SystemExit("Sections JSONL has no rows.")
    need_cols = {"doc_id", "section_number", "title", "text"}
    miss = sorted(c for c in need_cols if c not in sec_df.columns)
    if miss:
        raise SystemExit(f"Sections JSONL missing required columns: {miss}")

    sec_df["section_number"] = pd.to_numeric(sec_df["section_number"], errors="coerce")
    sec_df = sec_df.dropna(subset=["doc_id", "section_number"]).copy()
    sec_df["title"] = sec_df["title"].astype(str)
    sec_df["text"] = sec_df["text"].astype(str)
    sec_df["is_purpose_title"] = sec_df["title"].str.contains(_PURPOSE_TITLE_RE, na=False)
    sec_df = sec_df.sort_values(["doc_id", "section_number"]).reset_index(drop=True)

    selected_sec_rows: list[dict[str, Any]] = []
    for doc_id, g in sec_df.groupby("doc_id", dropna=False):
        g2 = g[g["is_purpose_title"] == True].copy()
        if g2.empty:
            pick = g.iloc[0]
            rule = "fallback_first_section"
        else:
            pick = g2.iloc[0]
            rule = "purpose_title_match"
        selected_sec_rows.append(
            {
                "doc_id": str(doc_id),
                "section_number": int(pick["section_number"]),
                "section_title": str(pick["title"]),
                "selection_rule": rule,
                "section_text": str(pick["text"]),
                "section_text_chars": int(len(str(pick["text"]))),
            }
        )
    sec_pick = pd.DataFrame.from_records(selected_sec_rows)
    sec_pick = docs.merge(sec_pick, on="doc_id", how="left")
    # Keep only truly non-empty section text rows; avoid treating NaN as the string "nan".
    sec_text = sec_pick["section_text"]
    mask_nonempty = sec_text.notna() & sec_text.astype(str).str.strip().ne("")
    sec_pick = sec_pick[mask_nonempty].copy()
    if sec_pick.empty:
        raise SystemExit("No selected section text rows after purpose-section selection.")

    # Purpose vectors from doc_purposes.
    purpose_rows = _iter_jsonl(purposes_jsonl)
    motives = sorted(
        {
            str(p.get("motive"))
            for row in purpose_rows
            for p in (row.get("purposes") or [])
            if isinstance(p, dict) and str(p.get("motive") or "").strip()
        }
    )
    if not motives:
        motives = ["other", "unclear"]
    motive_ix = {m: i for i, m in enumerate(motives)}
    purpose_vec: dict[str, np.ndarray] = {}
    for row in purpose_rows:
        doc_id = str(row.get("doc_id") or "").strip()
        if not doc_id:
            continue
        v = np.zeros((len(motives),), dtype=np.float64)
        for p in (row.get("purposes") or []):
            if not isinstance(p, dict):
                continue
            m = str(p.get("motive") or "").strip()
            if m not in motive_ix:
                continue
            conf = p.get("confidence")
            try:
                c = float(conf) if conf is not None else 1.0
            except Exception:
                c = 1.0
            if not math.isfinite(c):
                c = 1.0
            c = min(1.0, max(0.0, c))
            v[motive_ix[m]] += c
        purpose_vec[doc_id] = v

    # Section text embeddings via TF-IDF for pairwise cosine similarity.
    sec_pick = sec_pick.sort_values(["doc_id"]).reset_index(drop=True)
    tfidf = TfidfVectorizer(min_df=1, ngram_range=(1, 2), stop_words="english")
    x = tfidf.fit_transform(sec_pick["section_text"].astype(str).tolist())
    cos = cosine_similarity(x)

    idx_by_doc = {str(d): i for i, d in enumerate(sec_pick["doc_id"].astype(str).tolist())}
    pair_rows: list[dict[str, Any]] = []
    docs_rec = sec_pick.to_dict("records")
    for i, j in itertools.combinations(range(len(docs_rec)), 2):
        a = docs_rec[i]
        b = docs_rec[j]
        doc_a = str(a["doc_id"])
        doc_b = str(b["doc_id"])
        pv_a = purpose_vec.get(doc_a, np.zeros((len(motives),), dtype=np.float64))
        pv_b = purpose_vec.get(doc_b, np.zeros((len(motives),), dtype=np.float64))
        pair_rows.append(
            {
                "doc_id_a": doc_a,
                "doc_id_b": doc_b,
                "city_key_a": str(a.get("city_key") or ""),
                "city_key_b": str(b.get("city_key") or ""),
                "region_a": str(a.get("region") or ""),
                "region_b": str(b.get("region") or ""),
                "urbanicity_a": str(a.get("urbanicity_proxy") or ""),
                "urbanicity_b": str(b.get("urbanicity_proxy") or ""),
                "issue_year_a": int(a["issue_year"]) if pd.notna(a.get("issue_year")) else math.nan,
                "issue_year_b": int(b["issue_year"]) if pd.notna(b.get("issue_year")) else math.nan,
                "issue_decade_a": int(a["issue_decade"]) if pd.notna(a.get("issue_decade")) else math.nan,
                "issue_decade_b": int(b["issue_decade"]) if pd.notna(b.get("issue_decade")) else math.nan,
                "same_region": int(str(a.get("region")) == str(b.get("region"))),
                "same_urbanicity": int(str(a.get("urbanicity_proxy")) == str(b.get("urbanicity_proxy"))),
                "same_decade": int(a.get("issue_decade") == b.get("issue_decade")),
                "section_embedding_cosine": float(cos[i, j]),
                "purpose_js_similarity": float(_js_similarity(pv_a, pv_b)),
            }
        )

    pairs = pd.DataFrame.from_records(pair_rows)
    pairs = pairs.sort_values(["section_embedding_cosine", "purpose_js_similarity"], ascending=[False, False])
    if int(args.max_pairs) > 0:
        pairs = pairs.head(int(args.max_pairs)).copy()
    pairs_path = out_dir / "purpose_section_similarity_pairs.csv"
    pairs.to_csv(pairs_path, index=False, quoting=csv.QUOTE_MINIMAL)

    # Aggregated relation table by region/time/urbanicity relationship.
    rel = pairs.copy()
    rel["region_relation"] = rel["same_region"].map({1: "same_region", 0: "cross_region"})
    rel["urbanicity_relation"] = rel["same_urbanicity"].map({1: "same_urbanicity", 0: "cross_urbanicity"})
    rel["time_relation"] = rel["same_decade"].map({1: "same_decade", 0: "cross_decade"})
    agg = (
        rel.groupby(["region_relation", "urbanicity_relation", "time_relation"], dropna=False)
        .agg(
            n_pairs=("doc_id_a", "count"),
            mean_section_embedding_cosine=("section_embedding_cosine", "mean"),
            mean_purpose_js_similarity=("purpose_js_similarity", "mean"),
            p50_section_embedding_cosine=("section_embedding_cosine", "median"),
            p50_purpose_js_similarity=("purpose_js_similarity", "median"),
        )
        .reset_index()
        .sort_values(["region_relation", "urbanicity_relation", "time_relation"])
    )
    agg_path = out_dir / "purpose_similarity_by_region_time_urbanicity.csv"
    agg.to_csv(agg_path, index=False, quoting=csv.QUOTE_MINIMAL)

    ex = sec_pick[
        [
            "doc_id",
            "city_key",
            "city_name",
            "state_abbr",
            "region",
            "urbanicity_proxy",
            "issue_date",
            "section_number",
            "section_title",
            "selection_rule",
            "section_text_chars",
            "section_text",
        ]
    ].copy()
    ex["section_text"] = ex["section_text"].astype(str).str.slice(0, 2000)
    ex_path = out_dir / "purpose_section_examples.csv"
    ex.to_csv(ex_path, index=False, quoting=csv.QUOTE_MINIMAL)

    print(
        "Done. "
        f"docs={len(sec_pick)} pairs={len(pairs)} "
        f"pairs_csv={pairs_path} agg_csv={agg_path} examples_csv={ex_path}"
    )


if __name__ == "__main__":
    main()
