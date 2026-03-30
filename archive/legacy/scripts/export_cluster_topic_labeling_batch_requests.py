#!/usr/bin/env python3
"""
Export OpenAI Batch request shards for CLUSTER topic labeling (chunk-cluster → human label).

Inputs (RUN_ROOT from chunk-topic pipeline)
  - clusters_chunks/clusters.jsonl
      - one row per chunk with fields including:
          - chunk_id
          - cluster_id
          - cluster_prob
          - issue_id
          - doc_weight
  - requests/openai_requests_shard*.jsonl
      - embedding request lines that contain the full chunk text in body.input

Output
  - sharded OpenAI Batch request JSONLs for POST /v1/responses
  - mapping_shard*.jsonl with cluster metadata + the exact excerpts used in the prompt

Policy
  - Labels ALL clusters except noise (cluster_id == -1) by default.
  - Picks representative examples by highest cluster_prob, preferring distinct issue_id.
  - Fails loudly if any required example chunk text cannot be found in the embedding request shards.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from os.path import expanduser
from pathlib import Path
from typing import Any

import numpy as np

from newsvlm_analysis.cluster_topic_label_schema import SCHEMA_VERSIONS, openai_text_config


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _safe_stem(s: str) -> str:
    out = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in str(s))
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_") or "x"


def _load_prompt_text(path: Path) -> str:
    if not path.is_file():
        raise SystemExit(f"Prompt file not found: {path}")
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        raise SystemExit(f"Prompt file is empty: {path}")
    return text


@dataclass(frozen=True)
class ClusterChunk:
    chunk_id: str
    issue_id: str
    cluster_prob: float
    doc_weight: float


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Export OpenAI Batch request shards for cluster topic labeling.")
    ap.add_argument("--run-root", required=True, help="RUN_ROOT of an issue_topics_chunks* pipeline run on VAST")
    ap.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write request shards into (e.g. <RUN_ROOT>/cluster_topic_labels/requests)",
    )
    ap.add_argument(
        "--prompt-path",
        default="prompts/cluster_topic_labeler_prompt_text.txt",
        help="Prompt text file path",
    )

    ap.add_argument("--openai-model", default="gpt-5-nano", help="OpenAI model name for /v1/responses batch body")
    ap.add_argument(
        "--openai-text-format",
        choices=["json_schema", "json_object"],
        default="json_schema",
        help="OpenAI Responses text.format enforcement.",
    )
    ap.add_argument(
        "--schema-version",
        choices=list(SCHEMA_VERSIONS),
        default="v1",
        help=(
            "Output schema version for the cluster label JSON. "
            "v1=topic_name/topic_description/keywords. "
            "v2=adds structured mechanics tags for zoning/land-use."
        ),
    )

    ap.add_argument(
        "--max-examples-per-cluster",
        type=int,
        default=8,
        help="Max number of example excerpts to include for each cluster label prompt.",
    )
    ap.add_argument(
        "--example-max-chars",
        type=int,
        default=1100,
        help="Max characters of chunk text to include per example excerpt.",
    )
    ap.add_argument(
        "--candidate-pool-per-cluster",
        type=int,
        default=180,
        help=(
            "How many high-confidence chunks (by cluster_prob) to consider per cluster "
            "when selecting representative examples. Higher can yield more specific prompts, "
            "but requires loading more chunk texts."
        ),
    )
    ap.add_argument(
        "--cluster-keywords-k",
        type=int,
        default=12,
        help="How many auto keywords (TF‑IDF) to include in each cluster prompt (0 disables).",
    )
    ap.add_argument(
        "--example-selection",
        choices=["legacy", "informative"],
        default="informative",
        help=(
            "How to choose representative example chunks for each cluster prompt. "
            "legacy=rank by cluster_prob only (older behavior). "
            "informative=use keyword coverage + cluster_prob + TF‑IDF specificity."
        ),
    )
    ap.add_argument(
        "--no-mechanics-tags",
        action="store_true",
        help="Do not include heuristic mechanics tags in the prompt (keywords/examples only).",
    )
    ap.add_argument(
        "--excerpt-window-chars",
        type=int,
        default=240,
        help="For each matched keyword, include this many chars of context on each side when building excerpts.",
    )
    ap.add_argument(
        "--tfidf-max-chars",
        type=int,
        default=20000,
        help="Truncate candidate chunk texts to this many chars before fitting TF‑IDF (0 disables).",
    )
    ap.add_argument(
        "--min-cluster-size",
        type=int,
        default=1,
        help="Skip clusters with fewer than this many chunks (default 1 = label all clusters).",
    )
    ap.add_argument(
        "--max-clusters",
        type=int,
        default=0,
        help="Optional: cap number of clusters exported (0 disables). Useful for smoke tests.",
    )

    ap.add_argument("--requests-per-shard", type=int, default=5000, help="Maximum requests per output shard file")
    ap.add_argument(
        "--max-bytes-per-shard",
        type=int,
        default=180_000_000,
        help="Max bytes per OpenAI request shard file (mapping is not capped). Set 0 to disable.",
    )
    return ap.parse_args()


def _load_clusters(clusters_jsonl: Path) -> tuple[dict[int, list[ClusterChunk]], dict[int, dict[str, float]]]:
    if not clusters_jsonl.is_file():
        raise SystemExit(f"clusters.jsonl not found: {clusters_jsonl}")

    by_cluster: dict[int, list[ClusterChunk]] = defaultdict(list)
    stats: dict[int, dict[str, float]] = defaultdict(lambda: {"chunks": 0.0, "weight_total": 0.0})

    with clusters_jsonl.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            obj = json.loads(line)
            cid = obj.get("cluster_id")
            if not isinstance(cid, int):
                continue
            if cid == -1:
                continue
            chunk_id = obj.get("chunk_id")
            issue_id = obj.get("issue_id")
            if not isinstance(chunk_id, str) or not chunk_id.strip():
                raise SystemExit(f"Malformed clusters row missing chunk_id: {obj}")
            if not isinstance(issue_id, str) or not issue_id.strip():
                raise SystemExit(f"Malformed clusters row missing issue_id: {obj}")
            prob = obj.get("cluster_prob")
            if not isinstance(prob, (int, float)):
                prob = 0.0
            w = obj.get("doc_weight")
            if not isinstance(w, (int, float)):
                w = 1.0

            by_cluster[int(cid)].append(
                ClusterChunk(
                    chunk_id=chunk_id.strip(),
                    issue_id=issue_id.strip(),
                    cluster_prob=float(prob),
                    doc_weight=float(w),
                )
            )
            stats[int(cid)]["chunks"] += 1.0
            stats[int(cid)]["weight_total"] += float(w)

    if not by_cluster:
        raise SystemExit(f"No non-noise clusters found in {clusters_jsonl}")
    return dict(by_cluster), {k: dict(v) for k, v in stats.items()}


_PAGE_HEADER_RE = re.compile(r"^===\s*PAGE\b.*===\s*$", re.IGNORECASE)


def _strip_page_headers(text: str) -> str:
    """Remove synthetic per-page delimiter lines (=== PAGE ... ===) from chunk text."""
    if not isinstance(text, str) or not text.strip():
        return ""
    lines = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if _PAGE_HEADER_RE.match(line):
            continue
        lines.append(raw.rstrip())
    return "\n".join(lines).strip()


def _clean_for_tfidf(text: str, *, max_chars: int) -> str:
    """Light normalization to keep TF‑IDF stable on huge ordinance chunks."""
    t = _strip_page_headers(text)
    if max_chars > 0 and len(t) > max_chars:
        t = t[:max_chars].rstrip()
    return t


def _topk_sum_sparse_row(data: np.ndarray, k: int) -> float:
    if data.size <= 0:
        return 0.0
    if k <= 0 or data.size <= k:
        return float(np.sum(data))
    # Sum of k largest values without full sort.
    return float(np.partition(data, -k)[-k:].sum())

def _fit_tfidf_for_candidates(
    chunk_text_by_id: dict[str, str],
    *,
    tfidf_max_chars: int,
) -> tuple[list[str], Any, Any, dict[str, float]]:
    """
    Fit a TF‑IDF vectorizer over candidate chunk texts and return:
      - chunk_ids in matrix row order
      - fitted vectorizer
      - sparse tfidf matrix aligned with chunk_ids
      - per-chunk specificity scores (higher = more distinctive terms)
    """
    if not chunk_text_by_id:
        raise SystemExit("Internal error: no chunk texts to fit TF‑IDF")

    chunk_ids = sorted(chunk_text_by_id.keys())
    texts = [_clean_for_tfidf(chunk_text_by_id[cid], max_chars=tfidf_max_chars) for cid in chunk_ids]
    if not any(t.strip() for t in texts):
        raise SystemExit("All candidate chunk texts were empty after cleaning (unexpected)")

    # Local import so we don't require scikit-learn in legacy/example-only mode.
    from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore

    # Note: we keep this intentionally “small and stable”; the goal is *example selection*,
    # not perfect semantic representation.
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=50_000,
        min_df=2,
        max_df=0.98,
        ngram_range=(1, 2),
    )
    tfidf = vectorizer.fit_transform(texts)

    # Specificity score: sum of the top-k tfidf weights in the document.
    # Boilerplate-heavy docs tend to have many common tokens → lower top-k sum.
    scores: dict[str, float] = {}
    topk = 30
    for i, cid in enumerate(chunk_ids):
        row = tfidf.getrow(i)
        scores[cid] = _topk_sum_sparse_row(row.data, topk)

    return chunk_ids, vectorizer, tfidf, scores


def _cluster_top_terms(
    *,
    vectorizer: Any,
    tfidf: Any,
    chunk_ids_in_matrix_order: list[str],
    cluster_chunk_ids: list[str],
    k: int,
) -> list[str]:
    if k <= 0:
        return []
    want = set(cluster_chunk_ids)
    idx = [i for i, cid in enumerate(chunk_ids_in_matrix_order) if cid in want]
    if not idx:
        return []
    v = tfidf[idx].mean(axis=0)
    dense = np.asarray(v).ravel()
    if dense.size <= 0:
        return []
    kk = min(k, int(dense.size))
    top_idx = np.argpartition(-dense, kk - 1)[:kk]
    top_idx = top_idx[np.argsort(-dense[top_idx])]
    names = vectorizer.get_feature_names_out()
    out: list[str] = []
    for j in top_idx.tolist():
        term = str(names[int(j)]).strip()
        if not term:
            continue
        # Filter some ultra-generic legal boilerplate tokens that otherwise dominate.
        if term in {"section", "sections", "ordinance", "zoning", "shall", "hereby", "city", "town"}:
            continue
        out.append(term)
        if len(out) >= k:
            break
    return out


def _keyword_coverage_score(text: str, *, keywords: list[str]) -> int:
    if not keywords:
        return 0
    t = _strip_page_headers(text).lower()
    if not t:
        return 0
    score = 0
    for kw in keywords:
        k = (kw or "").strip().lower()
        if not k:
            continue
        if re.search(rf"\\b{re.escape(k)}\\b", t, flags=re.IGNORECASE):
            score += 1
    return score


def _pick_examples_legacy(rows: list[ClusterChunk], *, k: int) -> list[ClusterChunk]:
    """
    Previous behavior: choose examples by highest cluster_prob, preferring distinct issue_id.
    """
    if k <= 0:
        raise ValueError("k must be > 0")
    sorted_rows = sorted(rows, key=lambda r: (-r.cluster_prob, r.issue_id, r.chunk_id))
    chosen: list[ClusterChunk] = []
    seen_issues: set[str] = set()
    for r in sorted_rows:
        if r.issue_id in seen_issues:
            continue
        chosen.append(r)
        seen_issues.add(r.issue_id)
        if len(chosen) >= k:
            return chosen
    for r in sorted_rows:
        if r in chosen:
            continue
        chosen.append(r)
        if len(chosen) >= k:
            return chosen
    return chosen


def _pick_examples_informative(
    rows: list[ClusterChunk],
    *,
    k: int,
    chunk_text_by_id: dict[str, str],
    cluster_keywords: list[str],
    specificity_by_chunk_id: dict[str, float],
) -> list[ClusterChunk]:
    """
    Choose examples that are:
      - representative (high cluster_prob)
      - mechanically informative (contain cluster keywords)
      - distinctive (TF‑IDF specificity)
      - diverse across issues
    """
    if k <= 0:
        raise ValueError("k must be > 0")

    def sort_key(r: ClusterChunk) -> tuple[int, float, float, str, str]:
        text = chunk_text_by_id.get(r.chunk_id, "")
        kw_cov = _keyword_coverage_score(text, keywords=cluster_keywords)
        spec = float(specificity_by_chunk_id.get(r.chunk_id, 0.0))
        return (kw_cov, float(r.cluster_prob), spec, r.issue_id, r.chunk_id)

    sorted_rows = sorted(rows, key=lambda r: (-sort_key(r)[0], -sort_key(r)[1], -sort_key(r)[2], sort_key(r)[3], sort_key(r)[4]))

    chosen: list[ClusterChunk] = []
    seen_issues: set[str] = set()
    for r in sorted_rows:
        if r.issue_id in seen_issues:
            continue
        chosen.append(r)
        seen_issues.add(r.issue_id)
        if len(chosen) >= k:
            return chosen

    for r in sorted_rows:
        if r in chosen:
            continue
        chosen.append(r)
        if len(chosen) >= k:
            return chosen
    return chosen


def _load_chunk_texts_from_embedding_requests(*, request_dir: Path, needed_chunk_ids: set[str]) -> dict[str, str]:
    if not request_dir.is_dir():
        raise SystemExit(f"Embedding request dir not found: {request_dir}")

    found: dict[str, str] = {}
    shard_paths = sorted(request_dir.glob("openai_requests_shard*.jsonl"))
    if not shard_paths:
        raise SystemExit(f"No openai_requests_shard*.jsonl found under {request_dir}")

    for p in shard_paths:
        with p.open("r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                obj = json.loads(line)
                cid = obj.get("custom_id")
                if not isinstance(cid, str):
                    continue
                if cid not in needed_chunk_ids:
                    continue
                body = obj.get("body") or {}
                if not isinstance(body, dict):
                    raise SystemExit(f"Malformed embedding request body in {p} for custom_id={cid}")
                text = body.get("input")
                if not isinstance(text, str) or not text.strip():
                    raise SystemExit(f"Missing/empty embedding body.input in {p} for custom_id={cid}")
                found[cid] = text
                if len(found) >= len(needed_chunk_ids):
                    break
        if len(found) >= len(needed_chunk_ids):
            break

    missing = sorted(needed_chunk_ids - set(found))
    if missing:
        preview = ", ".join(missing[:20])
        more = "" if len(missing) <= 20 else f" (+{len(missing) - 20} more)"
        raise SystemExit(f"Missing chunk texts for {len(missing)} chunk_ids (example: {preview}{more})")
    return found


def _excerpt_head_raw(text: str, *, max_chars: int) -> str:
    t = (text or "").strip()
    if max_chars <= 0:
        return t
    if len(t) <= max_chars:
        return t
    return t[:max_chars].rstrip()


def _excerpt_keyword_windows(
    text: str,
    *,
    keywords: list[str],
    max_chars: int,
    window_chars: int,
) -> str:
    """
    Build a compact excerpt by extracting windows around cluster keywords.
    Falls back to a simple head excerpt when no keywords are found.
    """
    clean = _strip_page_headers(text)
    if max_chars <= 0:
        return clean.strip()
    if not clean.strip():
        return ""

    t = clean.strip()

    hits: list[tuple[int, int]] = []
    for kw in keywords:
        k = (kw or "").strip().lower()
        if not k:
            continue
        # Word-boundary-ish matching helps avoid spurious hits on short tokens.
        pat = re.compile(rf"\\b{re.escape(k)}\\b", re.IGNORECASE)
        m = pat.search(t)
        if not m:
            continue
        start = max(0, m.start() - window_chars)
        end = min(len(t), m.end() + window_chars)
        hits.append((start, end))
        if len(hits) >= 8:
            break

    if not hits:
        # Fallback: head excerpt, but prefer whole paragraphs.
        parts: list[str] = []
        remaining = max_chars
        for para in [p.strip() for p in t.split("\n\n") if p.strip()]:
            if remaining <= 0:
                break
            if len(para) + (2 if parts else 0) <= remaining:
                parts.append(para)
                remaining -= len(para) + (2 if parts else 0)
            else:
                parts.append(para[:remaining].rstrip())
                remaining = 0
        return "\n\n".join(parts).strip()[:max_chars].rstrip()

    # Merge overlapping windows.
    hits = sorted(hits)
    merged: list[tuple[int, int]] = []
    for s, e in hits:
        if not merged:
            merged.append((s, e))
            continue
        ps, pe = merged[-1]
        if s <= pe:
            merged[-1] = (ps, max(pe, e))
        else:
            merged.append((s, e))

    out_parts: list[str] = []
    used = 0
    for s, e in merged:
        seg = t[s:e].strip()
        if not seg:
            continue
        # Add separators between non-contiguous windows.
        sep = "\n...\n" if out_parts else ""
        add_len = len(sep) + len(seg)
        if used + add_len > max_chars:
            remaining = max_chars - used - len(sep)
            if remaining <= 0:
                break
            seg = seg[:remaining].rstrip()
            out_parts.append(sep + seg if sep else seg)
            used = max_chars
            break
        out_parts.append(sep + seg if sep else seg)
        used += add_len

        if used >= max_chars:
            break

    out = "".join(out_parts).strip()
    if not out:
        return t[:max_chars].rstrip()
    return out


def _build_cluster_prompt(
    *,
    prompt_text: str,
    cluster_id: int,
    cluster_keywords: list[str],
    mechanics_summary: dict[str, list[str]] | None,
    examples: list[dict[str, Any]],
) -> str:
    lines: list[str] = []
    lines.append(prompt_text.strip())
    lines.append("")
    lines.append(f"Cluster id: {cluster_id}")
    if cluster_keywords:
        lines.append("")
        lines.append("Cluster keywords (auto, from TF-IDF): " + ", ".join(cluster_keywords))
    if mechanics_summary:
        actions = mechanics_summary.get("action_tags") or []
        dims = mechanics_summary.get("dimension_tags") or []
        instr = mechanics_summary.get("instrument_tags") or []
        districts = mechanics_summary.get("district_tokens") or []
        parts = []
        if actions:
            parts.append("actions=" + ", ".join(actions))
        if dims:
            parts.append("dimensions=" + ", ".join(dims))
        if instr:
            parts.append("instruments=" + ", ".join(instr))
        if districts:
            parts.append("districts=" + ", ".join(districts))
        if parts:
            lines.append("")
            lines.append("Mechanics tags (auto): " + " | ".join(parts))
    lines.append("")
    lines.append("Examples (each is an excerpt from a chunk in this cluster):")
    for i, ex in enumerate(examples, start=1):
        lines.append("")
        lines.append(f"[EXAMPLE {i}] issue_id={ex.get('issue_id')} chunk_id={ex.get('chunk_id')}")
        lines.append(str(ex.get("excerpt") or "").strip())
    lines.append("")
    lines.append("Return the JSON now.")
    return "\n".join(lines).strip()


def main() -> None:
    args = _parse_args()

    run_root = Path(expanduser(args.run_root)).resolve()
    out_dir = Path(expanduser(args.output_dir)).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    prompt_path = Path(expanduser(args.prompt_path)).resolve()
    prompt_text = _load_prompt_text(prompt_path)

    clusters_jsonl = run_root / "clusters_chunks" / "clusters.jsonl"
    embedding_request_dir = run_root / "requests"

    by_cluster, stats = _load_clusters(clusters_jsonl)

    min_cluster_size = int(args.min_cluster_size)
    if min_cluster_size <= 0:
        raise SystemExit("--min-cluster-size must be >= 1")
    max_clusters = int(args.max_clusters) if int(args.max_clusters) > 0 else None

    max_examples = int(args.max_examples_per_cluster)
    if max_examples <= 0:
        raise SystemExit("--max-examples-per-cluster must be > 0")
    ex_max_chars = int(args.example_max_chars)
    if ex_max_chars <= 0:
        raise SystemExit("--example-max-chars must be > 0")
    candidate_pool = int(args.candidate_pool_per_cluster)
    if candidate_pool <= 0:
        raise SystemExit("--candidate-pool-per-cluster must be > 0")
    cluster_keywords_k = int(args.cluster_keywords_k)
    if cluster_keywords_k < 0:
        raise SystemExit("--cluster-keywords-k must be >= 0")
    example_selection = str(args.example_selection)
    if example_selection not in {"legacy", "informative"}:
        raise SystemExit(f"Unknown --example-selection: {example_selection}")
    include_mechanics_tags = not bool(args.no_mechanics_tags)
    summarize_mechanics_fn = None
    if include_mechanics_tags:
        # Lazy import: this module is optional when --no-mechanics-tags is set, and
        # we want the exporter to run in minimal environments without extra deps.
        from newsvlm_analysis.zoning_mechanics import summarize_mechanics as summarize_mechanics_fn  # type: ignore
    excerpt_window_chars = int(args.excerpt_window_chars)
    if excerpt_window_chars < 0:
        raise SystemExit("--excerpt-window-chars must be >= 0")
    tfidf_max_chars = int(args.tfidf_max_chars)
    if tfidf_max_chars < 0:
        raise SystemExit("--tfidf-max-chars must be >= 0")

    # Choose which clusters to label.
    cluster_ids = [cid for cid in sorted(by_cluster.keys()) if int(stats.get(cid, {}).get("chunks", 0)) >= min_cluster_size]
    if not cluster_ids:
        raise SystemExit(f"No clusters meet min_cluster_size={min_cluster_size}")
    if max_clusters is not None:
        cluster_ids = cluster_ids[:max_clusters]

    chosen_examples: dict[int, list[ClusterChunk]] = {}
    cluster_keywords_by_id: dict[int, list[str]] = {}
    cluster_mechanics_by_id: dict[int, dict[str, list[str]]] = {}

    # We only need TF‑IDF/scikit-learn if we are selecting examples "informatively" or
    # explicitly requesting cluster keywords.
    use_tfidf = (example_selection == "informative") or (cluster_keywords_k > 0)

    # Default to empty specificity scores; only populated in TF‑IDF mode.
    specificity_by_chunk_id: dict[str, float] = {}

    if use_tfidf:
        # First pass: pick a high-confidence candidate pool per cluster.
        # We will load texts for candidates and then choose examples based on a TF‑IDF specificity score.
        candidate_rows: dict[int, list[ClusterChunk]] = {}
        candidate_chunk_ids: set[str] = set()
        for cid in cluster_ids:
            rows = by_cluster[cid]
            # Sort by cluster_prob descending, then stable identifiers.
            ranked = sorted(rows, key=lambda r: (-r.cluster_prob, r.issue_id, r.chunk_id))
            pool = ranked[: min(candidate_pool, len(ranked))]
            candidate_rows[cid] = pool
            for r in pool:
                candidate_chunk_ids.add(r.chunk_id)

        chunk_text_by_id = _load_chunk_texts_from_embedding_requests(
            request_dir=embedding_request_dir, needed_chunk_ids=candidate_chunk_ids
        )

        # Fit TF‑IDF over the candidate pool so we can:
        #  - pick more informative examples (less boilerplate)
        #  - derive cluster keywords to give the labeler extra signal
        tfidf_max_chars_eff = tfidf_max_chars if tfidf_max_chars > 0 else 0
        chunk_ids_order, vectorizer, tfidf, specificity_by_chunk_id = _fit_tfidf_for_candidates(
            chunk_text_by_id,
            tfidf_max_chars=tfidf_max_chars_eff,
        )

        # Second pass: choose examples + cluster keywords per cluster.
        for cid in cluster_ids:
            pool = candidate_rows[cid]
            cluster_keywords = _cluster_top_terms(
                vectorizer=vectorizer,
                tfidf=tfidf,
                chunk_ids_in_matrix_order=chunk_ids_order,
                cluster_chunk_ids=[r.chunk_id for r in pool],
                k=cluster_keywords_k,
            )
            cluster_keywords_by_id[cid] = cluster_keywords
            if example_selection == "legacy":
                ex = _pick_examples_legacy(pool, k=min(max_examples, len(pool)))
            elif example_selection == "informative":
                ex = _pick_examples_informative(
                    pool,
                    k=min(max_examples, len(pool)),
                    chunk_text_by_id=chunk_text_by_id,
                    cluster_keywords=cluster_keywords,
                    specificity_by_chunk_id=specificity_by_chunk_id,
                )
            else:
                raise SystemExit(f"Unknown example_selection: {example_selection}")
            chosen_examples[cid] = ex
            # Deterministic mechanics tags to steer away from generic “zoning law” labels.
            # Summarize over the selected examples (the same text the labeler will see) to avoid
            # injecting tags that don't match the prompt excerpts.
            mech_chars = min(50_000, tfidf_max_chars_eff) if tfidf_max_chars_eff > 0 else 50_000
            if include_mechanics_tags:
                assert summarize_mechanics_fn is not None
                cluster_mechanics_by_id[cid] = summarize_mechanics_fn(
                    [_clean_for_tfidf(chunk_text_by_id.get(r.chunk_id, ""), max_chars=mech_chars) for r in ex],
                    top_k=6,
                )
    else:
        # Legacy (example-only) mode:
        # - Avoid scikit-learn dependency
        # - Pick examples purely by cluster_prob (preferring distinct issue_id)
        # - Do not compute TF‑IDF keywords
        needed_chunk_ids: set[str] = set()
        for cid in cluster_ids:
            rows = by_cluster[cid]
            ex = _pick_examples_legacy(rows, k=min(max_examples, len(rows)))
            chosen_examples[cid] = ex
            cluster_keywords_by_id[cid] = []
            for r in ex:
                needed_chunk_ids.add(r.chunk_id)

        chunk_text_by_id = _load_chunk_texts_from_embedding_requests(
            request_dir=embedding_request_dir, needed_chunk_ids=needed_chunk_ids
        )

        if include_mechanics_tags:
            mech_chars = 50_000
            for cid in cluster_ids:
                ex = chosen_examples[cid]
                assert summarize_mechanics_fn is not None
                cluster_mechanics_by_id[cid] = summarize_mechanics_fn(
                    [_clean_for_tfidf(chunk_text_by_id.get(r.chunk_id, ""), max_chars=mech_chars) for r in ex],
                    top_k=6,
                )

    # Record inputs/config for reproducibility.
    inputs_dir = out_dir / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    (inputs_dir / "run_root.txt").write_text(str(run_root) + "\n", encoding="utf-8")
    (inputs_dir / "clusters_jsonl_path.txt").write_text(str(clusters_jsonl) + "\n", encoding="utf-8")
    (inputs_dir / "prompt_path.txt").write_text(str(prompt_path) + "\n", encoding="utf-8")
    (inputs_dir / "prompt_text_sha256.txt").write_text(_sha256(prompt_text) + "\n", encoding="utf-8")
    (inputs_dir / "export_config.json").write_text(
        json.dumps(
            {
                "openai_model": str(args.openai_model),
                "openai_text_format": str(args.openai_text_format),
                "max_examples_per_cluster": max_examples,
                "example_max_chars": ex_max_chars,
                "candidate_pool_per_cluster": candidate_pool,
                "cluster_keywords_k": cluster_keywords_k,
                "example_selection": example_selection,
                "include_mechanics_tags": include_mechanics_tags,
                "excerpt_window_chars": excerpt_window_chars,
                "tfidf_max_chars": tfidf_max_chars,
                "min_cluster_size": min_cluster_size,
                "max_clusters": max_clusters,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    # Shard writers.
    shard_idx = 0
    requests_in_shard = 0
    bytes_in_shard = 0

    openai_path = out_dir / f"openai_requests_shard{shard_idx:03d}.jsonl"
    mapping_path = out_dir / f"mapping_shard{shard_idx:03d}.jsonl"
    openai_f = openai_path.open("w", encoding="utf-8")
    mapping_f = mapping_path.open("w", encoding="utf-8")

    request_limit = int(args.requests_per_shard)
    if request_limit <= 0:
        raise SystemExit("--requests-per-shard must be > 0")
    max_bytes = int(args.max_bytes_per_shard)
    if max_bytes < 0:
        raise SystemExit("--max-bytes-per-shard must be >= 0")

    def rotate_if_needed(next_openai_line: str) -> None:
        nonlocal shard_idx, requests_in_shard, bytes_in_shard, openai_f, mapping_f, openai_path, mapping_path
        if requests_in_shard <= 0:
            return
        if requests_in_shard < request_limit:
            if max_bytes <= 0:
                return
            if bytes_in_shard + len(next_openai_line.encode("utf-8")) <= max_bytes:
                return

        openai_f.close()
        mapping_f.close()
        shard_idx += 1
        requests_in_shard = 0
        bytes_in_shard = 0
        openai_path = out_dir / f"openai_requests_shard{shard_idx:03d}.jsonl"
        mapping_path = out_dir / f"mapping_shard{shard_idx:03d}.jsonl"
        openai_f = openai_path.open("w", encoding="utf-8")
        mapping_f = mapping_path.open("w", encoding="utf-8")

    schema_version = str(args.schema_version)
    text_cfg = openai_text_config(str(args.openai_text_format), schema_version=schema_version)
    if text_cfg is None:
        raise SystemExit("Internal error: missing OpenAI text config")

    clusters_written = 0
    t0 = time.perf_counter()

    for cid in cluster_ids:
        ex_rows = chosen_examples[cid]
        cluster_keywords = cluster_keywords_by_id.get(cid, [])
        mechanics_summary = cluster_mechanics_by_id.get(cid) if include_mechanics_tags else None
        ex_payload: list[dict[str, Any]] = []
        for r in ex_rows:
            full = chunk_text_by_id[r.chunk_id]
            excerpt: str
            if example_selection == "legacy":
                excerpt = _excerpt_head_raw(full, max_chars=ex_max_chars)
            else:
                excerpt = _excerpt_keyword_windows(
                    full,
                    keywords=cluster_keywords,
                    max_chars=ex_max_chars,
                    window_chars=excerpt_window_chars,
                )
            ex_payload.append(
                {
                    "chunk_id": r.chunk_id,
                    "issue_id": r.issue_id,
                    "cluster_prob": r.cluster_prob,
                    "tfidf_specificity": float(specificity_by_chunk_id.get(r.chunk_id, 0.0)),
                    "excerpt": excerpt,
                }
            )

        prompt = _build_cluster_prompt(
            prompt_text=prompt_text,
            cluster_id=cid,
            cluster_keywords=cluster_keywords,
            mechanics_summary=mechanics_summary,
            examples=ex_payload,
        )
        prompt_sha = _sha256(prompt)

        custom_id = f"cluster::{cid}"

        body: dict[str, Any] = {
            "model": str(args.openai_model),
            "input": [{"role": "user", "content": [{"type": "input_text", "text": prompt}]}],
            "text": text_cfg,
        }

        openai_line = {"custom_id": custom_id, "method": "POST", "url": "/v1/responses", "body": body}
        openai_line_s = json.dumps(openai_line, ensure_ascii=False) + "\n"

        rotate_if_needed(openai_line_s)

        mapping_line = {
            "id": custom_id,
            "custom_id": custom_id,
            "cluster_id": int(cid),
            "cluster_size_chunks": int(stats.get(cid, {}).get("chunks", 0.0)),
            "cluster_weight_total": float(stats.get(cid, {}).get("weight_total", 0.0)),
            "cluster_keywords": cluster_keywords,
            "examples": ex_payload,
            "run_root": str(run_root),
            "clusters_jsonl": str(clusters_jsonl),
            "embedding_request_dir": str(embedding_request_dir),
            "prompt_path": str(prompt_path),
            "prompt_text_sha256": _sha256(prompt_text),
            "final_prompt_sha256": prompt_sha,
            "openai_model": str(args.openai_model),
            "openai_text_format": str(args.openai_text_format),
            "schema_version": schema_version,
            "example_selection": {
                "candidate_pool_per_cluster": int(candidate_pool),
                "tfidf_max_chars": int(tfidf_max_chars),
                "cluster_keywords_k": int(cluster_keywords_k),
                "example_selection": str(example_selection),
                "include_mechanics_tags": bool(include_mechanics_tags),
                "excerpt_window_chars": int(excerpt_window_chars),
            },
            "exported_at": time.time(),
        }
        if include_mechanics_tags:
            mapping_line["cluster_mechanics"] = mechanics_summary

        openai_f.write(openai_line_s)
        mapping_f.write(json.dumps(mapping_line, ensure_ascii=False) + "\n")

        requests_in_shard += 1
        bytes_in_shard += len(openai_line_s.encode("utf-8"))
        clusters_written += 1

    openai_f.close()
    mapping_f.close()

    elapsed = round(time.perf_counter() - t0, 2)
    print(f"Done. clusters_written={clusters_written} shards={shard_idx + 1} elapsed_s={elapsed}")


if __name__ == "__main__":
    main()
