#!/usr/bin/env python3
"""
Build RHS-style panels from *clause-level* instrument clusters.

Background
----------
The existing RHS pipeline (scripts/build_regression_rhs_from_topics.py) builds
interpretable covariates from *chunk-topic clusters* + LLM-produced mechanics tags.

PI feedback on granularity suggests moving to a hybrid approach:
  sectioning -> motive labeling -> clause extraction -> clause clustering -> instrument labels

This script turns the downstream clause clustering output into regression-ready panels.

Inputs
------
- clauses.jsonl
    Produced by scripts/rehydrate_regulatory_clause_extraction_openai_batch_results.py
    Required fields:
      - clause_id
      - issue_id or doc_id
      - issue_date (or issue_id formatted as <slug>__YYYY-MM-DD)
      - slug (or parsable from issue_id)

- clusters.jsonl
    Produced by scripts/cluster_regulatory_clauses_local.py OR scripts/cluster_clause_embeddings.py
    Required fields:
      - clause_id
      - cluster_id (int; -1 = noise)
      - cluster_prob (optional)

- cluster_labels.jsonl (optional)
    Produced by scripts/rehydrate_regulatory_instrument_cluster_labels_openai_batch_results.py
    Used only to attach human-readable instrument_name/description.

Outputs
-------
Writes both .parquet and .csv:
- rhs_issue_cluster_long_<stamp>.{parquet,csv}
    One row per (issue_id, cluster_id) with:
      - n_clauses_cluster
      - cluster_share = n_clauses_cluster / n_clauses_non_noise (per-issue denominator)

- rhs_issue_cluster_summary_<stamp>.{parquet,csv}
    One row per issue with clause totals + noise/missing diagnostics.

- rhs_<unit>_<bucket>_panel_long_<stamp>.{parquet,csv}
    One row per (unit, bucket, cluster_id) with mean shares (including zeros).

No APIs are called.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
from collections import Counter, defaultdict
from os.path import expanduser
from pathlib import Path
from typing import Any, Iterable, Literal

import pandas as pd


Bucket = Literal["month", "year"]
Unit = Literal["paper", "city_state", "state"]


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.is_file():
        raise SystemExit(f"Missing input: {path}")
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                yield obj


def _bucket_from_date(date_s: str, bucket: Bucket) -> str:
    if not isinstance(date_s, str) or len(date_s) < 10:
        raise ValueError(f"Bad issue_date: {date_s!r}")
    if bucket == "year":
        return date_s[:4]
    if bucket == "month":
        return date_s[:7]
    raise ValueError(f"Unknown bucket: {bucket}")


def _safe_slug_from_issue_id(issue_id: str) -> str:
    if not isinstance(issue_id, str) or not issue_id.strip():
        return ""
    # Our pipeline uses `<slug>__<YYYY-MM-DD>`; keep it robust.
    return issue_id.split("__", 1)[0].strip()


def _safe_date_from_issue_id(issue_id: str) -> str:
    if not isinstance(issue_id, str) or "__" not in issue_id:
        return ""
    _, tail = issue_id.split("__", 1)
    tail = tail.strip()
    return tail if len(tail) >= 10 else ""


def _load_pub_metadata(path: Path) -> pd.DataFrame:
    """
    Load publication metadata (at minimum pub_slug). Supported:
      - .parquet
      - .jsonl
      - .csv
    """
    if not path.is_file():
        raise SystemExit(f"--pub-metadata not found: {path}")
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix.lower() in {".jsonl", ".json"}:
        df = pd.read_json(path, lines=True)
    elif path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        raise SystemExit(f"Unsupported --pub-metadata format: {path} (expected parquet/jsonl/csv)")

    if "pub_slug" not in df.columns:
        raise SystemExit(f"pub metadata missing required column pub_slug: {path}")

    # Keep a small, stable subset of columns (if present).
    keep = [
        c
        for c in [
            "pub_slug",
            "pub_title",
            "pub_id",
            "country_id",
            "country_abbr",
            "state_id",
            "state_abbr",
            "state_name",
            "city_id",
            "city_name",
            "pub_min_year",
            "pub_max_year",
        ]
        if c in df.columns
    ]
    out = df[keep].copy()
    out["pub_slug"] = out["pub_slug"].astype(str).str.strip()
    out = out[out["pub_slug"] != ""].drop_duplicates(subset=["pub_slug"], keep="first")
    return out


def _load_cluster_labels(path: Path | None) -> dict[int, dict[str, Any]]:
    if path is None or not path.is_file():
        return {}
    out: dict[int, dict[str, Any]] = {}
    for obj in _iter_jsonl(path):
        if obj.get("error") is not None:
            continue
        cid = obj.get("cluster_id")
        if isinstance(cid, int):
            out[int(cid)] = obj
    return out


def _write_df(df: pd.DataFrame, *, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = out_dir / f"{stem}.parquet"
    csv_path = out_dir / f"{stem}.csv"
    df.to_parquet(parquet_path, index=False)
    df.to_csv(csv_path, index=False, quoting=csv.QUOTE_MINIMAL)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build RHS panels from clause-level instrument clusters.")
    ap.add_argument("--clauses-jsonl", required=True, help="Path to clauses.jsonl")
    ap.add_argument("--clusters-jsonl", required=True, help="Path to clusters.jsonl (per-clause assignment)")
    ap.add_argument("--output-dir", required=True, help="Directory to write outputs into")

    ap.add_argument("--cluster-labels-jsonl", default="", help="Optional cluster_labels.jsonl for instrument names.")
    ap.add_argument("--pub-metadata", default="", help="Optional pub metadata (parquet/jsonl/csv) with pub_slug.")
    ap.add_argument("--bucket", choices=["month", "year"], default="month", help="Time bucket for aggregation.")
    ap.add_argument("--unit", choices=["paper", "city_state", "state"], default="paper", help="Panel unit.")
    ap.add_argument(
        "--by-classification-label",
        action="store_true",
        help="Include classification_label in the aggregated panel grouping keys.",
    )
    ap.add_argument("--include-noise", action="store_true", help="Include cluster_id=-1 in long panels.")
    ap.add_argument("--min-prob", type=float, default=0.0, help="Drop assignments with cluster_prob < this value.")
    ap.add_argument("--max-clauses", type=int, default=0, help="Optional cap on clauses processed (0 disables).")
    return ap.parse_args()


def main() -> None:
    args = _parse_args()

    clauses_path = Path(expanduser(args.clauses_jsonl)).resolve()
    clusters_path = Path(expanduser(args.clusters_jsonl)).resolve()
    out_dir = Path(expanduser(args.output_dir)).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    bucket: Bucket = str(args.bucket)  # type: ignore[assignment]
    unit: Unit = str(args.unit)  # type: ignore[assignment]
    include_noise = bool(args.include_noise)

    min_prob = float(args.min_prob)
    if min_prob < 0.0 or min_prob > 1.0:
        raise SystemExit("--min-prob must be in [0,1]")

    max_clauses = int(args.max_clauses)
    if max_clauses < 0:
        raise SystemExit("--max-clauses must be >= 0")

    labels_path = Path(expanduser(args.cluster_labels_jsonl)).resolve() if str(args.cluster_labels_jsonl).strip() else None
    cluster_labels = _load_cluster_labels(labels_path)

    # 1) Load clause_id -> (cluster_id, cluster_prob).
    assignments: dict[str, tuple[int, float]] = {}
    for obj in _iter_jsonl(clusters_path):
        clause_id = obj.get("clause_id")
        cluster_id = obj.get("cluster_id")
        if not isinstance(clause_id, str) or not clause_id.strip():
            continue
        if not isinstance(cluster_id, int):
            continue
        cid = clause_id.strip()
        if cid in assignments:
            raise SystemExit(f"Duplicate clause_id in clusters.jsonl: {cid}")
        prob_raw = obj.get("cluster_prob", 1.0)
        try:
            prob = float(prob_raw)
        except Exception:
            prob = 1.0
        assignments[cid] = (int(cluster_id), prob)

    if not assignments:
        raise SystemExit(f"No cluster assignments loaded from {clusters_path}")

    # 2) Stream clauses and accumulate per-issue counts.
    issue_meta: dict[str, dict[str, Any]] = {}
    issue_total = Counter()  # total clauses observed (rows in clauses.jsonl)
    issue_missing_cluster = Counter()  # clauses with no cluster assignment
    issue_assigned_total = Counter()  # clauses with cluster assignment (incl noise)
    issue_assigned_noise = Counter()
    issue_assigned_non_noise = Counter()
    issue_cluster_counts: Counter[tuple[str, int]] = Counter()

    processed = 0
    for row in _iter_jsonl(clauses_path):
        if max_clauses and processed >= max_clauses:
            break
        clause_id = row.get("clause_id")
        if not isinstance(clause_id, str) or not clause_id.strip():
            continue
        clause_id = clause_id.strip()

        issue_id = row.get("issue_id") or row.get("doc_id")
        if not isinstance(issue_id, str) or not str(issue_id).strip():
            raise SystemExit(f"Missing issue_id/doc_id for clause_id={clause_id}")
        issue_id = str(issue_id).strip()

        issue_total[issue_id] += 1

        asn = assignments.get(clause_id)
        if asn is None:
            issue_missing_cluster[issue_id] += 1
            processed += 1
            continue

        cluster_id, prob = asn
        if prob < min_prob:
            processed += 1
            continue

        issue_assigned_total[issue_id] += 1

        if cluster_id == -1:
            issue_assigned_noise[issue_id] += 1
            if include_noise:
                issue_cluster_counts[(issue_id, -1)] += 1
            processed += 1
            continue

        issue_assigned_non_noise[issue_id] += 1
        issue_cluster_counts[(issue_id, int(cluster_id))] += 1

        # Stash per-issue metadata once.
        if issue_id not in issue_meta:
            issue_date = row.get("issue_date")
            if not isinstance(issue_date, str) or not issue_date.strip():
                issue_date = _safe_date_from_issue_id(issue_id)
            if not isinstance(issue_date, str) or len(issue_date) < 10:
                raise SystemExit(f"Missing issue_date and cannot parse from issue_id={issue_id!r}")

            slug = row.get("slug")
            if not isinstance(slug, str) or not slug.strip():
                slug = _safe_slug_from_issue_id(issue_id)
            if not isinstance(slug, str) or not slug.strip():
                raise SystemExit(f"Missing slug and cannot parse from issue_id={issue_id!r}")

            issue_meta[issue_id] = {
                "issue_id": issue_id,
                "issue_date": issue_date.strip(),
                "slug": slug.strip(),
                "classification_label": str(row.get("classification_label") or "").strip() or None,
            }

        processed += 1

    if not issue_meta:
        raise SystemExit("No issue metadata collected (unexpected). Check inputs.")

    # 3) Build issue summary panel (one row per issue).
    issue_rows: list[dict[str, Any]] = []
    for issue_id, meta in sorted(issue_meta.items(), key=lambda kv: kv[0]):
        non_noise = int(issue_assigned_non_noise.get(issue_id, 0))
        assigned_total = int(issue_assigned_total.get(issue_id, 0))
        noise = int(issue_assigned_noise.get(issue_id, 0))
        missing = int(issue_missing_cluster.get(issue_id, 0))
        total = int(issue_total.get(issue_id, 0))

        row = {
            **meta,
            "bucket_month": _bucket_from_date(meta["issue_date"], "month"),
            "bucket_year": _bucket_from_date(meta["issue_date"], "year"),
            "n_clauses_total": total,
            "n_clauses_assigned": assigned_total,
            "n_clauses_non_noise": non_noise,
            "n_clauses_noise": noise,
            "n_clauses_missing_cluster": missing,
            # Shares over assigned_total (diagnostics).
            "share_noise_of_assigned": (float(noise) / float(assigned_total)) if assigned_total > 0 else 0.0,
            "share_missing_of_total": (float(missing) / float(total)) if total > 0 else 0.0,
        }
        issue_rows.append(row)

    issues_df = pd.DataFrame.from_records(issue_rows)

    # Optional publication metadata join.
    pub_meta_path = str(args.pub_metadata or "").strip()
    if pub_meta_path:
        pub_meta = _load_pub_metadata(Path(pub_meta_path).expanduser().resolve())
        issues_df = issues_df.merge(pub_meta, left_on="slug", right_on="pub_slug", how="left")

    # 4) Issue x cluster long panel (only non-zero cluster shares).
    long_rows: list[dict[str, Any]] = []
    for (issue_id, cluster_id), n in issue_cluster_counts.items():
        meta = issue_meta.get(issue_id)
        if meta is None:
            continue
        denom = int(issue_assigned_non_noise.get(issue_id, 0)) if cluster_id != -1 else int(issue_assigned_total.get(issue_id, 0))
        share = (float(n) / float(denom)) if denom > 0 else 0.0

        label = cluster_labels.get(int(cluster_id), {}) if cluster_id != -1 else {}
        long_rows.append(
            {
                **meta,
                "bucket_month": _bucket_from_date(meta["issue_date"], "month"),
                "bucket_year": _bucket_from_date(meta["issue_date"], "year"),
                "cluster_id": int(cluster_id),
                "n_clauses_cluster": int(n),
                "cluster_share": float(share),
                "instrument_name": label.get("instrument_name") if label else None,
                "instrument_description": label.get("instrument_description") if label else None,
            }
        )

    long_df = pd.DataFrame.from_records(long_rows)

    # Optional pub meta join for long_df too (kept consistent with issues_df).
    if pub_meta_path and "pub_slug" in issues_df.columns:
        # Join via slug; keep only the location columns from issues_df to avoid duplicating.
        loc_cols = [c for c in ["slug", "city_name", "state_abbr", "city_id", "state_id"] if c in issues_df.columns]
        if loc_cols:
            long_df = long_df.merge(issues_df[loc_cols].drop_duplicates(subset=["slug"]), on="slug", how="left")

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    _write_df(issues_df, out_dir=out_dir, stem=f"rhs_issue_cluster_summary_{stamp}")
    _write_df(long_df, out_dir=out_dir, stem=f"rhs_issue_cluster_long_{stamp}")

    # 5) Aggregated unit x bucket x cluster panel (mean shares, including zeros).
    bucket_col = "bucket_month" if bucket == "month" else "bucket_year"

    if unit == "paper":
        unit_cols = ["slug"]
    elif unit == "city_state":
        required = ["city_name", "state_abbr"]
        missing = [c for c in required if c not in issues_df.columns]
        if missing:
            raise SystemExit(
                f"--unit=city_state requires pub metadata with columns {required}; missing: {missing}. "
                "Pass --pub-metadata pointing at a publication locations file."
            )
        unit_cols = required
    elif unit == "state":
        if "state_abbr" not in issues_df.columns:
            raise SystemExit("--unit=state requires pub metadata with column state_abbr; pass --pub-metadata.")
        unit_cols = ["state_abbr"]
    else:
        raise SystemExit(f"Unknown --unit: {unit}")

    group_cols = unit_cols + [bucket_col]
    if bool(args.by_classification_label):
        group_cols.append("classification_label")

    # n_issues per group (denominator for mean including zeros).
    denom = issues_df.dropna(subset=group_cols).groupby(group_cols, as_index=False).agg({"issue_id": "nunique"})
    denom = denom.rename(columns={"issue_id": "n_issues"})

    # Sum of cluster_share across issues for each (group, cluster_id).
    numer = long_df.dropna(subset=group_cols).groupby(group_cols + ["cluster_id"], as_index=False).agg(
        {"cluster_share": "sum", "issue_id": "nunique", "n_clauses_cluster": "sum"}
    )
    numer = numer.rename(columns={"issue_id": "n_issues_with_cluster", "cluster_share": "sum_cluster_share"})

    agg = numer.merge(denom, on=group_cols, how="left")
    agg["mean_cluster_share"] = agg["sum_cluster_share"] / agg["n_issues"].astype(float)
    agg["mean_cluster_share_conditional"] = agg["sum_cluster_share"] / agg["n_issues_with_cluster"].astype(float)

    agg_stem = f"rhs_{unit}_{bucket}_panel_long_{stamp}"
    _write_df(agg, out_dir=out_dir, stem=agg_stem)

    # README for provenance.
    readme = out_dir / f"README_{stamp}.md"
    readme.write_text(
        "\n".join(
            [
                "# RHS from instrument clusters",
                "",
                f"Created: {stamp}",
                "",
                "Inputs:",
                f"- clauses_jsonl: {clauses_path}",
                f"- clusters_jsonl: {clusters_path}",
                f"- cluster_labels_jsonl: {labels_path if labels_path else '(none)'}",
                f"- pub_metadata: {pub_meta_path or '(none)'}",
                "",
                "Params:",
                f"- bucket: {bucket}",
                f"- unit: {unit}",
                f"- by_classification_label: {bool(args.by_classification_label)}",
                f"- include_noise: {include_noise}",
                f"- min_prob: {min_prob}",
                f"- max_clauses: {max_clauses}",
                "",
                "Outputs:",
                f"- rhs_issue_cluster_summary_{stamp}.parquet/.csv",
                f"- rhs_issue_cluster_long_{stamp}.parquet/.csv",
                f"- rhs_{unit}_{bucket}_panel_long_{stamp}.parquet/.csv",
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Done. issues={len(issues_df)} long_rows={len(long_df)} agg_rows={len(agg)} out={out_dir}")


if __name__ == "__main__":
    main()

