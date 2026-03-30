#!/usr/bin/env python3
"""
Build RHS feature panels for downstream regressions (e.g., diff-in-diff) from:
  - chunk-topic clustering outputs (UMAP/HDBSCAN clusters)
  - LLM cluster topic labels (v2 mechanics schema preferred)
  - optional newspaper metadata (publication -> city/state)

This is intentionally "artifact-first": it reads the pipeline outputs on disk and emits
machine-readable feature tables (Parquet/CSV) you can take straight into R/Stata/Python.

Per run root (issue-topic pipeline output), we expect:
  <RUN_ROOT>/
    clusters_chunks/clusters.jsonl
    <CLUSTER_LABEL_DIR>/outputs/cluster_labels.jsonl

The key idea:
  - Each issue is split into chunks.
  - Each chunk has a cluster_id and a doc_weight (1 / #chunks_in_issue).
  - Each cluster_id is labeled by the LLM with structured "mechanics tags".
  - We aggregate doc_weight mass by tag to produce interpretable, stable RHS covariates.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal

import pandas as pd

from newsvlm_analysis.cluster_topic_label_schema import (
    MECHANICS_ACTION_TAGS,
    MECHANICS_DECISION_BODY_TAGS,
    MECHANICS_DIMENSION_TAGS,
    MECHANICS_INSTRUMENT_TAGS,
    MECHANICS_ZONING_RELEVANCE,
)


Bucket = Literal["month", "year"]
MechanicsFilter = Literal["any", "zoning_or_mixed", "zoning_only"]


@dataclass(frozen=True)
class RunSpec:
    label: str
    root: Path


def _parse_run_spec(raw: str) -> RunSpec:
    # Format: LABEL:/path/to/run_root
    if ":" not in raw:
        raise SystemExit(f"Invalid --run '{raw}'. Expected LABEL:/path/to/run_root")
    label, path = raw.split(":", 1)
    label = label.strip()
    if not label:
        raise SystemExit(f"Invalid --run '{raw}': empty label")
    root = Path(path).expanduser().resolve()
    return RunSpec(label=label, root=root)


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.is_file():
        raise SystemExit(f"Missing required file: {path}")
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            yield json.loads(line)


def _bucket_from_date(date_s: str, bucket: Bucket) -> str:
    # date_s: YYYY-MM-DD
    if not isinstance(date_s, str) or len(date_s) < 10:
        raise ValueError(f"Bad issue_date: {date_s!r}")
    if bucket == "year":
        return date_s[:4]
    if bucket == "month":
        return date_s[:7]
    raise ValueError(f"Unknown bucket: {bucket}")


def _month_index(bucket_ym: str) -> int:
    # bucket_ym: YYYY-MM
    y, m = bucket_ym.split("-", 1)
    return int(y) * 12 + (int(m) - 1)


def _year_index(bucket_y: str) -> int:
    return int(bucket_y)


def _safe_slug_from_issue_id(issue_id: str) -> str:
    if not isinstance(issue_id, str) or not issue_id.strip():
        return ""
    # Our pipeline uses `<slug>__<YYYY-MM-DD>`; keep it robust.
    return issue_id.split("__", 1)[0].strip()


def _load_pub_metadata(path: Path) -> pd.DataFrame:
    """
    Load publication metadata (at minimum pub_slug). We support:
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

    # Normalize expected columns.
    if "pub_slug" not in df.columns:
        raise SystemExit(f"pub metadata missing required column pub_slug: {path}")

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


def _mechanics_filter_set(mode: MechanicsFilter) -> set[str]:
    allowed = set(MECHANICS_ZONING_RELEVANCE)
    if mode == "any":
        return allowed
    if mode == "zoning_or_mixed":
        return {"zoning", "mixed"}
    if mode == "zoning_only":
        return {"zoning"}
    raise ValueError(f"Unknown mechanics_filter: {mode}")


def _feature_columns(*, prefix: str, tags: tuple[str, ...]) -> list[str]:
    return [f"{prefix}{t}" for t in tags]


def _load_cluster_mechanics_map(labels_jsonl: Path) -> dict[int, dict[str, Any]]:
    """
    cluster_id -> {zoning_relevance, action_tags, dimension_tags, decision_body_tags, instrument_tags}
    Skips rows with errors.
    """
    out: dict[int, dict[str, Any]] = {}
    for row in _iter_jsonl(labels_jsonl):
        if row.get("error") is not None:
            continue
        cid = row.get("cluster_id")
        if not isinstance(cid, int):
            continue
        zr = row.get("zoning_relevance")
        if not isinstance(zr, str) or zr not in set(MECHANICS_ZONING_RELEVANCE):
            zr = "unknown"

        mechanics = row.get("mechanics") if isinstance(row.get("mechanics"), dict) else {}
        # v1 labels won't have mechanics; treat as empty.
        def _get_list(name: str) -> list[str]:
            v = mechanics.get(name) if isinstance(mechanics, dict) else None
            if not isinstance(v, list):
                return []
            return [str(x).strip() for x in v if str(x).strip()]

        out[int(cid)] = {
            "zoning_relevance": zr,
            "action_tags": _get_list("action_tags"),
            "dimension_tags": _get_list("dimension_tags"),
            "decision_body_tags": _get_list("decision_body_tags"),
            "instrument_tags": _get_list("instrument_tags"),
        }
    if not out:
        raise SystemExit(f"No valid cluster labels loaded from {labels_jsonl}")
    return out


def _build_issue_features_for_run(
    *,
    run: RunSpec,
    cluster_label_dir_name: str,
    mechanics_filter: MechanicsFilter,
) -> pd.DataFrame:
    clusters_jsonl = run.root / "clusters_chunks" / "clusters.jsonl"
    labels_jsonl = run.root / cluster_label_dir_name / "outputs" / "cluster_labels.jsonl"
    if not clusters_jsonl.is_file():
        raise SystemExit(f"Missing clusters file: {clusters_jsonl}")
    if not labels_jsonl.is_file():
        raise SystemExit(f"Missing cluster labels file: {labels_jsonl}")

    mechanics_by_cluster = _load_cluster_mechanics_map(labels_jsonl)
    allowed_zr = _mechanics_filter_set(mechanics_filter)

    action_tag_set = set(MECHANICS_ACTION_TAGS)
    dim_tag_set = set(MECHANICS_DIMENSION_TAGS)
    body_tag_set = set(MECHANICS_DECISION_BODY_TAGS)
    instr_tag_set = set(MECHANICS_INSTRUMENT_TAGS)

    action_cols = _feature_columns(prefix="action__", tags=MECHANICS_ACTION_TAGS)
    dim_cols = _feature_columns(prefix="dim__", tags=MECHANICS_DIMENSION_TAGS)
    body_cols = _feature_columns(prefix="body__", tags=MECHANICS_DECISION_BODY_TAGS)
    instr_cols = _feature_columns(prefix="instr__", tags=MECHANICS_INSTRUMENT_TAGS)
    feature_cols = action_cols + dim_cols + body_cols + instr_cols

    # issue_id -> accumulator row dict
    acc: dict[str, dict[str, Any]] = {}

    def _ensure_issue(issue_id: str) -> dict[str, Any]:
        row = acc.get(issue_id)
        if row is not None:
            return row
        row = {"issue_id": issue_id}
        # Pre-seed columns so we always emit stable schemas.
        for c in feature_cols:
            row[c] = 0.0
        row.update(
            {
                "issue_date": None,
                "bucket_month": None,
                "bucket_year": None,
                "classification_label": run.label,
                "slug": _safe_slug_from_issue_id(issue_id),
                "doc_weight_total": 0.0,
                "doc_weight_noise": 0.0,
                "doc_weight_missing_cluster_label": 0.0,
                "zr__zoning": 0.0,
                "zr__mixed": 0.0,
                "zr__non_zoning": 0.0,
                "zr__unknown": 0.0,
                "chunks_total": 0,
                "chunk_chars_total": 0,
                "chunk_count": None,
            }
        )
        acc[issue_id] = row
        return row

    for obj in _iter_jsonl(clusters_jsonl):
        issue_id = obj.get("issue_id")
        if not isinstance(issue_id, str) or not issue_id.strip():
            continue
        issue_id = issue_id.strip()
        row = _ensure_issue(issue_id)

        issue_date = obj.get("issue_date")
        if isinstance(issue_date, str) and issue_date.strip():
            # keep first observed value (should be constant)
            if row["issue_date"] is None:
                row["issue_date"] = issue_date
                try:
                    row["bucket_month"] = _bucket_from_date(issue_date, bucket="month")
                    row["bucket_year"] = _bucket_from_date(issue_date, bucket="year")
                except Exception:
                    # Leave buckets null; fail later if needed.
                    pass

        # Prefer the explicit label in the file if present.
        lab = obj.get("classification_label")
        if isinstance(lab, str) and lab.strip():
            row["classification_label"] = lab.strip()

        w = obj.get("doc_weight")
        if not isinstance(w, (int, float)) or not math.isfinite(float(w)):
            continue
        w = float(w)
        row["doc_weight_total"] += w
        row["chunks_total"] += 1

        ch = obj.get("chunk_chars")
        if isinstance(ch, int) and ch >= 0:
            row["chunk_chars_total"] += int(ch)

        cc = obj.get("chunk_count")
        if isinstance(cc, int) and cc > 0:
            prev = row.get("chunk_count")
            if prev is None:
                row["chunk_count"] = int(cc)
            elif int(prev) != int(cc):
                raise SystemExit(f"Inconsistent chunk_count for issue_id={issue_id}: saw {prev} then {cc} in {clusters_jsonl}")

        cluster_id = obj.get("cluster_id")
        if not isinstance(cluster_id, int):
            continue
        if int(cluster_id) == -1:
            row["doc_weight_noise"] += w
            continue

        mech = mechanics_by_cluster.get(int(cluster_id))
        if mech is None:
            row["doc_weight_missing_cluster_label"] += w
            row["zr__unknown"] += w
            continue

        zr = mech.get("zoning_relevance")
        if zr in {"zoning", "mixed", "non_zoning"}:
            row[f"zr__{zr}"] += w
        else:
            row["zr__unknown"] += w

        # Only attribute mechanics-tag features for clusters that are zoning-relevant (configurable).
        if zr not in allowed_zr:
            continue

        for t in mech.get("action_tags", []):
            if t in action_tag_set:
                row[f"action__{t}"] += w
        for t in mech.get("dimension_tags", []):
            if t in dim_tag_set:
                row[f"dim__{t}"] += w
        for t in mech.get("decision_body_tags", []):
            if t in body_tag_set:
                row[f"body__{t}"] += w
        for t in mech.get("instrument_tags", []):
            if t in instr_tag_set:
                row[f"instr__{t}"] += w

    if not acc:
        raise SystemExit(f"No issue rows produced for run={run.label} from {clusters_jsonl}")

    df = pd.DataFrame.from_records(list(acc.values()))
    # Basic sanity: doc_weight_total should be ~ (#issues) if doc_weight sums to 1 per issue.
    # We don't enforce hard equality; just keep it for diagnostics.
    return df


def _write_df(df: pd.DataFrame, *, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = out_dir / f"{stem}.parquet"
    csv_path = out_dir / f"{stem}.csv"
    df.to_parquet(parquet_path, index=False)
    df.to_csv(csv_path, index=False, quoting=csv.QUOTE_MINIMAL)


def _add_event_time(
    panel: pd.DataFrame,
    *,
    unit_cols: list[str],
    bucket_col: str,
    bucket: Bucket,
    event_feature: str,
    threshold: float,
) -> pd.DataFrame:
    if event_feature not in panel.columns:
        raise SystemExit(f"--event-feature not found in panel columns: {event_feature}")
    if bucket_col not in panel.columns:
        raise SystemExit(f"Internal error: missing bucket_col {bucket_col}")
    for c in unit_cols:
        if c not in panel.columns:
            raise SystemExit(f"Internal error: missing unit col {c}")

    # Find first bucket per unit where event_feature >= threshold.
    keys = unit_cols
    tmp = panel[keys + [bucket_col, event_feature]].copy()
    tmp = tmp.dropna(subset=keys + [bucket_col])
    tmp = tmp[tmp[event_feature].astype(float) >= float(threshold)]
    if tmp.empty:
        panel = panel.copy()
        panel["event_start_bucket"] = None
        panel["treated"] = 0
        panel["post"] = 0
        panel["event_time"] = None
        return panel

    tmp = tmp.sort_values(keys + [bucket_col])
    first = tmp.groupby(keys, as_index=False).first()[keys + [bucket_col]].rename(columns={bucket_col: "event_start_bucket"})

    out = panel.merge(first, on=keys, how="left")
    out["treated"] = out["event_start_bucket"].notna().astype(int)

    def _time_index(b: str) -> int:
        return _month_index(b) if bucket == "month" else _year_index(b)

    # Compute event_time as integer bucket diffs.
    def _compute_event_time(row: pd.Series) -> int | None:
        b = row.get(bucket_col)
        s = row.get("event_start_bucket")
        if not isinstance(b, str) or not isinstance(s, str) or not b or not s:
            return None
        try:
            return int(_time_index(b) - _time_index(s))
        except Exception:
            return None

    out["event_time"] = out.apply(_compute_event_time, axis=1)
    out["post"] = out["event_time"].apply(lambda x: int(isinstance(x, int) and x >= 0))
    return out


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build RHS regression features from clustering + mechanics labels.")
    ap.add_argument("--run", action="append", required=True, help="Run spec: LABEL:/path/to/run_root (repeatable).")
    ap.add_argument("--output-dir", required=True, help="Directory to write outputs into.")
    ap.add_argument(
        "--cluster-label-dir-name",
        default="cluster_topic_labels",
        help="Subdirectory under each run root that contains outputs/cluster_labels.jsonl",
    )
    ap.add_argument(
        "--mechanics-filter",
        choices=["any", "zoning_or_mixed", "zoning_only"],
        default="zoning_or_mixed",
        help="Which zoning_relevance values are allowed to contribute to mechanics-tag features.",
    )
    ap.add_argument(
        "--pub-metadata",
        default="",
        help="Optional publication metadata file (parquet/jsonl/csv) with at least pub_slug.",
    )
    ap.add_argument(
        "--bucket",
        choices=["month", "year"],
        default="month",
        help="Time bucket for the aggregated panel.",
    )
    ap.add_argument(
        "--unit",
        choices=["paper", "city_state", "state"],
        default="paper",
        help="Panel unit for aggregation. paper=pub_slug (from issue_id).",
    )
    ap.add_argument(
        "--event-feature",
        default="",
        help=(
            "Optional: derive event-time variables using the first bucket where this feature exceeds --event-threshold. "
            "Example: action__amend_map_rezone"
        ),
    )
    ap.add_argument("--event-threshold", type=float, default=0.20, help="Threshold for --event-feature (mean share).")
    ap.add_argument(
        "--by-classification-label",
        action="store_true",
        help="Include classification_label in the aggregated panel grouping keys.",
    )
    ap.add_argument(
        "--emit-deltas",
        action="store_true",
        help=(
            "Also emit first-difference columns in the aggregated panel (within each unit over time). "
            "Columns are prefixed with d_ (e.g., d_action__amend_map_rezone)."
        ),
    )
    return ap.parse_args()


def main() -> None:
    args = _parse_args()

    out_dir = Path(args.output_dir).expanduser().resolve()
    run_specs = [_parse_run_spec(r) for r in args.run]
    run_specs.sort(key=lambda r: r.label)

    mechanics_filter: MechanicsFilter = str(args.mechanics_filter)  # type: ignore[assignment]
    cluster_label_dir_name = str(args.cluster_label_dir_name)

    # 1) Issue-level features (one row per issue per run label).
    issue_frames: list[pd.DataFrame] = []
    for run in run_specs:
        df = _build_issue_features_for_run(
            run=run,
            cluster_label_dir_name=cluster_label_dir_name,
            mechanics_filter=mechanics_filter,
        )
        # Keep provenance.
        df["run_root"] = str(run.root)
        df["run_label"] = run.label
        issue_frames.append(df)

    issues = pd.concat(issue_frames, ignore_index=True)

    # Optional publication metadata join.
    pub_meta_path = str(args.pub_metadata or "").strip()
    if pub_meta_path:
        pub_meta = _load_pub_metadata(Path(pub_meta_path).expanduser().resolve())
        issues = issues.merge(pub_meta, left_on="slug", right_on="pub_slug", how="left")

    # Write issue-level panel.
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    issues_stem = f"rhs_issue_panel_{stamp}"
    _write_df(issues, out_dir=out_dir, stem=issues_stem)

    # 2) Aggregated panel: unit x bucket.
    bucket: Bucket = str(args.bucket)  # type: ignore[assignment]
    if bucket == "month":
        bucket_col = "bucket_month"
    else:
        bucket_col = "bucket_year"

    unit = str(args.unit)
    if unit == "paper":
        unit_cols = ["slug"]
    elif unit == "city_state":
        # Requires pub metadata join with city/state columns.
        required = ["city_name", "state_abbr"]
        missing = [c for c in required if c not in issues.columns]
        if missing:
            raise SystemExit(
                f"--unit=city_state requires pub metadata with columns {required}; missing: {missing}. "
                "Pass --pub-metadata pointing at newspaper-scrapping locations_headful.(parquet|jsonl)."
            )
        unit_cols = required
    elif unit == "state":
        if "state_abbr" not in issues.columns:
            raise SystemExit(
                "--unit=state requires pub metadata with column state_abbr; pass --pub-metadata pointing at locations_headful.(parquet|jsonl)."
            )
        unit_cols = ["state_abbr"]
    else:
        raise SystemExit(f"Unknown --unit: {unit}")

    group_cols = unit_cols + [bucket_col]
    if bool(args.by_classification_label):
        group_cols.append("classification_label")

    # Identify feature columns (mechanics tags + quality diagnostics).
    feature_prefixes = ("action__", "dim__", "body__", "instr__", "zr__", "doc_weight_", "chunks_total", "chunk_chars_total")
    feature_cols = [c for c in issues.columns if any(c.startswith(p) for p in feature_prefixes)]

    # Mean share features across issues in the bucket.
    agg = issues.dropna(subset=group_cols).groupby(group_cols, as_index=False).agg(
        {
            **{c: "mean" for c in feature_cols if c not in {"chunks_total", "chunk_chars_total"}},
            "chunks_total": "sum",
            "chunk_chars_total": "sum",
            "issue_id": "nunique",
        }
    )
    agg = agg.rename(columns={"issue_id": "n_issues"})

    # Enrich the city_state panel with stable numeric IDs if they exist in the pub metadata.
    if unit == "city_state" and {"state_id", "city_id"}.issubset(set(issues.columns)):
        loc_cols = ["city_name", "state_abbr", "state_id", "city_id"]
        loc = issues.dropna(subset=["city_name", "state_abbr"])[loc_cols].drop_duplicates()
        # Fail loudly if the metadata implies ambiguous mapping.
        loc_counts = loc.groupby(["city_name", "state_abbr"], as_index=False).agg(
            {"state_id": "nunique", "city_id": "nunique"}
        )
        if bool((loc_counts["state_id"] > 1).any() or (loc_counts["city_id"] > 1).any()):
            raise SystemExit("Ambiguous mapping for (city_name,state_abbr)->(state_id,city_id) in pub metadata join.")
        agg = agg.merge(loc, on=["city_name", "state_abbr"], how="left")

    # Optional: first differences within unit over time (useful for DiD-style specifications).
    if bool(args.emit_deltas):
        key_cols = unit_cols + (["classification_label"] if bool(args.by_classification_label) else [])
        if bucket == "month":
            agg["_bucket_idx"] = agg[bucket_col].astype(str).apply(_month_index)
        else:
            agg["_bucket_idx"] = agg[bucket_col].astype(str).apply(_year_index)

        # Only delta numeric feature columns (avoid grouping keys and metadata).
        delta_candidates = [
            c
            for c in agg.columns
            if any(c.startswith(p) for p in ("action__", "dim__", "body__", "instr__", "zr__"))
        ]
        agg = agg.sort_values(key_cols + ["_bucket_idx"])
        for c in delta_candidates:
            agg[f"d_{c}"] = agg.groupby(key_cols, sort=False)[c].diff()
        agg = agg.drop(columns=["_bucket_idx"])

    # Optional event-time variables.
    event_feature = str(args.event_feature or "").strip()
    if event_feature:
        agg = _add_event_time(
            agg,
            unit_cols=unit_cols + (["classification_label"] if bool(args.by_classification_label) else []),
            bucket_col=bucket_col,
            bucket=bucket,
            event_feature=event_feature,
            threshold=float(args.event_threshold),
        )

    agg_stem = f"rhs_{unit}_{bucket}_panel_{stamp}"
    _write_df(agg, out_dir=out_dir, stem=agg_stem)

    # Small README stub for provenance.
    readme = out_dir / f"README_{stamp}.md"
    readme.write_text(
        "\n".join(
            [
                "# RHS panel outputs",
                "",
                f"Created: {stamp}",
                "",
                "Inputs:",
                f"- cluster_label_dir_name: {cluster_label_dir_name}",
                f"- mechanics_filter: {mechanics_filter}",
                f"- bucket: {bucket}",
                f"- unit: {unit}",
                f"- by_classification_label: {bool(args.by_classification_label)}",
                f"- emit_deltas: {bool(args.emit_deltas)}",
                f"- event_feature: {event_feature or '(none)'}",
                f"- event_threshold: {float(args.event_threshold):.4f}",
                f"- pub_metadata: {pub_meta_path or '(none)'}",
                f"- runs: {len(run_specs)}",
                *[f"  - {r.label}: {r.root}" for r in run_specs],
                "",
                "Outputs:",
                f"- {issues_stem}.parquet / .csv (issue-level features)",
                f"- {agg_stem}.parquet / .csv (aggregated panel features)",
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Done. wrote={out_dir} issue_rows={len(issues)} panel_rows={len(agg)}")


if __name__ == "__main__":
    main()
