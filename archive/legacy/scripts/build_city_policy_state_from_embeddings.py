#!/usr/bin/env python3
"""
Build city-by-time "policy state" features from chunk embeddings.

This is intended for downstream panel regressions (DiD/event studies), but does NOT
choose a treatment definition. Instead it produces:
  - low-dimensional policy-state indices (PCA scores) at issue and city×time levels
  - change/intensity measures (within-city jump sizes over time)
  - optional step-function "active policy state" via forward fill

Inputs (per run root):
  <RUN_ROOT>/
    clusters_chunks/clusters.jsonl          # chunk_id, issue_id, issue_date, doc_weight, chunk_chars, ...
    embeddings_chunks/chunk_embeddings.npy  # aligned to embeddings_chunks/chunk_ids.txt
    embeddings_chunks/chunk_ids.txt

Important: this script is artifact-first and deterministic. It does not call an LLM.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


Bucket = Literal["month", "year"]
StateMode = Literal["observed", "step"]
Unit = Literal["paper", "city_state", "state"]


@dataclass(frozen=True)
class RunSpec:
    label: str
    root: Path


def _parse_run_spec(raw: str) -> RunSpec:
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
    # date_s expected: YYYY-MM-DD
    if not isinstance(date_s, str) or len(date_s) < 10:
        raise ValueError(f"Bad issue_date: {date_s!r}")
    if bucket == "year":
        return date_s[:4]
    if bucket == "month":
        return date_s[:7]
    raise ValueError(f"Unknown bucket: {bucket}")


def _month_index(bucket_ym: str) -> int:
    y, m = bucket_ym.split("-", 1)
    return int(y) * 12 + (int(m) - 1)


def _year_index(bucket_y: str) -> int:
    return int(bucket_y)


def _safe_slug_from_issue_id(issue_id: str) -> str:
    if not isinstance(issue_id, str) or not issue_id.strip():
        return ""
    return issue_id.split("__", 1)[0].strip()


def _load_pub_metadata(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise SystemExit(f"--pub-metadata not found: {path}")
    suf = path.suffix.lower()
    if suf == ".parquet":
        try:
            df = pd.read_parquet(path)
        except Exception as e:  # noqa: BLE001
            raise SystemExit(
                f"Failed to read parquet pub metadata (missing pyarrow/engine?): {path}\n"
                f"Error: {e}\n"
                "Fix: either install pyarrow in this environment or pass --pub-metadata as CSV/JSONL."
            ) from e
    elif suf in {".jsonl", ".json"}:
        df = pd.read_json(path, lines=True)
    elif suf == ".csv":
        df = pd.read_csv(path)
    else:
        raise SystemExit(f"Unsupported --pub-metadata format: {path} (expected parquet/jsonl/csv)")

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


def _load_chunk_id_order(path: Path) -> list[str]:
    if not path.is_file():
        raise SystemExit(f"Missing required file: {path}")
    ids: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            s = raw.strip()
            if not s:
                continue
            ids.append(s)
    if not ids:
        raise SystemExit(f"Empty chunk_ids file: {path}")
    return ids


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    # 1 - cosine similarity; returns 0 for identical direction, 2 for opposite.
    an = float(np.linalg.norm(a))
    bn = float(np.linalg.norm(b))
    if an == 0.0 or bn == 0.0:
        return float("nan")
    return float(1.0 - float(np.dot(a, b) / (an * bn)))


def _parse_chunk_idx(chunk_id: str) -> int | None:
    # Expect chunk_id like "...::chunk000" or "...::chunk012"
    m = re.search(r"::chunk(\d+)$", chunk_id)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _load_clusters_for_join(clusters_jsonl: Path) -> pd.DataFrame:
    rows = []
    for obj in _iter_jsonl(clusters_jsonl):
        chunk_id = obj.get("chunk_id")
        issue_id = obj.get("issue_id")
        issue_date = obj.get("issue_date")
        if not isinstance(chunk_id, str) or not chunk_id.strip():
            continue
        if not isinstance(issue_id, str) or not issue_id.strip():
            continue
        if not isinstance(issue_date, str) or not issue_date.strip():
            continue
        w = obj.get("doc_weight")
        if not isinstance(w, (int, float)) or not math.isfinite(float(w)):
            continue
        cc = obj.get("chunk_chars")
        if not isinstance(cc, int) or cc <= 0:
            # chunk_chars is used only for diagnostics; keep as NaN if absent.
            cc = None
        rows.append(
            {
                "chunk_id": chunk_id.strip(),
                "chunk_idx": _parse_chunk_idx(chunk_id.strip()),
                "issue_id": issue_id.strip(),
                "issue_date": issue_date.strip(),
                "doc_weight": float(w),
                "chunk_chars": cc,
                "cluster_id": obj.get("cluster_id"),
                "classification_label": obj.get("classification_label"),
            }
        )
    if not rows:
        raise SystemExit(f"No usable rows found in {clusters_jsonl}")
    df = pd.DataFrame.from_records(rows)
    return df


def _build_issue_embeddings_for_run(run: RunSpec) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Returns:
      - issues_df: one row per issue with issue_date, slug, classification_label, run_label, and embedding index.
      - issue_embeddings: float32 matrix [n_issues, dim] aligned to issues_df rows.
    """
    clusters_jsonl = run.root / "clusters_chunks" / "clusters.jsonl"
    emb_dir = run.root / "embeddings_chunks"
    chunk_ids_path = emb_dir / "chunk_ids.txt"
    emb_path = emb_dir / "chunk_embeddings.npy"

    if not clusters_jsonl.is_file():
        raise SystemExit(f"Missing clusters file: {clusters_jsonl}")
    if not chunk_ids_path.is_file():
        raise SystemExit(f"Missing chunk_ids file: {chunk_ids_path}")
    if not emb_path.is_file():
        raise SystemExit(f"Missing chunk_embeddings file: {emb_path}")

    clusters = _load_clusters_for_join(clusters_jsonl)
    chunk_ids = _load_chunk_id_order(chunk_ids_path)

    # Map chunk_id -> embedding row index.
    idx_by_chunk: dict[str, int] = {}
    for i, cid in enumerate(chunk_ids):
        if cid in idx_by_chunk:
            raise SystemExit(f"Duplicate chunk_id in {chunk_ids_path}: {cid}")
        idx_by_chunk[cid] = i

    # Keep only chunks with embeddings.
    clusters["emb_idx"] = clusters["chunk_id"].map(idx_by_chunk)
    missing = int(clusters["emb_idx"].isna().sum())
    if missing:
        raise SystemExit(f"{missing} chunks in {clusters_jsonl} missing embeddings in {chunk_ids_path}")

    # Open embeddings as a memmap to avoid loading the whole file at once.
    emb = np.load(emb_path, mmap_mode="r")
    if emb.ndim != 2:
        raise SystemExit(f"Expected 2D embedding matrix in {emb_path}, got shape={emb.shape}")
    dim = int(emb.shape[1])

    # Aggregate embeddings into issue-level mean vectors using doc_weight (sums to ~1 per issue).
    issues = clusters[["issue_id", "issue_date", "classification_label"]].drop_duplicates().copy()
    issues["slug"] = issues["issue_id"].apply(_safe_slug_from_issue_id)
    issues["run_label"] = run.label
    issues["run_root"] = str(run.root)

    # Preserve deterministic issue order.
    issues = issues.sort_values(["issue_date", "issue_id"]).reset_index(drop=True)
    issue_row_by_id = {iid: i for i, iid in enumerate(issues["issue_id"].tolist())}
    n_issues = int(len(issues))

    issue_emb = np.zeros((n_issues, dim), dtype=np.float32)
    weight_sum = np.zeros((n_issues,), dtype=np.float32)
    chunk_count = np.zeros((n_issues,), dtype=np.int32)

    # Stream through clusters rows and accumulate.
    for obj in clusters[["issue_id", "emb_idx", "doc_weight"]].itertuples(index=False):
        iid = str(obj.issue_id)
        row = issue_row_by_id.get(iid)
        if row is None:
            continue
        ei = int(obj.emb_idx)
        w = float(obj.doc_weight)
        issue_emb[row] += (w * emb[ei]).astype(np.float32, copy=False)
        weight_sum[row] += np.float32(w)
        chunk_count[row] += 1

    issues["doc_weight_sum"] = weight_sum.astype(np.float32)
    issues["chunks_in_issue"] = chunk_count.astype(np.int32)

    # Sanity: many issues should have weight_sum ~ 1.0; record as diagnostics (do not enforce).
    return issues, issue_emb


def _write_df(df: pd.DataFrame, *, out_dir: Path, stem: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{stem}.csv"
    df.to_csv(csv_path, index=False, quoting=csv.QUOTE_MINIMAL)
    pq = out_dir / f"{stem}.parquet"
    return pq


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build city×time policy state from chunk embeddings.")
    ap.add_argument("--run", action="append", required=True, help="Run spec: LABEL:/path/to/run_root (repeatable).")
    ap.add_argument("--output-dir", required=True, help="Directory to write outputs into.")
    ap.add_argument(
        "--pub-metadata",
        required=True,
        help="Publication metadata file (parquet/jsonl/csv) with at least pub_slug and city_name/state_abbr.",
    )
    ap.add_argument("--bucket", choices=["month", "year"], default="month", help="Time bucket for the city panel.")
    ap.add_argument("--unit", choices=["paper", "city_state", "state"], default="city_state", help="Panel unit.")
    ap.add_argument(
        "--state-mode",
        choices=["observed", "step"],
        default="observed",
        help="observed = only buckets with observed issues; step = forward-fill the latest observed state per unit.",
    )
    ap.add_argument(
        "--pca-components",
        type=int,
        default=10,
        help="How many PCA components to keep as low-dimensional indices (recommended: 5–20).",
    )
    ap.add_argument(
        "--by-classification-label",
        action="store_true",
        help="Keep classification_label separate (unit×time×label), instead of pooling them together.",
    )
    ap.add_argument(
        "--output-format",
        choices=["csv", "parquet", "both"],
        default="both",
        help=(
            "Output file format. "
            "parquet requires a parquet engine (pyarrow/fastparquet). "
            "csv is always available but larger."
        ),
    )
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    run_specs = [_parse_run_spec(r) for r in args.run]
    run_specs.sort(key=lambda r: r.label)

    pub_meta = _load_pub_metadata(Path(args.pub_metadata).expanduser().resolve())

    # 1) Build issue embeddings per run root and concatenate.
    issue_frames: list[pd.DataFrame] = []
    issue_embs: list[np.ndarray] = []
    for run in run_specs:
        issues_df, issue_emb = _build_issue_embeddings_for_run(run)
        issue_frames.append(issues_df)
        issue_embs.append(issue_emb)

    issues = pd.concat(issue_frames, ignore_index=True)
    emb = np.vstack(issue_embs)
    if int(len(issues)) != int(emb.shape[0]):
        raise SystemExit("Internal error: issue rows and embedding rows disagree")

    # 2) Join publication metadata -> city/state.
    issues = issues.merge(pub_meta, left_on="slug", right_on="pub_slug", how="left")

    unit: Unit = str(args.unit)  # type: ignore[assignment]
    if unit == "paper":
        unit_cols = ["slug"]
    elif unit == "city_state":
        required = ["city_name", "state_abbr"]
        missing = [c for c in required if c not in issues.columns]
        if missing:
            raise SystemExit(f"--unit=city_state requires pub metadata columns {required}; missing: {missing}")
        unit_cols = required
    elif unit == "state":
        if "state_abbr" not in issues.columns:
            raise SystemExit("--unit=state requires pub metadata column state_abbr")
        unit_cols = ["state_abbr"]
    else:
        raise SystemExit(f"Unknown --unit: {unit}")

    bucket: Bucket = str(args.bucket)  # type: ignore[assignment]
    bucket_col = "bucket_month" if bucket == "month" else "bucket_year"
    issues[bucket_col] = issues["issue_date"].astype(str).apply(lambda s: _bucket_from_date(s, bucket=bucket))

    # 3) PCA over issue embeddings -> low-dimensional indices.
    n_components = int(args.pca_components)
    if n_components <= 0:
        raise SystemExit("--pca-components must be > 0")
    if n_components > int(emb.shape[1]):
        raise SystemExit(f"--pca-components too large: {n_components} > embedding_dim={emb.shape[1]}")

    # Use randomized SVD for speed; keep deterministic output.
    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=0)
    scores = pca.fit_transform(emb.astype(np.float32, copy=False))

    # Record PCA diagnostics.
    pca_info = {
        "created_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "n_issues": int(len(issues)),
        "embedding_dim": int(emb.shape[1]),
        "n_components": int(n_components),
        "explained_variance_ratio": [float(x) for x in pca.explained_variance_ratio_],
    }
    (out_dir / "pca_info.json").write_text(json.dumps(pca_info, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    np.savez_compressed(
        out_dir / "pca_model.npz",
        components_=pca.components_.astype(np.float32),
        mean_=pca.mean_.astype(np.float32),
        explained_variance_ratio_=pca.explained_variance_ratio_.astype(np.float32),
    )

    for i in range(n_components):
        issues[f"pc{i+1}"] = scores[:, i].astype(np.float32)

    issues["emb_norm"] = np.linalg.norm(emb, axis=1).astype(np.float32)

    # 4) Write issue-level indices (no giant vectors).
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    issues_stem = f"issue_policy_indices_{stamp}"
    issues_pq = _write_df(issues, out_dir=out_dir, stem=issues_stem)
    if str(args.output_format) in {"parquet", "both"}:
        try:
            issues.to_parquet(issues_pq, index=False)
        except Exception as e:  # noqa: BLE001
            raise SystemExit(
                f"Failed to write parquet output: {issues_pq}\n"
                f"Error: {e}\n"
                "Fix: install pyarrow (recommended) or rerun with --output-format csv."
            ) from e

    # 5) Aggregate to unit×bucket (mean of issue scores).
    group_cols = unit_cols + [bucket_col]
    if bool(args.by_classification_label):
        group_cols.append("classification_label")

    feature_cols = [f"pc{i+1}" for i in range(n_components)] + ["emb_norm", "doc_weight_sum", "chunks_in_issue"]
    agg = issues.dropna(subset=group_cols).groupby(group_cols, as_index=False).agg(
        {
            **{c: "mean" for c in feature_cols},
            "issue_id": "nunique",
        }
    )
    agg = agg.rename(columns={"issue_id": "n_issues"})

    # Add stable numeric IDs when available (city_state only).
    if unit == "city_state" and {"state_id", "city_id"}.issubset(set(issues.columns)):
        loc_cols = ["city_name", "state_abbr", "state_id", "city_id"]
        loc = issues.dropna(subset=["city_name", "state_abbr"])[loc_cols].drop_duplicates()
        loc_counts = loc.groupby(["city_name", "state_abbr"], as_index=False).agg({"state_id": "nunique", "city_id": "nunique"})
        if bool((loc_counts["state_id"] > 1).any() or (loc_counts["city_id"] > 1).any()):
            raise SystemExit("Ambiguous mapping for (city_name,state_abbr)->(state_id,city_id) in pub metadata join.")
        agg = agg.merge(loc, on=["city_name", "state_abbr"], how="left")

    # 6) Optionally construct step-function "active policy state" by forward-filling within each unit.
    state_mode: StateMode = str(args.state_mode)  # type: ignore[assignment]
    if state_mode == "observed":
        panel = agg.copy()
        panel["observed"] = 1
    elif state_mode == "step":
        # Build a complete bucket grid per unit between its min and max observed buckets, then ffill the PCs.
        key_cols = unit_cols + (["classification_label"] if bool(args.by_classification_label) else [])
        rows = []
        idx_fn = _month_index if bucket == "month" else _year_index

        for key, g in agg.groupby(key_cols, sort=False):
            g = g.sort_values(bucket_col)
            bmin = str(g[bucket_col].min())
            bmax = str(g[bucket_col].max())
            imin = idx_fn(bmin)
            imax = idx_fn(bmax)
            if imax < imin:
                continue
            # Build bucket strings.
            if bucket == "year":
                buckets = [str(y) for y in range(imin, imax + 1)]
            else:
                buckets = []
                for mi in range(imin, imax + 1):
                    y = mi // 12
                    m = (mi % 12) + 1
                    buckets.append(f"{y:04d}-{m:02d}")
            key_dict = {k: v for k, v in zip(key_cols, key if isinstance(key, tuple) else (key,))}
            base = pd.DataFrame({**key_dict, bucket_col: buckets})
            merged = base.merge(g, on=key_cols + [bucket_col], how="left")
            merged["observed"] = merged["n_issues"].fillna(0).astype(int).clip(lower=0)
            # Forward-fill policy indices.
            for c in [f"pc{i+1}" for i in range(n_components)] + ["emb_norm"]:
                merged[c] = merged[c].ffill()
            rows.append(merged)
        if not rows:
            raise SystemExit("No step-function panel rows produced (unexpected).")
        panel = pd.concat(rows, ignore_index=True)
    else:
        raise SystemExit(f"Unknown --state-mode: {state_mode}")

    # 7) Compute within-unit time differences / jump sizes.
    key_cols = unit_cols + (["classification_label"] if bool(args.by_classification_label) else [])
    idx_fn = _month_index if bucket == "month" else _year_index
    panel["_t_idx"] = panel[bucket_col].astype(str).apply(idx_fn)
    panel = panel.sort_values(key_cols + ["_t_idx"]).reset_index(drop=True)

    # Jump size on PCA vectors (cosine distance + L2 norm).
    pc_cols = [f"pc{i+1}" for i in range(n_components)]

    def _jump_metrics(g: pd.DataFrame) -> pd.DataFrame:
        g = g.copy()
        prev = g[pc_cols].shift(1)
        cur = g[pc_cols]
        # cosine distance row-wise
        cd = []
        l2 = []
        for a, b in zip(cur.to_numpy(), prev.to_numpy()):
            if np.any(np.isnan(b)) or np.any(np.isnan(a)):
                cd.append(np.nan)
                l2.append(np.nan)
                continue
            cd.append(_cosine_distance(a.astype(np.float32), b.astype(np.float32)))
            l2.append(float(np.linalg.norm((a - b).astype(np.float32))))
        g["jump_cosine_pca"] = np.array(cd, dtype=np.float32)
        g["jump_l2_pca"] = np.array(l2, dtype=np.float32)
        return g

    panel = panel.groupby(key_cols, group_keys=False, sort=False).apply(_jump_metrics)
    panel = panel.drop(columns=["_t_idx"])

    panel_stem = f"{unit}_{bucket}_{state_mode}_policy_panel_{stamp}"
    panel_pq = _write_df(panel, out_dir=out_dir, stem=panel_stem)
    if str(args.output_format) in {"parquet", "both"}:
        try:
            panel.to_parquet(panel_pq, index=False)
        except Exception as e:  # noqa: BLE001
            raise SystemExit(
                f"Failed to write parquet output: {panel_pq}\n"
                f"Error: {e}\n"
                "Fix: install pyarrow (recommended) or rerun with --output-format csv."
            ) from e

    # 8) Quick feature catalog (data-driven “what varies?”) for numeric columns.
    num_cols = [c for c in panel.columns if pd.api.types.is_numeric_dtype(panel[c].dtype)]
    cat_rows = []
    for c in num_cols:
        s = pd.to_numeric(panel[c], errors="coerce")
        if s.notna().sum() == 0:
            continue
        cat_rows.append(
            {
                "column": c,
                "mean": float(s.mean()),
                "std": float(s.std(ddof=0)),
                "min": float(s.min()),
                "p25": float(s.quantile(0.25)),
                "p50": float(s.quantile(0.50)),
                "p75": float(s.quantile(0.75)),
                "max": float(s.max()),
                "pct_nonnull": float(s.notna().mean()),
                "pct_nonzero": float((s.fillna(0.0) != 0.0).mean()),
            }
        )
    catalog = pd.DataFrame.from_records(cat_rows).sort_values("std", ascending=False).reset_index(drop=True)
    catalog_path = out_dir / f"policy_feature_catalog_{stamp}.csv"
    catalog.to_csv(catalog_path, index=False, quoting=csv.QUOTE_MINIMAL)

    print(f"Done. wrote={out_dir} issues={len(issues)} panel_rows={len(panel)}")


if __name__ == "__main__":
    main()
