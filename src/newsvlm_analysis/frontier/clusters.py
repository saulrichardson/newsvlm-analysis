from __future__ import annotations

import json
import warnings
from collections import Counter
from pathlib import Path
from typing import Any

import hdbscan
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import NearestNeighbors

from newsvlm_analysis.zoning_mechanics import summarize_mechanics

from .contracts import FrontierExperiment, default_experiments
from .io_utils import clean_optional_str, normalize_for_fingerprint, slugify, write_json, write_parquet


HIERARCHICAL_ROW_LIMIT = 20000


def _ensure_finite_matrix(name: str, matrix: np.ndarray) -> np.ndarray:
    arr = np.asarray(matrix, dtype=np.float64)
    if not np.isfinite(arr).all():
        raise RuntimeError(f"{name} contains non-finite values with shape {arr.shape}")
    return arr


def _row_normalize(matrix: np.ndarray) -> np.ndarray:
    arr = _ensure_finite_matrix("cluster matrix", matrix)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.where(norms > 0.0, norms, 1.0)
    normalized = arr / norms
    if not np.isfinite(normalized).all():
        raise RuntimeError(f"row-normalized matrix contains non-finite values with shape {normalized.shape}")
    return normalized


def _cosine_similarity_to_centroid(matrix: np.ndarray, centroid: np.ndarray) -> np.ndarray:
    normalized_matrix = _row_normalize(matrix)
    normalized_centroid = _row_normalize(centroid)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        sims = normalized_matrix @ normalized_centroid.T
    if not np.isfinite(sims).all():
        raise RuntimeError(f"cosine similarity produced non-finite values with shape {sims.shape}")
    return sims.reshape(-1)


def _load_experiment_matrix(run_root: Path, level: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    level_dir = run_root / "embeddings" / level
    meta = pd.read_parquet(level_dir / "metadata.parquet")
    matrix = np.load(level_dir / "embedding_matrix.npy")
    umap_5d = np.load(level_dir / "umap_5d.npy")
    pca_df = pd.read_parquet(level_dir / "pca_coords.parquet")
    pca_cols = [c for c in pca_df.columns if c.startswith("pc")]
    pca = pca_df[pca_cols].to_numpy(dtype=np.float32)
    return meta, matrix, umap_5d, pca


def _cluster_hdbscan(matrix: np.ndarray, min_cluster_size: int) -> np.ndarray:
    if matrix.shape[0] == 0:
        return np.zeros((0,), dtype=int)
    if matrix.shape[0] <= 3:
        return np.zeros((matrix.shape[0],), dtype=int)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=max(2, int(min_cluster_size)), min_samples=max(1, int(min_cluster_size // 3)))
    return clusterer.fit_predict(matrix).astype(int)


def _cluster_graph(matrix: np.ndarray, k: int, similarity_threshold: float) -> np.ndarray:
    if matrix.shape[0] == 0:
        return np.zeros((0,), dtype=int)
    if matrix.shape[0] <= 2:
        return np.zeros((matrix.shape[0],), dtype=int)
    normalized = _row_normalize(matrix)
    nn = NearestNeighbors(n_neighbors=min(max(2, k), normalized.shape[0]), metric="euclidean")
    nn.fit(normalized)
    distances, indices = nn.kneighbors(normalized)
    rows: list[int] = []
    cols: list[int] = []
    for i in range(normalized.shape[0]):
        for dist, j in zip(distances[i], indices[i]):
            if i == j:
                continue
            sim = 1.0 - (float(dist) ** 2) / 2.0
            if sim >= float(similarity_threshold):
                rows.extend([i, j])
                cols.extend([j, i])
    if not rows:
        return np.arange(normalized.shape[0], dtype=int)
    graph = csr_matrix((np.ones(len(rows), dtype=int), (rows, cols)), shape=(normalized.shape[0], normalized.shape[0]))
    _, labels = connected_components(graph, directed=False)
    return labels.astype(int)


def _cluster_hierarchical(matrix: np.ndarray, distance_threshold: float) -> np.ndarray:
    if matrix.shape[0] == 0:
        return np.zeros((0,), dtype=int)
    if matrix.shape[0] <= 2:
        return np.zeros((matrix.shape[0],), dtype=int)
    model = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=float(distance_threshold),
        metric="cosine",
        linkage="average",
    )
    return model.fit_predict(matrix).astype(int)


def _hierarchical_meta_cluster_from_seed_assignments(
    *,
    run_root: Path,
    experiment: FrontierExperiment,
    meta: pd.DataFrame,
    dense_matrix: np.ndarray,
) -> tuple[np.ndarray, pd.DataFrame]:
    seed_path = run_root / "clusters" / "clause_instrument_hdbscan" / "assignments.parquet"
    if not seed_path.is_file():
        raise RuntimeError(
            "Scaled hierarchical clause clustering requires an existing clause_instrument_hdbscan assignments.parquet"
        )

    seed_assign = pd.read_parquet(seed_path).copy()
    if seed_assign.empty or "row_index" not in seed_assign.columns or "cluster_id" not in seed_assign.columns:
        raise RuntimeError("Seed clause_instrument_hdbscan assignments are missing row_index or cluster_id")

    seed_assign["cluster_id"] = seed_assign["cluster_id"].astype(int)
    seed_assign["row_index"] = seed_assign["row_index"].astype(int)
    usable = seed_assign[seed_assign["cluster_id"] >= 0].copy()
    if usable.empty:
        labels = np.full((len(meta),), -1, dtype=int)
        sweep = pd.DataFrame(
            [
                {
                    "parameter_name": "distance_threshold",
                    "parameter_value": float(experiment.params.get("distance_threshold", 0.85)),
                    "cluster_count": 0,
                    "noise_count": int(len(labels)),
                    "ari_vs_baseline": 1.0,
                }
            ]
        )
        return labels, sweep

    centroid_rows: list[dict[str, Any]] = []
    centroid_vectors: list[np.ndarray] = []
    for seed_cluster_id, group in usable.groupby("cluster_id", dropna=False):
        row_idx = group["row_index"].to_numpy(dtype=int)
        centroid_vectors.append(dense_matrix[row_idx].mean(axis=0))
        centroid_rows.append(
            {
                "seed_cluster_id": int(seed_cluster_id),
                "seed_cluster_size": int(len(group)),
            }
        )

    centroid_df = pd.DataFrame.from_records(centroid_rows).sort_values("seed_cluster_id").reset_index(drop=True)
    centroid_matrix = np.vstack(centroid_vectors).astype(np.float64)
    base_threshold = float(experiment.params.get("distance_threshold", 0.85))
    centroid_labels = _cluster_hierarchical(centroid_matrix, base_threshold)
    centroid_df["meta_cluster_id"] = centroid_labels
    meta_cluster_map = dict(zip(centroid_df["seed_cluster_id"], centroid_df["meta_cluster_id"]))

    labels = np.full((len(meta),), -1, dtype=int)
    for row in usable.itertuples(index=False):
        labels[int(row.row_index)] = int(meta_cluster_map[int(row.cluster_id)])

    sweep_rows: list[dict[str, Any]] = []
    for alt in sorted({round(max(0.2, base_threshold - 0.1), 2), round(base_threshold, 2), round(min(1.5, base_threshold + 0.1), 2)}):
        alt_labels = _cluster_hierarchical(centroid_matrix, alt)
        alt_map = dict(zip(centroid_df["seed_cluster_id"], alt_labels))
        row_alt = np.full((len(meta),), -1, dtype=int)
        for row in usable.itertuples(index=False):
            row_alt[int(row.row_index)] = int(alt_map[int(row.cluster_id)])
        sweep_rows.append(
            {
                "parameter_name": "distance_threshold",
                "parameter_value": alt,
                "cluster_count": int(len(set(int(x) for x in alt_labels if int(x) >= 0))),
                "noise_count": int((row_alt < 0).sum()),
                "ari_vs_baseline": float(adjusted_rand_score(labels, row_alt)),
            }
        )
    return labels, pd.DataFrame.from_records(sweep_rows)


def _top_keywords(texts: list[str], top_k: int = 6) -> list[str]:
    counter: Counter[str] = Counter()
    for text in texts:
        tokens = [tok for tok in normalize_for_fingerprint(text).split() if len(tok) >= 4]
        counter.update(tokens)
    return [tok for tok, _ in counter.most_common(top_k)]


def _label_cluster(cluster_df: pd.DataFrame, labeler: str) -> dict[str, Any]:
    texts = cluster_df["text_for_embedding"].astype(str).tolist()
    keywords = _top_keywords(texts)
    mechanics = summarize_mechanics(texts, top_k=5)
    if labeler == "mechanics":
        label = ", ".join(mechanics["dimension_tags"][:2] or mechanics["action_tags"][:2] or keywords[:2] or ["misc"])
        taxonomy = "mechanics"
    elif labeler == "instrument":
        hint_counter = cluster_df.get("instrument_type", pd.Series(dtype=str)).astype(str).value_counts()
        label = str(hint_counter.index[0]) if not hint_counter.empty else ", ".join(keywords[:2] or ["instrument"])
        taxonomy = "instrument"
    else:
        label = ", ".join(keywords[:3] or ["semantic cluster"])
        taxonomy = "semantic"
    return {
        "cluster_label": label,
        "cluster_label_slug": slugify(label),
        "label_taxonomy": taxonomy,
        "keywords": "|".join(keywords),
        "mechanics_action_tags": "|".join(mechanics["action_tags"]),
        "mechanics_dimension_tags": "|".join(mechanics["dimension_tags"]),
        "mechanics_instrument_tags": "|".join(mechanics["instrument_tags"]),
    }


def _assign_exemplars(cluster_df: pd.DataFrame, matrix: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    exemplar_rows: list[dict[str, Any]] = []
    outlier_rows: list[dict[str, Any]] = []
    for cluster_id, group in cluster_df.groupby("cluster_id", dropna=False):
        if int(cluster_id) < 0:
            continue
        idx = group["row_index"].to_numpy(dtype=int)
        sub = matrix[idx]
        centroid = sub.mean(axis=0, keepdims=True)
        sims = _cosine_similarity_to_centroid(sub, centroid)
        tmp = group.copy()
        tmp["centroid_similarity"] = sims
        for _, r in tmp.sort_values(["centroid_similarity", "item_id"], ascending=[False, True]).head(5).iterrows():
            exemplar_rows.append(r.to_dict())
        for _, r in tmp.sort_values(["centroid_similarity", "item_id"], ascending=[True, True]).head(5).iterrows():
            outlier_rows.append(r.to_dict())
    return pd.DataFrame.from_records(exemplar_rows), pd.DataFrame.from_records(outlier_rows)


def _sweep_rows(experiment: FrontierExperiment, base_labels: np.ndarray, matrix: np.ndarray) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if experiment.clusterer == "hdbscan":
        base = int(experiment.params.get("hdbscan_min_cluster_size", 10))
        for alt in sorted({max(2, base - 4), base, base + 4}):
            labels = _cluster_hdbscan(matrix, alt)
            rows.append(
                {
                    "parameter_name": "hdbscan_min_cluster_size",
                    "parameter_value": alt,
                    "cluster_count": int(len(set(int(x) for x in labels if int(x) >= 0))),
                    "noise_count": int((labels < 0).sum()),
                    "ari_vs_baseline": float(adjusted_rand_score(base_labels, labels)),
                }
            )
    elif experiment.clusterer == "graph":
        base = float(experiment.params.get("graph_similarity_threshold", 0.42))
        k = int(experiment.params.get("graph_k", 12))
        for alt in sorted({round(max(0.2, base - 0.05), 2), round(base, 2), round(min(0.9, base + 0.05), 2)}):
            labels = _cluster_graph(matrix, k, alt)
            rows.append(
                {
                    "parameter_name": "graph_similarity_threshold",
                    "parameter_value": alt,
                    "cluster_count": int(len(set(int(x) for x in labels if int(x) >= 0))),
                    "noise_count": 0,
                    "ari_vs_baseline": float(adjusted_rand_score(base_labels, labels)),
                }
            )
    elif experiment.clusterer == "hierarchical":
        base = float(experiment.params.get("distance_threshold", 0.85))
        for alt in sorted({round(max(0.2, base - 0.1), 2), round(base, 2), round(min(1.5, base + 0.1), 2)}):
            labels = _cluster_hierarchical(matrix, alt)
            rows.append(
                {
                    "parameter_name": "distance_threshold",
                    "parameter_value": alt,
                    "cluster_count": int(len(set(int(x) for x in labels if int(x) >= 0))),
                    "noise_count": 0,
                    "ari_vs_baseline": float(adjusted_rand_score(base_labels, labels)),
                }
            )
    return rows


def run_cluster_experiments(*, run_root: Path, experiments: list[FrontierExperiment] | None = None) -> dict[str, Any]:
    clusters_dir = run_root / "clusters"
    experiments = experiments or default_experiments()
    index_rows: list[dict[str, Any]] = []

    for experiment in experiments:
        if not experiment.enabled:
            continue
        out_dir = clusters_dir / experiment.name
        out_dir.mkdir(parents=True, exist_ok=True)
        meta, dense_matrix, umap_5d, pca = _load_experiment_matrix(run_root, experiment.representation_level)
        dense_matrix = _ensure_finite_matrix(f"{experiment.name} dense matrix", dense_matrix)
        feature_matrix = dense_matrix
        if experiment.reducer == "umap":
            feature_matrix = _ensure_finite_matrix(f"{experiment.name} umap matrix", umap_5d)
        elif experiment.reducer == "pca":
            feature_matrix = _ensure_finite_matrix(f"{experiment.name} pca matrix", pca[:, : min(8, pca.shape[1])])

        if experiment.clusterer == "hdbscan":
            labels = _cluster_hdbscan(feature_matrix, int(experiment.params.get("hdbscan_min_cluster_size", 10)))
        elif experiment.clusterer == "graph":
            labels = _cluster_graph(
                feature_matrix,
                int(experiment.params.get("graph_k", 12)),
                float(experiment.params.get("graph_similarity_threshold", 0.42)),
            )
        elif experiment.clusterer == "hierarchical":
            if feature_matrix.shape[0] > HIERARCHICAL_ROW_LIMIT and experiment.representation_level == "clause":
                labels, sweep_df = _hierarchical_meta_cluster_from_seed_assignments(
                    run_root=run_root,
                    experiment=experiment,
                    meta=meta,
                    dense_matrix=dense_matrix,
                )
            else:
                labels = _cluster_hierarchical(feature_matrix, float(experiment.params.get("distance_threshold", 0.85)))
                sweep_df = pd.DataFrame.from_records(_sweep_rows(experiment, labels, feature_matrix))
        else:
            labels = np.zeros((feature_matrix.shape[0],), dtype=int)
            sweep_df = pd.DataFrame.from_records(_sweep_rows(experiment, labels, feature_matrix))

        assign_df = meta.copy()
        assign_df["cluster_id"] = labels
        assign_df["row_index"] = np.arange(len(assign_df))
        exemplar_df, outlier_df = _assign_exemplars(assign_df, dense_matrix)

        cluster_rows: list[dict[str, Any]] = []
        for cluster_id, group in assign_df.groupby("cluster_id", dropna=False):
            label_payload = _label_cluster(group, experiment.labeler)
            cluster_rows.append(
                {
                    "cluster_id": int(cluster_id),
                    "cluster_size": int(len(group)),
                    **label_payload,
                }
            )
        cluster_df = pd.DataFrame.from_records(cluster_rows).sort_values(["cluster_id"]).reset_index(drop=True)
        assign_df = assign_df.merge(cluster_df, on="cluster_id", how="left")
        if experiment.clusterer != "hierarchical":
            sweep_df = pd.DataFrame.from_records(_sweep_rows(experiment, labels, feature_matrix))

        write_parquet(assign_df, out_dir / "assignments.parquet")
        write_parquet(cluster_df, out_dir / "cluster_labels.parquet")
        write_parquet(exemplar_df, out_dir / "cluster_exemplars.parquet")
        write_parquet(outlier_df, out_dir / "cluster_outliers.parquet")
        write_parquet(sweep_df, out_dir / "stability_sweep.parquet")
        write_json(
            out_dir / "summary.json",
            {
                "experiment": experiment.to_dict(),
                "item_count": int(len(assign_df)),
                "cluster_count": int(len([x for x in assign_df["cluster_id"].unique().tolist() if int(x) >= 0])),
                "noise_count": int((assign_df["cluster_id"].astype(int) < 0).sum()),
            },
        )
        index_rows.append(
            {
                "experiment_name": experiment.name,
                "representation_level": experiment.representation_level,
                "labeler": experiment.labeler,
                "clusterer": experiment.clusterer,
                "item_count": int(len(assign_df)),
                "cluster_count": int(len([x for x in assign_df["cluster_id"].unique().tolist() if int(x) >= 0])),
                "noise_count": int((assign_df["cluster_id"].astype(int) < 0).sum()),
            }
        )

    index_df = pd.DataFrame.from_records(index_rows).sort_values("experiment_name").reset_index(drop=True)
    write_parquet(index_df, clusters_dir / "experiment_index.parquet")
    write_json(clusters_dir / "cluster_index.json", {"experiments": index_rows})
    return {"experiment_count": int(len(index_rows))}
