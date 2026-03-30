from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import umap
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from .io_utils import clean_optional_str, write_json, write_parquet


def _assert_finite(name: str, matrix: np.ndarray) -> np.ndarray:
    if not np.isfinite(matrix).all():
        raise RuntimeError(f"{name} produced non-finite values with shape {matrix.shape}")
    return matrix.astype(np.float32)


def _representation_df(run_root: Path, level: str) -> tuple[pd.DataFrame, str]:
    if level == "ordinance":
        df = pd.read_parquet(run_root / "corpus" / "ordinances.parquet").copy()
        if "jurisdiction_resolution_status" in df.columns:
            df = df[df["jurisdiction_resolution_status"].astype(str) == "resolved_with_census"].copy()
        df["text_for_embedding"] = df["clean_text"].astype(str)
        df["item_id"] = df["ordinance_doc_id"].astype(str)
        df = df[df["text_for_embedding"].str.strip().astype(bool)].copy()
        return df, "text_for_embedding"
    if level == "section":
        df = pd.read_parquet(run_root / "structure" / "sections.parquet").copy()
        df["text_for_embedding"] = df["text"].astype(str)
        df["item_id"] = df["section_id"].astype(str)
        df = df[df["text_for_embedding"].str.strip().astype(bool)].copy()
        return df, "text_for_embedding"
    if level == "clause":
        semantics_path = run_root / "semantics" / "operative_clause_semantics.parquet"
        source_name = "operative_clause_semantics"
        if semantics_path.is_file():
            df = pd.read_parquet(semantics_path).copy()
            df["text_for_embedding"] = df["clause_text"].astype(str)
        else:
            fallback_semantics = run_root / "semantics" / "clause_semantics.parquet"
            if fallback_semantics.is_file():
                df = pd.read_parquet(fallback_semantics).copy()
                df["text_for_embedding"] = df["clause_text"].astype(str)
                source_name = "clause_semantics"
            else:
                df = pd.read_parquet(run_root / "structure" / "clauses.parquet").copy()
                df["text_for_embedding"] = df["clause_text"].astype(str)
                source_name = "structure_clauses"
        df["item_id"] = df["clause_id"].astype(str)
        df = df[df["text_for_embedding"].str.strip().astype(bool)].copy()
        df["clause_embedding_source"] = source_name
        return df, "text_for_embedding"
    raise SystemExit(f"Unknown representation level: {level}")


def _dense_embedding_matrix(texts: list[str]) -> tuple[np.ndarray, dict[str, Any]]:
    if not texts:
        return np.zeros((0, 0), dtype=np.float32), {"vectorizer": "empty"}
    if len(texts) == 1:
        return np.ones((1, 1), dtype=np.float32), {"vectorizer": "single_item"}

    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=8000, min_df=1)
    X = vectorizer.fit_transform(texts)
    if X.shape[0] <= 16 or X.shape[1] <= 16:
        dense = X.toarray().astype(np.float32)
        return dense, {
            "vectorizer": "tfidf_dense",
            "tfidf_features": int(X.shape[1]),
            "svd_components": int(dense.shape[1]),
            "vocabulary_sample": sorted(vectorizer.vocabulary_.keys())[:50],
        }
    n_components = min(64, max(2, X.shape[0] - 1), max(2, X.shape[1] - 1))
    if n_components <= 1:
        dense = X.toarray().astype(np.float32)
        return dense, {"vectorizer": "tfidf", "svd_components": dense.shape[1]}
    svd = TruncatedSVD(n_components=n_components, random_state=0)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"sklearn\..*")
        dense = _assert_finite("TruncatedSVD", svd.fit_transform(X))
    return dense, {
        "vectorizer": "tfidf_svd",
        "tfidf_features": int(X.shape[1]),
        "svd_components": int(n_components),
        "explained_variance_ratio_sum": float(np.asarray(getattr(svd, "explained_variance_ratio_", [])).sum()),
        "vocabulary_sample": sorted(vectorizer.vocabulary_.keys())[:50],
    }


def _safe_pca(matrix: np.ndarray, n_components: int) -> np.ndarray:
    if matrix.shape[0] <= 1:
        return np.zeros((matrix.shape[0], max(1, n_components)), dtype=np.float32)
    n = min(n_components, matrix.shape[0], matrix.shape[1])
    if n <= 0:
        return np.zeros((matrix.shape[0], 1), dtype=np.float32)
    pca = PCA(n_components=n, random_state=0)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"sklearn\..*")
        coords = _assert_finite("PCA", pca.fit_transform(matrix))
    if coords.shape[1] < n_components:
        pad = np.zeros((coords.shape[0], n_components - coords.shape[1]), dtype=np.float32)
        coords = np.hstack([coords, pad])
    return coords


def _pad_embedding(coords: np.ndarray, n_components: int) -> np.ndarray:
    if coords.shape[1] < n_components:
        pad = np.zeros((coords.shape[0], n_components - coords.shape[1]), dtype=np.float32)
        coords = np.hstack([coords.astype(np.float32), pad])
    return coords.astype(np.float32)


def _safe_umap(matrix: np.ndarray, n_components: int) -> np.ndarray:
    if matrix.shape[0] == 0:
        return np.zeros((0, n_components), dtype=np.float32)
    max_rank = max(1, min(matrix.shape[0] - 1, matrix.shape[1] if matrix.ndim == 2 else 1))
    if matrix.shape[0] <= max(3, n_components + 1) or max_rank <= 1:
        return _pad_embedding(_safe_pca(matrix, min(n_components, max_rank)), n_components)
    reducer = umap.UMAP(
        n_neighbors=min(25, max(3, matrix.shape[0] - 1)),
        min_dist=0.0,
        n_components=n_components,
        metric="cosine",
        init="random",
        random_state=0,
    )
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"n_jobs value 1 overridden to 1 by setting random_state\. Use no seed for parallelism\.",
                category=UserWarning,
            )
            return _assert_finite("UMAP", reducer.fit_transform(matrix))
    except Exception:
        return _pad_embedding(_safe_pca(matrix, min(n_components, max_rank)), n_components)


def build_embedding_outputs(*, run_root: Path) -> dict[str, Any]:
    embeddings_dir = run_root / "embeddings"
    summary: dict[str, Any] = {}
    for level in ("ordinance", "section", "clause"):
        level_dir = embeddings_dir / level
        level_dir.mkdir(parents=True, exist_ok=True)
        df, text_col = _representation_df(run_root, level)
        texts = [clean_optional_str(x) for x in df[text_col].tolist()]
        matrix, meta = _dense_embedding_matrix(texts)
        pca_coords = _safe_pca(matrix, 10)
        umap_2d = _safe_umap(matrix, 2)
        umap_5d = _safe_umap(matrix, 5)

        np.save(level_dir / "embedding_matrix.npy", matrix)
        np.save(level_dir / "umap_5d.npy", umap_5d)

        meta_df = df.copy()
        meta_df["embedding_row_index"] = np.arange(len(meta_df))
        write_parquet(meta_df, level_dir / "metadata.parquet")
        write_parquet(
            pd.DataFrame({"item_id": meta_df["item_id"], "umap_x": umap_2d[:, 0] if len(umap_2d) else [], "umap_y": umap_2d[:, 1] if len(umap_2d) else []}),
            level_dir / "umap_2d.parquet",
        )
        pca_cols = {f"pc{i+1}": pca_coords[:, i] for i in range(pca_coords.shape[1])}
        pca_cols["item_id"] = meta_df["item_id"].tolist()
        write_parquet(pd.DataFrame(pca_cols), level_dir / "pca_coords.parquet")
        level_summary = {"level": level, "item_count": int(len(meta_df)), **meta}
        if level == "clause" and "clause_embedding_source" in meta_df.columns:
            level_summary["clause_embedding_source"] = clean_optional_str(meta_df["clause_embedding_source"].iloc[0])
        write_json(level_dir / "summary.json", level_summary)
        summary[level] = level_summary

    write_json(embeddings_dir / "embedding_index.json", summary)
    return summary
