"""
faiss_hnsw.py

Approximate nearest-neighbor (ANN) retrieval using FAISS HNSW for content-based
recommendations. Designed to be a drop-in alternative to the scikit-learn
NearestNeighbors backend used in the existing content-based models.

Key features
------------
- Supports inner-product (cosine equivalent with normalized vectors) or L2.
- HNSW graph with tunable construction/search parameters.
- String ID support via external mapping persisted as JSON.
- Graceful import handling: raises clear ImportError if `faiss` is missing.
- Simple interface: fit / query / batch_query / save / load.

Usage example
-------------
if __name__ == "__main__":
    import numpy as np
    rng = np.random.default_rng(0)
    X = rng.standard_normal((1000, 128)).astype("float32")
    ids = [f"movie_{i}" for i in range(len(X))]

    idx = FaissHNSWIndex(dim=128, metric="ip")
    idx.fit(X, ids)
    q = X[0]
    print("Top-5:", idx.query(q, top_k=5, exclude_ids={ids[0]}))
"""

from __future__ import annotations

import json
import os
from typing import List, Optional, Sequence, Set, Tuple
import numpy as np


def _safe_import_faiss():
    try:
        import faiss  # type: ignore
        return faiss
    except Exception as e:
        raise ImportError(
            "faiss is required for FaissHNSWIndex but is not installed. "
            "Install with `pip install faiss-cpu` (or faiss-gpu) and retry."
        ) from e


def normalize_vectors(X: np.ndarray) -> np.ndarray:
    """
    L2-normalize rows of X (in-place safe) for cosine/inner-product searches.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, dim), dtype float32

    Returns
    -------
    np.ndarray : normalized copy (float32)
    """
    if X.dtype != np.float32:
        X = X.astype("float32", copy=False)
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    return X / norms


class FaissHNSWIndex:
    """
    HNSW index wrapper with FAISS.

    Parameters
    ----------
    dim : int
        Embedding dimensionality.
    metric : {"ip", "l2"}, default "ip"
        "ip" for inner-product (use with normalized vectors to emulate cosine),
        "l2" for Euclidean distance.
    hnsw_m : int, default 32
        Number of neighbors in HNSW graph (M).
    ef_search : int, default 50
        Search beam width; higher -> better recall, higher latency.
    ef_construction : int, default 200
        Construction beam width; higher -> better quality, higher build time.
    """

    def __init__(
        self,
        dim: int,
        metric: str = "ip",
        hnsw_m: int = 32,
        ef_search: int = 50,
        ef_construction: int = 200,
    ) -> None:
        if metric not in {"ip", "l2"}:
            raise ValueError("metric must be 'ip' or 'l2'")
        self.dim = dim
        self.metric = metric
        self.hnsw_m = hnsw_m
        self.ef_search = ef_search
        self.ef_construction = ef_construction

        self._faiss = None
        self.index = None  # faiss index object
        self.ids: List[str] = []  # row -> movie_id mapping

    # -----------------------------
    # lifecycle
    # -----------------------------

    def fit(self, X: np.ndarray, ids: Sequence[str]) -> None:
        """
        Build the HNSW index.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, dim), float32
            Feature matrix.
        ids : Sequence[str]
            External IDs (e.g., movie_id) aligned with X rows.
        """
        self._faiss = _safe_import_faiss()

        if X.dtype != np.float32:
            X = X.astype("float32", copy=False)

        if self.metric == "ip":
            X = normalize_vectors(X)

        if X.ndim != 2 or X.shape[1] != self.dim:
            raise ValueError(f"X must have shape (n, {self.dim})")

        self.ids = [str(x) for x in ids]

        if self.metric == "ip":
            index = self._faiss.IndexHNSWFlat(
                self.dim, self.hnsw_m, self._faiss.METRIC_INNER_PRODUCT)
        else:
            index = self._faiss.IndexHNSWFlat(
                self.dim, self.hnsw_m, self._faiss.METRIC_L2)

        # construction parameter
        index.hnsw.efConstruction = self.ef_construction

        index.add(X)
        # search parameter
        index.hnsw.efSearch = self.ef_search

        self.index = index

    def query(
        self,
        vector: np.ndarray,
        top_k: int = 10,
        exclude_ids: Optional[Set[str]] = None,
    ) -> List[Tuple[str, float]]:
        """
        Single-query ANN search.

        Parameters
        ----------
        vector : np.ndarray, shape (dim,)
        top_k : int
        exclude_ids : set[str] or None
            IDs to exclude (e.g., the query's own movie_id).

        Returns
        -------
        list[(id, score)]
            For metric='ip' we return similarity (higher is better).
            For metric='l2' we return negative distance as a score (higher better).
        """
        if self.index is None:
            raise RuntimeError(
                "Index not built or loaded. Call fit() or load().")

        v = vector.astype("float32", copy=False).reshape(1, -1)
        if self.metric == "ip":
            v = normalize_vectors(v)

        D, I = self.index.search(v, k=max(top_k, 1))
        scores = D[0].tolist()
        idxs = I[0].tolist()

        results: List[Tuple[str, float]] = []
        for j, score in zip(idxs, scores):
            if j == -1:
                continue
            mid = self.ids[j]
            if exclude_ids and mid in exclude_ids:
                continue
            # Convert to "similarity-like" score if L2 (higher=better)
            sim = float(score if self.metric == "ip" else -score)
            results.append((mid, sim))
            if len(results) >= top_k:
                break
        return results

    def batch_query(
        self,
        vectors: np.ndarray,
        top_k: int = 10,
        exclude_ids: Optional[Set[str]] = None,
    ) -> List[List[Tuple[str, float]]]:
        """
        Batch ANN search.

        Parameters
        ----------
        vectors : np.ndarray, shape (n_queries, dim)
        top_k : int
        exclude_ids : set[str] or None

        Returns
        -------
        list[list[(id, score)]]
        """
        if self.index is None:
            raise RuntimeError(
                "Index not built or loaded. Call fit() or load().")

        V = vectors.astype("float32", copy=False)
        if self.metric == "ip":
            V = normalize_vectors(V)

        D, I = self.index.search(V, k=max(top_k, 1))
        out: List[List[Tuple[str, float]]] = []
        for row_scores, row_idxs in zip(D, I):
            results: List[Tuple[str, float]] = []
            for j, score in zip(row_idxs.tolist(), row_scores.tolist()):
                if j == -1:
                    continue
                mid = self.ids[j]
                if exclude_ids and mid in exclude_ids:
                    continue
                sim = float(score if self.metric == "ip" else -score)
                results.append((mid, sim))
                if len(results) >= top_k:
                    break
            out.append(results)
        return out

    # -----------------------------
    # persistence
    # -----------------------------

    def save(self, index_path: str, meta_path: str) -> None:
        """
        Save index and ID mapping.

        Parameters
        ----------
        index_path : str
        meta_path : str (JSON with {"ids": [...], "dim": int, "metric": str, "hnsw_m": int, ...})
        """
        if self.index is None:
            raise RuntimeError("Index not built; nothing to save.")

        self._faiss = self._faiss or _safe_import_faiss()
        self._faiss.write_index(self.index, index_path)

        meta = {
            "ids": self.ids,
            "dim": self.dim,
            "metric": self.metric,
            "hnsw_m": self.hnsw_m,
            "ef_search": self.ef_search,
            "ef_construction": self.ef_construction,
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f)

    def load(self, index_path: str, meta_path: str) -> None:
        """
        Load index and ID mapping.
        """
        self._faiss = _safe_import_faiss()
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        self.ids = [str(x) for x in meta["ids"]]
        self.dim = int(meta["dim"])
        self.metric = str(meta["metric"])
        self.hnsw_m = int(meta["hnsw_m"])
        self.ef_search = int(meta["ef_search"])
        self.ef_construction = int(meta["ef_construction"])

        self.index = self._faiss.read_index(index_path)
        # (Re-)apply search parameter from meta for clarity
        self.index.hnsw.efSearch = self.ef_search
