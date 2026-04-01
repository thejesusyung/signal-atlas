from __future__ import annotations

import numpy as np

NOISE_LABEL = -1


def cluster_embeddings(matrix: np.ndarray, min_cluster_size: int = 5) -> np.ndarray:
    """
    Run HDBSCAN on an L2-normalised embedding matrix.
    Returns int array of cluster labels; NOISE_LABEL (-1) = unclustered.
    Falls back to all-noise if corpus is too small for the requested min_cluster_size.
    """
    import hdbscan

    if len(matrix) < min_cluster_size:
        return np.full(len(matrix), NOISE_LABEL, dtype=int)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=max(1, min_cluster_size // 2),
        metric="euclidean",  # L2-normalised embeddings: euclidean ≈ cosine
        cluster_selection_method="eom",
    )
    return clusterer.fit_predict(matrix).astype(int)


def label_cluster(cluster_id: int, titles: list[str], max_titles: int = 4) -> str:
    """
    Heuristic cluster label: first few representative titles joined by ' · '.
    Returns 'unclustered' for noise points.
    """
    if cluster_id == NOISE_LABEL:
        return "unclustered"
    sample = [t[:55] for t in titles[:max_titles]]
    return " · ".join(sample)[:200]
