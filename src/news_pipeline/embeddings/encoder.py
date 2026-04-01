from __future__ import annotations

from functools import lru_cache

import numpy as np

MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384


@lru_cache(maxsize=1)
def _get_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(MODEL_NAME)


def encode_texts(texts: list[str], batch_size: int = 64) -> np.ndarray:
    """Encode texts; returns float32 array shape (N, 384), L2-normalised."""
    model = _get_model()
    return model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )


def vector_to_db(vec: np.ndarray) -> list[float]:
    """Convert a numpy vector to a plain list for storage via VectorType."""
    return vec.tolist()


def vector_from_db(value) -> np.ndarray:
    """Convert a DB value (list or array) back to a numpy float32 array."""
    return np.array(value, dtype=np.float32)
