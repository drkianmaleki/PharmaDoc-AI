from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    """Small wrapper around SentenceTransformers."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return normalized embeddings as a NumPy array."""

        if not texts:
            return np.empty((0, 0), dtype=float)

        embeddings = self.model.encode(texts, normalize_embeddings=True)
        return np.asarray(embeddings, dtype=float)
