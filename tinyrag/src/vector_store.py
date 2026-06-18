"""
src/vector_store.py
--------------------
The simplest possible vector store: a Python list of
(text, vector, metadata) records, searched by brute-force cosine
similarity. No HNSW index, no approximate nearest neighbor search,
no persistence to disk.

This is conceptually exactly what Chroma/FAISS/Pinecone do for you,
minus the optimizations that make them fast at millions of vectors.
At the scale of a few hundred or thousand chunks, brute force is
plenty fast, and seeing it written out makes "vector database" stop
feeling like a black box.

Tradeoff worth noting out loud (interviewers like this answer):
everything here lives in RAM and disappears when the process exits.
Real vector DBs add persistence and indexing for speed at scale -
that's *all* they're adding conceptually.
"""

import numpy as np


class TinyVectorStore:
    def __init__(self):
        self._records = []  # each: {"text": str, "vector": np.ndarray, "metadata": dict}

    def add(self, text: str, vector: list[float], metadata: dict | None = None) -> None:
        self._records.append({
            "text": text,
            "vector": np.array(vector, dtype=np.float32),
            "metadata": metadata or {},
        })

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """
        cos(theta) = (a . b) / (||a|| * ||b||)

        Measures the angle between two vectors, ignoring their
        magnitude - so it cares about *direction* (meaning) rather
        than how long the embedding vector happens to be.
        """
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(dot_product / (norm_a * norm_b))

    def search(self, query_vector: list[float], top_k: int = 3) -> list[tuple[float, dict]]:
        """
        Returns the top_k records most similar to query_vector,
        as (similarity_score, record) tuples, sorted descending.
        """
        query_vector = np.array(query_vector, dtype=np.float32)
        scored = [
            (self._cosine_similarity(query_vector, r["vector"]), r)
            for r in self._records
        ]
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return scored[:top_k]

    def __len__(self) -> int:
        return len(self._records)
