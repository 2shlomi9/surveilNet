from __future__ import annotations
import os
import faiss
import numpy as np
from typing import Tuple

class FaissIndex:
    """
    Simple FAISS wrapper using IndexFlatIP (exact search).
    Assumes input vectors are already L2-normalized.
    """
    def __init__(self, dim: int, index_path: str, vectors_path: str):
        self.dim = dim
        self.index_path = index_path
        self.vectors_path = vectors_path
        self.index = faiss.IndexFlatIP(dim)
        self._vectors = None  # numpy array of shape (N, dim)

        if os.path.exists(index_path) and os.path.exists(vectors_path):
            self._load()

    def _load(self):
        self.index = faiss.read_index(self.index_path)
        self._vectors = np.load(self.vectors_path)
        # sanity: make sure index.ntotal matches vectors count
        if self.index.ntotal != self._vectors.shape[0]:
            raise RuntimeError("FAISS index and vectors count mismatch.")

    def _save(self):
        faiss.write_index(self.index, self.index_path)
        np.save(self.vectors_path, self._vectors)

    @property
    def size(self) -> int:
        return int(self.index.ntotal)

    def add(self, vectors: np.ndarray) -> Tuple[int, int]:
        """
        Add vectors (L2-normalized). Returns (start_id, end_id_exclusive).
        """
        if vectors.dtype != np.float32:
            vectors = vectors.astype(np.float32)
        start_id = self.size
        self.index.add(vectors)
        if self._vectors is None:
            self._vectors = vectors.copy()
        else:
            self._vectors = np.vstack([self._vectors, vectors])
        self._save()
        end_id = self.size
        return start_id, end_id

    def search(self, query: np.ndarray, top_k: int = 10):
        """
        Search top-k for a single query vector (must be L2-normalized).
        Returns (scores, indices).
        """
        if query.dtype != np.float32:
            query = query.astype(np.float32)
        D, I = self.index.search(query.reshape(1,-1), top_k)
        return D[0], I[0]
