# models/absent.py
# Simple person model used by FaceDatabase and the API.
# Goals:
#  - Always define .embeddings (list[np.ndarray]) and .mean_embedding (np.ndarray or None)
#  - Keep serialization (to_dict) free of raw numpy arrays
#  - Be minimal and framework-agnostic

from __future__ import annotations

from typing import List, Optional, Any, Dict
import numpy as np


class Absent:
    """
    Represents a person in memory.

    Fields:
      - id: unique string identifier (UUID or external)
      - first_name, last_name: basic identity fields
      - age: optional string / number (kept as string in most APIs)
      - images: list of local image paths used to compute embeddings
      - main_img_path: canonical image (local path or remote URL if uploaded)
      - embeddings: list of np.ndarray (512,) computed from 'images'
      - mean_embedding: single np.ndarray (512,) averaged over 'embeddings', or None
    """

    def __init__(
        self,
        person_id: str,
        first_name: str,
        last_name: str,
        age: Optional[str] = None,
    ) -> None:
        self.id: str = person_id
        self.first_name: str = first_name
        self.last_name: str = last_name
        self.age: Optional[str] = age

        # Local image paths used to build embeddings
        self.images: List[str] = []

        # May be a local path OR a remote URL (e.g., Cloudinary)
        self.main_img_path: Optional[str] = None

        # Always defined:
        self.embeddings: List[np.ndarray] = []   # list of 512-d vectors
        self.mean_embedding: Optional[np.ndarray] = None  # single 512-d vector or None

    # ---------------------------- Helpers ---------------------------- #

    @property
    def full_name(self) -> str:
        return f"{self.first_name} {self.last_name}".strip()

    def has_embeddings(self) -> bool:
        return bool(self.embeddings) or (self.mean_embedding is not None)

    def set_embeddings(self, vectors: List[np.ndarray]) -> None:
        """
        Replace the current embeddings list with 'vectors' (each np.ndarray (512,)).
        Also updates mean_embedding accordingly.
        """
        self.embeddings = [np.asarray(v, dtype=np.float32) for v in (vectors or [])]
        if self.embeddings:
            self.mean_embedding = np.mean(np.stack(self.embeddings, axis=0), axis=0).astype(np.float32)
        else:
            self.mean_embedding = None

    # ---------------------------- Serialization ---------------------------- #

    def to_dict(self) -> Dict[str, Any]:
        """
        Safe dict for Firestore or JSON (no raw numpy arrays inside).
        Note:
          - We do NOT return 'embeddings' (list of vectors) here to avoid large payloads
            and Firestore nested-array limitations.
          - If you want to store the embedding, store ONLY 'mean_embedding' as a flat list.
            FaceDatabase handles this when uploading.
        """
        return {
            "id": self.id,
            "first_name": self.first_name,
            "last_name": self.last_name,
            "age": self.age,
            "main_img_path": self.main_img_path,
            # 'embedding' (flat list) is handled by FaceDatabase.upload_person_to_firestore
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Absent":
        """
        Build an Absent from a dict (e.g., Firestore doc). This does NOT hydrate numpy arrays.
        The FaceDatabase.load_from_firestore will populate embeddings/mean_embedding correctly.
        """
        person = cls(
            person_id=data.get("id") or "",
            first_name=data.get("first_name") or "",
            last_name=data.get("last_name") or "",
            age=data.get("age"),
        )
        person.main_img_path = data.get("main_img_path")
        # embeddings / mean_embedding are set by the loader (FaceDatabase.load_from_firestore)
        return person

    def __repr__(self) -> str:
        emb_count = len(self.embeddings) if self.embeddings is not None else 0
        has_mean = self.mean_embedding is not None
        return f"Absent(id={self.id!r}, name={self.full_name!r}, emb_count={emb_count}, mean={has_mean})"
