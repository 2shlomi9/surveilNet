# models/face_database.py
# Maintains a local people database, computes face embeddings for new persons,
# and uploads/loads single persons to/from Firestore.
# Design goals:
#  - Each person has: .embeddings (list[np.ndarray]) and .mean_embedding (np.ndarray or None)
#  - Firestore stores a single flat "embedding" (list[float]) per person (no nested arrays)
#  - upload_person_to_firestore(db, person): uploads ONLY the newly added person
#  - load_from_firestore(db): backward-compatible with legacy "embeddings" if present

from __future__ import annotations

import os
from typing import List, Optional

import numpy as np
import torch
from PIL import Image

from facenet_pytorch import MTCNN, InceptionResnetV1

# Person model
from models.absent import Absent  # Ensure Absent defines: id, first_name, last_name, age, images, main_img_path, embeddings, mean_embedding

# Optional Cloudinary integration
_CLOUDINARY_AVAILABLE = False
try:
    import cloudinary
    import cloudinary.uploader
    # Optional config module (keys/tokens). If absent, we rely on env vars.
    try:
        from configs.cloudinary_config import CLOUDINARY_CONFIG  # {'cloud_name':..., 'api_key':..., 'api_secret':...}
        cloudinary.config(
            cloud_name=CLOUDINARY_CONFIG.get("cloud_name"),
            api_key=CLOUDINARY_CONFIG.get("api_key"),
            api_secret=CLOUDINARY_CONFIG.get("api_secret"),
            secure=True,
        )
    except Exception:
        # If a config module is not present, assume the environment variables are already set.
        pass
    _CLOUDINARY_AVAILABLE = True
except Exception:
    _CLOUDINARY_AVAILABLE = False


class FaceDatabase:
    """
    In-memory people DB + embedding utilities + Firestore/Cloudinary helpers.
    """

    def __init__(self) -> None:
        self.people: List[Absent] = []

        # Device + face models
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mtcnn = MTCNN(image_size=160, margin=20, post_process=True, device=self.device)
        self.resnet = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)

    # ------------------------- Public API ------------------------- #

    def add_person(
        self,
        person_id: str,
        first_name: str,
        last_name: str,
        image_paths: List[str],
        age: Optional[str] = None,
    ) -> Absent:
        """
        Create a person object, compute embeddings from the given images,
        set mean_embedding, and append to self.people. Returns the created person.
        """
        person = Absent(person_id, first_name, last_name, age)
        person.images = image_paths or []
        if image_paths:
            # Choose the first image as the main one by default
            person.main_img_path = image_paths[0]

        # Compute embeddings for each valid image
        person.embeddings = []
        for p in image_paths:
            emb = self._image_to_embedding(p)
            if emb is not None:
                person.embeddings.append(emb)

        # Compute a single representative embedding (mean) to store in Firestore
        person.mean_embedding = None
        if person.embeddings:
            person.mean_embedding = (
                np.mean(np.stack(person.embeddings, axis=0), axis=0).astype(np.float32)
            )

        self.people.append(person)
        print(f"[INFO] Added person: {first_name} {last_name} (ID={person_id}) | embeddings={len(person.embeddings)}")
        return person

    def upload_person_to_firestore(self, db, person: Absent) -> None:
        """
        Upload ONLY the provided person to Firestore.
        - Stores a flat 'embedding' (list of floats) — not nested arrays.
        - Optionally uploads the main image to Cloudinary (if available) and overwrites main_img_path with URL.
        """
        # Optional main image upload to Cloudinary
        main_url = None
        if _CLOUDINARY_AVAILABLE and person.main_img_path and os.path.isfile(person.main_img_path):
            try:
                up = cloudinary.uploader.upload(
                    person.main_img_path,
                    folder="faces_main",
                    overwrite=True,
                    resource_type="image",
                )
                main_url = up.get("secure_url") or up.get("url")
            except Exception as e:
                print(f"[WARN] Cloudinary upload failed for {person.first_name} {person.last_name}: {e}")

        if main_url:
            person.main_img_path = main_url  # store the hosted URL

        # Serialize a single flat embedding (mean)
        embedding_list = None
        if getattr(person, "mean_embedding", None) is not None:
            embedding_list = person.mean_embedding.tolist()

        doc = {
            "id": person.id,
            "first_name": person.first_name,
            "last_name": person.last_name,
            "age": person.age,
            "main_img_path": person.main_img_path,
            "embedding": embedding_list,                        # flat list (OK for Firestore)
            "embeddings_count": len(person.embeddings or []),   # metadata only
        }

        db.collection("people").document(person.id).set(doc)
        print(f"[INFO] Saved {person.first_name} {person.last_name} (ID={person.id}) to Firestore")

    def load_from_firestore(self, db) -> None:
        """
        Load all people from Firestore into memory.
        Backward compatible:
          - preferred: 'embedding' (flat list)
          - legacy:   'embeddings' (list of lists) -> mean will be computed
        """
        self.people.clear()
        docs = db.collection("people").stream()
        count = 0
        for d in docs:
            data = d.to_dict() or {}

            person = Absent(
                person_id=data.get("id") or d.id,
                first_name=data.get("first_name", ""),
                last_name=data.get("last_name", ""),
                age=data.get("age"),
            )
            person.main_img_path = data.get("main_img_path")
            person.embeddings = []      # always present
            person.mean_embedding = None

            # Preferred new schema: flat 'embedding'
            emb_flat = data.get("embedding")
            # Legacy schema: 'embeddings' (nested lists)
            legacy_embs = data.get("embeddings")

            if isinstance(emb_flat, list) and emb_flat:
                arr = np.array(emb_flat, dtype=np.float32)
                person.mean_embedding = arr
                # Keep at least one embedding in .embeddings for compatibility with code that expects a list
                person.embeddings.append(arr)

            elif isinstance(legacy_embs, list) and legacy_embs:
                mats = []
                for x in legacy_embs:
                    if isinstance(x, list) and x:
                        mats.append(np.array(x, dtype=np.float32))
                if mats:
                    stacked = np.stack(mats, axis=0)
                    person.mean_embedding = np.mean(stacked, axis=0).astype(np.float32)
                    person.embeddings.append(person.mean_embedding)

            self.people.append(person)
            count += 1

        print(f"[INFO] Loaded {count} people from Firestore.")

    # --------------- Optional bulk (keep for one-time seeding) --------------- #
    def upload_to_firestore(self, db) -> None:
        """
        Bulk upload of all people in self.people.
        Not recommended for per-request use; keep only for one-time seeding/migration.
        """
        for person in self.people:
            self.upload_person_to_firestore(db, person)
        print(f"[INFO] Uploaded {len(self.people)} people to Firestore.")

    # ------------------------- Internal helpers ------------------------- #

    def _image_to_embedding(self, image_path: str) -> Optional[np.ndarray]:
        """
        Open image, align with MTCNN, and compute a 512-dim embedding using InceptionResnetV1.
        Returns None if face alignment fails.
        """
        try:
            img = Image.open(image_path).convert("RGB")
            aligned = self.mtcnn(img)  # torch.Tensor [3,160,160] or None
            if aligned is None:
                return None
            aligned = aligned.unsqueeze(0).to(self.device)  # [1,3,160,160]
            with torch.no_grad():
                emb = self.resnet(aligned)  # [1,512]
            return emb.squeeze(0).cpu().numpy().astype(np.float32)
        except Exception as e:
            print(f"[WARN] Failed to embed image '{image_path}': {e}")
            return None
