import os
import torch
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mtcnn = MTCNN(image_size=160, margin=0, device=device)
model = InceptionResnetV1(pretrained="vggface2").eval().to(device)


class Absent:
    def __init__(self, id, first_name, last_name, img_paths, age=None, main_img_path=None):
        self.id = id
        self.first_name = first_name
        self.last_name = last_name
        self.img_paths = img_paths if isinstance(img_paths, list) else [img_paths]
        self.main_img_path = main_img_path or (self.img_paths[0] if self.img_paths else None)
        self.age = age
        self._embeddings = None

    def get_embs(self):
        """Compute embeddings for all images and cache them."""
        if self._embeddings is not None:
            return self._embeddings

        embs = []
        for img_path in self.img_paths:
            if not os.path.exists(img_path):
                continue
            try:
                img = Image.open(img_path).convert("RGB")
                face = mtcnn(img)
                if face is None:
                    continue
                emb = model(face.unsqueeze(0).to(device)).detach().cpu().numpy()[0]
                embs.append(emb)
            except Exception as e:
                print(f"[ERROR] Failed to process image {img_path}: {e}")

        self._embeddings = embs
        return embs

    def to_dict(self):
        """Convert person data to Firestore-friendly dict."""
        return {
            "id": self.id,
            "first_name": self.first_name,
            "last_name": self.last_name,
            "age": self.age,
            "main_img_path": self.main_img_path,
            "embeddings": [{"vector": emb.tolist()} for emb in (self._embeddings or [])],
        }
