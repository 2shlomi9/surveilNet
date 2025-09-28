import os
from PIL import Image
import torch
import numpy as np
from facenet_pytorch import InceptionResnetV1, MTCNN

# Device & models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mtcnn = MTCNN(image_size=160, margin=0, device=device)
model = InceptionResnetV1(pretrained="vggface2").eval().to(device)


class Absent:
    def __init__(self, id, first_name, last_name, img_paths, age=None):
        self.id = id
        self.first_name = first_name
        self.last_name = last_name
        self.img_paths = img_paths if isinstance(img_paths, list) else [img_paths]
        self.age = age
        self._embeddings = None  # cache embeddings list

    def get_embs(self):
        """Compute embeddings for all images once and cache them"""
        if self._embeddings is None:
            self._embeddings = []
            for path in self.img_paths:
                if not os.path.exists(path):
                    print(f"[ERROR] Image not found: {path}")
                    continue

                img = Image.open(path).convert("RGB")
                face_tensor = mtcnn(img)
                if face_tensor is None:
                    print(f"[WARNING] Face not detected in image: {path}")
                    continue

                face_tensor = face_tensor.unsqueeze(0).to(device)
                with torch.no_grad():
                    emb = model(face_tensor).cpu().numpy()[0]

                self._embeddings.append(emb)

        return self._embeddings

    def to_string(self):
        return f"id:{self.id}, first name:{self.first_name}, last name:{self.last_name}"

    def to_dict(self):
        """Convert person to dictionary (for Firestore or JSON)"""
        return {
            "id": self.id,
            "first_name": self.first_name,
            "last_name": self.last_name,
            "age": self.age,
            "embeddings": [{"vector": emb.tolist()} for emb in self.get_embs()]
        }

