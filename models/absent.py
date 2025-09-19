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
    def __init__(self, id, first_name, last_name, img_path=None, age=None, embedding=None, main_img=None):
        self.id = id
        self.first_name = first_name
        self.last_name = last_name
        self.img_path = img_path
        self.age = age
        self.main_img = main_img
        self._embedding = np.array(embedding, dtype=np.float32) if embedding is not None else None  

    def get_emb(self):
        """Compute embedding once (if not provided)"""
        if self._embedding is None and self.img_path:
            if not os.path.exists(self.img_path):
                print(f"[ERROR] Image not found: {self.img_path}")
                return None

            img = Image.open(self.img_path).convert("RGB")
            face_tensor = mtcnn(img)
            if face_tensor is None:
                print(f"[WARNING] Face not detected in image: {self.img_path}")
                return None

            face_tensor = face_tensor.unsqueeze(0).to(device)
            with torch.no_grad():
                emb = model(face_tensor).cpu().numpy()[0]

            self._embedding = emb
        return self._embedding
    
    def to_string(self):
        return f"id:{self.id}, first name:{self.first_name}, last name:{self.last_name}"

    def to_dict(self):
        """Convert person to dictionary (for Firestore or JSON)"""
        return {
            "id": self.id,
            "first_name": self.first_name,
            "last_name": self.last_name,
            "age": self.age,
            "embedding": self.get_emb().tolist() if self.get_emb() is not None else None
        }


