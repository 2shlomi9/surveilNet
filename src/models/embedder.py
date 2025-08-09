# src/models/embedder.py
from __future__ import annotations
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1

class FaceEmbedder:
    """
    Baseline embedder using FaceNet (InceptionResnetV1 pretrained on VGGFace2).
    Produces 512-d embeddings. Input must be an aligned RGB PIL.Image.
    """
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.model = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)
        self.preprocess = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
        ])

    def embed(self, img_rgb: Image.Image) -> np.ndarray:
        t = self.preprocess(img_rgb).unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb = self.model(t)
        v = emb.cpu().numpy()[0]
        # L2 normalize to use inner-product ≡ cosine similarity
        norm = np.linalg.norm(v) + 1e-8
        return v / norm
