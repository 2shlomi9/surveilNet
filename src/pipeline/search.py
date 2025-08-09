from __future__ import annotations
import os
import numpy as np
import cv2
from typing import List, Tuple
from PIL import Image
from sqlalchemy import text
from sqlalchemy.engine import Engine
from src.utils.config import load_config
from src.models.detector import detect_faces_retina
from src.models.align import align_by_eyes
from src.models.embedder import FaceEmbedder
from src.store.indexer import FaissIndex
from src.store.sql import get_engine

def load_and_align_query(img_path: str, align_size=(112,112)) -> Image.Image:
    """
    Load a query image, detect a single face, and align it.
    If detection fails, fall back to center crop.
    """
    bgr = cv2.imread(img_path)
    if bgr is None:
        raise RuntimeError(f"Failed to read image: {img_path}")
    dets = detect_faces_retina(bgr, conf_thresh=0.5)
    if dets:
        # take the best-scoring detection
        dets = sorted(dets, key=lambda d: d.score, reverse=True)
        aligned = align_by_eyes(bgr, dets[0].bbox, dets[0].landmarks, output_size=align_size)
    else:
        # fallback: center crop and resize
        h, w = bgr.shape[:2]
        side = min(h, w)
        y = (h - side) // 2
        x = (w - side) // 2
        crop = cv2.resize(bgr[y:y+side, x:x+side], align_size)
        aligned = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    return aligned

def search_image(img_path: str, top_k: int = 10, conf_path: str = "configs/default.yaml"):
    cfg = load_config(conf_path)
    engine: Engine = get_engine(cfg["sqlserver"]["conn_str"])

    embedder = FaceEmbedder()
    index = FaissIndex(dim=cfg["embed"]["dim"],
                       index_path=cfg["index"]["faiss_path"],
                       vectors_path=cfg["index"]["vectors_path"])

    aligned = load_and_align_query(img_path, align_size=tuple(cfg["align"]["size"]))
    q = embedder.embed(aligned)  # L2-normalized
    scores, idxs = index.search(q.astype(np.float32), top_k=top_k)

    # fetch metadata from SQL Server
    with engine.begin() as conn:
        results = []
        for rank, (score, vec_id) in enumerate(zip(scores, idxs), start=1):
            row = conn.execute(
                text("SELECT AppearanceId, ThumbPath, CameraId, VideoId, FrameIndex, TimestampMs "
                     "FROM dbo.FaceAppearances WHERE VectorId = :vid"),
                {"vid": int(vec_id)}
            ).fetchone()
            if row:
                results.append({
                    "rank": rank,
                    "score": float(score),
                    "appearance_id": int(row.AppearanceId),
                    "thumb_path": row.ThumbPath,
                    "camera_id": int(row.CameraId) if row.CameraId is not None else None,
                    "video_id": int(row.VideoId) if row.VideoId is not None else None,
                    "frame_index": int(row.FrameIndex),
                    "timestamp_ms": int(row.TimestampMs),
                })
    return results
