# src/pipeline/index_video.py
from __future__ import annotations
import os
import cv2
import numpy as np
from typing import Optional
from PIL import Image
from sqlalchemy.engine import Engine
from src.utils.config import load_config, ensure_dirs
from src.pipeline.ingestion import frames_from_video
from src.models.detector import detect_faces_retina
from src.models.align import align_by_eyes
from src.models.embedder import FaceEmbedder
from src.store.indexer import FaissIndex
from src.store.sql import get_engine, run_schema, insert_video, insert_face_appearance, update_vector_map
from src.utils.paths import thumb_path

def quality_score(face_rgb: "np.ndarray") -> float:
    """
    Very simple quality proxy: variance of Laplacian (sharpness) + size factor.
    """
    gray = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2GRAY)
    sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    h, w = face_rgb.shape[:2]
    size = float(min(h, w))
    return 0.5 * sharp + 0.5 * size

def save_thumbnail(img_rgb: "np.ndarray", path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pil = Image.fromarray(img_rgb)
    pil.save(path, quality=92)

def index_video(video_path: str,
                camera_id: Optional[int] = None,
                conf_path: str = "configs/default.yaml") -> None:
    cfg = load_config(conf_path)
    media_root = cfg["paths"]["media_root"]
    db_root = cfg["paths"]["db_root"]
    ensure_dirs(media_root, db_root, os.path.join(media_root, "thumbs"))

    # SQL Server
    engine: Engine = get_engine(cfg["sqlserver"]["conn_str"])
    run_schema(engine)  # safe to call repeatedly

    # Register video (optional)
    with engine.begin() as conn:
        video_id = insert_video(conn, camera_id=camera_id, source_path=video_path)

    # Models
    embedder = FaceEmbedder()
    index = FaissIndex(dim=cfg["embed"]["dim"],
                       index_path=cfg["index"]["faiss_path"],
                       vectors_path=cfg["index"]["vectors_path"])

    # Iterate frames
    for frame_idx, ts_ms, frame_bgr in frames_from_video(video_path, frame_interval=cfg["video"]["frame_interval"]):
        detections = detect_faces_retina(frame_bgr, conf_thresh=cfg["detect"]["conf_thresh"])
        if not detections:
            continue

        for det in detections:
            x1,y1,x2,y2 = det.bbox
            min_px = min(x2-x1, y2-y1)
            if min_px < cfg["detect"]["min_face_px"]:
                continue

            # align
            aligned = align_by_eyes(frame_bgr, det.bbox, det.landmarks, tuple(cfg["align"]["size"]))
            emb = embedder.embed(aligned)  # L2-normalized

            # quality
            aligned_rgb = np.array(aligned)  # RGB
            q = quality_score(aligned_rgb)

            # save appearance + thumb
            with engine.begin() as conn:
                appearance_id = insert_face_appearance(
                    conn, camera_id=camera_id, video_id=video_id,
                    frame_index=frame_idx, ts_ms=ts_ms,
                    bbox=det.bbox, quality=q,
                    thumb_path=None  # set after saving file
                )

            thumb_p = thumb_path(media_root, appearance_id)
            save_thumbnail(aligned_rgb, thumb_p)

            # update thumb path
            with engine.begin() as conn:
                conn.exec_driver_sql(
                    "UPDATE dbo.FaceAppearances SET ThumbPath = ? WHERE AppearanceId = ?",
                    (thumb_p, appearance_id,)
                )

            # add to FAISS
            start_id, end_id = index.add(emb.reshape(1,-1).astype(np.float32))
            vector_id = start_id  # one vector added
            with engine.begin() as conn:
                l2 = float(np.linalg.norm(emb))
                update_vector_map(conn, appearance_id=appearance_id, vector_id=vector_id, l2norm=l2)

    print("Indexing complete.")
