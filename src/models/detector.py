# src/models/detector.py
from __future__ import annotations
from typing import List, Optional, Tuple
from pydantic import BaseModel
from retinaface import RetinaFace
import numpy as np

class FaceDetection(BaseModel):
    bbox: Tuple[int,int,int,int]  # (x1,y1,x2,y2)
    landmarks: Optional[dict]     # keys: left_eye, right_eye, nose, mouth_left, mouth_right
    score: float

def detect_faces_retina(frame_bgr: "np.ndarray", conf_thresh: float = 0.9) -> List[FaceDetection]:
    """
    Run RetinaFace on a BGR frame and return detections with bbox and landmarks.
    """
    detections = RetinaFace.detect_faces(frame_bgr)
    out: List[FaceDetection] = []
    if isinstance(detections, dict):
        for _, info in detections.items():
            x1, y1, x2, y2 = map(int, info["facial_area"])
            score = float(info.get("score", 1.0))
            if score < conf_thresh:
                continue
            landmarks = info.get("landmarks", None)
            out.append(FaceDetection(bbox=(x1,y1,x2,y2), landmarks=landmarks, score=score))
    return out
