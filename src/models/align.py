from __future__ import annotations
import cv2
import numpy as np
from typing import Tuple, Optional
from PIL import Image

def _to_tuple(p):
    return (int(p[0]), int(p[1]))

def align_by_eyes(frame_bgr: "np.ndarray",
                  bbox: Tuple[int,int,int,int],
                  landmarks: Optional[dict],
                  output_size: Tuple[int,int] = (112,112)) -> Image.Image:
    """
    Align a face crop using eye landmarks if available; otherwise, simple center crop & resize.
    Returns a PIL.Image in RGB.
    """
    x1,y1,x2,y2 = bbox
    face_bgr = frame_bgr[y1:y2, x1:x2]
    if landmarks and "left_eye" in landmarks and "right_eye" in landmarks:
        le = _to_tuple(landmarks["left_eye"])
        re = _to_tuple(landmarks["right_eye"])
        le = (le[0]-x1, le[1]-y1)
        re = (re[0]-x1, re[1]-y1)
        dy, dx = re[1]-le[1], re[0]-le[0]
        angle = np.degrees(np.arctan2(dy, dx))
        center = (int((le[0]+re[0])//2), int((le[1]+re[1])//2))
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(face_bgr, M, (face_bgr.shape[1], face_bgr.shape[0]))
        # simple square crop around eyes center
        half = min(rotated.shape[0], rotated.shape[1]) // 2
        x = max(0, center[0]-half)
        y = max(0, center[1]-half)
        crop = rotated[y:y+2*half, x:x+2*half]
    else:
        crop = face_bgr
    crop = cv2.resize(crop, output_size, interpolation=cv2.INTER_LINEAR)
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    return Image.fromarray(crop_rgb)
