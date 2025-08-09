from __future__ import annotations
import cv2
from typing import Iterator, Tuple

def frames_from_video(video_path: str, frame_interval: int = 5) -> Iterator[Tuple[int, int, "np.ndarray"]]:
    """
    Yield (frame_index, timestamp_ms, frame_bgr) every 'frame_interval' frames.
    Uses OpenCV VideoCapture. Frames are returned in BGR color space.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1
            if frame_idx % frame_interval != 0:
                continue
            ts_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
            yield frame_idx, ts_ms, frame  # frame is BGR
    finally:
        cap.release()
