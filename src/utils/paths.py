import os
from uuid import uuid4
from datetime import datetime

def video_path(media_root: str, camera_id: int, dt_utc: datetime, video_id: str) -> str:
    return os.path.join(
        media_root, "videos", str(camera_id),
        f"{dt_utc.year:04d}", f"{dt_utc.month:02d}", f"{dt_utc.day:02d}",
        f"{video_id}.mp4"
    )

def thumb_path(media_root: str, appearance_id: int) -> str:
    return os.path.join(media_root, "thumbs", f"{appearance_id}.jpg")

def face_path(media_root: str, appearance_id: int) -> str:
    return os.path.join(media_root, "faces", f"{appearance_id}.jpg")

def frame_path(media_root: str, video_id: str, frame_idx: int) -> str:
    return os.path.join(media_root, "frames", video_id, f"{frame_idx:06d}.jpg")

def clip_path(media_root: str, video_id: str, start_ms: int, end_ms: int) -> str:
    return os.path.join(media_root, "clips", f"{video_id}_{start_ms}_{end_ms}.mp4")

def query_image_path(media_root: str, query_id: int) -> str:
    return os.path.join(media_root, "queries", str(query_id), f"{uuid4()}.jpg")
