from __future__ import annotations
import os
import csv
from pathlib import Path
from typing import List, Tuple, Dict, Any

import cv2
import numpy as np
from retinaface import RetinaFace
from PIL import Image


# ---------------- Configuration ----------------
VIDEOS_DIR = "videos_database"         # input videos directory
OUTPUT_ROOT = "extracted_faces"        # output root directory
FRAME_INTERVAL = 5                     # process every Nth frame
CONF_THRESH = 0.9                      # detection confidence threshold
MIN_FACE_PX = 60                       # minimum face box side in pixels
OUTPUT_SIZE: tuple[int, int] | None = None  # e.g., (160,160) for uniform crops, or None to keep original size
VIDEO_EXTS = {".mp4", ".avi", ".mkv", ".mov"}
# ------------------------------------------------


def list_video_files(videos_dir: str) -> List[str]:
    """Return a list of video file paths under videos_dir with allowed extensions."""
    out = []
    for name in os.listdir(videos_dir):
        p = os.path.join(videos_dir, name)
        if not os.path.isfile(p):
            continue
        if os.path.splitext(name.lower())[1] in VIDEO_EXTS:
            out.append(p)
    return sorted(out)


def ensure_dir(path: str) -> None:
    """Create directory if it does not exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


def detect_faces(frame_bgr: np.ndarray) -> Dict[str, Any] | None:
    """Run RetinaFace detection on a BGR frame. Returns dict (key->info) or None."""
    return RetinaFace.detect_faces(frame_bgr)


def save_face_crop(frame_bgr: np.ndarray, bbox: Tuple[int, int, int, int], save_path: str) -> bool:
    """
    Crop a face from a BGR frame and save it as JPEG (RGB).
    Returns True if saved, False otherwise.
    """
    x1, y1, x2, y2 = bbox
    h, w = frame_bgr.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return False

    crop_bgr = frame_bgr[y1:y2, x1:x2]
    if crop_bgr.size == 0:
        return False

    if OUTPUT_SIZE is not None:
        crop_bgr = cv2.resize(crop_bgr, OUTPUT_SIZE, interpolation=cv2.INTER_LINEAR)

    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    Image.fromarray(crop_rgb).save(save_path, quality=92)
    return True


def process_single_video(video_path: str, out_dir: str) -> dict:
    """
    Process a single video: iterate frames, detect faces, save crops, write a CSV of detections.
    Returns a summary dict.
    """
    ensure_dir(out_dir)
    csv_path = os.path.join(out_dir, "detections.csv")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0 else 0
    frame_idx = 0
    saved_faces = 0
    written_rows = 0

    with open(csv_path, "w", newline="", encoding="utf-8") as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow(["frame_index", "timestamp_ms", "x1", "y1", "x2", "y2", "score", "face_path"])

        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            frame_idx += 1
            if FRAME_INTERVAL > 1 and (frame_idx % FRAME_INTERVAL) != 0:
                continue

            ts_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
            detections = detect_faces(frame_bgr)

            if not isinstance(detections, dict):
                continue

            face_num_in_frame = 0
            for _, info in detections.items():
                x1, y1, x2, y2 = map(int, info["facial_area"])
                score = float(info.get("score", 1.0))

                if score < CONF_THRESH:
                    continue
                if min(x2 - x1, y2 - y1) < MIN_FACE_PX:
                    continue

                fname = f"frame{frame_idx:06d}_face{face_num_in_frame:02d}.jpg"
                save_path = os.path.join(out_dir, fname)

                if save_face_crop(frame_bgr, (x1, y1, x2, y2), save_path):
                    writer.writerow([frame_idx, ts_ms, x1, y1, x2, y2, f"{score:.4f}", save_path])
                    saved_faces += 1
                    written_rows += 1
                    face_num_in_frame += 1

            # optional: simple progress print every few hundred processed frames
            if FRAME_INTERVAL > 0 and (frame_idx % (FRAME_INTERVAL * 100) == 0):
                print(f"[{os.path.basename(video_path)}] processed frame {frame_idx}/{total_frames}, "
                      f"faces saved so far: {saved_faces}")

    cap.release()
    return {
        "video": os.path.basename(video_path),
        "frames_processed": frame_idx,
        "faces_saved": saved_faces,
        "csv": csv_path
    }


def process_all_videos(videos_dir: str, output_root: str) -> List[dict]:
    """
    Iterate all videos in `videos_dir`, extract faces to per-video folders under `output_root`.
    Returns a list of summary dicts per video.
    """
    ensure_dir(output_root)
    videos = list_video_files(videos_dir)
    if not videos:
        print(f"No video files found in: {videos_dir}")
        return []

    summaries = []
    for vpath in videos:
        base = os.path.splitext(os.path.basename(vpath))[0]
        out_dir = os.path.join(output_root, base)
        print(f"==> Processing video: {vpath}")
        try:
            summary = process_single_video(vpath, out_dir)
            summaries.append(summary)
            print(f"    Done: faces_saved={summary['faces_saved']}  csv={summary['csv']}")
        except Exception as e:
            print(f"    ERROR processing {vpath}: {e}")
    return summaries


def main() -> None:
    ensure_dir(OUTPUT_ROOT)
    summaries = process_all_videos(VIDEOS_DIR, OUTPUT_ROOT)
    print("\nSummary:")
    for s in summaries:
        print(f"  - {s['video']}: faces_saved={s['faces_saved']}  csv={s['csv']}")


if __name__ == "__main__":
    main()
