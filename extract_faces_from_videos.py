# extract_faces_from_videos_timed.py
from __future__ import annotations

# ---- Env flags for TF/RetinaFace (must be set before importing retinaface/tensorflow) ----
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")       # reduce TF verbosity
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")  # avoid pre-allocating full VRAM

import csv
from pathlib import Path
from typing import List, Tuple, Dict, Any
from time import perf_counter

import cv2
import numpy as np
from retinaface import RetinaFace
from PIL import Image


# ---------------- Configuration ----------------
VIDEOS_DIR = "videos_database"          # input videos directory
OUTPUT_ROOT = "extracted_faces"         # output root directory

FRAME_INTERVAL = 5                      # process every Nth frame
CONF_THRESH = 0.90                      # detection confidence threshold
MIN_FACE_PX = 60                        # minimum face box side in pixels

# Downscale large frames before detection to save VRAM/time.
# Typical good range: 960-1280; increase if faces are very small.
MAX_DET_SIDE = 960

# If you want uniform crops, set a size like (160, 160); otherwise None keeps original crop size.
OUTPUT_SIZE: tuple[int, int] | None = None

VIDEO_EXTS = {".mp4", ".avi", ".mkv", ".mov"}

# Print progress every this many processed frames (after skipping)
PROGRESS_EVERY = 100
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


def resize_for_detection(frame_bgr: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    """
    Resize (downscale) the frame so that max(height, width) <= max_side, preserving aspect ratio.
    Returns the resized frame and the applied scale factor (resized / original).
    """
    h, w = frame_bgr.shape[:2]
    scale = 1.0
    max_hw = max(h, w)
    if max_hw > max_side:
        scale = max_side / float(max_hw)
        new_w, new_h = int(w * scale), int(h * scale)
        frame_bgr = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return frame_bgr, scale


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
    Process a single video: iterate frames (with skip), downscale for detection, detect faces,
    save crops, write a CSV of detections, and collect timing stats.
    Returns a summary dict.
    """
    ensure_dir(out_dir)
    csv_path = os.path.join(out_dir, "detections.csv")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0 else 0

    # Timing accumulators (only for processed frames, i.e., after frame skipping)
    t_wall_start = perf_counter()
    processed_frames = 0
    det_ms_acc = 0.0
    crop_ms_acc = 0.0

    frame_idx = 0
    saved_faces = 0

    with open(csv_path, "w", newline="", encoding="utf-8") as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow(["frame_index", "timestamp_ms", "x1", "y1", "x2", "y2", "score", "face_path"])

        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            frame_idx += 1

            # skip frames to reduce compute
            if FRAME_INTERVAL > 1 and (frame_idx % FRAME_INTERVAL) != 0:
                continue

            processed_frames += 1
            ts_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))

            # Downscale before detection to save VRAM/time
            frame_small, scale = resize_for_detection(frame_bgr, MAX_DET_SIDE)

            # Detection timing
            t0 = perf_counter()
            detections = detect_faces(frame_small)
            t1 = perf_counter()
            det_ms = (t1 - t0) * 1000.0
            det_ms_acc += det_ms

            if not isinstance(detections, dict):
                # Progress print
                if PROGRESS_EVERY > 0 and (processed_frames % PROGRESS_EVERY == 0):
                    avg_det = det_ms_acc / max(1, processed_frames)
                    elapsed = perf_counter() - t_wall_start
                    fps = processed_frames / max(1e-9, elapsed)
                    print(f"[{os.path.basename(video_path)}] processed={processed_frames}, "
                          f"avg_det_ms={avg_det:.1f}, fps~{fps:.2f}, faces_saved={saved_faces}")
                continue

            face_num_in_frame = 0
            # Cropping/saving timing per frame (accumulate across faces)
            t_crop_start = perf_counter()

            for _, info in detections.items():
                x1, y1, x2, y2 = map(int, info["facial_area"])
                score = float(info.get("score", 1.0))

                if score < CONF_THRESH:
                    continue
                # Map bbox back to original frame coordinates if we downscaled
                if scale != 1.0:
                    inv = 1.0 / scale
                    x1 = int(x1 * inv); y1 = int(y1 * inv); x2 = int(x2 * inv); y2 = int(y2 * inv)

                if min(x2 - x1, y2 - y1) < MIN_FACE_PX:
                    continue

                fname = f"frame{frame_idx:06d}_face{face_num_in_frame:02d}.jpg"
                save_path = os.path.join(out_dir, fname)

                if save_face_crop(frame_bgr, (x1, y1, x2, y2), save_path):
                    writer.writerow([frame_idx, ts_ms, x1, y1, x2, y2, f"{score:.4f}", save_path])
                    saved_faces += 1
                    face_num_in_frame += 1

            t_crop_end = perf_counter()
            crop_ms_acc += (t_crop_end - t_crop_start) * 1000.0

            # Progress print
            if PROGRESS_EVERY > 0 and (processed_frames % PROGRESS_EVERY == 0):
                avg_det = det_ms_acc / max(1, processed_frames)
                avg_crop = crop_ms_acc / max(1, processed_frames)
                elapsed = perf_counter() - t_wall_start
                fps = processed_frames / max(1e-9, elapsed)
                print(f"[{os.path.basename(video_path)}] processed={processed_frames}, "
                      f"avg_det_ms={avg_det:.1f}, avg_crop_ms={avg_crop:.1f}, fps~{fps:.2f}, "
                      f"faces_saved={saved_faces}")

    cap.release()

    # Final stats
    t_wall = perf_counter() - t_wall_start
    avg_det_ms = det_ms_acc / max(1, processed_frames)
    avg_crop_ms = crop_ms_acc / max(1, processed_frames)
    fps = processed_frames / max(1e-9, t_wall)

    return {
        "video": os.path.basename(video_path),
        "frames_total": total_frames,
        "frames_processed": processed_frames,
        "faces_saved": saved_faces,
        "csv": csv_path,
        "avg_det_ms": avg_det_ms,
        "avg_crop_ms": avg_crop_ms,
        "fps": fps,
        "wall_secs": t_wall
    }


def process_all_videos(videos_dir: str, output_root: str) -> List[dict]:
    """
    Iterate all videos in `videos_dir`, extract faces to per-video folders under `output_root`.
    Returns a list of summary dicts per video (with timing).
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
            print(f"    Done: faces_saved={summary['faces_saved']}  csv={summary['csv']}  "
                  f"fps~{summary['fps']:.2f}  avg_det_ms={summary['avg_det_ms']:.1f}")
        except Exception as e:
            print(f"    ERROR processing {vpath}: {e}")
    return summaries


def main() -> None:
    ensure_dir(OUTPUT_ROOT)
    summaries = process_all_videos(VIDEOS_DIR, OUTPUT_ROOT)
    print("\nSummary:")
    for s in summaries:
        print(f"  - {s['video']}: faces_saved={s['faces_saved']}, "
              f"frames_processed={s['frames_processed']}/{s['frames_total']}, "
              f"fps~{s['fps']:.2f}, avg_det_ms={s['avg_det_ms']:.1f}, "
              f"avg_crop_ms={s['avg_crop_ms']:.1f}, wall_secs={s['wall_secs']:.2f}, "
              f"csv={s['csv']}")


if __name__ == "__main__":
    main()
