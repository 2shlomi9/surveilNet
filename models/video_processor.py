# models/video_processor.py
# Video processing in "store only" mode: extract face embeddings + thumbnails + per-frame metadata
# and save them under frame_store/<video_name>/{embeds,thumbs,meta.jsonl}.
# No person matching happens during processing. Matching is done later, on-demand.

from __future__ import annotations

import os
import sys
import json
from typing import List, Dict, Optional

import cv2
import numpy as np
import torch
from PIL import Image
from datetime import datetime, timezone, timedelta
from pathlib import Path

# Face detection + embedding backends
from retinaface import RetinaFace
from facenet_pytorch import MTCNN, InceptionResnetV1


# ----------------------------- small utils ----------------------------- #

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two float vectors."""
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    na = float(np.linalg.norm(a) + 1e-8)
    nb = float(np.linalg.norm(b) + 1e-8)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _parse_iso(iso_str: Optional[str]) -> Optional[datetime]:
    """Accept both ...Z and offset forms."""
    if not iso_str:
        return None
    try:
        if iso_str.endswith("Z"):
            iso_str = iso_str.replace("Z", "+00:00")
        return datetime.fromisoformat(iso_str)
    except Exception:
        return None


def _render_progress(cur: int, total: int, prefix: str = "[PROCESS]") -> None:
    """
    Render a single-line progress bar to stdout.
    cur is 0-based frame index; shown to user as 1-based.
    """
    total = max(int(total or 0), 1)
    cur_disp = min(cur + 1, total)
    ratio = cur_disp / total
    width = 30  # bar width in chars
    filled = int(width * ratio)
    bar = "█" * filled + "-" * (width - filled)
    percent = int(round(ratio * 100))
    sys.stdout.write(f"\r{prefix} frame {cur_disp}/{total} |{bar}| {percent:3d}%")
    sys.stdout.flush()


# ----------------------------- main class ----------------------------- #

class VideoProcessor:
    """
    Process a video, detect faces, compute embeddings, and STORE them with thumbnails + metadata.
    This class does NOT perform person matching during processing.
    """

    def __init__(
        self,
        matcher,  # kept for API compatibility (not used when store_only=True)
        output_folder: str,
        frame_skip: int = 5,
        store_only: bool = True,
        video_meta: Optional[Dict] = None,
        frame_store_root: str = "frame_store",
        device: Optional[torch.device] = None,
        stop_event=None,  # threading.Event (optional)
        progress_cb=None,  # callable(frame_idx:int, total:int) for UI progress
    ):
        self.matcher = matcher
        self.output_folder = output_folder
        self.frame_skip = max(1, int(frame_skip))
        self.store_only = store_only
        self.video_meta = video_meta or {}
        self.frame_store_root = frame_store_root
        self.stop_event = stop_event
        self.progress_cb = progress_cb

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Face alignment + embedding
        self.mtcnn = MTCNN(image_size=160, margin=20, post_process=True, device=self.device)
        self.resnet = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)

    # ----------------------------- pipeline ----------------------------- #

    def process_video(self, video_path: str) -> Dict:
        """
        Read frames, detect faces, compute embeddings, and store them (store_only mode).
        Prints current frame and a terminal progress bar updated per frame.
        Returns stats dict.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[ERROR] Failed to open video: {video_path}")
            return {"frames_total": 0, "stored_faces": 0, "fps": 0.0}

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total_frames <= 0:
            total_frames = 1  # avoid div by zero; progress still updates

        video_name = os.path.basename(video_path)
        base_store = os.path.join(self.frame_store_root, os.path.splitext(video_name)[0])
        embeds_dir = os.path.join(base_store, "embeds")
        thumbs_dir = os.path.join(base_store, "thumbs")
        _ensure_dir(embeds_dir)
        _ensure_dir(thumbs_dir)

        meta_path = os.path.join(base_store, "meta.jsonl")
        meta_fp = open(meta_path, "a", encoding="utf-8")

        start_iso = self.video_meta.get("start_time")
        start_dt = _parse_iso(start_iso)
        location = self.video_meta.get("location", "Unknown")

        frame_idx = 0
        stored_faces = 0

        try:
            while True:
                if self.stop_event and self.stop_event.is_set():
                    print("\n[INFO] Processing interrupted (stop_event set).")
                    break

                ok, frame = cap.read()
                h, w = frame.shape[:2]
                if not ok:
                    # finalize progress to 100%
                    _render_progress(total_frames - 1, total_frames)
                    if self.progress_cb:
                        self.progress_cb(total_frames - 1, total_frames)
                    break

                # Print progress to terminal + fire UI callback
                _render_progress(frame_idx, total_frames)
                if self.progress_cb:
                    self.progress_cb(frame_idx, total_frames)

                if frame_idx % self.frame_skip != 0:
                    frame_idx += 1
                    continue

                boxes = self.detect_faces(frame)  # list of [x1,y1,x2,y2]
                # save per det
                for det_i, box in enumerate(boxes):
                    if self.stop_event and self.stop_event.is_set():
                        print("\n[INFO] Processing interrupted inside boxes loop.")
                        break

                    face_img = self.crop_face(frame, box)  # BGR
                    if face_img is None or face_img.size == 0:
                        continue

                    emb = self.get_embedding_from_image(face_img)  # np.ndarray (512,)
                    if emb is None:
                        continue

                    # unique file names
                    base_name = f"{frame_idx}_{det_i}"
                    emb_path = os.path.join(embeds_dir, f"{base_name}.npy")
                    thumb_path = os.path.join(thumbs_dir, f"{base_name}.jpg")

                    # save to disk
                    np.save(emb_path, emb.astype(np.float32))
                    cv2.imwrite(thumb_path, face_img)

                    # calculate time
                    time_iso = None
                    if start_dt:
                        dt = start_dt + timedelta(seconds=float(frame_idx) / max(1.0, fps))
                        time_iso = (
                            dt.astimezone(timezone.utc)
                            .replace(tzinfo=timezone.utc)
                            .isoformat()
                            .replace("+00:00", "Z")
                        )

                    # meta data
                    rec = {
                        "video": video_name,
                        "frame_idx": frame_idx,
                        "det_i": det_i,
                        "fps": fps,
                        "place": location,
                        "time_iso": time_iso,
                        "thumb_path": os.path.abspath(thumb_path),
                        "embed_path": os.path.abspath(emb_path),
                        "box": [int(v) for v in box],  # [x1,y1,x2,y2]
                        "frame_h": frame.shape[0],
                        "frame_w": frame.shape[1],
                    }
                    meta_fp.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    stored_faces += 1

                frame_idx += 1

        finally:
            cap.release()
            meta_fp.close()
            _render_progress(total_frames - 1, total_frames)
            sys.stdout.write("\n")
            sys.stdout.flush()

        return {
            "frames_total": int(total_frames),
            "stored_faces": int(stored_faces),
            "fps": float(fps),
            "store_dir": base_store,
        }

    # ----------------------------- face utils ----------------------------- #

    def detect_faces(self, frame_bgr: np.ndarray) -> List[List[int]]:
        """
        Run RetinaFace and return list of [x1, y1, x2, y2] boxes.
        """
        try:
            detections = RetinaFace.detect_faces(frame_bgr)
        except Exception:
            detections = None

        boxes: List[List[int]] = []
        if not detections or detections == "No face detected":
            return boxes

        for _, det in detections.items():
            area = det.get("facial_area")
            if area and len(area) == 4:
                x1, y1, x2, y2 = [int(v) for v in area]
                boxes.append([max(0, x1), max(0, y1), max(x2, 0), max(y2, 0)])
        return boxes

    def crop_face(self, frame_bgr: np.ndarray, box: List[int]) -> Optional[np.ndarray]:
        """
        Crop the face region safely. Returns BGR image or None.
        """
        h, w = frame_bgr.shape[:2]
        x1, y1, x2, y2 = box
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        if x2 <= x1 or y2 <= y1:
            return None
        return frame_bgr[y1:y2, x1:x2].copy()

    def get_embedding_from_image(self, face_bgr: np.ndarray) -> Optional[np.ndarray]:
        """
        Align with MTCNN to 160x160 and compute 512-d embedding with InceptionResnetV1.
        """
        try:
            rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            aligned = self.mtcnn(pil_img)  # torch.Tensor [3,160,160] or None
            if aligned is None:
                return None
            aligned = aligned.unsqueeze(0).to(self.device)
            with torch.no_grad():
                emb = self.resnet(aligned)  # [1,512]
            return emb.squeeze(0).cpu().numpy()
        except Exception:
            return None


# ----------------------------- search helpers ----------------------------- #

def search_best_match_for_person(person, base_folder: str = "frame_store") -> Optional[Dict]:
    """
    Iterate over stored frame embeddings and return the single best match.
    Prefers person's mean_embedding if available; otherwise uses max over person's embeddings.
    Adds 'frame_url' for direct <img src> (served by Flask route /frame_store/<...>).
    Also returns fps and box.
    """
    has_mean = getattr(person, "mean_embedding", None) is not None
    has_list = bool(getattr(person, "embeddings", None))
    if not person or (not has_mean and not has_list):
        return None
    if not os.path.exists(base_folder):
        return None

    def _to_frame_url(abs_or_rel_path: str) -> Optional[str]:
        if not abs_or_rel_path:
            return None
        p = Path(abs_or_rel_path.replace("\\", "/"))
        if not p.is_absolute():
            p = Path(base_folder) / p
        try:
            p = p.resolve()
            root = Path(base_folder).resolve()
            rel = p.relative_to(root)
            return "/frame_store/" + rel.as_posix()
        except Exception:
            return None

    def _score_vs_person(emb_vec: np.ndarray) -> float:
        if has_mean:
            return _cosine_sim(emb_vec, person.mean_embedding)
        return max(_cosine_sim(emb_vec, e) for e in person.embeddings)

    best = None

    for video_dir in os.listdir(base_folder):
        path = os.path.join(base_folder, video_dir)
        meta_path = os.path.join(path, "meta.jsonl")
        if not os.path.isfile(meta_path):
            continue

        with open(meta_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    embed_path = rec.get("embed_path")
                    if not embed_path:
                        continue
                    if not os.path.isabs(embed_path):
                        embed_path = os.path.join(path, embed_path)
                    if not os.path.exists(embed_path):
                        continue
                    emb = np.load(embed_path)
                except Exception:
                    continue

                score = float(_score_vs_person(emb))
                if (best is None) or (score > best["score"]):
                    frame_img_path = rec.get("thumb_path")
                    if frame_img_path and not os.path.isabs(frame_img_path):
                        frame_img_path = os.path.join(path, frame_img_path)
                    best = {
                        "score": score,
                        "frame_image": frame_img_path,
                        "frame_url": _to_frame_url(frame_img_path),
                        "place": rec.get("place", "Unknown"),
                        "time_iso": rec.get("time_iso"),
                        "video": rec.get("video"),
                        "frame_idx": rec.get("frame_idx"),
                        "fps": rec.get("fps"),
                        "box": rec.get("box"),
                    }

    return best


def search_matches_for_person(
    person,
    base_folder: str = "frame_store",
    top_k: int = 10,
    min_score: float = 0.55,
) -> List[Dict]:
    """
    Return a list of matches sorted by score desc (for MatchPage).
    Each item includes frame_url (served by /frame_store/*), and now also fps and box.
    Filters by min_score and trims to top_k.
    """
    results: List[Dict] = []

    has_mean = getattr(person, "mean_embedding", None) is not None
    has_list = bool(getattr(person, "embeddings", None))
    if not person or (not has_mean and not has_list):
        return results
    if not os.path.exists(base_folder):
        return results

    def _to_frame_url(abs_or_rel_path: str) -> Optional[str]:
        if not abs_or_rel_path:
            return None
        p = Path(abs_or_rel_path.replace("\\", "/"))
        if not p.is_absolute():
            p = Path(base_folder) / p
        try:
            p = p.resolve()
            root = Path(base_folder).resolve()
            rel = p.relative_to(root)
            return "/frame_store/" + rel.as_posix()
        except Exception:
            return None

    def _score_vs_person(emb_vec: np.ndarray) -> float:
        if has_mean:
            return _cosine_sim(emb_vec, person.mean_embedding)
        return max(_cosine_sim(emb_vec, e) for e in person.embeddings)

    for video_dir in os.listdir(base_folder):
        path = os.path.join(base_folder, video_dir)
        meta_path = os.path.join(path, "meta.jsonl")
        if not os.path.isfile(meta_path):
            continue

        with open(meta_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    embed_path = rec.get("embed_path")
                    if not embed_path:
                        continue
                    if not os.path.isabs(embed_path):
                        embed_path = os.path.join(path, embed_path)
                    if not os.path.exists(embed_path):
                        continue
                    emb = np.load(embed_path)
                except Exception:
                    continue

                score = float(_score_vs_person(emb))
                if score < float(min_score):
                    continue

                frame_img_path = rec.get("thumb_path")
                if frame_img_path and not os.path.isabs(frame_img_path):
                    frame_img_path = os.path.join(path, frame_img_path)

                results.append({
                    "score": score,
                    "frame_image": frame_img_path,
                    "frame_url": _to_frame_url(frame_img_path),
                    "place": rec.get("place", "Unknown"),
                    "time_iso": rec.get("time_iso"),
                    "video": rec.get("video"),
                    "frame_idx": rec.get("frame_idx"),
                    "fps": rec.get("fps"),
                    "box": rec.get("box"),
                })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:max(1, int(top_k))]
