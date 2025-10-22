# app.py
# Flask API for SurveilNet:
# - Video upload
# - Async video processing (store-only) + terminal progress print
# - Add person: store in Firestore, search matches, append bests to feed
# - Serve frame_store files and short video snippets around a matched frame
# - Matches feed (JSONL) with delete / best-per-person
# NOTE: comments are in English only.

from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
import signal, sys, threading, subprocess
from pathlib import Path
import mimetypes
import os
import uuid
import json
import time
import torch
import cv2  # <-- OpenCV for video snippets
from typing import Optional
import imageio.v2 as iio
import subprocess, shutil
import imageio_ffmpeg
from datetime import datetime
import numpy as np

# Project models
from models.face_database import FaceDatabase
from models.face_matcher import FaceMatcher
from models.video_processor import (
    VideoProcessor,
    search_best_match_for_person,
    search_matches_for_person,
)

# Firebase
import firebase_admin
from firebase_admin import credentials, firestore


# ----------------------------- App & Config ----------------------------- #

app = Flask(__name__)
CORS(
    app,
    resources={r"/*": {"origins": ["http://localhost:3000"]}},
    methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type"],
    supports_credentials=False,
)


# Always add CORS headers (also on errors)
@app.after_request
def add_cors_headers(resp):
    resp.headers["Access-Control-Allow-Origin"] = "http://localhost:3000"
    resp.headers["Access-Control-Allow-Methods"] = "GET,POST,DELETE,OPTIONS"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return resp

# OPTIONS preflight ok
@app.before_request
def handle_preflight():
    if request.method == "OPTIONS":
        from flask import make_response
        resp = make_response("", 200)
        resp.headers["Access-Control-Allow-Origin"] = "http://localhost:3000"
        resp.headers["Access-Control-Allow-Methods"] = "GET,POST,DELETE,OPTIONS"
        resp.headers["Access-Control-Allow-Headers"] = request.headers.get(
            "Access-Control-Request-Headers", "Content-Type"
        )
        return resp

STOP_EVENT = threading.Event()

# Folders
UPLOAD_FOLDER = "uploads"
VIDEO_FOLDER = "videos_database"
VIDEO_TMP_FOLDER = "videos_tmp"
MATCHES_FOLDER = "matches"
FRAME_STORE_ROOT = "frame_store"
SNIPPETS_FOLDER = "snippets"  # for 10s clips

# Keep only the best matches in the feed:
FEED_MIN_SCORE = 0.50
FEED_TOP_K = 1
FEED_PER_VIDEO = True
FEED_MIN_FRAME_GAP = 15

# Feed file
MATCHES_FEED_PATH = Path(MATCHES_FOLDER) / "matches.jsonl"

# Allowed extensions
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp", "mp4"}

# Ensure required folders exist
for d in [UPLOAD_FOLDER, VIDEO_FOLDER, VIDEO_TMP_FOLDER, MATCHES_FOLDER, FRAME_STORE_ROOT, SNIPPETS_FOLDER]:
    os.makedirs(d, exist_ok=True)
MATCHES_FEED_PATH.parent.mkdir(parents=True, exist_ok=True)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Firebase init (adjust path to your service account)
cred = credentials.Certificate("configs/accountKey.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

# In-memory people DB
face_db = FaceDatabase()


# ----------------------------- Utils ----------------------------- #

def _handle_sigint(signum, frame):
    print("\n[CTRL-C] Stopping now...", flush=True)
    STOP_EVENT.set()
    os._exit(130)

signal.signal(signal.SIGINT, _handle_sigint)

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def _new_progress(filename: str):
    return {
        "filename": filename,
        "frames_done": 0,
        "frames_total": 0,
        "percent": 0,
        "done": False,
        "canceled": False,
        "error": None,
        "started_at": datetime.utcnow().isoformat() + "Z",
        "finished_at": None,
    }

def _set_progress(job_id: str, **kw):
    with JOBS_LOCK:
        pr = JOBS.get(job_id, {}).get("progress")
        if pr is not None:
            pr.update(kw)

def append_matches_to_feed(person, matches: list) -> None:
    """
    Append selected matches to JSONL feed.
    Also persists fps and box (if available) so snippet can be created later.
    """
    if not matches:
        return
    now = int(time.time())
    with open(MATCHES_FEED_PATH, "a", encoding="utf-8") as fp:
        for m in matches:
            rec = {
                "id": str(uuid.uuid4()),
                "ts": now,
                "person_id": person.id,
                "person_name": f"{person.first_name} {person.last_name}".strip(),
                "person_main_image": getattr(person, "main_img_path", None),
                "score": m.get("score"),
                "place": m.get("place"),
                "time": m.get("time_iso"),
                "video": m.get("video"),
                "frame_idx": m.get("frame_idx"),
                "frame_url": m.get("frame_url"),
                "frame_image": m.get("frame_image"),
                "fps": m.get("fps"),
                "box": m.get("box"),
            }
            fp.write(json.dumps(rec, ensure_ascii=False) + "\n")


def select_best_matches(matches, min_score=0.50, top_k=5, per_video=True, min_frame_gap=0):
    """
    Filter/select best matches by policy.
    """
    if not matches:
        return []

    filt = [m for m in matches if (m.get("score") or 0) >= float(min_score)]
    if not filt:
        return []

    if per_video:
        best_by_video = {}
        for m in filt:
            vid = m.get("video") or "__no_video__"
            if vid not in best_by_video or m["score"] > best_by_video[vid]["score"]:
                best_by_video[vid] = m
        filt = list(best_by_video.values())

    if min_frame_gap and min_frame_gap > 0:
        grouped = {}
        for m in filt:
            grouped.setdefault(m.get("video") or "__no_video__", []).append(m)
        pruned = []
        for vid, items in grouped.items():
            items.sort(key=lambda x: (x.get("frame_idx") is None, x.get("frame_idx", 0)))
            kept = []
            last_frame = None
            for it in items:
                fi = it.get("frame_idx")
                if fi is None or last_frame is None or abs(int(fi) - int(last_frame)) >= int(min_frame_gap):
                    kept.append(it)
                    last_frame = fi
            pruned.extend(kept)
        filt = pruned

    filt.sort(key=lambda x: x["score"], reverse=True)
    return filt[:max(1, int(top_k))]


# ----------------------------- Health ----------------------------- #

@app.route("/api/health", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy", "device": str(device)}), 200


# ----------------------------- People ----------------------------- #

@app.route("/api/people", methods=["GET"])
def get_all_people():
    try:
        return jsonify({"people": [p.to_dict() for p in face_db.people]}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/people/<person_id>", methods=["GET"])
def get_person(person_id):
    for person in face_db.people:
        if person.id == person_id:
            return jsonify(person.to_dict()), 200
    return jsonify({"error": "Person not found"}), 404


@app.route("/api/people", methods=["POST"])
def add_person():
    """
    Add a single person and upload ONLY that person to Firestore.
    Then search stored frames and append best matches to the feed.
    """
    try:
        first_name = request.form.get("first_name")
        last_name  = request.form.get("last_name")
        age        = request.form.get("age")

        if not first_name or not last_name:
            return jsonify({"error": "first_name and last_name are required"}), 400

        if "images" not in request.files:
            return jsonify({"error": "At least one image is required"}), 400

        images = request.files.getlist("images")
        if not images:
            return jsonify({"error": "At least one image is required"}), 400

        person_id = str(uuid.uuid4())
        person_folder = os.path.join(UPLOAD_FOLDER, f"{person_id}_{first_name}_{last_name}")
        os.makedirs(person_folder, exist_ok=True)

        img_paths = []
        for image in images:
            if image and allowed_file(image.filename):
                filename = secure_filename(image.filename)
                save_path = os.path.join(person_folder, filename)
                image.save(save_path)
                img_paths.append(save_path)

        if not img_paths:
            return jsonify({"error": "No valid images provided"}), 400

        new_person = face_db.add_person(person_id, first_name, last_name, img_paths, age)
        face_db.upload_person_to_firestore(db, new_person)

        has_embed = (getattr(new_person, "mean_embedding", None) is not None) or bool(getattr(new_person, "embeddings", None))
        best = None
        if has_embed:
            matches_list = search_matches_for_person(
                new_person,
                base_folder=FRAME_STORE_ROOT,
                top_k=50,
                min_score=0.50
            ) or []

            if matches_list:
                best_for_feed = select_best_matches(
                    matches_list,
                    min_score=FEED_MIN_SCORE,
                    top_k=FEED_TOP_K,
                    per_video=FEED_PER_VIDEO,
                    min_frame_gap=FEED_MIN_FRAME_GAP,
                )
                if best_for_feed:
                    best = best_for_feed[0]
                    append_matches_to_feed(new_person, best_for_feed)

        return jsonify({
            "message": f"Added {first_name} {last_name}",
            "id": person_id,
            "last_seen": best
        }), 201

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ----------------------------- Videos ----------------------------- #

@app.route("/api/videos", methods=["GET"])
def get_videos():
    videos = [f for f in os.listdir(VIDEO_FOLDER) if allowed_file(f) and f.lower().endswith(".mp4")]
    return jsonify({"videos": videos}), 200


@app.route("/api/upload_video", methods=["POST"])
def upload_video():
    """
    Upload a video file with required metadata: start_time, location.
    Writes to a temporary .part file and then renames to final name.
    """
    try:
        if "video" not in request.files:
            return jsonify({"error": "No video file provided"}), 400
        video = request.files["video"]
        if not video or not allowed_file(video.filename):
            return jsonify({"error": "Invalid video file"}), 400

        start_time = request.form.get("start_time")
        location   = request.form.get("location")
        if not start_time or not location:
            return jsonify({"error": "start_time and location are required"}), 400

        filename = secure_filename(video.filename)
        upload_id = str(uuid.uuid4())
        tmp_path   = os.path.join(VIDEO_TMP_FOLDER, f"{upload_id}_{filename}.part")
        final_path = os.path.join(VIDEO_FOLDER, filename)

        chunk_size = 1024 * 1024
        bytes_written = 0
        try:
            with open(tmp_path, "wb") as out:
                while True:
                    if STOP_EVENT.is_set():
                        out.close()
                        try:
                            if os.path.exists(tmp_path):
                                os.remove(tmp_path)
                        finally:
                            return jsonify({"status": "canceled"}), 499
                    chunk = video.stream.read(chunk_size)
                    if not chunk:
                        break
                    out.write(chunk)
                    bytes_written += len(chunk)
        except (ConnectionResetError, BrokenPipeError, OSError):
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            finally:
                return jsonify({"status": "canceled", "bytes_written": bytes_written}), 499
        except Exception as e:
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            finally:
                return jsonify({"error": str(e)}), 500

        os.replace(tmp_path, final_path)

        try:
            meta = {
                "filename": filename,
                "start_time": start_time,
                "location": location,
                "bytes": bytes_written,
            }
            with open(os.path.join(VIDEO_FOLDER, f"{filename}.json"), "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[WARN] Failed to write video metadata: {e}")

        return jsonify({"message": f"Video {filename} uploaded successfully", "filename": filename, "bytes": bytes_written}), 201

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ----------------------------- Async processing (progress) ----------------------------- #

PROCESS_JOBS = {}  # job_id -> {"current": int, "total": int, "done": bool, "filename": str}
PROCESS_LOCK = threading.Lock()

# === Job manager (in-memory) ===
JOBS = {}  # job_id -> {"thread": Thread, "stop": Event, "progress": {...}}
JOBS_LOCK = threading.Lock()
def make_progress_cb(job_id: str):
    def _cb(cur: int, total: int):
        with PROCESS_LOCK:
            job = PROCESS_JOBS.get(job_id)
            if job:
                job["current"] = int(cur) + 1
                job["total"] = max(int(total), 1)
                job["done"] = (job["current"] >= job["total"])
    return _cb

def _run_processing_job(job_id: str, filename: str, video_meta: dict):
    try:
        matcher = FaceMatcher(face_db)
        processor = VideoProcessor(
            matcher=matcher,
            output_folder=MATCHES_FOLDER,
            frame_skip=5,
            store_only=True,
            video_meta=video_meta,
            frame_store_root=FRAME_STORE_ROOT,
            device=device,
            stop_event=STOP_EVENT,
            progress_cb=make_progress_cb(job_id),
        )
        video_path = os.path.join(VIDEO_FOLDER, filename)
        processor.process_video(video_path)
    except Exception as e:
        print(f"[ERROR] job {job_id} failed: {e}")
    finally:
        with PROCESS_LOCK:
            job = PROCESS_JOBS.get(job_id)
            if job:
                total = max(job.get("total", 1), 1)
                job["current"] = total
                job["total"] = total
                job["done"] = True

@app.route("/api/process_video_async", methods=["POST"])
def process_video_async():
    data = request.get_json() or {}
    filename = secure_filename(data.get("filename", ""))

    if not filename:
        return jsonify({"error": "filename is required"}), 400

    video_path = os.path.join(VIDEO_FOLDER, filename)
    if not os.path.exists(video_path):
        return jsonify({"error": f"Video not found: {filename}"}), 404

    # צור job חדש
    job_id = str(uuid.uuid4())
    stop_ev = threading.Event()
    progress = _new_progress(filename)

    def progress_cb(done, total):
        pct = int((done / total) * 100) if total else 0
        _set_progress(job_id, frames_done=done, frames_total=total, percent=pct)

    def worker():
        try:
            # video_meta (אופציונלי)
            meta_path = os.path.join(VIDEO_FOLDER, f"{filename}.json")
            video_meta = {}
            if os.path.exists(meta_path):
                try:
                    with open(meta_path, "r", encoding="utf-8") as f:
                        video_meta = json.load(f)
                except Exception:
                    video_meta = {}

            matcher = FaceMatcher(face_db)
            processor = VideoProcessor(
                matcher=matcher,
                output_folder=MATCHES_FOLDER,
                frame_skip=5,
                store_only=True,
                video_meta=video_meta,
                frame_store_root=FRAME_STORE_ROOT,
                device=device,
                stop_event=stop_ev,
                progress_cb=progress_cb,
            )

            stats = processor.process_video(video_path)
            _set_progress(job_id, done=True, percent=100, finished_at=datetime.utcnow().isoformat() + "Z")
        except Exception as e:
            _set_progress(job_id, error=str(e), done=True, finished_at=datetime.utcnow().isoformat() + "Z")

    th = threading.Thread(target=worker, name=f"process-{filename}", daemon=True)
    with JOBS_LOCK:
        JOBS[job_id] = {"thread": th, "stop": stop_ev, "progress": progress}
    th.start()

    return jsonify({"job_id": job_id}), 202

@app.route("/api/process_status", methods=["GET"])
def process_status():
    job_id = request.args.get("job_id", "")
    if not job_id:
        return jsonify({"error": "job_id is required"}), 400

    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return jsonify({"error": "job not found"}), 404
        cur = int(job.get("current", 0))
        total = max(int(job.get("total", 1)), 1)
        done = bool(job.get("done", False))
        percent = int(round(min(cur, total) * 100 / total))

    return jsonify({
        "filename": job.get("filename"),
        "current": cur,
        "total": total,
        "percent": percent,
        "done": done
    }), 200

@app.route("/api/process_cancel", methods=["POST"])
def process_cancel():
    job_id = request.args.get("job_id", "")
    if not job_id:
        try:
            body = request.get_json() or {}
            job_id = body.get("job_id", "")
        except Exception:
            pass
    if not job_id:
        return jsonify({"error": "job_id is required"}), 400

    with JOBS_LOCK:
        job = JOBS.get(job_id)

    if not job:
        return jsonify({"error": "job not found"}), 404

    # בקש מהעיבוד להיעצר
    job["stop"].set()

    # ננסה להצטרף ל־thread לזמן קצר (לא חוסם לנצח)
    job["thread"].join(timeout=2.0)

    _set_progress(job_id, canceled=True, done=True, finished_at=datetime.utcnow().isoformat() + "Z")

    return jsonify({"message": "processing canceled", "job_id": job_id}), 200

# ----------------------------- Matches / Feed ----------------------------- #

@app.route("/api/matches_feed", methods=["GET"])
def get_matches_feed():
    """
    Return saved matches as a list (newest first).
    If best_per_person=1, return only the top-scoring match per person.
    """
    items = []
    if MATCHES_FEED_PATH.exists():
        with open(MATCHES_FEED_PATH, "r", encoding="utf-8") as fp:
            for line in fp:
                try:
                    items.append(json.loads(line))
                except Exception:
                    continue

    items.sort(key=lambda x: x.get("ts", 0), reverse=True)

    best_only = str(request.args.get("best_per_person", "0")).lower() in ("1", "true", "yes")
    if best_only and items:
        best_by_person = {}
        for m in items:
            pid = m.get("person_id") or "__unknown__"
            cur = best_by_person.get(pid)
            if cur is None:
                best_by_person[pid] = m
            else:
                if (m.get("score", 0) > cur.get("score", 0)) or (
                    m.get("score", 0) == cur.get("score", 0) and m.get("ts", 0) > cur.get("ts", 0)
                ):
                    best_by_person[pid] = m
        items = list(best_by_person.values())
        items.sort(key=lambda x: x.get("ts", 0), reverse=True)

    return jsonify({"matches": items}), 200

@app.route("/api/matches_feed/<match_id>", methods=["DELETE", "OPTIONS"])
def delete_match_from_feed(match_id):
    if request.method == "OPTIONS":
        return ("", 200)

    if not MATCHES_FEED_PATH.exists():
        return jsonify({"error": "Feed file not found"}), 404

    kept, removed = [], 0
    with open(MATCHES_FEED_PATH, "r", encoding="utf-8") as fp:
        for line in fp:
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if obj.get("id") == match_id:
                removed += 1
            else:
                kept.append(obj)

    with open(MATCHES_FEED_PATH, "w", encoding="utf-8") as fp:
        for obj in kept:
            fp.write(json.dumps(obj, ensure_ascii=False) + "\n")

    if removed == 0:
        return jsonify({"error": "Match not found"}), 404
    return jsonify({"message": f"Deleted {removed} match item", "deleted": removed}), 200

def _person_mean_embedding(person):
    """
    Return a single embedding vector (np.ndarray, L2-normalized) for the person:
    prefer person.mean_embedding if exists; otherwise average over person.embeddings.
    Return None if missing.
    """
    vec = None
    try:
        if getattr(person, "mean_embedding", None) is not None:
            vec = np.array(person.mean_embedding, dtype=np.float32)
        elif getattr(person, "embeddings", None):
            arrs = [np.array(e, dtype=np.float32) for e in person.embeddings if e is not None]
            if arrs:
                vec = np.mean(arrs, axis=0).astype(np.float32)
        if vec is None:
            return None
        # L2 normalize
        n = np.linalg.norm(vec)
        if n > 0:
            vec = vec / n
        return vec
    except Exception:
        return None


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


@app.route("/api/match_boxes", methods=["GET"])
def api_match_boxes():
    """
    Return the best (highest-score) face box per frame within [start_idx, end_idx] for a given person & video.
    Query params:
      - person_id (required)
      - video (required)  # the video file name used in frame_store/<video_name_noext>
      - start_idx (required, int)
      - end_idx   (required, int)  # inclusive
      - threshold (optional, float; default 0.60)
    Response:
    {
      "video": "...",
      "fps": <float|None>,
      "frame_w": <int|None>,
      "frame_h": <int|None>,
      "boxes": [
        {"frame_idx": i, "box": [x, y, w, h], "score": 0.82}
      ]
    }
    Notes:
    - Picks at most ONE box per frame: the detection whose cosine vs person is maximal and >= threshold.
    - Uses embeddings and boxes saved by the store-only pipeline in frame_store/<video_noext>/meta.jsonl
    """
    try:
        person_id = request.args.get("person_id", "").strip()
        video_name = request.args.get("video", "").strip()
        start_idx = int(request.args.get("start_idx", "-1"))
        end_idx   = int(request.args.get("end_idx", "-1"))
        threshold = float(request.args.get("threshold", "0.60"))

        if not person_id or not video_name or start_idx < 0 or end_idx < 0:
            return jsonify({"error": "person_id, video, start_idx, end_idx are required"}), 400
        if end_idx < start_idx:
            return jsonify({"error": "end_idx must be >= start_idx"}), 400

        # find the person in memory
        person = None
        for p in face_db.people:
            if p.id == person_id:
                person = p
                break
        if person is None:
            return jsonify({"error": "person not found"}), 404

        pvec = _person_mean_embedding(person)
        if pvec is None:
            return jsonify({"error": "person has no embeddings"}), 400

        # derive frame_store path
        video_noext = os.path.splitext(video_name)[0]
        root_dir = Path(FRAME_STORE_ROOT) / video_noext
        meta_path = root_dir / "meta.jsonl"
        if not meta_path.is_file():
            return jsonify({"error": "meta.jsonl not found for video"}), 404

        # optional global info (fps, frame size) – if saved in the first line; else fallback to None
        fps_val = None
        frame_w = None
        frame_h = None

        best_per_frame = {}  # frame_idx -> {"box":[x,y,w,h], "score":float}

        with open(meta_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except Exception:
                    continue

                fi = rec.get("frame_idx")
                if fi is None:
                    continue
                if fi < start_idx or fi > end_idx:
                    continue

                embed_path = rec.get("embed_path")
                if not embed_path or not os.path.exists(embed_path):
                    continue

                # load embedding for this detection
                try:
                    emb = np.load(embed_path).astype(np.float32)
                except Exception:
                    continue

                # cosine vs person
                score = _cosine_sim(emb, pvec)
                if score < threshold:
                    # below threshold → ignore this detection for the frame
                    continue

                # track the best candidate in this frame
                cur = best_per_frame.get(fi)
                if (cur is None) or (score > cur["score"]):
                    # ensure we provide a box; if missing, skip
                    box = rec.get("box")
                    if not box or not isinstance(box, (list, tuple)) or len(box) != 4:
                        continue
                    best_per_frame[fi] = {
                        "box": [float(box[0]), float(box[1]), float(box[2]), float(box[3])],
                        "score": float(score)
                    }

                # collect optional globals if present
                if fps_val is None and rec.get("fps") is not None:
                    try:
                        fps_val = float(rec.get("fps"))
                    except Exception:
                        fps_val = None
                if frame_w is None and rec.get("frame_w") is not None:
                    try:
                        frame_w = int(rec.get("frame_w"))
                    except Exception:
                        frame_w = None
                if frame_h is None and rec.get("frame_h") is not None:
                    try:
                        frame_h = int(rec.get("frame_h"))
                    except Exception:
                        frame_h = None

        # pack results sorted by frame_idx
        boxes = []
        for fi in range(start_idx, end_idx + 1):
            if fi in best_per_frame:
                entry = best_per_frame[fi]
                boxes.append({"frame_idx": fi, "box": entry["box"], "score": entry["score"]})

        return jsonify({
            "video": video_name,
            "fps": fps_val,
            "frame_w": frame_w,
            "frame_h": frame_h,
            "boxes": boxes
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ----------------------------- frame_store & snippets serving ----------------------------- #

@app.route("/frame_store/<path:subpath>")
def serve_frame_store(subpath):
    """
    Serve files from frame_store by relative URL.
    Example: /frame_store/<video_name>/thumbs/215.jpg
    """
    safe_sub = subpath.replace("\\", "/")
    root = Path(FRAME_STORE_ROOT).resolve()
    p = (root / safe_sub).resolve()

    if root not in p.parents and p != root:
        return jsonify({"error": "path outside of frame_store"}), 403
    if not p.is_file():
        return jsonify({"error": f"file not found: {p.name}"}), 404

    mime = mimetypes.guess_type(str(p))[0] or "application/octet-stream"
    return send_file(str(p), mimetype=mime)

@app.route("/api/frame_image")
def serve_frame_image():
    raw_path = request.args.get("path") or ""
    if not raw_path:
        return jsonify({"error": "path query param is required"}), 400

    raw_path = raw_path.replace("\\", "/")
    p = Path(raw_path)
    if not p.is_absolute():
        p = Path(FRAME_STORE_ROOT) / p

    root = Path(FRAME_STORE_ROOT).resolve()
    try:
        p = p.resolve()
    except Exception as e:
        return jsonify({"error": f"bad path: {e}"}), 400

    if root not in p.parents and p != root:
        return jsonify({"error": "path outside of frame_store"}), 403
    if not p.is_file():
        return jsonify({"error": f"file not found: {p.name}"}), 404

    mime = mimetypes.guess_type(str(p))[0] or "application/octet-stream"
    return send_file(str(p), mimetype=mime)

# Serve pre-generated snippets
@app.route("/snippets/<path:filename>")
def serve_snippet(filename):
    safe = filename.replace("\\", "/")
    root = Path(SNIPPETS_FOLDER).resolve()
    p = (root / safe).resolve()
    if root not in p.parents and p != root:
        return jsonify({"error": "path outside of snippets"}), 403
    if not p.is_file():
        return jsonify({"error": "file not found"}), 404
    mime = "video/mp4"
    return send_file(str(p), mimetype=mime)


# ----------------------------- Video snippet API (OpenCV) ----------------------------- #

def _find_meta_record(video_name_noext: str, frame_idx: int) -> Optional[dict]:
    """
    Load frame_store/<video_name_noext>/meta.jsonl and return record with matching frame_idx.
    """
    meta_path = Path(FRAME_STORE_ROOT) / video_name_noext / "meta.jsonl"
    if not meta_path.is_file():
        return None
    try:
        with open(meta_path, "r", encoding="utf-8") as fp:
            for line in fp:
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                if int(rec.get("frame_idx", -1)) == int(frame_idx):
                    return rec
    except Exception:
        return None
    return None

def _open_writer_mp4(out_path: Path, fps: float, size: tuple) -> Optional[cv2.VideoWriter]:
    """
    Try to open a cv2.VideoWriter for MP4 using several common codecs.
    Returns an opened writer or None.
    """
    w, h = size
    # Try mp4v (most common on Windows)
    for fourcc_str in ("mp4v", "avc1"):
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        writer = cv2.VideoWriter(str(out_path), fourcc, float(max(fps, 1.0)), (int(w), int(h)))
        if writer.isOpened():
            return writer
        writer.release()
    return None

def _write_mp4_opencv(out_path: Path, cap, start_frame, end_frame, fps, size, draw_box):
    writer = _open_writer_mp4(out_path, fps, size)
    if writer is None:
        return 0
    count = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    cur = start_frame
    while cur <= end_frame:
        ok, frame = cap.read()
        if not ok:
            break
        if draw_box:
            x1, y1, x2, y2 = draw_box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
        writer.write(frame)
        count += 1
        cur += 1
    writer.release()
    return count

def _write_gif(out_path: Path, cap, start_frame, end_frame, fps, draw_box):
    frames = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    cur = start_frame
    while cur <= end_frame:
        ok, frame = cap.read()
        if not ok:
            break
        if draw_box:
            x1, y1, x2, y2 = draw_box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
        # BGR -> RGB for GIF
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
        cur += 1
    if not frames:
        return 0
    # duration per frame in seconds
    per = 1.0 / max(fps, 1.0)
    iio.mimsave(out_path, frames, duration=per, loop=0)  # infinite loop
    return len(frames)

def ensure_h264(mp4_in: Path) -> Path:
    """
    If mp4_in isn't H.264, create (once) an H.264 copy next to it and return the new path.
    Uses imageio-ffmpeg bundled ffmpeg — no system install needed.
    """
    mp4_in = Path(mp4_in)
    out = mp4_in.with_name(mp4_in.stem + "_h264.mp4")
    if out.is_file() and out.stat().st_size > 0:
        return out

    ff = imageio_ffmpeg.get_ffmpeg_exe()  # bundled ffmpeg path
    # Fast transcode to H.264 baseline (compatible with browsers) / yuv420p / faststart
    cmd = [
        ff, "-y",
        "-i", str(mp4_in),
        "-c:v", "libx264", "-preset", "veryfast", "-profile:v", "baseline",
        "-level", "3.0", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-an",  # no audio
        str(out)
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        if out.is_file() and out.stat().st_size > 0:
            return out
    except Exception as e:
        print("[H264] transcode failed:", e)
    # fallback
    return mp4_in


@app.route("/api/video_snippet", methods=["GET"])
def api_video_snippet():
    """
    Create (or return cached) ~10s MP4 clip around a matched frame.
    Flow:
      1) Read frames with OpenCV and write MP4 (mp4v).
      2) Transcode to H.264 (baseline/yuv420p) via imageio-ffmpeg for browser playback.
      3) Return URL under /snippets/.
    Query params:
      - video: filename in videos_database (e.g. video.mp4)
      - frame_idx: integer
      - window: seconds before/after (default 5)
      - annotate: 1/0 draw face box
      - (optional) fps: override FPS
      - (optional) box: "x1,y1,x2,y2"
    """
    video = request.args.get("video") or ""
    frame_idx_q = request.args.get("frame_idx") or ""
    window = float(request.args.get("window", 5))
    annotate = str(request.args.get("annotate", "1")).lower() in ("1", "true", "yes")

    if not video or frame_idx_q == "":
        return jsonify({"error": "video and frame_idx are required"}), 400
    try:
        frame_idx = int(frame_idx_q)
    except Exception:
        return jsonify({"error": "frame_idx must be integer"}), 400

    filename = secure_filename(video)
    video_path = Path(VIDEO_FOLDER) / filename
    if not video_path.is_file():
        return jsonify({"error": f"Video not found: {filename}"}), 404

    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return jsonify({"error": "Failed to open video"}), 500

    # fps / frame count / size
    cap_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if cap_fps <= 0:
        cap_fps = request.args.get("fps", type=float) or 25.0  # fallback
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if width == 0 or height == 0:
        ok, frm = cap.read()
        if ok:
            height, width = frm.shape[:2]
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Box
    box_str = request.args.get("box")
    if annotate and not box_str:
        meta_rec = _find_meta_record(Path(filename).stem, frame_idx)
        if meta_rec and meta_rec.get("box"):
            b = meta_rec["box"]
            if isinstance(b, list) and len(b) == 4:
                box_str = ",".join(map(str, b))
    draw_box = None
    if annotate and box_str:
        try:
            x1, y1, x2, y2 = [int(v) for v in box_str.split(",")]
            draw_box = (x1, y1, x2, y2)
        except Exception:
            draw_box = None

    # Frame window (robust by frames)
    window_frames = max(int(round(window * cap_fps)), 1)
    start_frame = max(frame_idx - window_frames, 0)
    end_frame = min(frame_idx + window_frames, max(total_frames - 1, 0))
    if end_frame <= start_frame:
        end_frame = min(start_frame + 1, max(total_frames - 1, 1))

    start_sec = start_frame / cap_fps
    duration = max((end_frame - start_frame + 1) / cap_fps, 0.04)

    # Cache keys
    base_key = f"{Path(filename).stem}__f{frame_idx}__w{int(window)}__ann{1 if draw_box else 0}"
    mp4_path = Path(SNIPPETS_FOLDER) / (base_key + ".mp4")
    h264_path = Path(SNIPPETS_FOLDER) / (base_key + "_h264.mp4")

    # If H.264 already exists, return it immediately
    if h264_path.is_file() and h264_path.stat().st_size > 0:
        cap.release()
        return jsonify({"kind": "video", "url": f"/snippets/{h264_path.name}",
                        "start": start_sec, "duration": duration, "annotated": bool(draw_box)}), 200

    # If raw mp4 exists but no H.264 yet, try to ensure H.264 and return
    if mp4_path.is_file() and mp4_path.stat().st_size > 0 and not h264_path.is_file():
        out = ensure_h264(mp4_path)
        cap.release()
        return jsonify({"kind": "video", "url": f"/snippets/{out.name}",
                        "start": start_sec, "duration": duration, "annotated": bool(draw_box)}), 200

    # Otherwise, write raw MP4 with OpenCV first
    if width <= 0 or height <= 0:
        cap.release()
        return jsonify({"error": "Invalid video dimensions"}), 500

    writer = _open_writer_mp4(mp4_path, cap_fps, (width, height))
    if writer is None:
        cap.release()
        return jsonify({"error": "Failed to initialize video writer"}), 500

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    cur = start_frame
    frames_written = 0
    try:
        while cur <= end_frame:
            ok, frame = cap.read()
            if not ok:
                break
            if draw_box:
                x1, y1, x2, y2 = draw_box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            writer.write(frame)
            frames_written += 1
            cur += 1
    finally:
        writer.release()
        cap.release()

    if not mp4_path.is_file() or mp4_path.stat().st_size == 0 or frames_written == 0:
        return jsonify({"error": "Failed to create snippet (no frames)"}), 500

    # Transcode to H.264 for browser playback
    out = ensure_h264(mp4_path)
    return jsonify({"kind": "video", "url": f"/snippets/{out.name}",
                    "start": start_sec, "duration": duration, "annotated": bool(draw_box)}), 200

# ----------------------------- Legacy / Misc ----------------------------- #

@app.route("/matches/<filename>")
def serve_match(filename):
    return send_from_directory(MATCHES_FOLDER, filename)

@app.route("/api/matches/<filename>", methods=["DELETE"])
def delete_match(filename):
    try:
        file_path = os.path.join(MATCHES_FOLDER, filename)
        if not os.path.exists(file_path):
            return jsonify({"error": "File not found"}), 404
        os.remove(file_path)
        return jsonify({"message": f"File {filename} deleted successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/matches", methods=["GET"])
def get_matches():
    files = [f for f in os.listdir(MATCHES_FOLDER) if allowed_file(f) and f.lower().endswith((".jpg", ".png"))]
    return jsonify({"matches": files}), 200

@app.route("/api/matches", methods=["DELETE"])
def delete_all_matches():
    try:
        deleted_files = []
        for filename in os.listdir(MATCHES_FOLDER):
            if allowed_file(filename) and filename.lower().endswith((".jpg", ".png")):
                file_path = os.path.join(MATCHES_FOLDER, filename)
                os.remove(file_path)
                deleted_files.append(filename)
        if not deleted_files:
            return jsonify({"message": "No matches found to delete"}), 200
        return jsonify({"message": f"Deleted {len(deleted_files)} match files", "deleted_files": deleted_files}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ----------------------------- Main ----------------------------- #

if __name__ == "__main__":
    USE_FIRESTORE = True
    if USE_FIRESTORE:
        face_db.load_from_firestore(db)
        print(f"[INFO] Loaded {len(face_db.people)} people from Firestore.")
    app.run(debug=True, use_reloader=False, host="127.0.0.1", port=5000)
