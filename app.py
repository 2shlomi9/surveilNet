# app.py
# Flask API for SurveilNet:
# - Video upload
# - Async video processing (store-only) + terminal progress print
# - Add person: store in Firestore, search matches, append bests to feed
# - Serve frame_store files and short video snippets around a matched frame
# - Matches feed (JSONL) with delete / best-per-person
# NOTE: comments are in English only.

from typing import Optional
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
SNIPPETS_FOLDER = "snippets"  # <--- new: for 10s clips

# Keep only the best matches in the feed:
FEED_MIN_SCORE = 0.70
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


def append_matches_to_feed(person, matches: list) -> None:
    """
    Append selected matches to JSONL feed.
    Now also persists fps and box (if available) so snippet can be created later.
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
                # NEW FIELDS:
                "fps": m.get("fps"),
                "box": m.get("box"),
            }
            fp.write(json.dumps(rec, ensure_ascii=False) + "\n")


def select_best_matches(matches, min_score=0.7, top_k=5, per_video=True, min_frame_gap=0):
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
    data = request.get_json()
    if not data or "filename" not in data:
        return jsonify({"error": "Filename is required in JSON body"}), 400

    filename = secure_filename(data["filename"])
    video_path = os.path.join(VIDEO_FOLDER, filename)
    if not os.path.exists(video_path):
        return jsonify({"error": f"Video file not found: {filename}"}), 404

    meta_path = os.path.join(VIDEO_FOLDER, f"{filename}.json")
    video_meta = {}
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                video_meta = json.load(f)
        except Exception:
            video_meta = {}

    job_id = str(uuid.uuid4())
    with PROCESS_LOCK:
        PROCESS_JOBS[job_id] = {"current": 0, "total": 1, "done": False, "filename": filename}

    t = threading.Thread(target=_run_processing_job, args=(job_id, filename, video_meta), daemon=True)
    t.start()

    return jsonify({"job_id": job_id}), 202

@app.route("/api/process_status", methods=["GET"])
def process_status():
    job_id = request.args.get("job_id", "")
    if not job_id:
        return jsonify({"error": "job_id is required"}), 400

    with PROCESS_LOCK:
        job = PROCESS_JOBS.get(job_id)
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

# NEW: serve pre-generated snippets
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


# ----------------------------- Video snippet API ----------------------------- #

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

def _ffmpeg_exists() -> bool:
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        return True
    except Exception:
        return False

@app.route("/api/video_snippet", methods=["GET"])
def api_video_snippet():
    """
    Create (or return cached) 10s clip around a matched frame.
    Params:
      - video: filename (e.g., video.mp4)
      - frame_idx: integer frame index
      - window: seconds before/after (default 5)
      - annotate: 1/0 draw face box if available (default 1)
    Returns JSON: { url, start, duration, annotated }
    """
    video = request.args.get("video") or ""
    frame_idx = request.args.get("frame_idx") or ""
    window = float(request.args.get("window", 5))
    annotate = str(request.args.get("annotate", "1")).lower() in ("1", "true", "yes")

    if not video or frame_idx == "":
        return jsonify({"error": "video and frame_idx are required"}), 400

    try:
        frame_idx = int(frame_idx)
    except Exception:
        return jsonify({"error": "frame_idx must be integer"}), 400

    filename = secure_filename(video)
    video_path = Path(VIDEO_FOLDER) / filename
    if not video_path.is_file():
        return jsonify({"error": f"Video not found: {filename}"}), 404

    # try to get fps/box from request (optional)
    fps = request.args.get("fps", type=float)
    box = request.args.get("box")  # expected "x1,y1,x2,y2" if provided

    # fallback: read meta.jsonl for frame record
    if fps is None or (annotate and not box):
        meta_rec = _find_meta_record(Path(filename).stem, frame_idx)
        if meta_rec:
            if fps is None:
                fps = float(meta_rec.get("fps") or 25.0)
            if annotate and not box and meta_rec.get("box"):
                b = meta_rec["box"]
                if isinstance(b, list) and len(b) == 4:
                    box = ",".join(map(str, b))

    if fps is None:
        fps = 25.0

    center_t = max(frame_idx, 0) / max(fps, 1.0)
    start = max(center_t - window, 0.0)
    duration = max(window * 2.0, 0.1)

    # cache filename
    key = f"{Path(filename).stem}__f{frame_idx}__w{int(window)}__ann{1 if annotate and box else 0}.mp4"
    out_path = Path(SNIPPETS_FOLDER) / key
    if out_path.is_file():
        return jsonify({"url": f"/snippets/{out_path.name}", "start": start, "duration": duration, "annotated": bool(annotate and box)}), 200

    # build ffmpeg command
    cmd = ["ffmpeg", "-y", "-ss", f"{start:.3f}", "-i", str(video_path), "-t", f"{duration:.3f}"]
    draw = None
    if annotate and box:
        try:
            x1, y1, x2, y2 = [int(v) for v in box.split(",")]
            w = max(1, x2 - x1)
            h = max(1, y2 - y1)
            draw = f"drawbox=x={x1}:y={y1}:w={w}:h={h}:color=red@0.8:thickness=4"
        except Exception:
            draw = None

    if draw:
        cmd += ["-vf", draw]

    # Re-encode (veryfast) for compatibility
    cmd += ["-c:v", "libx264", "-preset", "veryfast", "-c:a", "aac", str(out_path)]

    if _ffmpeg_exists():
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return jsonify({"url": f"/snippets/{out_path.name}", "start": start, "duration": duration, "annotated": bool(draw)}), 200
        except Exception as e:
            # fallback later
            print(f"[WARN] ffmpeg failed: {e}")

    # Fallback (rare): just return error if ffmpeg unavailable/failed
    return jsonify({"error": "Failed to create snippet (ffmpeg missing or failed)."}), 500


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
