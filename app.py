# app.py
# Flask API for SurveilNet:
# - Video upload (no progress bar on frontend)
# - Video processing in "store only" mode (saves embeddings+thumbs+meta into frame_store)
# - Add person: computes embeddings, uploads ONLY that person to Firestore, searches matches,
#               appends ALL matches to a feed (for MatchPage), returns best to caller.
# - Static serving of frame_store files for <img src> usage in the UI.

from flask import Flask, request, jsonify, send_from_directory, send_file, make_response
from flask_cors import CORS
from werkzeug.utils import secure_filename
import signal, sys, threading
from pathlib import Path
import mimetypes
import os
import uuid
import json
import time
import torch
import threading

# Project models
from models.face_database import FaceDatabase
from models.face_matcher import FaceMatcher  # kept for VideoProcessor ctor shape
from models.video_processor import VideoProcessor, search_best_match_for_person, search_matches_for_person

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

@app.after_request
def add_cors_headers(resp):
    resp.headers["Access-Control-Allow-Origin"] = "http://localhost:3000"
    resp.headers["Access-Control-Allow-Methods"] = "GET,POST,DELETE,OPTIONS"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return resp

# להחזיר 200 לכל preflight (OPTIONS) לכל נתיב
@app.before_request
def handle_preflight():
    if request.method == "OPTIONS":
        resp = make_response("", 200)
        resp.headers["Access-Control-Allow-Origin"] = "http://localhost:3000"
        resp.headers["Access-Control-Allow-Methods"] = "GET,POST,DELETE,OPTIONS"
        resp.headers["Access-Control-Allow-Headers"] = request.headers.get(
            "Access-Control-Request-Headers", "Content-Type"
        )
        return resp

STOP_EVENT = threading.Event()

PROCESS_JOBS = {}  # job_id -> {"current": int, "total": int, "done": bool, "filename": str}
PROCESS_LOCK = threading.Lock()

# Folders
UPLOAD_FOLDER = "uploads"
VIDEO_FOLDER = "videos_database"
VIDEO_TMP_FOLDER = "videos_tmp"
MATCHES_FOLDER = "matches"
FRAME_STORE_ROOT = "frame_store"

# Keep only the best matches in the feed:
FEED_MIN_SCORE = 0.70      # store only matches with score >= 0.70
FEED_TOP_K = 1             # keep at most top 5 matches overall
FEED_PER_VIDEO = True      # keep only the best match per video
FEED_MIN_FRAME_GAP = 15    # (optional) min frame distance between kept matches in same video

# Feed file for storing all matches (JSONL)
MATCHES_FEED_PATH = Path(MATCHES_FOLDER) / "matches.jsonl"

# Allowed extensions
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp", "mp4"}

# Ensure required folders exist
for d in [UPLOAD_FOLDER, VIDEO_FOLDER, VIDEO_TMP_FOLDER, MATCHES_FOLDER, FRAME_STORE_ROOT]:
    os.makedirs(d, exist_ok=True)
MATCHES_FEED_PATH.parent.mkdir(parents=True, exist_ok=True)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Firebase init
# NOTE: adjust path to your own service account if needed
cred = credentials.Certificate("configs/accountKey.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

# In-memory people DB
face_db = FaceDatabase()


# ----------------------------- Utils ----------------------------- #

def _handle_sigint(signum, frame):
    print("\n[CTRL-C] Stopping now...", flush=True)
    STOP_EVENT.set()
    # Hard-exit prevents hanging threads; use 130 (SIGINT)
    os._exit(130)


signal.signal(signal.SIGINT, _handle_sigint)

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
        matcher = FaceMatcher(face_db)  # not used in store_only
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
def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def append_matches_to_feed(person, matches: list) -> None:
    """
    Append matches to a JSONL feed so the UI (MatchPage) can render a history.
    Each line is a JSON object with person & match details.
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
            }
            fp.write(json.dumps(rec, ensure_ascii=False) + "\n")

def select_best_matches(matches, min_score=0.7, top_k=5, per_video=True, min_frame_gap=0):
    """
    Filter and select only the 'best' matches:
      - score >= min_score
      - if per_video: keep only the single best match per video
      - enforce a minimal frame gap (optional) to avoid near-duplicate frames
      - finally, sort by score desc and keep top_k
    """
    if not matches:
        return []

    # 1) threshold
    filt = [m for m in matches if (m.get("score") or 0) >= float(min_score)]

    if not filt:
        return []

    # 2) optionally pick only the best per video
    if per_video:
        best_by_video = {}
        for m in filt:
            vid = m.get("video") or "__no_video__"
            if vid not in best_by_video or m["score"] > best_by_video[vid]["score"]:
                best_by_video[vid] = m
        filt = list(best_by_video.values())

    # 3) optionally enforce minimal frame gap (per video)
    if min_frame_gap and min_frame_gap > 0:
        grouped = {}
        for m in filt:
            grouped.setdefault(m.get("video") or "__no_video__", []).append(m)

        pruned = []
        for vid, items in grouped.items():
            # sort by frame index ascending (fallback to score)
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

    # 4) sort by score desc and cut to top_k
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
    After success, search stored frame embeddings (from /api/process_video) and:
      - append ALL matches to feed (for MatchPage)
      - return the BEST match in "last_seen" (or None)
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

        # Create the person and compute embeddings locally
        new_person = face_db.add_person(person_id, first_name, last_name, img_paths, age)

        # Upload ONLY this person to Firestore (no bulk)
        face_db.upload_person_to_firestore(db, new_person)

        # Optional: run search against stored frames (frame_store) if embeddings exist
        has_embed = (getattr(new_person, "mean_embedding", None) is not None) or bool(getattr(new_person, "embeddings", None))
        best = None
        if has_embed:
            # get top matches (for feed) + best (for immediate UI)
            matches_list = search_matches_for_person(
                new_person,
                base_folder=FRAME_STORE_ROOT,
                top_k=50,  # pull a generous set from disk first
                min_score=0.50  # lower prefilter; we'll re-filter tighter for the feed
            ) or []

            best = None
            if matches_list:
                # keep only the best ones for the feed, based on our stricter policy:
                best_for_feed = select_best_matches(
                    matches_list,
                    min_score=FEED_MIN_SCORE,
                    top_k=FEED_TOP_K,
                    per_video=FEED_PER_VIDEO,
                    min_frame_gap=FEED_MIN_FRAME_GAP,
                )
                if best_for_feed:
                    best = best_for_feed[0]  # return best to the caller
                    append_matches_to_feed(new_person, best_for_feed)  # persist only the best ones

        return jsonify({
            "message": f"Added {first_name} {last_name}",
            "id": person_id,
            "last_seen": best  # may be None
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
    If client aborts the upload, returns 499 and removes the partial file.
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
                        # cleanup and exit immediately
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

        # finalize move
        os.replace(tmp_path, final_path)

        # store metadata next to video
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


@app.route("/api/process_video", methods=["POST"])
def process_video():
    """
    Process a video to STORE embeddings & thumbnails ONLY (no matching here).
    Requires JSON body: { "filename": "<name.mp4>" }
    """
    data = request.get_json()
    if not data or "filename" not in data:
        return jsonify({"error": "Filename is required in JSON body"}), 400

    filename = secure_filename(data["filename"])
    video_path = os.path.join(VIDEO_FOLDER, filename)
    if not os.path.exists(video_path):
        return jsonify({"error": f"Video file not found: {filename}"}), 404

    # load metadata saved at upload time (optional)
    video_meta = {}
    meta_path = os.path.join(VIDEO_FOLDER, f"{filename}.json")
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                video_meta = json.load(f)
        except Exception:
            video_meta = {}

    # Create processor in store-only mode (no matching)
    matcher = FaceMatcher(face_db)  # not used in store_only mode
    processor = VideoProcessor(
        matcher=matcher,
        output_folder=MATCHES_FOLDER,
        frame_skip=5,
        store_only=True,
        video_meta=video_meta,
        frame_store_root=FRAME_STORE_ROOT,
        device=device,
        stop_event=STOP_EVENT,
    )
    stats = processor.process_video(video_path)

    return jsonify({"message": "Video processed (embeddings stored)", "video": filename, "stats": stats}), 200


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

    # newest first
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
                # prefer higher score; if tie, prefer newer (higher ts)
                if (m.get("score", 0) > cur.get("score", 0)) or (
                    m.get("score", 0) == cur.get("score", 0) and m.get("ts", 0) > cur.get("ts", 0)
                ):
                    best_by_person[pid] = m
        items = list(best_by_person.values())
        # keep consistent order: newest first
        items.sort(key=lambda x: x.get("ts", 0), reverse=True)

    return jsonify({"matches": items}), 200

@app.route("/api/matches_feed/<match_id>", methods=["DELETE", "OPTIONS"])
def delete_match_from_feed(match_id):
    if request.method == "OPTIONS":
        return ("", 200)  # preflight OK

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

# ----------------------------- Static serving for frame_store ----------------------------- #

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


# (Optional) fallback route if frontend still uses ?path=...
@app.route("/api/frame_image")
def serve_frame_image():
    """
    Serve a frame image by absolute or relative path (query param 'path').
    Enforces that the file is under frame_store.
    """
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


# ----------------------------- Legacy / Misc (kept if used elsewhere) ----------------------------- #

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

@app.route("/api/process_video_async", methods=["POST"])
def process_video_async():
    data = request.get_json()
    if not data or "filename" not in data:
        return jsonify({"error": "Filename is required in JSON body"}), 400

    filename = secure_filename(data["filename"])
    video_path = os.path.join(VIDEO_FOLDER, filename)
    if not os.path.exists(video_path):
        return jsonify({"error": f"Video file not found: {filename}"}), 404

    # load optional meta (saved at upload)
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

# ----------------------------- Main ----------------------------- #

if __name__ == "__main__":
    # Initialize people DB from Firestore on startup
    USE_FIRESTORE = True
    if USE_FIRESTORE:
        face_db.load_from_firestore(db)
        print(f"[INFO] Loaded {len(face_db.people)} people from Firestore.")
    else:
        # Optional: local bootstrap (not recommended when using Firestore)
        gallery_folder = "database"
        # face_db.build_from_folder(gallery_folder)
        # face_db.upload_to_firestore(db)
        pass

    app.run(debug=True, use_reloader=False, host="127.0.0.1", port=5000)
