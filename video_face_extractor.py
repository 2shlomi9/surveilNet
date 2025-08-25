import os
import cv2
import numpy as np
from PIL import Image
from facenet_pytorch import InceptionResnetV1, MTCNN
from retinaface import RetinaFace
import torch
import pyodbc
import uuid
import configparser

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(image_size=160, margin=0, device=device)
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

def _load_sqlserver_conn_str(cfg_path="configs/host_info.ini"):
    """
    Build a pyodbc SQL Server connection string from an INI file.

    INI format (see config/config.example.ini):
    [sqlserver]
    driver=ODBC Driver 17 for SQL Server
    server=localhost
    database=FaceRecognitionDB
    trusted_connection=yes
    username=
    password=

    Returns:
        str: pyodbc connection string.
    Raises:
        FileNotFoundError/KeyError: if config or required keys are missing.
    """
    cp = configparser.ConfigParser()
    if not cp.read(cfg_path, encoding="utf-8"):
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    s = cp["sqlserver"]
    driver = s.get("driver", "ODBC Driver 17 for SQL Server")
    server = s["server"]
    database = s["database"]
    trusted = s.get("trusted_connection", "yes").strip().lower()

    if trusted in ("yes", "true", "1", "y"):
        # Windows Integrated Security
        return f"Driver={{{driver}}};Server={server};Database={database};Trusted_Connection=yes;"
    else:
        username = s.get("username", "")
        password = s.get("password", "")
        return f"Driver={{{driver}}};Server={server};Database={database};Uid={username};Pwd={password};"

# SQL Connection
conn = pyodbc.connect(_load_sqlserver_conn_str())
cursor = conn.cursor()

def normalize(v):
    """
    L2-normalize a vector.

    Args:
        v (np.ndarray): input vector.

    Returns:
        np.ndarray: L2-normalized vector (same shape). If norm is 0 or non-finite, returns v unchanged.
    """
    n = np.linalg.norm(v)
    if n == 0 or not np.isfinite(n):
        return v
    return v / n


def cosine_similarity(a, b):
    """
    Compute cosine similarity between two vectors.

    Args:
        a (np.ndarray): vector A.
        b (np.ndarray): vector B.

    Returns:
        float: cosine similarity in [-1, 1].
    """
    return np.dot(normalize(a), normalize(b))


def get_all_gallery_embeddings():
    """
    Load all (PersonID, AverageEmbedding) pairs from FaceGallery.

    Returns:
        list[tuple[str, np.ndarray]]: list of (person_id, embedding[512]).
    """
    cursor.execute("SELECT PersonID, AverageEmbedding FROM FaceGallery")
    return [(str(pid), np.frombuffer(emb, dtype=np.float32)) for pid, emb in cursor.fetchall()]


def update_or_insert_gallery(person_id, new_emb):
    """
    Upsert the gallery average embedding for a person:
    - If exists: average old and new (then normalize) and update.
    - If not: insert the (normalized) new embedding.

    Args:
        person_id (str): person unique ID.
        new_emb (np.ndarray): embedding[512] to merge into the gallery.
    """
    cursor.execute("SELECT AverageEmbedding FROM FaceGallery WHERE PersonID=?", person_id)
    row = cursor.fetchone()
    if row:
        existing_emb = np.frombuffer(row[0], dtype=np.float32)
        avg_emb = normalize((existing_emb + new_emb) / 2)
        print(f"Updating PersonID {person_id} avg embedding.")
        cursor.execute("UPDATE FaceGallery SET AverageEmbedding=? WHERE PersonID=?",
                       bytearray(avg_emb.astype(np.float32).tobytes()), person_id)
    else:
        print(f"Inserting new PersonID {person_id} into gallery.")
        cursor.execute("INSERT INTO FaceGallery (PersonID, AverageEmbedding) VALUES (?, ?)",
                       person_id, bytearray(normalize(new_emb).astype(np.float32).tobytes()))
    conn.commit()


def save_embedding_to_db(video_name, frame_num, person_id, embedding, x1, y1, x2, y2):
    """
    Insert a single face embedding row into FaceEmbeddings.

    Args:
        video_name (str): video file name.
        frame_num (int): frame index.
        person_id (str): person unique ID.
        embedding (np.ndarray): face embedding[512].
        x1,y1,x2,y2 (int): face bbox (inclusive-exclusive right/bottom).
    """
    sql = """
    INSERT INTO FaceEmbeddings (VideoName, FrameNumber, PersonID, Embedding, BBox_X1, BBox_Y1, BBox_X2, BBox_Y2)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """
    cursor.execute(sql, video_name, frame_num, person_id,
                   bytearray(embedding.astype(np.float32).tobytes()), x1, y1, x2, y2)
    conn.commit()


def _clamp_bbox_xyxy(x1, y1, x2, y2, W, H):
    """
    Clamp (x1,y1,x2,y2) to image bounds and reject invalid/tiny boxes.

    Args:
        x1,y1,x2,y2 (int/float): bbox corners.
        W,H (int): frame width/height.

    Returns:
        tuple[int,int,int,int] | None: clamped bbox or None if invalid.
    """
    if not all(map(np.isfinite, [x1, y1, x2, y2])):
        return None
    x1 = int(max(0, min(int(round(x1)), W - 1)))
    y1 = int(max(0, min(int(round(y1)), H - 1)))
    x2 = int(max(0, min(int(round(x2)), W)))
    y2 = int(max(0, min(int(round(y2)), H)))
    if x2 <= x1 or y2 <= y1:
        return None
    if (x2 - x1) < 5 or (y2 - y1) < 5:
        return None
    return x1, y1, x2, y2


def _clamp_bbox_xywh(x, y, w, h, W, H):
    """
    Clamp (x,y,w,h) to image bounds and reject invalid/tiny boxes.

    Args:
        x,y,w,h (int/float): bbox top-left and size.
        W,H (int): frame width/height.

    Returns:
        tuple[int,int,int,int] | None: clamped (x,y,w,h) or None if invalid.
    """
    if not all(map(np.isfinite, [x, y, w, h])):
        return None
    x = int(round(x)); y = int(round(y))
    w = int(round(w)); h = int(round(h))
    x = max(0, min(x, W - 1))
    y = max(0, min(y, H - 1))
    x2 = max(x + 1, min(x + w, W))
    y2 = max(y + 1, min(y + h, H))
    w = x2 - x
    h = y2 - y
    if w < 5 or h < 5:
        return None
    return x, y, w, h


def detect_faces_retina(img_bgr):
    """
    Detect faces with RetinaFace, align with MTCNN, and embed with FaceNet.

    Args:
        img_bgr (np.ndarray): BGR frame.

    Returns:
        list[tuple[np.ndarray, tuple[int,int,int,int], np.ndarray]]:
            list of (embedding[512], (x1,y1,x2,y2), face_bgr_crop).
    """
    faces = []
    detections = RetinaFace.detect_faces(img_bgr)
    if isinstance(detections, dict):
        H, W = img_bgr.shape[:2]
        for det in detections.values():
            # get and clamp bbox
            x1, y1, x2, y2 = map(int, det['facial_area'])
            safe = _clamp_bbox_xyxy(x1, y1, x2, y2, W, H)
            if safe is None:
                continue
            x1c, y1c, x2c, y2c = safe

            face_bgr = img_bgr[y1c:y2c, x1c:x2c]
            if face_bgr.size == 0:
                continue

            # convert and align via MTCNN
            face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
            face_pil = Image.fromarray(face_rgb)
            face_tensor = mtcnn(face_pil)
            if face_tensor is not None:
                face_tensor = face_tensor.unsqueeze(0).to(device)
                with torch.no_grad():
                    emb = model(face_tensor).cpu().numpy()[0]
                faces.append((emb, (x1c, y1c, x2c, y2c), face_bgr))
    return faces


def match_to_gallery(embedding, gallery, threshold=0.7):
    """
    Match an embedding to the gallery by cosine similarity.

    Args:
        embedding (np.ndarray): query embedding[512].
        gallery (list[tuple[str,np.ndarray]]): loaded gallery (person_id, embedding).
        threshold (float): minimal cosine similarity to accept as same person.

    Returns:
        str: matched person_id or a new UUID if no match above threshold.
    """
    for person_id, gallery_emb in gallery:
        sim = cosine_similarity(embedding, gallery_emb)
        if sim > threshold:
            return person_id
    return str(uuid.uuid4())  # new person


def process_video(video_path, frame_skip=5):
    """
    Process a video:
    - Every 'frame_skip' frames: run RetinaFace detection + MTCNN align + FaceNet embed.
    - Between detections: update CSRT trackers, crop safely with clamped bboxes, re-embed.

    Args:
        video_path (str): path to video file.
        frame_skip (int): detect every N frames; in-between use trackers.
    """
    video_name = os.path.basename(video_path)
    cap = cv2.VideoCapture(video_path)
    frame_num = 0

    gallery = get_all_gallery_embeddings()
    trackers = {}  # person_id -> tracker

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_num += 1

        if frame_num % frame_skip != 0:
            # Update existing trackers (safe clamp to avoid empty ROI)
            to_remove = []
            H, W = frame.shape[:2]
            for pid, tracker in list(trackers.items()):
                ok, bbox = tracker.update(frame)
                if ok:
                    safe = _clamp_bbox_xywh(*bbox, W, H)
                    if safe is None:
                        to_remove.append(pid)
                        continue
                    x, y, w, h = safe
                    cropped = frame[y:y+h, x:x+w]
                    if cropped.size == 0:
                        to_remove.append(pid)
                        continue

                    cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(cropped_rgb)
                    face_tensor = mtcnn(pil_img)
                    if face_tensor is None:
                        to_remove.append(pid)
                        continue
                    face_tensor = face_tensor.unsqueeze(0).to(device)
                    with torch.no_grad():
                        emb = model(face_tensor).cpu().numpy()[0]
                    save_embedding_to_db(video_name, frame_num, pid, emb, x, y, x+w, y+h)
                else:
                    to_remove.append(pid)
            for pid in to_remove:
                trackers.pop(pid, None)
            continue

        print(f"[Frame {frame_num}] Detecting faces")
        faces = detect_faces_retina(frame)
        for emb, (x1, y1, x2, y2), face_crop in faces:
            person_id = match_to_gallery(emb, gallery)
            update_or_insert_gallery(person_id, emb)
            save_embedding_to_db(video_name, frame_num, person_id, emb, x1, y1, x2, y2)

            # Initialize tracker (kept identical API; creation may require legacy in some OpenCV builds)
            tracker = cv2.TrackerCSRT_create()
            tracker.init(frame, (x1, y1, x2 - x1, y2 - y1))
            trackers[person_id] = tracker

    cap.release()
    print(f"Finished processing {video_name}")


if __name__ == "__main__":
    videos_dir = "videos_database"
    for filename in os.listdir(videos_dir):
        if filename.endswith(".mp4"):
            process_video(os.path.join(videos_dir, filename))
    conn.close()
