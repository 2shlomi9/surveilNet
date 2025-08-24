import os
import cv2
import numpy as np
from PIL import Image
from facenet_pytorch import InceptionResnetV1, MTCNN
from retinaface import RetinaFace
import torch
import pyodbc
import uuid

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(image_size=160, margin=0, device=device)
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# SQL Connection
conn = pyodbc.connect("Driver={ODBC Driver 17 for SQL Server};Server=localhost;Database=FaceRecognitionDB;Trusted_Connection=yes;")
cursor = conn.cursor()

def normalize(v): return v / np.linalg.norm(v)
def cosine_similarity(a, b): return np.dot(normalize(a), normalize(b))

def get_all_gallery_embeddings():
    cursor.execute("SELECT PersonID, AverageEmbedding FROM FaceGallery")
    return [(str(pid), np.frombuffer(emb, dtype=np.float32)) for pid, emb in cursor.fetchall()]

def update_or_insert_gallery(person_id, new_emb):
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
                       person_id, bytearray(new_emb.astype(np.float32).tobytes()))
    conn.commit()

def save_embedding_to_db(video_name, frame_num, person_id, embedding, x1, y1, x2, y2):
    sql = """
    INSERT INTO FaceEmbeddings (VideoName, FrameNumber, PersonID, Embedding, BBox_X1, BBox_Y1, BBox_X2, BBox_Y2)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """
    cursor.execute(sql, video_name, frame_num, person_id,
                   bytearray(embedding.astype(np.float32).tobytes()), x1, y1, x2, y2)
    conn.commit()

def detect_faces_retina(img_bgr):
    faces = []
    detections = RetinaFace.detect_faces(img_bgr)
    if isinstance(detections, dict):
        for det in detections.values():
            x1, y1, x2, y2 = map(int, det['facial_area'])
            face_bgr = img_bgr[y1:y2, x1:x2]
            face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
            face_pil = Image.fromarray(face_rgb)
            face_tensor = mtcnn(face_pil)
            if face_tensor is not None:
                face_tensor = face_tensor.unsqueeze(0).to(device)
                with torch.no_grad():
                    emb = model(face_tensor).cpu().numpy()[0]
                faces.append((emb, (x1, y1, x2, y2), face_bgr))
    return faces

def match_to_gallery(embedding, gallery, threshold=0.7):
    for person_id, gallery_emb in gallery:
        sim = cosine_similarity(embedding, gallery_emb)
        if sim > threshold:
            return person_id
    return str(uuid.uuid4())  # new person

def process_video(video_path, frame_skip=5):
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
            # Update existing trackers
            to_remove = []
            for pid, tracker in trackers.items():
                ok, bbox = tracker.update(frame)
                if ok:
                    x, y, w, h = map(int, bbox)
                    cropped = frame[y:y+h, x:x+w]
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
                trackers.pop(pid)
            continue

        print(f"[Frame {frame_num}] Detecting faces")
        faces = detect_faces_retina(frame)
        for emb, (x1, y1, x2, y2), face_crop in faces:
            person_id = match_to_gallery(emb, gallery)
            update_or_insert_gallery(person_id, emb)
            save_embedding_to_db(video_name, frame_num, person_id, emb, x1, y1, x2, y2)

            # Initialize tracker
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
