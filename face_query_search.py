import torch
import numpy as np
import pyodbc
from PIL import Image
from facenet_pytorch import InceptionResnetV1, MTCNN
import argparse
import os
import cv2
from video_face_extractor import _load_sqlserver_conn_str

# Init
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(image_size=160, margin=0, device=device)
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

VIDEOS_FOLDER = "videos_database"
OUTPUT_FOLDER = "matches"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
SIMILARITY_THRESHOLD = 0.5


def get_embedding(img_path):
    img = Image.open(img_path).convert("RGB")
    face = mtcnn(img)
    if face is None:
        raise ValueError("Face not detected")
    face = face.unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model(face).cpu().numpy()[0]
    return embedding


def cosine_similarity(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return np.dot(a, b)


def search_db(embedding, topk=10):
    # SQL Connection
    conn = pyodbc.connect(_load_sqlserver_conn_str())
    cursor = conn.cursor()
    cursor.execute("SELECT VideoName, FrameNumber, PersonID, Embedding, BBox_X1, BBox_Y1, BBox_X2, BBox_Y2 FROM FaceEmbeddings")

    matches = []
    for row in cursor.fetchall():
        emb_bytes = bytes(row.Embedding)
        db_emb = np.frombuffer(emb_bytes, dtype=np.float32)
        if len(db_emb) != 512:
            continue  # corrupted or invalid
        sim = cosine_similarity(embedding, db_emb)
        matches.append((sim, row.VideoName, row.FrameNumber, row.PersonID, row.BBox_X1, row.BBox_Y1, row.BBox_X2, row.BBox_Y2))

    conn.close()
    matches.sort(reverse=True)
    return matches[:topk]


def show_matches(results):
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    for i, (sim, video, frame, pid, x1, y1, x2, y2) in enumerate(results):
        if sim < SIMILARITY_THRESHOLD:
            print(f"Skipping match {i + 1} due to low similarity ({sim:.2f})")
            continue

        video_path = os.path.join(VIDEOS_FOLDER, video)
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ret, img = cap.read()
        cap.release()

        if not ret:
            print(f"Could not read frame {frame} from {video}")
            continue

        # Crop the detected face
        face_crop = img[y1:y2, x1:x2]

        # Resize for display (e.g., width = 300px)
        display_width = 300
        h, w = face_crop.shape[:2]
        scale = display_width / w
        resized_face = cv2.resize(face_crop, (display_width, int(h * scale)))

        # Show face
        window_name = f"Match {i + 1}"
        cv2.imshow(window_name, resized_face)
        cv2.waitKey(0)

        # Safe destroy
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) >= 1:
            cv2.destroyWindow(window_name)

        # Save full-res face crop
        output_path = os.path.join(OUTPUT_FOLDER, f"match_{i + 1}_sim_{int(sim * 100)}.jpg")
        cv2.imwrite(output_path, face_crop)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True)
    parser.add_argument('--topk', type=int, default=5)
    args = parser.parse_args()

    emb = get_embedding(args.image)
    results = search_db(emb, topk=args.topk)

    for sim, video, frame, pid, x1, y1, x2, y2 in results:
        print(f"[{sim:.2f}] Video: {video}, Frame: {frame}, PersonID: {pid}, Box: ({x1},{y1})–({x2},{y2})")

    show_matches(results)
