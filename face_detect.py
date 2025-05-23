import os
from PIL import Image
from facenet_pytorch import InceptionResnetV1, MTCNN
import torch
import numpy as np
import cv2
from retinaface import RetinaFace
from collections import defaultdict

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

mtcnn = MTCNN(image_size=160, margin=0, device=device)
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

def get_embedding_from_image(img):
    face_tensor = mtcnn(img)
    if face_tensor is None:
        return None
    face_tensor = face_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model(face_tensor)
    return embedding.cpu().numpy()[0]

def build_database(folder_path):
    database = defaultdict(list)
    for filename in os.listdir(folder_path):
        path = os.path.join(folder_path, filename)
        name = filename.split("_")[0] 
        img = Image.open(path).convert("RGB")
        face = mtcnn(img)
        if face is not None:
            emb = get_embedding_from_image(img)
            if emb is not None:
                database[name].append((emb, img))
    return database

def save_image(img, path):
    img.save(path, quality=100)

def extract_faces_retina(img_bgr):
    detections = RetinaFace.detect_faces(img_bgr)
    faces = []
    if isinstance(detections, dict):
        for key in detections:
            face_info = detections[key]
            x1, y1, x2, y2 = map(int, face_info['facial_area'])
            face_bgr = img_bgr[y1:y2, x1:x2]
            face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
            face_pil = Image.fromarray(face_rgb)
            emb = get_embedding_from_image(face_pil)  
            if emb is not None:
                faces.append((emb, face_pil))
    return faces

def process_video_and_save_matches(video_path, database, threshold=0.65, output_folder="matches", frame_skip=5):
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    similarity_scores = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if frame_count % frame_skip != 0:
            continue

        print(f"Processing frame {frame_count}...")

        retina_faces = extract_faces_retina(frame)
        if not retina_faces:
            print(f"No faces detected in frame {frame_count}")
            continue

        print(f"Detected {len(retina_faces)} faces in frame {frame_count}")

        for i, (embedding, face_img) in enumerate(retina_faces):
            best_score = -1
            best_match = None
            best_match_img = None
            for person_name, emb_list in database.items():
                for emb, orig_img in emb_list:
                    embedding_norm = embedding / np.linalg.norm(embedding)
                    emb_norm = emb / np.linalg.norm(emb)
                    sim = np.dot(embedding_norm, emb_norm)
                    if sim > best_score:
                        best_score = sim
                        best_match = person_name
                        best_match_img = orig_img

            similarity_scores.append(best_score)
            print("SCORE:", best_score, "  THRESHOLD:", threshold)

            if best_score > threshold:
                print(f"Frame {frame_count}: Found match {best_match} with similarity {best_score:.2f}")
                match_face_path = os.path.join(output_folder, f"frame{frame_count}_face{i}.jpg")
                save_image(face_img, match_face_path)
                matched_path = os.path.join(output_folder, f"frame{frame_count}_match_{best_match}.jpg")
                save_image(best_match_img, matched_path)

    cap.release()
    print("Finished processing video.")

def main():
    gallery_folder = "face_database"
    video_path = "videos_database/video.mp4"
    print("Building face embeddings database...")
    database = build_database(gallery_folder)
    print(f"Database contains {len(database)} people.")
    process_video_and_save_matches(video_path, database)

if __name__ == "__main__":
    main()
