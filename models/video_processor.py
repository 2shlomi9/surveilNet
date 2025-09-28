import cv2
from PIL import Image
from retinaface import RetinaFace
import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mtcnn = MTCNN(image_size=160, margin=0, device=device)
model = InceptionResnetV1(pretrained="vggface2").eval().to(device)

def get_embedding_from_image(img):
    face_tensor = mtcnn(img)
    if face_tensor is None:
        return None
    face_tensor = face_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model(face_tensor).cpu().numpy()[0]
    return emb

class VideoProcessor:
    def __init__(self, face_matcher, output_folder="matches", frame_skip=5, threshold=0.65):
        self.matcher = face_matcher
        self.output_folder = output_folder
        self.frame_skip = frame_skip
        self.threshold = threshold
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

    def extract_faces_retina(self, frame_bgr):
        detections = RetinaFace.detect_faces(frame_bgr)
        faces = []
        if isinstance(detections, dict):
            for key in detections:
                face_info = detections[key]
                x1, y1, x2, y2 = map(int, face_info["facial_area"])
                face_bgr = frame_bgr[y1:y2, x1:x2]
                face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
                face_pil = Image.fromarray(face_rgb)
                emb = get_embedding_from_image(face_pil)
                if emb is not None:
                    faces.append((emb, face_pil))
        return faces

    def process_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            if frame_count % self.frame_skip != 0:
                continue

            print(f"[INFO] Processing frame {frame_count}...")
            try:
                retina_faces = self.extract_faces_retina(frame)
            except Exception as e:
                print(f"[WARNING] Frame {frame_count} skipped due to detection error: {e}")
                continue

            if not retina_faces:
                print(f"[INFO] Frame {frame_count} has no faces detected, skipping...")
                continue

            seen_ids = set()  # Track IDs already printed in this frame
            for embedding, face_img in retina_faces:
                matches = self.matcher.match_embedding(embedding, threshold=0.0)
                if matches:
                    best_person, best_score = matches[0]
                    if best_person.id in seen_ids:
                        continue  # Skip duplicate ID for this frame
                    seen_ids.add(best_person.id)

                    if best_score >= self.threshold:
                        filename = f"frame{frame_count}_id{best_person.id}_{best_person.first_name}_{best_person.last_name}.jpg"
                        save_path = os.path.join(self.output_folder, filename)
                        face_img.save(save_path, quality=100)
                        print(f"[MATCH] Frame {frame_count}: {best_person.first_name} {best_person.last_name} "
                            f"(ID={best_person.id}) | Score={best_score:.2f} [SAVED]")
                    else:
                        print(f"[NO MATCH] Frame {frame_count}: {best_person.first_name} {best_person.last_name} "
                            f"(ID={best_person.id}) | Score={best_score:.2f}")

        cap.release()
        print("[INFO] Video processing completed.")
