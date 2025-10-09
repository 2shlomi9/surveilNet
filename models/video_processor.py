import cv2
from PIL import Image
from retinaface import RetinaFace
import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import os
import requests

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
        self.best_matches = {}  # Dictionary to track best match per personId

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

        # Log all people in the database
        print(f"[DEBUG] Loaded {len(self.matcher.db.people)} people from database:")
        for person in self.matcher.db.people:
            embs = person.get_embs()
            print(f"[DEBUG] Person {person.first_name} {person.last_name} (ID={person.id}): {len(embs)} embeddings")

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

            for embedding, face_img in retina_faces:
                matches = self.matcher.match_embedding(embedding, threshold=self.threshold)
                if matches:
                    print(f"[DEBUG] Frame {frame_count}: Found {len(matches)} matches above threshold {self.threshold}")
                    for person, score in matches:
                        print(f"[DEBUG] Match: {person.first_name} {person.last_name} (ID={person.id}) | Score={score:.2f}")

                        # Check if this is the best match for this person
                        if person.id not in self.best_matches or score > self.best_matches[person.id]["score"]:
                            match_filename = f"frame{frame_count}_id{person.id}_{person.first_name}_{person.last_name}.jpg"
                            match_save_path = os.path.join(self.output_folder, match_filename)
                            face_img.save(match_save_path, quality=100)
                            print(f"[MATCH] Frame {frame_count}: {person.first_name} {person.last_name} "
                                  f"(ID={person.id}) | Score={score:.2f} [SAVED to {match_save_path}]")

                            # Update best match for this person
                            self.best_matches[person.id] = {
                                "filename": match_filename,
                                "score": score,
                                "image": face_img,
                                "first_name": person.first_name,
                                "last_name": person.last_name,
                                "reference_saved": self.best_matches.get(person.id, {}).get("reference_saved", False)
                            }

                            # Download and save the reference image (only once per person)
                            if person.main_img_path and not self.best_matches[person.id]["reference_saved"]:
                                print(f"[INFO] Attempting to download reference image from {person.main_img_path}")
                                try:
                                    ref_filename = f"reference_id{person.id}_{person.first_name}_{person.last_name}.jpg"
                                    ref_save_path = os.path.join(self.output_folder, ref_filename)
                                    response = requests.get(person.main_img_path, stream=True)
                                    if response.status_code == 200:
                                        with open(ref_save_path, 'wb') as f:
                                            for chunk in response.iter_content(1024):
                                                f.write(chunk)
                                        print(f"[REFERENCE] Saved reference image for {person.first_name} {person.last_name} "
                                              f"(ID={person.id}) to {ref_save_path}")
                                        self.best_matches[person.id]["reference_saved"] = True
                                    else:
                                        print(f"[ERROR] Failed to download reference image from {person.main_img_path}: HTTP {response.status_code}")
                                except Exception as e:
                                    print(f"[ERROR] Failed to save reference image for {person.first_name} {person.last_name}: {e}")
                            elif not person.main_img_path:
                                print(f"[WARNING] No main_img_path for {person.first_name} {person.last_name} (ID={person.id})")
                        else:
                            print(f"[MATCH] Frame {frame_count}: {person.first_name} {person.last_name} "
                                  f"(ID={person.id}) | Score={score:.2f} [SKIPPED, lower score than {self.best_matches[person.id]['score']:.2f}]")
                else:
                    print(f"[NO MATCH] Frame {frame_count}: No match found above threshold {self.threshold}")

        # Clean up: Remove old match files, keep only the best
        for person_id, match_data in self.best_matches.items():
            for file in os.listdir(self.output_folder):
                if file.startswith(f"frame") and f"id{person_id}" in file and file != match_data["filename"]:
                    try:
                        os.remove(os.path.join(self.output_folder, file))
                        print(f"[CLEANUP] Removed old match file: {file}")
                    except Exception as e:
                        print(f"[ERROR] Failed to remove old match file {file}: {e}")

        cap.release()
        print("[INFO] Video processing completed.")