
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

            seen_ids = set()  # Track IDs already processed in this frame
            for embedding, face_img in retina_faces:
                matches = self.matcher.match_embedding(embedding, threshold=0.0)
                if matches:
                    best_person, best_score = matches[0]
                    if best_person.id in seen_ids:
                        continue  # Skip duplicate ID for this frame
                    seen_ids.add(best_person.id)

                    if best_score >= self.threshold:
                        # Check if this is the best match for this person
                        if best_person.id not in self.best_matches or best_score > self.best_matches[best_person.id]["score"]:
                            match_filename = f"frame{frame_count}_id{best_person.id}_{best_person.first_name}_{best_person.last_name}.jpg"
                            match_save_path = os.path.join(self.output_folder, match_filename)
                            face_img.save(match_save_path, quality=100)
                            print(f"[MATCH] Frame {frame_count}: {best_person.first_name} {best_person.last_name} "
                                  f"(ID={best_person.id}) | Score={best_score:.2f} [SAVED to {match_save_path}]")

                            # Update best match for this person
                            self.best_matches[best_person.id] = {
                                "filename": match_filename,
                                "score": best_score,
                                "image": face_img,
                                "first_name": best_person.first_name,
                                "last_name": best_person.last_name
                            }

                            # Download and save the reference image (only once per person)
                            if best_person.main_img_path and "reference_saved" not in self.best_matches[best_person.id]:
                                print(f"[INFO] Attempting to download reference image from {best_person.main_img_path}")
                                try:
                                    ref_filename = f"reference_id{best_person.id}_{best_person.first_name}_{best_person.last_name}.jpg"
                                    ref_save_path = os.path.join(self.output_folder, ref_filename)
                                    response = requests.get(best_person.main_img_path, stream=True)
                                    if response.status_code == 200:
                                        with open(ref_save_path, 'wb') as f:
                                            for chunk in response.iter_content(1024):
                                                f.write(chunk)
                                        print(f"[REFERENCE] Saved reference image for {best_person.first_name} {best_person.last_name} "
                                              f"(ID={best_person.id}) to {ref_save_path}")
                                        self.best_matches[best_person.id]["reference_saved"] = True
                                    else:
                                        print(f"[ERROR] Failed to download reference image from {best_person.main_img_path}: HTTP {response.status_code}")
                                except Exception as e:
                                    print(f"[ERROR] Failed to save reference image for {best_person.first_name} {best_person.last_name}: {e}")
                            elif not best_person.main_img_path:
                                print(f"[WARNING] No main_img_path for {best_person.first_name} {best_person.last_name} (ID={best_person.id})")
                        else:
                            print(f"[MATCH] Frame {frame_count}: {best_person.first_name} {best_person.last_name} "
                                  f"(ID={best_person.id}) | Score={best_score:.2f} [SKIPPED, lower score than {self.best_matches[best_person.id]['score']:.2f}]")
                    else:
                        print(f"[NO MATCH] Frame {frame_count}: {best_person.first_name} {best_person.last_name} "
                              f"(ID={best_person.id}) | Score={best_score:.2f}")

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


# import cv2
# from PIL import Image
# from retinaface import RetinaFace
# import numpy as np
# import torch
# from facenet_pytorch import MTCNN, InceptionResnetV1
# import os
# import requests

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# mtcnn = MTCNN(image_size=160, margin=0, device=device)
# model = InceptionResnetV1(pretrained="vggface2").eval().to(device)

# def get_embedding_from_image(img):
#     face_tensor = mtcnn(img)
#     if face_tensor is None:
#         return None
#     face_tensor = face_tensor.unsqueeze(0).to(device)
#     with torch.no_grad():
#         emb = model(face_tensor).cpu().numpy()[0]
#     return emb

# class VideoProcessor:
#     def __init__(self, face_matcher, output_folder="matches", frame_skip=5, threshold=0.65):
#         self.matcher = face_matcher
#         self.output_folder = output_folder
#         self.frame_skip = frame_skip
#         self.threshold = threshold
#         if not os.path.exists(self.output_folder):
#             os.makedirs(self.output_folder)
#         self.best_matches = {}  # Dictionary to track best match per personId

#     def extract_faces_retina(self, frame_bgr):
#         detections = RetinaFace.detect_faces(frame_bgr)
#         faces = []
#         if isinstance(detections, dict):
#             for key in detections:
#                 face_info = detections[key]
#                 x1, y1, x2, y2 = map(int, face_info["facial_area"])
#                 face_bgr = frame_bgr[y1:y2, x1:x2]
#                 face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
#                 face_pil = Image.fromarray(face_rgb)
#                 emb = get_embedding_from_image(face_pil)
#                 if emb is not None:
#                     faces.append((emb, face_pil))
#         return faces

#     def process_video(self, video_path):
#         cap = cv2.VideoCapture(video_path)
#         frame_count = 0

#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 break
#             frame_count += 1
#             if frame_count % self.frame_skip != 0:
#                 continue

#             print(f"[INFO] Processing frame {frame_count}...")
#             try:
#                 retina_faces = self.extract_faces_retina(frame)
#             except Exception as e:
#                 print(f"[WARNING] Frame {frame_count} skipped due to detection error: {e}")
#                 continue

#             if not retina_faces:
#                 print(f"[INFO] Frame {frame_count} has no faces detected, skipping...")
#                 continue

#             seen_ids = set()  # Track IDs already processed in this frame
#             for embedding, face_img in retina_faces:
#                 matches = self.matcher.match_embedding(embedding, threshold=0.0)
#                 if matches:
#                     best_person, best_score = matches[0]
#                     if best_person.id in seen_ids:
#                         continue  # Skip duplicate ID for this frame
#                     seen_ids.add(best_person.id)

#                     if best_score >= self.threshold:
#                         # Check if this is the best match for this person
#                         if best_person.id not in self.best_matches or best_score > self.best_matches[best_person.id]["score"]:
#                             match_filename = f"frame{frame_count}_id{best_person.id}_{best_person.first_name}_{best_person.last_name}.jpg"
#                             match_save_path = os.path.join(self.output_folder, match_filename)
#                             face_img.save(match_save_path, quality=100)
#                             print(f"[MATCH] Frame {frame_count}: {best_person.first_name} {best_person.last_name} "
#                                   f"(ID={best_person.id}) | Score={best_score:.2f} [SAVED to {match_save_path}]")

#                             # Update best match for this person
#                             self.best_matches[best_person.id] = {
#                                 "filename": match_filename,
#                                 "score": best_score,
#                                 "image": face_img,
#                                 "first_name": best_person.first_name,
#                                 "last_name": best_person.last_name
#                             }

#                             # Download and save the reference image (only once per person)
#                             if best_person.main_img_path and "reference_saved" not in self.best_matches[best_person.id]:
#                                 print(f"[INFO] Attempting to download reference image from {best_person.main_img_path}")
#                                 try:
#                                     ref_filename = f"reference_id{best_person.id}_{best_person.first_name}_{best_person.last_name}.jpg"
#                                     ref_save_path = os.path.join(self.output_folder, ref_filename)
#                                     response = requests.get(best_person.main_img_path, stream=True)
#                                     if response.status_code == 200:
#                                         with open(ref_save_path, 'wb') as f:
#                                             for chunk in response.iter_content(1024):
#                                                 f.write(chunk)
#                                         print(f"[REFERENCE] Saved reference image for {best_person.first_name} {best_person.last_name} "
#                                               f"(ID={best_person.id}) to {ref_save_path}")
#                                         self.best_matches[best_person.id]["reference_saved"] = True
#                                     else:
#                                         print(f"[ERROR] Failed to download reference image from {best_person.main_img_path}: HTTP {response.status_code}")
#                                 except Exception as e:
#                                     print(f"[ERROR] Failed to save reference image for {best_person.first_name} {best_person.last_name}: {e}")
#                             elif not best_person.main_img_path:
#                                 print(f"[WARNING] No main_img_path for {best_person.first_name} {best_person.last_name} (ID={best_person.id})")
#                         else:
#                             print(f"[MATCH] Frame {frame_count}: {best_person.first_name} {best_person.last_name} "
#                                   f"(ID={best_person.id}) | Score={best_score:.2f} [SKIPPED, lower score than {self.best_matches[best_person.id]['score']:.2f}]")
#                     else:
#                         print(f"[NO MATCH] Frame {frame_count}: {best_person.first_name} {best_person.last_name} "
#                               f"(ID={best_person.id}) | Score={best_score:.2f}")

#         # Clean up: Remove old match files, keep only the best
#         for person_id, match_data in self.best_matches.items():
#             for file in os.listdir(self.output_folder):
#                 if file.startswith(f"frame") and f"id{person_id}" in file and file != match_data["filename"]:
#                     try:
#                         os.remove(os.path.join(self.output_folder, file))
#                         print(f"[CLEANUP] Removed old match file: {file}")
#                     except Exception as e:
#                         print(f"[ERROR] Failed to remove old match file {file}: {e}")

#         cap.release()
#         print("[INFO] Video processing completed.")