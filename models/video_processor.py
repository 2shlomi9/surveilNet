import cv2
from PIL import Image
from retinaface import RetinaFace
import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import os
import cv2
import numpy as np
from PIL import Image
from retinaface import RetinaFace
from facenet_pytorch import InceptionResnetV1, MTCNN
import torch

# --- Device & models ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mtcnn = MTCNN(image_size=160, margin=0, device=device)
model = InceptionResnetV1(pretrained="vggface2").eval().to(device)


def get_embedding_from_image(img):
    """Extract embedding from a PIL image"""
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
        """Extract faces from frame using RetinaFace and return embeddings + face crops"""
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
        """Run face recognition on video and save matched faces"""
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
            retina_faces = self.extract_faces_retina(frame)

            for i, (embedding, face_img) in enumerate(retina_faces):
                best_match, score = self.matcher.match_embedding(embedding, self.threshold)
                if best_match is not None:
                    print(f"[MATCH] Frame {frame_count}: {best_match.first_name} {best_match.last_name} "
                          f"(ID={best_match.id}) | Score={score:.2f}")

                    # Save face with person details in filename
                    filename = f"frame{frame_count}_id{best_match.id}_{best_match.first_name}_{best_match.last_name}.jpg"
                    save_path = os.path.join(self.output_folder, filename)
                    face_img.save(save_path, quality=100)

        cap.release()
        print("[INFO] Video processing completed.")

