# main.py

from models.face_database import FaceDatabase
from models.face_matcher import FaceMatcher
from models.video_processor import VideoProcessor
import torch
import firebase_admin
from firebase_admin import credentials, firestore

# --- Firebase init ---
cred = credentials.Certificate("configs/accountKey.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

# --- Device setup ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Build database from local folder ---
gallery_folder = "database"
face_db = FaceDatabase()
face_db.build_from_folder(gallery_folder)  

# --- Upload to Firestore ---
face_db.upload_to_firestore(db)            

# # --- Matcher ---
# matcher = FaceMatcher(face_db)

# # --- Video Processing ---
# video_path = "videos_database/video.mp4"
# video_processor = VideoProcessor(matcher, output_folder="matches", frame_skip=5)
# video_processor.process_video(video_path)

# print("[DONE] Pipeline finished successfully.")
