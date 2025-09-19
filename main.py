from models.face_database import FaceDatabase
from models.face_matcher import FaceMatcher
from models.video_processor import VideoProcessor
import torch
import firebase_admin
from firebase_admin import credentials, firestore

# --- Firebase initialization ---
cred = credentials.Certificate("configs/accountKey.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

# --- Device setup ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Choose mode ---
USE_FIRESTORE = True   # True → load from Firestore, False → build locally and upload

face_db = FaceDatabase()

if USE_FIRESTORE:
    # Load embeddings directly from Firestore
    face_db.load_from_firestore(db)
    print(f"[INFO] Loaded {len(face_db.people)} people from Firestore.")
else:
    # Build database locally from folder and upload to Firestore
    gallery_folder = "face_database"
    face_db.build_from_folder(gallery_folder)
    face_db.upload_to_firestore(db)
    print(f"[INFO] Built and uploaded {len(face_db.people)} people from local folder.")

# --- Face matcher ---
matcher = FaceMatcher(face_db)

# --- Video processing ---
video_path = "videos_database/video.mp4"
video_processor = VideoProcessor(matcher, output_folder="matches", frame_skip=5)
video_processor.process_video(video_path)

