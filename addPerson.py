from models.face_database import FaceDatabase
from models.face_matcher import FaceMatcher
from models.video_processor import VideoProcessor
from models.absent import Absent
import torch
import firebase_admin
from firebase_admin import credentials, firestore

# --- Firebase initialization ---
cred = credentials.Certificate("configs/accountKey.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

person = Absent()