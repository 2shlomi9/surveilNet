from models.face_database import FaceDatabase
from models.absent import Absent
import firebase_admin
from firebase_admin import credentials, firestore
import os

def add_person_existing_db(face_db, person_id, first_name, last_name, folder_path, db, age=None):
    """Add one person using Absent class"""
    if not os.path.exists(folder_path):
        raise ValueError(f"Folder not found: {folder_path}")

    images = [os.path.join(folder_path, f) 
              for f in os.listdir(folder_path) 
              if f.lower().endswith((".jpg", ".png"))][:5]

    if not images:
        raise ValueError(f"No images found in {folder_path}")

    person = Absent(person_id, first_name, last_name, images, age)
    person.get_embs()  

    face_db.people.append(person)

    doc_ref = db.collection("Users").document(str(person.id))
    if doc_ref.get().exists:
        print(f"[INFO] Person {person_id} already exists. Updating embeddings...")
    else:
        print(f"[INFO] Adding new person {person_id} to Firestore...")

    doc_ref.set(person.to_dict())
    print(f"[INFO] Done uploading {first_name} {last_name} (ID={person_id}) to Firestore.")


if __name__ == "__main__":
    # --- Firebase init ---
    cred = credentials.Certificate("configs/accountKey.json")
    firebase_admin.initialize_app(cred)
    db = firestore.client()

    # --- Load existing database in memory ---
    face_db = FaceDatabase()
    face_db.load_from_firestore(db)
    print(f"[INFO] Loaded {len(face_db.people)} people from Firestore.")

    # --- Path to the new person's folder ---
    folder = "database/05_sali_sharfman"

    # --- Add new person ---
    add_person_existing_db(face_db, "05", "sali", "sharfman", folder, db, age=25)
