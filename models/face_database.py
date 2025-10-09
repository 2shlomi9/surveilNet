import os
import numpy as np
import cloudinary
import cloudinary.uploader
from models.absent import Absent
from configs.cloudinary_config import cloudinary  # make sure you have this file

class FaceDatabase:
    def __init__(self):
        self.people = []

    def build_from_folder(self, folder_path):
        """Build database from folders formatted as: id_first_last."""
        for person_folder in os.listdir(folder_path):
            full_path = os.path.join(folder_path, person_folder)
            if not os.path.isdir(full_path):
                continue

            try:
                parts = person_folder.split("_")
                person_id = parts[0]
                first_name = parts[1]
                last_name = parts[2] if len(parts) > 2 else "Unknown"
            except Exception:
                print(f"[WARNING] Skipping folder {person_folder}, invalid format.")
                continue

            images = [f for f in os.listdir(full_path) if f.lower().endswith((".jpg", ".png"))]
            if not images:
                print(f"[WARNING] No images found for {person_folder}")
                continue

            selected_images = images[:5]
            img_paths = [os.path.join(full_path, img) for img in selected_images]

            main_img_path = img_paths[0]
            person = Absent(person_id, first_name, last_name, img_paths, main_img_path=main_img_path)
            self.people.append(person)

        print(f"[INFO] Built database with {len(self.people)} people.")

    def add_person(self, person_id, first_name, last_name, img_path, age=None):
        """Add a single person manually"""
        person = Absent(person_id, first_name, last_name, img_path, age)
        person.get_embs()
        self.people.append(person)
        print(f"[INFO] Added person: {first_name} {last_name} (ID={person_id})")

    def upload_to_firestore(self, firestore_client):
        """Upload main image to Cloudinary and embeddings to Firestore."""
        for person in self.people:
            if not person.main_img_path:
                print(f"[WARNING] No main image for {person.first_name} {person.last_name} (ID={person.id})")
                continue

            try:
                print(f"[INFO] Uploading {person.main_img_path} to Cloudinary for {person.first_name} {person.last_name}")
                upload_result = cloudinary.uploader.upload(person.main_img_path, folder="faces_main")
                image_url = upload_result.get("secure_url")
                if not image_url:
                    print(f"[ERROR] No secure_url returned for {person.main_img_path}")
                    continue
                person.main_img_path = image_url
                print(f"[INFO] Uploaded to Cloudinary: {image_url}")
            except Exception as e:
                print(f"[ERROR] Failed to upload {person.main_img_path} to Cloudinary: {e}")
                continue

            person.get_embs()
            if not person._embeddings:
                print(f"[WARNING] No embeddings generated for {person.first_name} {person.last_name} (ID={person.id})")

            doc_ref = firestore_client.collection("Users").document(str(person.id))
            doc_ref.set(person.to_dict())
            print(f"[INFO] Saved {person.first_name} {person.last_name} (ID={person.id}) to Firestore")

        print(f"[INFO] Uploaded {len(self.people)} people to Firestore.")

    def load_from_firestore(self, firestore_client):
        """Load people from Firestore."""
        self.people = []
        try:
            docs = firestore_client.collection("Users").stream()
            for doc in docs:
                data = doc.to_dict()
                person = Absent(
                    id=data["id"],
                    first_name=data["first_name"],
                    last_name=data["last_name"],
                    img_paths=[],  # Not loading local paths from Firestore
                    age=data.get("age"),
                    main_img_path=data.get("main_img_path")
                )
                embeddings = [np.array(emb["vector"]) for emb in data.get("embeddings", [])]
                person._embeddings = embeddings
                self.people.append(person)
            print(f"[INFO] Loaded {len(self.people)} people from Firestore.")
        except Exception as e:
            print(f"[ERROR] Failed to load from Firestore: {e}")