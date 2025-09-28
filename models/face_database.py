
import os
from models.absent import Absent
import numpy as np

class FaceDatabase:
    def __init__(self):
        self.people = []

    def build_from_folder(self, folder_path):
        """Build database from folders formatted as: id_first_last"""
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

            person = Absent(person_id, first_name, last_name, img_paths)
            self.people.append(person)

        print(f"[INFO] Built database with {len(self.people)} people.")

    def add_person(self, person_id, first_name, last_name, img_path, age=None):
        """Add a single person manually"""
        person = Absent(person_id, first_name, last_name, img_path, age)
        person.get_embs()
        self.people.append(person)
        print(f"[INFO] Added person: {first_name} {last_name} (ID={person_id})")

    def upload_to_firestore(self, firestore_client):
        """Upload all people to Firestore"""
        for person in self.people:
            doc_ref = firestore_client.collection("Users").document(str(person.id))
            doc_ref.set(person.to_dict())
        print(f"[INFO] Uploaded {len(self.people)} people to Firestore.")

    def load_from_firestore(self, firestore_client):
        """Load all people from Firestore and rebuild FaceDatabase"""
        self.people = []
        users_ref = firestore_client.collection("Users").stream()

        for doc in users_ref:
            data = doc.to_dict()
            person_id = data["id"]
            first_name = data.get("first_name", "Unknown")
            last_name = data.get("last_name", "Unknown")
            age = data.get("age")

            embeddings_data = data.get("embeddings", [])
            embeddings = []
            for emb_obj in embeddings_data:
                vector = emb_obj.get("vector")
                if vector:
                    embeddings.append(np.array(vector))

            person = Absent(person_id, first_name, last_name, img_paths=[], age=age)
            person._embeddings = embeddings

            self.people.append(person)

        print(f"[INFO] Loaded {len(self.people)} people from Firestore.")
