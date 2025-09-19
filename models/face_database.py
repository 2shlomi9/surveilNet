import os
from models.absent import Absent


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

            img_path = os.path.join(full_path, images[0])
            person = Absent(person_id, first_name, last_name, img_path)
            person.get_emb()  
            self.people.append(person)

        print(f"[INFO] Built database with {len(self.people)} people.")

    def add_person(self, person_id, first_name, last_name, img_path, age=None):
        """Add a single person manually"""
        person = Absent(person_id, first_name, last_name, img_path, age)
        person.get_emb()
        self.people.append(person)
        print(f"[INFO] Added person: {first_name} {last_name} (ID={person_id})")

    def upload_to_firestore(self, firestore_client):
        """Upload all people to Firestore"""
        for person in self.people:
            doc_ref = firestore_client.collection("Users").document(str(person.id))
            doc_ref.set(person.to_dict())
        print(f"[INFO] Uploaded {len(self.people)} people to Firestore.")

    def load_from_firestore(self, firestore_client):
        """Load all people from Firestore"""
        docs = firestore_client.collection("Users").stream()
        self.people = []
        for doc in docs:
            data = doc.to_dict()
            if not data.get("embedding"):
                continue
            person = Absent(
                id=data.get("id"),
                first_name=data.get("first_name"),
                last_name=data.get("last_name"),
                age=data.get("age"),
                embedding=data.get("embedding")  
            )
            self.people.append(person)
        print(f"[INFO] Loaded {len(self.people)} people from Firestore.")

    def print_list(self):
        str = ""
        for p in self.people:
            str += p.to_string()
            str += '\n'
        print(str)

