import numpy as np

class FaceMatcher:
    def __init__(self, face_database):
        self.db = face_database

    def match_embedding(self, embedding, threshold=0.65):
        """
        Find all matches in the database for the given embedding
        that are above the similarity threshold.
        Returns a list of tuples: [(person, score), ...]
        """
        matches = []

        embedding_norm = embedding / np.linalg.norm(embedding)

        for person in self.db.people:
            db_embs = person.get_embs()
            if not db_embs:
                continue

            for db_emb in db_embs:
                db_emb_norm = db_emb / np.linalg.norm(db_emb)
                sim = np.dot(embedding_norm, db_emb_norm)

                if sim >= threshold:
                    matches.append((person, sim))

        # Sort matches by similarity descending
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches