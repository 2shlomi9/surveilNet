# import numpy as np

# class FaceMatcher:
#     def __init__(self, face_database):
#         self.db = face_database

#     def match_embedding(self, embedding, threshold=0.65):
#         """Find the best match in the database for the given embedding"""
#         best_score = -1
#         best_match = None

#         for person in self.db.people:
#             db_emb = person.get_emb()
#             if db_emb is None:
#                 continue

#             embedding_norm = embedding / np.linalg.norm(embedding)
#             db_emb_norm = db_emb / np.linalg.norm(db_emb)
#             sim = np.dot(embedding_norm, db_emb_norm)

#             if sim > best_score:
#                 best_score = sim
#                 best_match = person

#         if best_score >= threshold:
#             return best_match, best_score
#         return None, best_score


import numpy as np

class FaceMatcher:
    def __init__(self, face_database):
        self.db = face_database

    def match_embedding(self, embedding, threshold=0.65):
        """
        Find the best match in the database for the given embedding.
        Each person may have multiple embeddings.
        """
        best_score = -1
        best_match = None

        embedding_norm = embedding / np.linalg.norm(embedding)

        for person in self.db.people:
            db_embeddings = person.get_emb()  # list of embeddings
            if not db_embeddings:
                continue

            for db_emb in db_embeddings:
                db_emb_norm = db_emb / np.linalg.norm(db_emb)
                sim = np.dot(embedding_norm, db_emb_norm)
                if sim > best_score:
                    best_score = sim
                    best_match = person

        if best_score >= threshold:
            return best_match, best_score
        return None, best_score
