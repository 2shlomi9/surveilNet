# face_search.py
from __future__ import annotations
import os
import csv
import argparse
import shutil
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1, fixed_image_standardization

import faiss

from sqlalchemy import create_engine, Table, Column, Integer, BigInteger, LargeBinary, String, Float, MetaData, DateTime, text, BINARY
from sqlalchemy.exc import IntegrityError

# ---------- Config ----------
OUTPUT_ROOT = "extracted_faces"               # per-video subfolders
INDEX_DIR = os.path.join(OUTPUT_ROOT, "_index")
EMBED_NPY = os.path.join(INDEX_DIR, "embeddings.npy")
IDS_CSV = os.path.join(INDEX_DIR, "index_ids.csv")
FAISS_INDEX = os.path.join(INDEX_DIR, "faces.index")
SEARCH_RESULTS_DIR = os.path.join(OUTPUT_ROOT, "_search_results")

MODEL_NAME = "inceptionresnetv1_vggface2"
EMBED_DIM = 512
BATCH_SIZE = 64
IMAGE_SIZE = 160  # Facenet input size
# ---------------------------


def ensure_dir(p: str) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)


def gather_face_records(output_root: str) -> List[Dict[str, Any]]:
    """
    Scan OUTPUT_ROOT/*/detections.csv and collect face_path + metadata.
    Expected columns include: frame_index,timestamp_ms,x1,y1,x2,y2,score,face_path
    """
    records: List[Dict[str, Any]] = []
    root = Path(output_root)
    if not root.exists():
        print(f"[gather] OUTPUT_ROOT not found: {output_root}")
        return records

    for sub in sorted(root.iterdir()):
        if not sub.is_dir():
            continue
        csv_path = sub / "detections.csv"
        if not csv_path.exists():
            continue

        with csv_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                face_path = row.get("face_path", "")
                if not face_path:
                    continue
                fp = Path(face_path)
                if not fp.is_file():
                    fp = (sub / face_path).resolve()
                if not fp.is_file():
                    continue

                records.append({
                    "video_dir": str(sub),
                    "video": sub.name,
                    "frame_index": int(row.get("frame_index", 0)),
                    "timestamp_ms": int(row.get("timestamp_ms", 0)),
                    "x1": int(row.get("x1", 0)), "y1": int(row.get("y1", 0)),
                    "x2": int(row.get("x2", 0)), "y2": int(row.get("y2", 0)),
                    "score": float(row.get("score", 0.0)),
                    "face_path": str(fp),
                })
    print(f"[gather] Found {len(records)} face crops across {output_root}")
    return records


def load_facenet(device: str = "cpu"):
    model = InceptionResnetV1(pretrained="vggface2").eval().to(device)
    tfm = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        fixed_image_standardization,
    ])
    return model, tfm


def _load_rgb(path: str) -> Image.Image:
    img = Image.open(path).convert("RGB")
    return img


def compute_embeddings(records: List[Dict[str, Any]], batch_size: int = BATCH_SIZE, device: str | None = None) -> np.ndarray:
    if not records:
        return np.zeros((0, EMBED_DIM), dtype=np.float32)
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model, tfm = load_facenet(device=device)

    embs: List[np.ndarray] = []
    batch: List[torch.Tensor] = []

    for rec in tqdm(records, desc="[embed]"):
        img = _load_rgb(rec["face_path"])
        tensor = tfm(img)  # [3,160,160]
        batch.append(tensor)
        if len(batch) >= batch_size:
            with torch.no_grad():
                b = torch.stack(batch, dim=0).to(device)
                e = model(b)
                e = torch.nn.functional.normalize(e, p=2, dim=1)
            embs.append(e.cpu().numpy())
            batch = []

    if batch:
        with torch.no_grad():
            b = torch.stack(batch, dim=0).to(device)
            e = model(b)
            e = torch.nn.functional.normalize(e, p=2, dim=1)
        embs.append(e.cpu().numpy())

    all_embs = np.concatenate(embs, axis=0).astype(np.float32) if embs else np.zeros((0, EMBED_DIM), dtype=np.float32)
    print(f"[embed] Computed embeddings: {all_embs.shape}")
    return all_embs


def save_index(embeddings: np.ndarray, records: List[Dict[str, Any]]) -> None:
    ensure_dir(INDEX_DIR)
    np.save(EMBED_NPY, embeddings)
    print(f"[index] Saved embeddings: {EMBED_NPY}")

    with open(IDS_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "video", "frame_index", "timestamp_ms", "x1", "y1", "x2", "y2", "score", "face_path"])
        for idx, rec in enumerate(records):
            w.writerow([
                idx, rec["video"], rec["frame_index"], rec["timestamp_ms"],
                rec["x1"], rec["y1"], rec["x2"], rec["y2"],
                f"{rec['score']:.4f}", rec["face_path"]
            ])
    print(f"[index] Saved ids: {IDS_CSV}")

    if embeddings.shape[0] > 0:
        d = embeddings.shape[1]
        index = faiss.IndexFlatIP(d)   # cosine via inner product on L2-normalized vectors
        index.add(embeddings)
        faiss.write_index(index, FAISS_INDEX)
        print(f"[index] Saved FAISS index: {FAISS_INDEX}")
    else:
        print("[index] No embeddings to index.")


def load_index() -> Tuple[np.ndarray, faiss.IndexFlatIP, List[Dict[str, Any]]]:
    if not os.path.isfile(EMBED_NPY) or not os.path.isfile(IDS_CSV):
        raise RuntimeError("Index files not found. Run 'python face_search.py build' first.")

    embeddings = np.load(EMBED_NPY).astype(np.float32)

    id_records: List[Dict[str, Any]] = []
    with open(IDS_CSV, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            id_records.append({
                "id": int(row["id"]),
                "video": row["video"],
                "frame_index": int(row["frame_index"]),
                "timestamp_ms": int(row["timestamp_ms"]),
                "x1": int(row["x1"]), "y1": int(row["y1"]),
                "x2": int(row["x2"]), "y2": int(row["y2"]),
                "score": float(row["score"]),
                "face_path": row["face_path"],
            })

    if not os.path.isfile(FAISS_INDEX):
        raise RuntimeError("FAISS index file not found. Re-run 'python face_search.py build'.")

    index = faiss.read_index(FAISS_INDEX)
    return embeddings, index, id_records


def embed_single_image(img_path: str, device: str | None = None) -> np.ndarray:
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model, tfm = load_facenet(device=device)
    img = _load_rgb(img_path)
    with torch.no_grad():
        t = tfm(img).unsqueeze(0).to(device)  # [1,3,160,160]
        e = model(t)                          # [1,512]
        e = torch.nn.functional.normalize(e, p=2, dim=1)
    return e.cpu().numpy().astype(np.float32)  # [1,512]


def search_topk(query_emb: np.ndarray, index: faiss.IndexFlatIP, k: int) -> Tuple[np.ndarray, np.ndarray]:
    scores, idxs = index.search(query_emb, k)  # cosine on normalized vectors
    return scores, idxs


def save_search_results(results: List[Dict[str, Any]], run_name: str) -> str:
    out_dir = os.path.join(SEARCH_RESULTS_DIR, run_name)
    ensure_dir(out_dir)
    csv_path = os.path.join(out_dir, "results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rank", "score", "video", "frame_index", "timestamp_ms", "bbox", "face_path", "copied_to"])
        for r in results:
            src = r["face_path"]
            rank = r["rank"]
            score = f"{r['score']:.4f}"
            name = f"rank{rank:02d}_score{score}_vid-{r['video']}_f{r['frame_index']:06d}.jpg"
            dst = os.path.join(out_dir, name)
            try:
                shutil.copyfile(src, dst)
                copied_to = dst
            except Exception:
                copied_to = ""
            w.writerow([
                rank, score, r["video"], r["frame_index"], r["timestamp_ms"],
                f"({r['x1']},{r['y1']},{r['x2']},{r['y2']})",
                src, copied_to
            ])
    print(f"[search] Saved results to: {out_dir}")
    return out_dir


# ---------------------- DB (SQL Server) ----------------------

def make_engine(conn_str: str):
    """
    conn_str example:
    mssql+pyodbc://USER:PASS@SERVERNAME/DBNAME?driver=ODBC+Driver+17+for+SQL+Server&TrustServerCertificate=yes
    """
    return create_engine(conn_str, pool_pre_ping=True, pool_size=10, max_overflow=20)


def get_face_table(meta: MetaData) -> Table:
    return Table(
        "FaceEmbedding", meta,
        Column("EmbId", BigInteger, primary_key=True, autoincrement=True),
        Column("Video", String(255), nullable=False),
        Column("FrameIndex", Integer, nullable=False),
        Column("TimestampMs", BigInteger, nullable=False),
        Column("X1", Integer, nullable=False), Column("Y1", Integer, nullable=False),
        Column("X2", Integer, nullable=False), Column("Y2", Integer, nullable=False),
        Column("Score", Float, nullable=False),
        Column("FacePath", String(400), nullable=False),
        Column("ModelName", String(64), nullable=False),
        Column("Dim", Integer, nullable=False),
        Column("L2Norm", Float, nullable=False),
        Column("VecSHA1", BINARY(20), nullable=False, unique=True),
        Column("Vector", LargeBinary, nullable=False),
        Column("CreatedAt", DateTime, server_default=text("SYSDATETIME()")),
    )


def vec_sha1(vec_f32: np.ndarray) -> bytes:
    """Return SHA1(bytes) of float32 vector."""
    return hashlib.sha1(vec_f32.tobytes()).digest()


def cmd_build(args):
    records = gather_face_records(OUTPUT_ROOT)
    if not records:
        print("[build] No face records found. Run your extractor first.")
        return
    embs = compute_embeddings(records, batch_size=BATCH_SIZE)
    save_index(embs, records)


def cmd_search(args):
    _, index, id_records = load_index()

    query_path = args.image
    if not os.path.isfile(query_path):
        raise RuntimeError(f"Query image not found: {query_path}")
    q = embed_single_image(query_path)  # [1,512]

    topk = args.topk
    scores, idxs = search_topk(q, index, topk)

    results: List[Dict[str, Any]] = []
    for rank, (score, idx) in enumerate(zip(scores[0], idxs[0]), start=1):
        rec = id_records[int(idx)]
        results.append({
            "rank": rank,
            "score": float(score),
            "video": rec["video"],
            "frame_index": rec["frame_index"],
            "timestamp_ms": rec["timestamp_ms"],
            "x1": rec["x1"], "y1": rec["y1"], "x2": rec["x2"], "y2": rec["y2"],
            "face_path": rec["face_path"],
        })

    print("\n[search] Top matches:")
    for r in results:
        print(f"  #{r['rank']:02d}  score={r['score']:.4f}  video={r['video']}  "
              f"frame={r['frame_index']}  ts_ms={r['timestamp_ms']}  path={r['face_path']}")

    if args.save:
        run_name = Path(query_path).stem
        save_search_results(results, run_name)


def cmd_pushdb(args):
    """
    Read embeddings + ids from _index and insert into SQL Server table dbo.FaceEmbedding.
    Skips duplicates by VecSHA1 unique key.
    """
    # Load index files
    embeddings = np.load(EMBED_NPY).astype(np.float32)
    id_rows: List[Dict[str, Any]] = []
    with open(IDS_CSV, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            id_rows.append(row)

    # Prepare DB
    engine = make_engine(args.connection_string)
    meta = MetaData()
    face_tbl = get_face_table(meta)
    with engine.begin() as conn:
        meta.create_all(conn)  # create if not exists

        # Insert rows
        print("[pushdb] Inserting rows into dbo.FaceEmbedding ...")
        for i, row in enumerate(tqdm(id_rows, desc="[pushdb]")):
            vec = embeddings[int(row["id"])]  # [512]
            l2 = float(np.linalg.norm(vec))
            sha = vec_sha1(vec)

            ins = face_tbl.insert().values(
                Video=row["video"],
                FrameIndex=int(row["frame_index"]),
                TimestampMs=int(row["timestamp_ms"]),
                X1=int(row["x1"]), Y1=int(row["y1"]),
                X2=int(row["x2"]), Y2=int(row["y2"]),
                Score=float(row["score"]),
                FacePath=row["face_path"],
                ModelName=MODEL_NAME,
                Dim=EMBED_DIM,
                L2Norm=l2,
                VecSHA1=sha,
                Vector=vec.tobytes(),  # float32 bytes length=2048
            )
            try:
                conn.execute(ins)
            except IntegrityError:
                # duplicate (VecSHA1) — skip
                continue

    print("[pushdb] Done.")

def main():
    parser = argparse.ArgumentParser(description="Face search over extracted face crops.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_build = sub.add_parser("build", help="Build embeddings + FAISS index from extracted_faces/*/detections.csv")
    p_build.set_defaults(func=cmd_build)

    p_search = sub.add_parser("search", help="Search matches for a query image")
    p_search.add_argument("--image", required=True, help="Path to a face image (already cropped if possible)")
    p_search.add_argument("--topk", type=int, default=20, help="Number of matches to return")
    p_search.add_argument("--save", action="store_true", help="Copy matched images and write results.csv")
    p_search.set_defaults(func=cmd_search)

    p_push = sub.add_parser("pushdb", help="Push all embeddings into SQL Server (dbo.FaceEmbedding)")
    p_push.add_argument("--connection-string", required=True, help="SQLAlchemy conn str, e.g. mssql+pyodbc://matan:matanaankl123@localhost/FaceRecognitionDB?driver=ODBC+Driver+17+for+SQL+Server&TrustServerCertificate=yes")
    p_push.set_defaults(func=cmd_pushdb)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
