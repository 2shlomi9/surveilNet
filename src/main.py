# src/main.py
"""
Command-line entrypoint for the Missing-ID project.

Subcommands:
  - index-video: index a video file (detect faces -> align -> embed -> store -> add to FAISS)
  - search:      search by one or multiple query images (few-shot) against the FAISS index

Notes:
  * SQL Server is used for metadata; FAISS holds the vector index on disk.
"""

from __future__ import annotations
import argparse
import os
import sys
import numpy as np

# Ensure project root is importable when running `python -m src.main`
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sqlalchemy import text
from sqlalchemy.engine import Engine

from src.utils.config import load_config, ensure_dirs
from src.pipeline.index_video import index_video
from src.pipeline.search import load_and_align_query
from src.models.embedder import FaceEmbedder
from src.store.indexer import FaissIndex
from src.store.sql import get_engine, run_schema


def _ensure_env(cfg: dict) -> None:
    """
    Ensure base folders exist (media/db) and SQL schema is in place.
    """
    media_root = cfg["paths"]["media_root"]
    db_root = cfg["paths"]["db_root"]
    ensure_dirs(media_root, db_root)


def cmd_init_db(args: argparse.Namespace) -> None:
    """
    Initialize SQL Server schema only.
    """
    cfg = load_config(args.config)
    _ensure_env(cfg)
    engine: Engine = get_engine(cfg["sqlserver"]["conn_str"])
    run_schema(engine)
    print("SQL Server schema initialized.")


def cmd_index_video(args: argparse.Namespace) -> None:
    """
    Index a single video file.
    """
    cfg = load_config(args.config)
    _ensure_env(cfg)

    video_path = args.video
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    index_video(video_path=video_path, camera_id=args.camera_id, conf_path=args.config)


def _search_core(query_vec: np.ndarray, cfg: dict) -> list[dict]:
    """
    Core search routine against FAISS + metadata fetch from SQL Server.
    Returns a list of dicts with match metadata (rank, score, appearance info).
    """
    # Load FAISS index
    index = FaissIndex(dim=cfg["embed"]["dim"],
                       index_path=cfg["index"]["faiss_path"],
                       vectors_path=cfg["index"]["vectors_path"])
    if index.size == 0:
        raise RuntimeError("FAISS index is empty. Index videos first via `index-video`.")

    # Top-K search
    top_k = int(cfg["search"]["top_k"])
    scores, idxs = index.search(query_vec.astype(np.float32), top_k=top_k)

    # Fetch metadata from SQL Server
    engine: Engine = get_engine(cfg["sqlserver"]["conn_str"])
    results = []
    with engine.begin() as conn:
        for rank, (score, vec_id) in enumerate(zip(scores, idxs), start=1):
            row = conn.execute(
                text("SELECT AppearanceId, ThumbPath, CameraId, VideoId, FrameIndex, TimestampMs "
                     "FROM dbo.FaceAppearances WHERE VectorId = :vid"),
                {"vid": int(vec_id)}
            ).fetchone()
            if row:
                results.append({
                    "rank": rank,
                    "score": float(score),
                    "appearance_id": int(row.AppearanceId),
                    "thumb_path": row.ThumbPath,
                    "camera_id": int(row.CameraId) if row.CameraId is not None else None,
                    "video_id": int(row.VideoId) if row.VideoId is not None else None,
                    "frame_index": int(row.FrameIndex),
                    "timestamp_ms": int(row.TimestampMs),
                })
    return results


def cmd_search(args: argparse.Namespace) -> None:
    """
    Search by one or multiple query images (few-shot).
    We average normalized embeddings from all provided images.
    """
    cfg = load_config(args.config)
    _ensure_env(cfg)

    # Prepare models
    embedder = FaceEmbedder()
    align_size = tuple(cfg["align"]["size"])

    # Build a few-shot query vector: average of normalized embeddings
    vecs = []
    for img_path in args.images:
        if not os.path.isfile(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")
        aligned = load_and_align_query(img_path, align_size=align_size)
        v = embedder.embed(aligned)  # already L2-normalized
        vecs.append(v)

    if len(vecs) == 0:
        raise RuntimeError("No valid query images provided.")

    # Average normalized embeddings, then normalize again
    q = np.mean(np.stack(vecs, axis=0), axis=0)
    q = q / (np.linalg.norm(q) + 1e-8)

    results = _search_core(q, cfg)

    # Print human-readable output
    print(f"\nTop-{cfg['search']['top_k']} matches:")
    for r in results:
        print(
            f"[{r['rank']:02d}] score={r['score']:.4f}  "
            f"appearance_id={r['appearance_id']}  "
            f"camera_id={r['camera_id']}  video_id={r['video_id']}  "
            f"frame={r['frame_index']}  ts_ms={r['timestamp_ms']}  "
            f"thumb='{r['thumb_path']}'"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="missing-id",
        description="Missing-ID: face indexing and search CLI"
    )
    parser.add_argument(
        "--config", "-c",
        default="configs/default.yaml",
        help="Path to YAML config file (default: configs/default.yaml)"
    )

    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init-db", help="Initialize SQL Server schema")
    p_init.set_defaults(func=cmd_init_db)

    p_index = sub.add_parser("index-video", help="Index a video file into the system")
    p_index.add_argument("--video", "-v", required=True, help="Path to video file (mp4)")
    p_index.add_argument("--camera-id", type=int, default=None, help="Optional CameraId to associate")
    p_index.set_defaults(func=cmd_index_video)

    p_search = sub.add_parser("search", help="Search by one or multiple query images (few-shot)")
    p_search.add_argument("--images", "-i", nargs="+", required=True, help="Path(s) to query image(s)")
    p_search.set_defaults(func=cmd_search)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
