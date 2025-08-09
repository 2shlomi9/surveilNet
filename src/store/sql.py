from __future__ import annotations
import os
from typing import Optional, Tuple, Dict, Any
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine, Connection

def get_engine(conn_str: str) -> Engine:
    """
    Create a SQLAlchemy Engine for SQL Server via pyodbc.
    """
    engine = create_engine(conn_str, pool_pre_ping=True, fast_executemany=True)
    return engine

def run_schema(engine: Engine, schema_path: str = "db/schema.sql") -> None:
    """
    Execute the SQL schema file to create tables and indexes.
    """
    with engine.begin() as conn:
        with open(schema_path, "r", encoding="utf-8") as f:
            sql = f.read()
        # split on GO if present (some scripts use it), naive split
        for chunk in sql.split("GO"):
            stmt = chunk.strip()
            if stmt:
                conn.exec_driver_sql(stmt)

def insert_video(conn: Connection, camera_id: Optional[int], source_path: str,
                 start_time_utc: Optional[str] = None, end_time_utc: Optional[str] = None) -> int:
    """
    Insert a video row and return VideoId.
    """
    res = conn.execute(
        text("""
        INSERT INTO dbo.Videos (CameraId, SourcePath, StartTimeUtc, EndTimeUtc)
        OUTPUT INSERTED.VideoId
        VALUES (:camera_id, :source_path, :start_time, :end_time)
        """),
        {"camera_id": camera_id, "source_path": source_path, "start_time": start_time_utc, "end_time": end_time_utc}
    )
    return int(res.scalar_one())

def insert_face_appearance(conn: Connection, camera_id: Optional[int], video_id: Optional[int],
                           frame_index: int, ts_ms: int, bbox: Tuple[int,int,int,int],
                           quality: Optional[float], thumb_path: Optional[str]) -> int:
    """
    Insert a face appearance and return AppearanceId.
    """
    x1,y1,x2,y2 = bbox
    res = conn.execute(
        text("""
        INSERT INTO dbo.FaceAppearances
        (CameraId, VideoId, FrameIndex, TimestampMs, BboxX1, BboxY1, BboxX2, BboxY2, QualityScore, ThumbPath)
        OUTPUT INSERTED.AppearanceId
        VALUES (:camera_id, :video_id, :frame_index, :ts_ms, :x1, :y1, :x2, :y2, :quality, :thumb_path)
        """),
        {"camera_id": camera_id, "video_id": video_id, "frame_index": frame_index, "ts_ms": ts_ms,
         "x1": x1, "y1": y1, "x2": x2, "y2": y2, "quality": quality, "thumb_path": thumb_path}
    )
    return int(res.scalar_one())

def update_vector_map(conn: Connection, appearance_id: int, vector_id: int, l2norm: float) -> None:
    """
    Update FaceAppearances with VectorId and insert into VectorMap.
    """
    conn.execute(
        text("UPDATE dbo.FaceAppearances SET VectorId = :vid WHERE AppearanceId = :aid"),
        {"vid": vector_id, "aid": appearance_id}
    )
    conn.execute(
        text("""
        INSERT INTO dbo.VectorMap (VectorId, AppearanceId, L2Norm)
        VALUES (:vid, :aid, :l2)
        """),
        {"vid": vector_id, "aid": appearance_id, "l2": l2norm}
    )
