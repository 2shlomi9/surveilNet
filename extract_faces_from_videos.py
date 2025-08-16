from retinaface import RetinaFace
import cv2
import os
import csv
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

INPUT_DIR = "videos_database"
OUTPUT_DIR = "extracted_faces"
CONF_THRESH = 0.90
FRAME_INTERVAL = 5
MIN_FACE_PX = 60
MAX_DET_SIDE = 960  # Resize long side to this before detection

def extract_faces_from_video(video_path, output_subdir):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[error] Failed to open video: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    detections = []
    frame_idx = -1

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_idx % FRAME_INTERVAL != 0:
            continue

        # Resize if too large
        h, w = frame.shape[:2]
        scale = 1.0
        if max(h, w) > MAX_DET_SIDE:
            scale = MAX_DET_SIDE / max(h, w)
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

        # Extract faces with alignment
        try:
            faces = RetinaFace.extract_faces(frame, align=True, threshold=CONF_THRESH)
        except Exception as e:
            print(f"[warn] Failed on frame {frame_idx}: {e}")
            continue

        timestamp_ms = int(1000 * frame_idx / fps)

        for i, face in enumerate(faces):
            fh, fw = face.shape[:2]
            if min(fh, fw) < MIN_FACE_PX:
                continue

            crop_name = f"f{frame_idx:06d}_t{timestamp_ms}_face{i}.jpg"
            crop_path = os.path.join(output_subdir, crop_name)
            cv2.imwrite(crop_path, face)

            detections.append({
                "frame_index": frame_idx,
                "timestamp_ms": timestamp_ms,
                "x1": 0, "y1": 0, "x2": fw, "y2": fh,  # unknown after align
                "score": 1.0,
                "face_path": crop_name,
            })

    cap.release()

    if detections:
        csv_path = os.path.join(output_subdir, "detections.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=detections[0].keys())
            writer.writeheader()
            writer.writerows(detections)

        print(f"[done] {len(detections)} face(s) saved to: {output_subdir}")
    else:
        print(f"[done] No faces found in: {video_path}")

def main():
    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    videos = list(Path(INPUT_DIR).glob("*.mp4"))

    for video_path in tqdm(videos, desc="Processing videos"):
        name = video_path.stem
        output_subdir = os.path.join(OUTPUT_DIR, name)
        Path(output_subdir).mkdir(exist_ok=True)
        extract_faces_from_video(video_path, output_subdir)

if __name__ == "__main__":
    main()
