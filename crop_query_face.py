import cv2
from retinaface import RetinaFace
import argparse
import os

def extract_and_save_face(image_path: str, output_path: str, align: bool = True) -> None:
    # Detect and extract faces
    try:
        faces = RetinaFace.extract_faces(img_path=image_path, align=align)
    except Exception as e:
        print(f"[error] Failed to extract faces: {e}")
        return

    if not faces:
        print("[error] No face detected in input image.")
        return

    # Save the first face found
    face = faces[0]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, cv2.cvtColor(face, cv2.COLOR_RGB2BGR))
    print(f"[done] Saved cropped face to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crop and align face from query image.")
    parser.add_argument("--input", required=True, help="Path to original image with face")
    parser.add_argument("--output", default="cropped_face.jpg", help="Output cropped face path")
    parser.add_argument("--no-align", action="store_true", help="Disable face alignment")

    args = parser.parse_args()
    extract_and_save_face(args.input, args.output, align=not args.no_align)
