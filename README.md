# SurveilNet – Video Face Indexing & Search

## 1. What is this project?

SurveilNet is an end-to-end system for **video face detection**, **face recognition**, and **match visualization**, developed as a university final project.
**Core capabilities:**
- Video ingestion → faces + embeddings written to DB  
- Gallery maintenance → per-person average embedding  
- Query by image → find top matches across all stored frames  
- Operational robustness (ROI clamping, configurable SQL connection, reproducible SQL schema)

### The system:

✅ Uploads and processes CCTV-like video streams  
✅ Detects and indexes faces frame-by-frame  
✅ Uploads a person photo and searches across all processed videos  
✅ Displays the matching video segment with a **dynamic moving face box**

This provides a realistic proof-of-concept for missing-person search and visual surveillance investigation workflows.

---
## ✅ Core Capabilities

| Feature | Description |
|--------|-------------|
| Video ingestion | Frame-by-frame face detection + embedding extraction |
| Firestore database | Store enrolled people with embeddings |
| Search | Upload an image → find best-matching frames in processed videos |
| Visualization | Play snippet with **moving face box** synced to tracked identity |
| Multi-user support | Parallel uploads and searches using async task pool |
| Cancel-safety | Upload + processing can be individually canceled |
| Robustness | ROI clamping, FPS normalization, snippet caching |

## 2. Project goals

- **Accurate face search** from unconstrained video (pose/lighting/scale changes)  
- **Stable, comparable embeddings** via alignment + FaceNet  
- **Simple, explainable pipeline** using well-known models  
- **Operational robustness** (no ROI crashes, handle GPU/CPU constraints)  
- **Concurrency Jobs** using multithreaded flask server

---

## 3. Algorithms & why we use them

### RetinaFace (TensorFlow)
- **What**: Strong face detector with 5 facial landmarks  
- **Why**: Provides reliable bounding boxes and landmarks for alignment  
- **Role**: Detect faces per frame

### MTCNN (facenet-pytorch)
- **What**: Lightweight detector/aligner  
- **Why**: Produces normalized 160×160 crops for stable embeddings  
- **Role**: Align face crops for FaceNet input
- 
➡ *This step greatly improves match reliability. Without alignment — similarity drops dramatically.*

### FaceNet – InceptionResnetV1
- **What**: Embedding model (512-D vectors)  
- **Why**: Compact, comparable embeddings; widely validated  
- **Role**: Generate embeddings for DB and search

### Cosine Similarity
- **Why:** Simple, fast, scale-invariant
- **Role:** Person search → find best scores per frame

---

## 4. Installation & dependencies

### 4.1 Prerequisites
- Python 3.9+
- (Optional) NVIDIA GPU + CUDA/cuDNN  

### 4.2 Python packages
```bash
pip install opencv-contrib-python numpy Pillow pyodbc
pip install torch torchvision torchaudio
pip install facenet-pytorch
pip install retinaface tensorflow==2.*
```
See requirements.txt for more information.

## 🎥 Video Snippet Playback & Moving Face Box

When a match is found:

1️⃣ A short H.264 snippet is generated around the matched frame  
2️⃣ For each frame in the snippet — the best matching detection is loaded  
3️⃣ A **sticky** tracking system keeps box visible even in low-confidence frames  

✅ Result: The box **follows the same identity**, not random faces in the video.
---

## ⚙️ Processing Pipeline (Video)

For each uploaded video:

1. Read frames (downsample by skip=5 for speed)
2. Detect faces → RetinaFace  
3. Clamp bounding boxes to frame
4. Align crops → MTCNN  
5. Generate embedding → FaceNet  
6. Save `{frame_idx, bbox, embedding, fps}` to disk (frame_store/)  
7. Track async progress (polling from frontend)

✅ GPU is used when available  
✅ Multiple concurrent jobs via semaphore-limited worker threads

---

## 👤 Person Enrollment Pipeline

1. User uploads 1+ images
2. RetinaFace detects faces in the images
3. MTCNN aligns each face crop
4. FaceNet → embeddings
5. Average embedding saved in Firestore for that identity
6. Immediately search over all processed frames
7. Show best match (if score > threshold)

---

## 🛠 Tech Stack

| Area | Tool |
|------|-----|
| Backend | Flask (Python) |
| Face Detection | RetinaFace |
| Alignment | MTCNN |
| Embeddings | FaceNet (facenet-pytorch) |
| Frontend | React.js |
| Storage (people) | Google Firestore |
| Storage (frames) | Local frame_store/ |
| Snippet Encoding | OpenCV + ffmpeg auto-H.264 |
| Parallelism | Python threading + semaphore |
| Modal UX | Custom React player w/ overlay canvas |


Output:
- Console similarity scores + metadata
- Cropped face results saved in matches/

## Performance tips
- Downscale frames before detection
- Use frame skipping (detect every N frames)
- Split TF on CPU & FaceNet on GPU (or vice-versa)


## Why these choices?

- RetinaFace → reliable detection + landmarks
- MTCNN → convenient alignment
- FaceNet → proven embeddings
- Cosine similarity → simple + effective
