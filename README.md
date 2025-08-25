# SurveilNet – Video Face Indexing & Search

## 1. What is this project?

SurveilNet ingests videos, detects faces frame-by-frame, aligns each face to a canonical view, extracts a 512-D embedding per face, and stores results in SQL Server.  
Later, you can query the database with a single face image and retrieve matching frames/videos.

**Core capabilities:**
- Video ingestion → faces + embeddings written to DB  
- Gallery maintenance → per-person average embedding  
- Query by image → find top matches across all stored frames  
- Operational robustness (ROI clamping, configurable SQL connection, reproducible SQL schema)

---

## 2. Project goals

- **Accurate face search** from unconstrained video (pose/lighting/scale changes)  
- **Stable, comparable embeddings** via alignment + FaceNet  
- **Simple, explainable pipeline** using well-known models  
- **Operational robustness** (no ROI crashes, handle GPU/CPU constraints)  
- **Reproducible setup** (config file, SQL scripts, clear commands)  

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

### FaceNet – InceptionResnetV1
- **What**: Embedding model (512-D vectors)  
- **Why**: Compact, comparable embeddings; widely validated  
- **Role**: Generate embeddings for DB and search

### Cosine Similarity
- **What**: Measure similarity of normalized vectors  
- **Why**: Simple and effective metric for embeddings  
- **Role**: Compare new faces against gallery/DB

### OpenCV Trackers (optional)
- **What**: Appearance-based tracking between detections  
- **Why**: Reduce detection frequency  
- **Trade-off**: Trackers drift; optional in our pipeline

### ROI Clamping
- **What**: Clamp bboxes to frame, discard invalids  
- **Why**: Prevent empty crops and crashes  
- **Role**: Stabilize pipeline

---

## 4. Installation & dependencies

### 4.1 Prerequisites
- Python 3.9+  
- Microsoft ODBC Driver 17/18 for SQL Server  
- (Optional) NVIDIA GPU + CUDA/cuDNN  

### 4.2 Python packages
```bash
pip install opencv-contrib-python numpy Pillow pyodbc
pip install torch torchvision torchaudio
pip install facenet-pytorch
pip install retinaface tensorflow==2.*
```
See requirements.txt for more information.

### 4.3 Configuration

`config/host_info.ini` (gitignored).

**Windows Authentication:**
```ini
[sqlserver]
driver = ODBC Driver 17 for SQL Server
server = YourServer
database = YourDB
trusted_connection = yes - if using Windows Authentication, no - if using SQL SERVER Authentication
username = fill if trusted_connection = no
password = fill if trusted_connection = no
```
# Database & Scripts Documentation

## 4.4 Database schema

Run the following SQL scripts in **SQL Server Management Studio** to create and prepare the database schema:
-sql_queries/create_tables.sql
-sql_queries/indexes.sql

---

## 6. Scripts

### 6.1 video_face_extractor.py

**Purpose**:  
Process videos in `videos_database/`, detect faces, embed them, and insert results into SQL Server.

**Flow**:
1. Connect to DB (from config)  
2. For each `.mp4` file:  
   - Detect with RetinaFace  
   - Clamp boxes  
   - Align with MTCNN  
   - Embed with FaceNet  
   - Match to gallery (cosine sim, threshold)  
   - Insert row in `FaceEmbeddings` + update `FaceGallery`  
   - Print logs for monitoring  

**Run**:
```bash
python video_face_extractor.py
```

### 6.2 face_query_search.py

Purpose:
Search the database for matches to a query image.

Flow:
- Load query image
- Detect/align with MTCNN
- Embed with FaceNet
- Fetch embeddings from DB
- Compare with cosine similarity
- Display/save top matches

Run:
```bash
python face_query_search.py --image path/to/query.jpg --topk 5
```

Output:
- Console similarity scores + metadata
- Cropped face results saved in matches/

7. Performance tips
- Downscale frames before detection
- Use frame skipping (detect every N frames)
- Split TF on CPU & FaceNet on GPU (or vice-versa)
- Use ROI clamping (already included)
- Batch DB writes to reduce overhead

8. Troubleshooting

- Empty ROI crash → fixed by clamp logic
- GPU OOM → run one framework on CPU, other on GPU
- Missing CSRT tracker → use opencv-contrib-python or legacy API
- ODBC SSL error → add Encrypt=yes;TrustServerCertificate=yes;

9. Why these choices?

- RetinaFace → reliable detection + landmarks
- MTCNN → convenient alignment
- FaceNet → proven embeddings
- Cosine similarity → simple + effective
- SQL Server → reliable storage, easy queries
