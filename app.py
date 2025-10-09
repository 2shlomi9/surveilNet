from flask import Flask, request, jsonify, send_from_directory
from models.face_database import FaceDatabase
from models.face_matcher import FaceMatcher
from models.video_processor import VideoProcessor
import firebase_admin
from firebase_admin import credentials, firestore
import torch
import os
from werkzeug.utils import secure_filename
import uuid
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)
CORS(app, origins=["http://localhost:3000"])  # React dev server

# Firebase initialization
cred = credentials.Certificate("configs/accountKey.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize FaceDatabase
face_db = FaceDatabase()

# Configuration
UPLOAD_FOLDER = 'uploads'
VIDEO_FOLDER = 'videos_database'
MATCHES_FOLDER = 'matches'
ALLOWED_EXTENSIONS = {'jpg', 'png', 'mp4'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MATCHES_FOLDER, exist_ok=True)
os.makedirs(VIDEO_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ---------------------- API ENDPOINTS ----------------------

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'device': str(device)}), 200

@app.route('/api/people', methods=['GET'])
def get_all_people():
    try:
        return jsonify({'people': [p.to_dict() for p in face_db.people]}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/people/<person_id>', methods=['GET'])
def get_person(person_id):
    for person in face_db.people:
        if person.id == person_id:
            return jsonify(person.to_dict()), 200
    return jsonify({'error': 'Person not found'}), 404

@app.route('/api/people', methods=['POST'])
def add_person():
    try:
        if not request.form.get('first_name') or not request.form.get('last_name'):
            return jsonify({'error': 'first_name and last_name are required'}), 400

        person_id = str(uuid.uuid4())
        first_name = request.form['first_name']
        last_name = request.form['last_name']
        age = request.form.get('age')

        if 'images' not in request.files:
            return jsonify({'error': 'At least one image is required'}), 400

        images = request.files.getlist('images')
        img_paths = []
        person_folder = os.path.join(app.config['UPLOAD_FOLDER'], f"{person_id}_{first_name}_{last_name}")
        os.makedirs(person_folder, exist_ok=True)

        for image in images:
            if image and allowed_file(image.filename):
                filename = secure_filename(image.filename)
                save_path = os.path.join(person_folder, filename)
                image.save(save_path)
                img_paths.append(save_path)

        if not img_paths:
            return jsonify({'error': 'No valid images provided'}), 400

        face_db.add_person(person_id, first_name, last_name, img_paths, age)
        face_db.upload_to_firestore(db)

        return jsonify({'message': f'Added {first_name} {last_name}', 'id': person_id}), 201

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/videos', methods=['GET'])
def get_videos():
    videos = [f for f in os.listdir(VIDEO_FOLDER) if allowed_file(f) and f.endswith('.mp4')]
    return jsonify({'videos': videos})

@app.route('/api/upload_video', methods=['POST'])
def upload_video():
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400

        video = request.files['video']
        if video and allowed_file(video.filename):
            filename = secure_filename(video.filename)
            save_path = os.path.join(VIDEO_FOLDER, filename)
            video.save(save_path)
            return jsonify({'message': f'Video {filename} uploaded successfully', 'filename': filename}), 201
        else:
            return jsonify({'error': 'Invalid video file'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/matches', methods=['GET'])
def get_matches():
    matches = [f for f in os.listdir(MATCHES_FOLDER) if allowed_file(f) and f.endswith(('.jpg', '.png'))]
    return jsonify({'matches': matches})

@app.route('/api/process_video', methods=['POST'])
def process_video():
    data = request.get_json()
    if not data or 'filename' not in data:
        return jsonify({'error': 'Filename is required in JSON body'}), 400

    filename = secure_filename(data['filename'])
    video_path = os.path.join(VIDEO_FOLDER, filename)
    if not os.path.exists(video_path):
        return jsonify({'error': f'Video file not found: {filename}'}), 404

    matcher = FaceMatcher(face_db)
    processor = VideoProcessor(matcher, output_folder=MATCHES_FOLDER, frame_skip=5)
    processor.process_video(video_path)

    return jsonify({'message': 'Video processed', 'video': filename}), 200

@app.route('/matches/<filename>')
def serve_match(filename):
    return send_from_directory(MATCHES_FOLDER, filename)

@app.route('/api/matches/<filename>', methods=['DELETE'])
def delete_match(filename):
    try:
        file_path = os.path.join(MATCHES_FOLDER, filename)
        if not os.path.exists(file_path):
            return jsonify({'error': 'File not found'}), 404
        os.remove(file_path)
        return jsonify({'message': f'File {filename} deleted successfully'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/matches', methods=['DELETE'])
def delete_all_matches():
    try:
        deleted_files = []
        for filename in os.listdir(MATCHES_FOLDER):
            if allowed_file(filename) and filename.endswith(('.jpg', '.png')):
                file_path = os.path.join(MATCHES_FOLDER, filename)
                os.remove(file_path)
                deleted_files.append(filename)
        if not deleted_files:
            return jsonify({'message': 'No matches found to delete'}), 200
        return jsonify({'message': f'Deleted {len(deleted_files)} match files', 'deleted_files': deleted_files}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ---------------------- MAIN ----------------------
if __name__ == '__main__':
    # Load database
    USE_FIRESTORE = True
    if USE_FIRESTORE:
        face_db.load_from_firestore(db)
        print(f"[INFO] Loaded {len(face_db.people)} people from Firestore.")
    else:
        gallery_folder = "database"
        face_db.build_from_folder(gallery_folder)
        face_db.upload_to_firestore(db)
        print(f"[INFO] Built and uploaded {len(face_db.people)} people from local folder.")

    app.run(debug=True, host='127.0.0.1', port=5000)