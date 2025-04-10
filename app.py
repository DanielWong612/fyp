from flask import Flask, render_template, request, Response, jsonify
import os
import json
import shutil
from recognitionDemo.classroom_Monitor_System import generate_processed_frames, detected_faces, known_faces, auto_capture,capture_face_from_current_frame
import cv2

app = Flask(__name__)

STUDENTS_FILE = 'students.json'
PAIRINGS_FILE = 'face_pairings.json'
FACE_DB_PATH = 'static/face_database'
THUMB_DIR = 'static/thumbnails'
DEFAULT_AVATAR = 'default_avatar.png'

def load_students():
    with open(STUDENTS_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_pairings():
    if not os.path.exists(PAIRINGS_FILE):
        return {}
    with open(PAIRINGS_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_pairing(image, sid):
    pairings = load_pairings()
    pairings[image] = sid
    with open(PAIRINGS_FILE, 'w', encoding='utf-8') as f:
        json.dump(pairings, f, indent=2)

def get_first_face_images():
    if not os.path.exists(THUMB_DIR):
        os.makedirs(THUMB_DIR)
    pairings = load_pairings()
    user_faces = []
    
    for filename in sorted(os.listdir(FACE_DB_PATH)):
        if filename.startswith('user') and filename.lower().endswith('.jpg'):
            src_path = os.path.join(FACE_DB_PATH, filename)
            if os.path.isfile(src_path): 
                thumb_name = filename
                dst_path = os.path.join(THUMB_DIR, thumb_name)
                if not os.path.exists(dst_path):
                    shutil.copy(src_path, dst_path)
                if thumb_name not in pairings:
                    user_faces.append({'label': filename, 'filepath': src_path})
    
    return user_faces

def map_student_faces(students, pairings):
    sid_to_image = {v: k for k, v in pairings.items()}
    for student in students:
        image_name = sid_to_image.get(student['sid'])
        student['avatar'] = f"thumbnails/{image_name}" if image_name else DEFAULT_AVATAR
    return students

@app.route('/')
def index():
    students = load_students()
    pairings = load_pairings()
    students = map_student_faces(students, pairings)
    unpaired_faces = get_first_face_images()
    return render_template('index.html', students=students, unpaired_faces=unpaired_faces)

@app.route('/pair', methods=['POST'])
def pair():
    image = request.form['image']  
    student_sid = request.form['student_sid']  
    # 移動圖片到學生的目錄
    src_path = os.path.join(FACE_DB_PATH, image)  # recognitionDemo/face_database/user1.jpg
    dst_dir = os.path.join(FACE_DB_PATH, student_sid)  # recognitionDemo/face_database/220675880
    os.makedirs(dst_dir, exist_ok=True)
    dst_path = os.path.join(dst_dir, image)  # recognitionDemo/face_database/220675880/user1.jpg
    shutil.move(src_path, dst_path)
    save_pairing(image, student_sid)
    return jsonify({'success': True, 'image': image, 'student_sid': student_sid})

@app.route('/manual_capture', methods=['POST'])
def manual_capture_route():
    selected_student = request.form['student_sid']
    return Response(generate_processed_frames(selected_student=selected_student, manual_capture_trigger=True), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/processed_video_feed')
def processed_video_feed():
    return Response(generate_processed_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture_face', methods=['POST'])
def capture_face():
    captured_faces = capture_face_from_current_frame()
    return jsonify({'success': True, 'captured_faces': len(captured_faces)})

@app.route('/capture_all_faces', methods=['POST'])
def capture_all_faces():
    for i, face_img in enumerate(detected_faces):
        filename = f"user_{i + 1}.png"
        filepath = os.path.join(FACE_DB_PATH, filename)
        cv2.imwrite(filepath, face_img)
        
        # Checking for Matched Students
        pairings = load_pairings()
        if filename in pairings:
            student_sid = pairings[filename]
            # Create student folder (if it does not exist)
            student_dir = os.path.join(FACE_DB_PATH, student_sid)
            os.makedirs(student_dir, exist_ok=True)
            # Moving Photos to Student Folders
            dst_path = os.path.join(student_dir, filename)
            shutil.move(filepath, dst_path)
    
    detected_faces.clear()  # Clear List
    return jsonify({'success': True})

if __name__ == '__main__':
    app.run(debug=True)