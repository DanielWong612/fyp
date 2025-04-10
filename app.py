from flask import Flask, render_template, request, Response, jsonify
import os
import json
import shutil
from recognitionDemo.classroom_Monitor_System import (
    generate_processed_frames, 
    detected_faces, 
    known_faces, 
    auto_capture, 
    capture_face_from_current_frame,
    yolo_model,
    sid_to_name
)
import cv2
import numpy as np
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
from deepface import DeepFace

app = Flask(__name__)

# Enable the 'do' extension for Jinja2
app.jinja_env.add_extension('jinja2.ext.do')

STUDENTS_FILE = 'students.json'
PAIRINGS_FILE = 'face_pairings.json'
FACE_DB_PATH = 'static/face_database'
DEFAULT_AVATAR = 'default_avatar.png'
ATTENDANCE_FILE = 'attendance.json'
similarity_threshold = 0.6

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
    pairings = load_pairings()
    user_faces = []
    for filename in sorted(os.listdir(FACE_DB_PATH)):
        if filename.startswith('user') and filename.lower().endswith('.jpg'):
            src_path = os.path.join(FACE_DB_PATH, filename)
            if os.path.isfile(src_path): 
                if filename not in pairings:
                    user_faces.append({'label': filename, 'filepath': src_path})
    return user_faces

def map_student_faces(students, pairings):
    for student in students:
        student_dir = os.path.join(FACE_DB_PATH, student['sid'])
        if os.path.exists(student_dir):
            for filename in os.listdir(student_dir):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    student['avatar'] = f"face_database/{student['sid']}/{filename}"
                    break
            else:
                student['avatar'] = DEFAULT_AVATAR
        else:
            student['avatar'] = DEFAULT_AVATAR
    return students

def load_attendance_history():
    """Load existing attendance records or initialize empty list"""
    if not os.path.exists(ATTENDANCE_FILE):
        return []
    
    try:
        with open(ATTENDANCE_FILE, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if not content:
                return []
            return json.loads(content)
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Error loading attendance history: {e}")
        return []

def save_attendance_history(attendance_history):
    """Save attendance history to file"""
    with open(ATTENDANCE_FILE, 'w', encoding='utf-8') as f:
        json.dump(attendance_history, f, ensure_ascii=False, indent=4)

@app.route('/')
def index():
    students = load_students()
    pairings = load_pairings()
    students = map_student_faces(students, pairings)
    unpaired_faces = get_first_face_images()
    max_recognizable_people = len(students)
    current_recognized_people = len([student for student in students if student.get('avatar') != 'default_avatar.png'])
    return render_template('index.html', students=students, unpaired_faces=unpaired_faces, 
                         max_recognizable_people=max_recognizable_people, 
                         current_recognized_people=current_recognized_people)

@app.route('/pair', methods=['POST'])
def pair():
    image = request.form['image']  
    student_sid = request.form['student_sid']  
    src_path = os.path.join(FACE_DB_PATH, image)
    dst_dir = os.path.join(FACE_DB_PATH, student_sid)
    os.makedirs(dst_dir, exist_ok=True)
    dst_path = os.path.join(dst_dir, image)
    shutil.move(src_path, dst_path)
    save_pairing(image, student_sid)
    return jsonify({'success': True, 'image': image, 'student_sid': student_sid})

@app.route('/manual_capture', methods=['POST'])
def manual_capture_route():
    selected_student = request.form['student_sid']
    return Response(generate_processed_frames(selected_student=selected_student, manual_capture_trigger=True), 
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/processed_video_feed')
def processed_video_feed():
    mode = request.args.get('mode', 'face_emotion')
    return Response(generate_processed_frames(mode=mode), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture_face', methods=['POST'])
def capture_face():
    try:
        captured_faces = capture_face_from_current_frame()
        return jsonify({'success': True, 'captured_faces': len(captured_faces)})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/capture_all_faces', methods=['POST'])
def capture_all_faces():
    for i, face_img in enumerate(detected_faces):
        filename = f"user_{i + 1}.jpg"
        filepath = os.path.join(FACE_DB_PATH, filename)
        cv2.imwrite(filepath, face_img)
        
        pairings = load_pairings()
        if filename in pairings:
            student_sid = pairings[filename]
            student_dir = os.path.join(FACE_DB_PATH, student_sid)
            os.makedirs(student_dir, exist_ok=True)
            dst_path = os.path.join(student_dir, filename)
            shutil.move(filepath, dst_path)
    
    detected_faces.clear()
    return jsonify({'success': True})

@app.route('/start_attendance', methods=['POST'])
def start_attendance():
    interval = request.form.get('interval', type=int)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return jsonify({'success': False, 'error': 'Cannot access camera'}), 500
    
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        return jsonify({'success': False, 'error': 'Failed to capture frame'}), 500
    
    recognized_students = detect_students_in_frame(frame)
    
    # Map student IDs to names
    recognized_students_with_names = [
        {'sid': sid, 'name': sid_to_name.get(sid, 'Unknown')}
        for sid in recognized_students
    ]
    
    new_record = {
        'timestamp': datetime.now().isoformat(),
        'recognized_students': recognized_students_with_names,
        'total_students': len(load_students()),
        'present_count': len(recognized_students)
    }
    
    attendance_history = load_attendance_history()
    attendance_history.append(new_record)
    save_attendance_history(attendance_history)
    
    return jsonify({
        'success': True, 
        'interval': interval,
        'recognized_count': len(recognized_students),
        'recognized_students': recognized_students
    })

@app.route('/attendance')
def attendance_page():
    attendance_history = load_attendance_history()
    return render_template('attendance.html', attendance_history=attendance_history)

@app.route('/get_attendance_history', methods=['GET'])
def get_attendance_history():
    attendance_history = load_attendance_history()
    return jsonify({'success': True, 'history': attendance_history})

@app.route('/upload_photo', methods=['POST'])
def upload_photo():
    if 'photo' not in request.files:
        return jsonify({'success': False, 'error': 'No photo uploaded'}), 400
    
    student_sid = request.form.get('student_sid')
    if not student_sid:
        return jsonify({'success': False, 'error': 'No student SID provided'}), 400

    is_replace = request.form.get('is_replace') == 'true'

    photo = request.files['photo']
    if photo.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'}), 400

    if not photo.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        return jsonify({'success': False, 'error': 'Invalid file format. Only JPG, JPEG, and PNG are allowed'}), 400

    # Save the uploaded photo temporarily to check for faces
    temp_filename = f"temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    temp_filepath = os.path.join(FACE_DB_PATH, temp_filename)
    photo.save(temp_filepath)

    # Log the file path and check if the file exists
    print(f"Temporary file saved at: {temp_filepath}")
    if not os.path.exists(temp_filepath):
        return jsonify({'success': False, 'error': 'Failed to save the uploaded file'}), 500

    # Check if the image contains a detectable face using different backends
    backends = ['opencv', 'ssd', 'dlib', 'mtcnn', 'retinaface']
    face_detected = False
    for backend in backends:
        try:
            print(f"Trying face detection with backend: {backend}")
            DeepFace.represent(temp_filepath, model_name='Facenet', detector_backend=backend, enforce_detection=True)
            face_detected = True
            print(f"Face detected with backend: {backend}")
            break
        except Exception as e:
            print(f"Face detection failed with backend {backend}: {str(e)}")
            continue

    # If DeepFace fails, try OpenCV Haar Cascade as a fallback
    if not face_detected:
        print("Falling back to OpenCV Haar Cascade for face detection")
        try:
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            img = cv2.imread(temp_filepath)
            if img is None:
                raise Exception("Failed to read the image")
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            if len(faces) > 0:
                face_detected = True
                print(f"Face detected with OpenCV Haar Cascade: {len(faces)} faces found")
            else:
                print("No faces detected with OpenCV Haar Cascade")
        except Exception as e:
            print(f"OpenCV face detection failed: {str(e)}")

    if not face_detected:
        os.remove(temp_filepath)
        return jsonify({'success': False, 'error': 'No face detected in the uploaded photo. Tried all methods.'}), 400

    # If replacing, find and move the old photo
    if is_replace:
        student_dir = os.path.join(FACE_DB_PATH, student_sid)
        old_photo = None
        if os.path.exists(student_dir):
            for filename in os.listdir(student_dir):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    old_photo = filename
                    break
        
        if old_photo:
            old_photo_path = os.path.join(student_dir, old_photo)
            new_photo_path = os.path.join(FACE_DB_PATH, old_photo)
            # Move the old photo to static/face_database
            shutil.move(old_photo_path, new_photo_path)
            print(f"Moved old photo to: {new_photo_path}")
            # Update face_pairings.json to remove the old pairing
            pairings = load_pairings()
            if old_photo in pairings:
                del pairings[old_photo]
                with open(PAIRINGS_FILE, 'w', encoding='utf-8') as f:
                    json.dump(pairings, f, indent=2)

    # Save the new photo to the student's directory
    student_dir = os.path.join(FACE_DB_PATH, student_sid)
    os.makedirs(student_dir, exist_ok=True)
    
    # Generate a unique filename for the new photo
    filename = f"{student_sid}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    filepath = os.path.join(student_dir, filename)
    
    # Move the temporary file to the student's directory
    shutil.move(temp_filepath, filepath)

    # Update the face pairings
    save_pairing(filename, student_sid)

    return jsonify({'success': True})

@app.route('/unmatch_student', methods=['POST'])
def unmatch_student():
    student_sid = request.form.get('student_sid')
    if not student_sid:
        return jsonify({'success': False, 'error': 'No student SID provided'}), 400

    # Find the student's photo in their directory
    student_dir = os.path.join(FACE_DB_PATH, student_sid)
    photo_to_unmatch = None
    if os.path.exists(student_dir):
        for filename in os.listdir(student_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                photo_to_unmatch = filename
                break

    if not photo_to_unmatch:
        return jsonify({'success': False, 'error': 'No photo found for this student'}), 400

    # Move the photo back to static/face_database
    old_photo_path = os.path.join(student_dir, photo_to_unmatch)
    new_photo_path = os.path.join(FACE_DB_PATH, photo_to_unmatch)
    shutil.move(old_photo_path, new_photo_path)
    print(f"Moved photo to: {new_photo_path}")

    # Update face_pairings.json to remove the pairing
    pairings = load_pairings()
    if photo_to_unmatch in pairings:
        del pairings[photo_to_unmatch]
        with open(PAIRINGS_FILE, 'w', encoding='utf-8') as f:
            json.dump(pairings, f, indent=2)

    return jsonify({'success': True})

def detect_students_in_frame(frame):
    """Detect and recognize students in a single frame"""
    results = yolo_model.predict(source=frame, conf=0.25, imgsz=640, verbose=False)
    recognized_sids = set()
    
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy() if result.boxes.xyxy is not None else []
        for box in boxes:
            x1, y1, x2, y2 = map(int, box[:4])
            face_img = frame[y1:y2, x1:x2]
            
            try:
                embedding = DeepFace.represent(
                    face_img,
                    model_name='Facenet',
                    enforce_detection=False
                )[0]["embedding"]
                
                label, similarity = recognize_face(embedding, known_faces)
                if label and similarity >= similarity_threshold and label in sid_to_name:
                    recognized_sids.add(label)
                    print(f"Recognized student: {label} (similarity: {similarity})")
                else:
                    print(f"Face detected but not recognized (similarity: {similarity})")
                    
            except Exception as e:
                print(f"Error processing face: {e}")
                continue
    
    print(f"Total recognized students: {len(recognized_sids)} - {recognized_sids}")
    return list(recognized_sids)

def recognize_face(embedding, known_faces, threshold=similarity_threshold):
    best_label = None
    best_similarity = 0
    for label, embeddings in known_faces.items():
        for known_embedding in embeddings:
            similarity = cosine_similarity([embedding], [known_embedding])[0][0]
            if similarity > best_similarity:
                best_similarity = similarity
                best_label = label if similarity > threshold else None
    return best_label, best_similarity

if __name__ == '__main__':
    app.run(debug=True)