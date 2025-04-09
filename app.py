from flask import Flask, render_template, request, redirect, url_for, Response, jsonify
import os, json, shutil, cv2

app = Flask(__name__)

# File paths and constants
STUDENTS_FILE = 'students.json'        # JSON file storing student data
PAIRINGS_FILE = 'face_pairings.json'   # JSON file storing face-student pairings
FACE_DB_PATH = 'recognitionDemo/face_database'  # Directory for face images
THUMB_DIR = 'static/thumbnails'        # Directory for thumbnail images
DEFAULT_AVATAR = 'default_avatar.png'  # Default avatar if no face is paired

# Load student data from JSON file
def load_students():
    with open(STUDENTS_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)

# Load face-student pairings from JSON file
def load_pairings():
    if not os.path.exists(PAIRINGS_FILE):
        return {}
    with open(PAIRINGS_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)

# Save a new face-student pairing to the JSON file
def save_pairing(image, sid):
    pairings = load_pairings()
    pairings[image] = sid
    with open(PAIRINGS_FILE, 'w', encoding='utf-8') as f:
        json.dump(pairings, f, indent=2)

# Get list of unpaired face images from the face database
def get_first_face_images():
    if not os.path.exists(THUMB_DIR):
        os.makedirs(THUMB_DIR)
    pairings = load_pairings()
    user_faces = []
    for user_folder in sorted(os.listdir(FACE_DB_PATH)):
        user_path = os.path.join(FACE_DB_PATH, user_folder)
        if os.path.isdir(user_path):
            images = [f for f in os.listdir(user_path) if f.lower().endswith(('.jpg', '.png'))]
            if images:
                first_img = images[0]
                src_path = os.path.join(user_path, first_img)
                thumb_name = f"{user_folder}_{first_img}"
                dst_path = os.path.join(THUMB_DIR, thumb_name)
                if not os.path.exists(dst_path):
                    shutil.copy(src_path, dst_path)
                if thumb_name not in pairings:
                    user_faces.append(thumb_name)
    return user_faces

# Map face images to students for avatar display
def map_student_faces(students, pairings):
    sid_to_image = {v: k for k, v in pairings.items()}  # Reverse mapping: {sid: image}
    for student in students:
        image_name = sid_to_image.get(student['sid'])
        student['avatar'] = f"thumbnails/{image_name}" if image_name else DEFAULT_AVATAR
    return students

# Generate video frames for the live camera feed
def generate_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Route for the homepage
@app.route('/')
def index():
    students = load_students()
    pairings = load_pairings()
    students = map_student_faces(students, pairings)
    unpaired_faces = get_first_face_images()
    return render_template('index.html', students=students, unpaired_faces=unpaired_faces)

# Route to handle pairing requests (returns JSON instead of redirecting)
@app.route('/pair', methods=['POST'])
def pair():
    image = request.form['image']           # Face image filename
    student_sid = request.form['student_sid']  # Student ID
    save_pairing(image, student_sid)        # Save the pairing
    # Return a JSON response to confirm success
    return jsonify({'success': True, 'image': image, 'student_sid': student_sid})

# Route for the live video feed
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)