import cv2
import numpy as np
from ultralytics import YOLO
from deepface import DeepFace
from sklearn.metrics.pairwise import cosine_similarity
from mtcnn import MTCNN
import os
from datetime import datetime
import time
import tensorflow as tf
from tensorflow.keras.models import load_model
import json

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Set parameters
yolo_model_path = "recognitionDemo/Yolo/yolov8n-face.pt"
emotion_model_path = "recognitionDemo/FER/model.h5"
capture_dir = "static/face_database"
delay_frames = 5
similarity_threshold = 0.6
high_similarity_threshold = 0.8
capture_interval = 10
frontal_check_interval = 5

# Load models
yolo_model = YOLO(yolo_model_path)
emotion_model = load_model(emotion_model_path)
print("Emotion model input shape:", emotion_model.input_shape)
mtcnn = MTCNN()

# Emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Known face database (initially empty)
known_faces = {}
last_capture_time = {}

# Initialize detected_faces list
detected_faces = []

# Load student data
def load_students():
    try:
        with open('students.json', 'r', encoding='utf-8') as f:
            students = json.load(f)
        sid_to_name = {student['sid']: student['name'] for student in students}
        print("Loaded sid_to_name:", sid_to_name)
        return students, sid_to_name
    except FileNotFoundError:
        print("students.json not found.")
        return [], {}

students, sid_to_name = load_students()
max_recognizable_people = len(students)
print(f"Number of students loaded: {len(students)}")

# Load known face features from directory
def load_known_faces(capture_dir):
    for user_dir in os.listdir(capture_dir):
        user_path = os.path.join(capture_dir, user_dir)
        if os.path.isdir(user_path):
            known_faces[user_dir] = []
            for img_file in os.listdir(user_path):
                img_path = os.path.join(user_path, img_file)
                try:
                    embedding = DeepFace.represent(img_path, model_name='Facenet', enforce_detection=False)[0]["embedding"]
                    known_faces[user_dir].append(embedding)
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")

# Preprocess image for emotion recognition
def preprocess_image(face_image, target_size=(48, 48)):
    face_image = cv2.resize(face_image, target_size)
    face_image = np.stack([face_image] * 3, axis=-1)
    face_image = face_image.astype('float32') / 255.0
    face_image = np.expand_dims(face_image, axis=0)
    return face_image

# Check if the face is frontal
def is_frontal_face(face_img):
    try:
        detections = mtcnn.detect_faces(face_img)
        if detections:
            keypoints = detections[0]['keypoints']
            nose = keypoints['nose']
            img_width = face_img.shape[1]
            if 0.25 * img_width < nose[0] < 0.75 * img_width:
                return True
    except Exception:
        return False
    return False

# Save face image
def auto_capture(face_img, label, capture_dir):
    try:
        embedding = DeepFace.represent(
            face_img,
            model_name='Facenet',
            enforce_detection=False
        )[0]["embedding"]
        existing_label = get_new_label(embedding, known_faces)
        if existing_label != label:
            label = existing_label
    except Exception as e:
        print(f"Feature extraction error during auto_capture: {e}")

    base_filename = f"{label}.jpg"
    filepath = os.path.join(capture_dir, base_filename)
    counter = 1
    while os.path.exists(filepath):
        new_filename = f"{label}_{counter}.jpg"
        filepath = os.path.join(capture_dir, new_filename)
        counter += 1
    cv2.imwrite(filepath, face_img)
    print(f"Saved face to: {filepath}")
    return filepath

def recognize_face(embedding, known_faces, threshold=similarity_threshold):
    best_label = None
    best_similarity = 0
    for label, embeddings in known_faces.items():
        for known_embedding in embeddings:
            similarity = cosine_similarity([embedding], [known_embedding])[0][0]
            if similarity > best_similarity:
                best_similarity = similarity
                best_label = label if similarity > threshold else None
    if best_label and best_similarity < high_similarity_threshold:
        return best_label, best_similarity
    return best_label, best_similarity

# Check if face is a duplicate
def is_duplicate_face(embedding, seen_faces, threshold=0.8):
    current_time = time.time()
    for face_id, data in seen_faces.items():
        similarity = cosine_similarity([embedding], [data['embedding']])[0][0]
        if similarity > threshold and current_time - data['timestamp'] < 5:
            return True
    return False

# Track face
def track_face(embedding, tracked_faces, current_box, threshold=similarity_threshold):
    for face_id, data in tracked_faces.items():
        tracked_embedding = data['embedding']
        similarity = cosine_similarity([embedding], [tracked_embedding])[0][0]
        tracked_box = data.get('box', [0, 0, 0, 0])
        distance = np.sqrt(sum((a - b) ** 2 for a, b in zip(current_box, tracked_box)))
        if similarity > threshold and distance < 50:
            return face_id
    return None

# Predict emotion
def predict_emotion(face_img):
    try:
        face_img = cv2.resize(face_img, (48, 48))
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        face_img = face_img.astype('float32') / 255.0
        face_img = np.expand_dims(face_img, axis=0)
        
        predictions = emotion_model.predict(face_img, verbose=0)
        emotion_index = np.argmax(predictions)
        emotion_label = emotion_labels[emotion_index]
        confidence = np.max(predictions)
        
        if confidence > 0.5:
            return f"{emotion_label}: {confidence:.2f}"
        return "Neutral"
    except Exception as e:
        print(f"Emotion prediction error: {e}")
        return "Unknown"

# Get or reuse label for new face
def get_new_label(embedding, known_faces):
    for label, embeddings in known_faces.items():
        for known_embedding in embeddings:
            similarity = cosine_similarity([embedding], [known_embedding])[0][0]
            if similarity > high_similarity_threshold:
                return label
    new_label = f"user_{len(known_faces) + 1}"
    return new_label

# Main function
def main():
    load_known_faces(capture_dir)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Camera not opened.")
        return

    window_name = "Face and Emotion Recognition"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    tracked_faces = {}
    seen_faces = {}
    face_id_counter = 0
    frame_count = 0
    last_frontal_result = {}
    tracked_people = set()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        results = yolo_model.predict(source=frame, conf=0.25, imgsz=640, verbose=False)
        current_faces = {}

        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy() if result.boxes.xyxy is not None else []
            for box in boxes:
                x1, y1, x2, y2 = map(int, box[:4])
                face_img = frame[y1:y2, x1:x2]
                try:
                    embedding = DeepFace.represent(face_img, model_name='Facenet', enforce_detection=False)[0]["embedding"]
                except Exception as e:
                    print(f"Feature extraction error: {e}")
                    continue

                emotion_label = predict_emotion(face_img)
                recognized_label, similarity = recognize_face(embedding, known_faces)

                if recognized_label and recognized_label not in tracked_people:
                    tracked_people.add(recognized_label)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, recognized_label, (x1, y1-25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(frame, emotion_label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    if frame_count % frontal_check_interval == 0 or recognized_label not in last_frontal_result:
                        is_frontal = is_frontal_face(face_img)
                        last_frontal_result[recognized_label] = is_frontal
                    else:
                        is_frontal = last_frontal_result.get(recognized_label, False)
                    if is_frontal:
                        current_time = time.time()
                        if similarity > high_similarity_threshold and (recognized_label not in last_capture_time or current_time - last_capture_time[recognized_label] > capture_interval):
                            auto_capture(face_img, recognized_label, capture_dir)
                            known_faces[recognized_label].append(embedding)
                            last_capture_time[recognized_label] = current_time
                    continue

                face_id = track_face(embedding, tracked_faces, (x1, y1, x2, y2))
                if face_id is None and not is_duplicate_face(embedding, seen_faces):
                    face_id = f"face_{face_id_counter}"
                    face_id_counter += 1
                    tracked_faces[face_id] = {
                        'embedding': embedding,
                        'frame_count': 1,
                        'image': face_img,
                        'label': "Unknown",
                        'box': (x1, y1, x2, y2)
                    }
                    seen_faces[face_id] = {'embedding': embedding, 'timestamp': time.time()}
                    detected_faces.append(face_img)
                else:
                    tracked_faces[face_id]['frame_count'] += 1
                    tracked_faces[face_id]['embedding'] = embedding
                    tracked_faces[face_id]['image'] = face_img
                    tracked_faces[face_id]['box'] = (x1, y1, x2, y2)

                current_faces[face_id] = (x1, y1, x2, y2)

                if tracked_faces[face_id]['frame_count'] >= delay_frames:
                    if frame_count % frontal_check_interval == 0 or face_id not in last_frontal_result:
                        is_frontal = is_frontal_face(tracked_faces[face_id]['image'])
                        last_frontal_result[face_id] = is_frontal
                    else:
                        is_frontal = last_frontal_result.get(face_id, False)

                    if is_frontal:
                        new_label = get_new_label(embedding, known_faces)
                        auto_capture(tracked_faces[face_id]['image'], new_label, capture_dir)
                        if new_label not in known_faces:
                            known_faces[new_label] = [embedding]
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, new_label, (x1, y1-25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        cv2.putText(frame, emotion_label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                        del tracked_faces[face_id]
                    else:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        cv2.putText(frame, "Detected", (x1, y1-25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                        cv2.putText(frame, emotion_label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        tracked_faces = {k: v for k, v in tracked_faces.items() if k in current_faces}
        seen_faces = {k: v for k, v in seen_faces.items() if time.time() - v['timestamp'] < 10}

        tracked_people.clear()

        cv2.imshow(window_name, frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    detected_faces.clear()
    seen_faces.clear()

def capture_unique_unknown_faces(frame):
    """Capture unique unknown faces from a single frame and return them."""
    results = yolo_model.predict(source=frame, conf=0.25, imgsz=640, verbose=False)
    unique_faces = {}
    face_embeddings = []

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy() if result.boxes.xyxy is not None else []
        for box in boxes:
            x1, y1, x2, y2 = map(int, box[:4])
            face_img = frame[y1:y2, x1:x2]
            box_key = f"{x1}_{y1}_{x2}_{y2}"

            try:
                embedding = DeepFace.represent(
                    face_img,
                    model_name='Facenet',
                    enforce_detection=False
                )[0]["embedding"]
            except Exception as e:
                print(f"Feature extraction error: {e}")
                continue

            is_duplicate = False
            for existing_embedding in face_embeddings:
                similarity = cosine_similarity([embedding], [existing_embedding])[0][0]
                if similarity > high_similarity_threshold:
                    is_duplicate = True
                    break

            if is_duplicate:
                continue

            recognized_label, similarity = recognize_face(embedding, known_faces)
            if recognized_label is None:
                if box_key not in unique_faces:
                    unique_faces[box_key] = face_img
                    face_embeddings.append(embedding)

    captured_faces = []
    for i, (box_key, face_img) in enumerate(unique_faces.items()):
        embedding = DeepFace.represent(
            face_img,
            model_name='Facenet',
            enforce_detection=False
        )[0]["embedding"]
        new_label = get_new_label(embedding, known_faces)
        filepath = auto_capture(face_img, new_label, capture_dir)
        if new_label not in known_faces:
            known_faces[new_label] = []
        known_faces[new_label].append(embedding)
        captured_faces.append({'label': new_label, 'filepath': filepath})

    return captured_faces

def capture_face_from_current_frame():
    """Capture all unique unknown faces from the current frame."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot access camera.")
        return []
    
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("Error: Failed to read frame.")
        return []
    
    return capture_unique_unknown_faces(frame)

def generate_processed_frames(selected_student=None, manual_capture_trigger=False, mode='face_emotion'):
    load_known_faces(capture_dir)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot access camera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame.")
            break

        # Initialize emotion counter for the current frame
        emotion_counter = {label: 0 for label in emotion_labels}
        total_students = 0  # Counter for unique students (after duplicate filtering)
        total_faces_in_frame = 0  # Counter for all faces detected in the frame

        results = yolo_model.predict(source=frame, conf=0.25, imgsz=640, verbose=False)
        frame_embeddings = []
        recognized_sids = set()

        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy() if result.boxes.xyxy is not None else []
            total_faces_in_frame = len(boxes)  # Count all faces detected by YOLO
            for box in boxes:
                x1, y1, x2, y2 = map(int, box[:4])
                face_img = frame[y1:y2, x1:x2]
                try:
                    embedding = DeepFace.represent(face_img, model_name='Facenet', enforce_detection=False)[0]["embedding"]
                except Exception as e:
                    print(f"Feature extraction error: {e}")
                    continue

                # Check for duplicates in current frame
                is_duplicate = any(cosine_similarity([embedding], [e])[0][0] > high_similarity_threshold 
                                for e in frame_embeddings)
                
                if not is_duplicate:
                    frame_embeddings.append(embedding)
                    total_students += 1  # Increment total students for each unique face

                    # Determine what to display based on the mode
                    display_label = "Unknown"
                    color = (255, 255, 0)  # Yellow for unknown by default
                    emotion_label = ""

                    try:
                        if mode in ['face_emotion', 'face']:  # Face recognition
                            label, similarity = recognize_face(embedding, known_faces)
                            if label and similarity >= similarity_threshold:
                                recognized_sids.add(label)
                                student_name = sid_to_name.get(label, "")
                                display_label = f"{student_name} ({label})"
                                color = (0, 255, 0)  # Green for recognized

                        if mode in ['face_emotion', 'emotion']:  # Emotion recognition
                            emotion_label = predict_emotion(face_img)
                            # Extract the emotion name (without confidence) for counting
                            emotion_name = emotion_label.split(':')[0].strip()
                            if emotion_name in emotion_counter:
                                emotion_counter[emotion_name] += 1
                            else:
                                # Handle "Neutral" or "Unknown" cases
                                if emotion_label == "Neutral":
                                    emotion_counter["Neutral"] += 1
                                elif emotion_label == "Unknown":
                                    # Optionally handle "Unknown" emotions
                                    pass

                        # Draw the bounding box and labels
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, display_label, (x1, y1-25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        if emotion_label:
                            cv2.putText(frame, emotion_label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                        if manual_capture_trigger and selected_student == label and is_frontal_face(face_img):
                            auto_capture(face_img, label, capture_dir)
                            manual_capture_trigger = False
                    except Exception as e:
                        print(f"Feature extraction error: {e}")
                        continue

        # Display the total students, total faces in frame, and emotion counter in the top-right corner
        frame_height, frame_width = frame.shape[:2]
        text_x = frame_width - 150  # Position 150 pixels from the right edge
        text_y = 30  # Start 30 pixels from the top
        line_spacing = 20  # Space between lines

        # Calculate the dimensions of the counter area
        counter_height = (len(emotion_labels) + 2) * line_spacing + 10  # +2 for "Total Students" and "Total Faces in Frame"
        counter_width = 140  # Width of the counter area
        counter_x = frame_width - 160  # Slightly more padding on the left
        counter_y = 20  # Slightly more padding on the top

        # Draw a semi-transparent white rectangle as the background
        overlay = frame.copy()
        cv2.rectangle(overlay, (counter_x, counter_y), 
                      (counter_x + counter_width, counter_y + counter_height), 
                      (255, 255, 255), -1)  # White rectangle
        alpha = 0.7  # Transparency factor
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # Display "Total Students" and "Total Faces in Frame"
        total_faces_text = f"Total Faces: {total_faces_in_frame}"
        text_y += line_spacing
        cv2.putText(frame, total_faces_text, (text_x, text_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        text_y += line_spacing + 10  # Add extra spacing after counts

        # Display all emotions
        for i, (emotion, count) in enumerate(emotion_counter.items()):
            text = f"{emotion}: {count}"
            cv2.putText(frame, text, (text_x, text_y + i * line_spacing), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

def initialize_directories():
    for student in students:
        user_dir = os.path.join(capture_dir, student['sid'])
        os.makedirs(user_dir, exist_ok=True)

initialize_directories()

if __name__ == "__main__":
    main()