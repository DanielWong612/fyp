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

# Set parameters
yolo_model_path = "recognitionDemo/Yolo/yolov8n-face.pt"  # Path to YOLO face detection model
emotion_model_path = "recognitionDemo/FER/model.h5"  # Path to emotion recognition model
capture_dir = "recognitionDemo/face_database"  # Root directory for storing faces
delay_frames = 5  # Number of frames to delay capture (new users)
similarity_threshold = 0.6  # Recognition similarity threshold
high_similarity_threshold = 0.8  # High similarity threshold for auto-capturing new features
capture_interval = 10  # Time interval for auto-capturing new features (seconds)
frontal_check_interval = 5  # Check for frontal face every few frames

# Load models
yolo_model = YOLO(yolo_model_path)
emotion_model = load_model(emotion_model_path)
print("Emotion model input shape:", emotion_model.input_shape)  # Verify model input shape
mtcnn = MTCNN()

# Emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Known face database (initially empty)
known_faces = {}  # {label: [embedding1, embedding2, ...]}
last_capture_time = {}  # {label: timestamp}

# Load student data
def load_students():
    try:
        with open('students.json', 'r', encoding='utf-8') as f:
            students = json.load(f)
        sid_to_name = {student['sid']: student['name'] for student in students}
        print("Loaded students:", students)
        return students, sid_to_name
    except FileNotFoundError:
        print("students.json not found.")
        return [], {}

students, sid_to_name = load_students()

# Load known face features from directory
def load_known_faces(capture_dir):
    for user_dir in os.listdir(capture_dir):
        user_path = os.path.join(capture_dir, user_dir)
        if os.path.isdir(user_path):
            known_faces[user_dir] = []
            for img_file in os.listdir(user_path):
                img_path = os.path.join(user_path, img_file)
                try:
                    embedding = DeepFace.represent(
                        img_path,
                        model_name='Facenet',
                        enforce_detection=False
                    )[0]["embedding"]
                    known_faces[user_dir].append(embedding)
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")

# Preprocess image for emotion recognition
def preprocess_image(face_image, target_size=(48, 48)):
    """Preprocess the image to fit the emotion recognition model"""
    face_image = cv2.resize(face_image, target_size)  # Resize to 48x48
    face_image = np.stack([face_image] * 3, axis=-1)  # Convert grayscale to RGB
    face_image = face_image.astype('float32') / 255.0  # Normalize to [0, 1]
    face_image = np.expand_dims(face_image, axis=0)  # Add batch dimension
    return face_image

# Check if the face is frontal
def is_frontal_face(face_img):
    try:
        detections = mtcnn.detect_faces(face_img)
        if detections:
            keypoints = detections[0]['keypoints']
            nose = keypoints['nose']
            img_width = face_img.shape[1]
            # Check if the nose is between 25% and 75% of the image width
            if 0.25 * img_width < nose[0] < 0.75 * img_width:
                return True
    except Exception:
        return False
    return False

# Save face image
def auto_capture(face_img, label, capture_dir):
    user_dir = os.path.join(capture_dir, label)
    os.makedirs(user_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{label}_{timestamp}.jpg"
    filepath = os.path.join(user_dir, filename)
    cv2.imwrite(filepath, face_img)
    print(f"Saved frontal face to: {filepath}")
    return filepath

# Recognize face
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

# Track face
def track_face(embedding, tracked_faces, threshold=similarity_threshold):
    for face_id, data in tracked_faces.items():
        tracked_embedding = data['embedding']
        similarity = cosine_similarity([embedding], [tracked_embedding])[0][0]
        if similarity > threshold:
            return face_id
    return None

# Predict emotion
def predict_emotion(face_img):
    try:
        gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        processed_face = preprocess_image(gray_face)
        predictions = emotion_model.predict(processed_face, verbose=0)  # Disable progress bar
        emotion_index = np.argmax(predictions)
        emotion_label = emotion_labels[emotion_index]
        confidence = np.max(predictions)
        return f"{emotion_label}: {confidence:.2f}"
    except Exception as e:
        print(f"Emotion prediction error: {e}")
        return "Unknown"

# Main function
def main():
    # Initially load known faces
    load_known_faces(capture_dir)

    # Start the camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Camera not opened.")
        return

    # Create display window
    window_name = "Face and Emotion Recognition"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # Tracking data buffer
    tracked_faces = {}  # {face_id: {'embedding': embedding, 'frame_count': int, 'image': face_img, 'label': str}}
    face_id_counter = 0  # Used to generate unique face_id
    frame_count = 0  # Control the frequency of frontal checks
    last_frontal_result = {}  # Store the most recent frontal check result for each face_id

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Use YOLO to detect faces
        results = yolo_model.predict(source=frame, conf=0.25, imgsz=640, verbose=False)

        current_faces = {}  # Faces detected in the current frame

        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy() if result.boxes.xyxy is not None else []
            
            for box in boxes:
                x1, y1, x2, y2 = map(int, box[:4])
                face_img = frame[y1:y2, x1:x2]

                # Extract features using DeepFace
                try:
                    embedding = DeepFace.represent(
                        face_img,
                        model_name='Facenet',
                        enforce_detection=False
                    )[0]["embedding"]
                except Exception as e:
                    print(f"Feature extraction error: {e}")
                    continue

                # Predict emotion
                emotion_label = predict_emotion(face_img)

                # Attempt to recognize the face
                recognized_label, similarity = recognize_face(embedding, known_faces)

                if recognized_label:
                    # If recognition succeeds, display label and emotion
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, recognized_label, (x1, y1-25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(frame, emotion_label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                    # Check if it's frontal and auto-capture new features
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

                # Attempt to track existing faces
                face_id = track_face(embedding, tracked_faces)

                if face_id is None:
                    # New face, assign a new ID
                    face_id = f"face_{face_id_counter}"
                    face_id_counter += 1
                    tracked_faces[face_id] = {
                        'embedding': embedding,
                        'frame_count': 1,
                        'image': face_img,
                        'label': "Unknown"
                    }
                else:
                    # Update tracked face data
                    tracked_faces[face_id]['frame_count'] += 1
                    tracked_faces[face_id]['embedding'] = embedding
                    tracked_faces[face_id]['image'] = face_img

                current_faces[face_id] = (x1, y1, x2, y2)

                # Delay capture logic (new users)
                if tracked_faces[face_id]['frame_count'] >= delay_frames:
                    # Check if it's a frontal face
                    if frame_count % frontal_check_interval == 0 or face_id not in last_frontal_result:
                        is_frontal = is_frontal_face(tracked_faces[face_id]['image'])
                        last_frontal_result[face_id] = is_frontal
                    else:
                        is_frontal = last_frontal_result.get(face_id, False)

                    if is_frontal:
                        new_label = f"user_{len(known_faces) + 1}"
                        auto_capture(tracked_faces[face_id]['image'], new_label, capture_dir)
                        known_faces[new_label] = [tracked_faces[face_id]['embedding']]
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, new_label, (x1, y1-25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        cv2.putText(frame, emotion_label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                        del tracked_faces[face_id]
                    else:
                        # Non-frontal face, display detection result
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        cv2.putText(frame, "Detected", (x1, y1-25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                        cv2.putText(frame, emotion_label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                else:
                    # Not yet reached delay frame count, display "Detecting"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
                    cv2.putText(frame, "Detecting", (x1, y1-25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    cv2.putText(frame, emotion_label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Clean up tracking data for faces not in the current frame
        tracked_faces = {k: v for k, v in tracked_faces.items() if k in current_faces}

        # Display the result
        cv2.imshow(window_name, frame)
        if cv2.waitKey(1) & 0xFF == 27:  # Press ESC to exit
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

def generate_processed_frames(selected_student=None, manual_capture_trigger=False):
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

        results = yolo_model.predict(source=frame, conf=0.25, imgsz=640, verbose=False)
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

                if recognized_label:
                    student_name = sid_to_name.get(recognized_label, "")
                    display_label = f"{student_name} ({recognized_label}) - {emotion_label}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, display_label, (x1, y1-25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    #print(f"Recognized: {display_label}")
                    if manual_capture_trigger and selected_student == recognized_label and is_frontal_face(face_img):
                        print(f"Capturing face for {recognized_label}")
                        manual_capture(face_img, recognized_label, capture_dir)
                        manual_capture_trigger = False
                else:
                    display_label = f"{emotion_label}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
                    cv2.putText(frame, display_label, (x1, y1-25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

if __name__ == "__main__":
    main()
