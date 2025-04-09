import cv2
import numpy as np
from ultralytics import YOLO
from deepface import DeepFace
from sklearn.metrics.pairwise import cosine_similarity
from mtcnn import MTCNN
import os
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.models import load_model

# Set parameters
yolo_model_path = "recognitionDemo/Yolo/yolov8n-face.pt"
emotion_model_path = "recognitionDemo/FER/model.h5"
capture_dir = "recognitionDemo/face_database"
similarity_threshold = 0.6
frontal_check_interval = 5

# Load models
yolo_model = YOLO(yolo_model_path)
try:
    emotion_model = load_model(emotion_model_path, compile=False)
    emotion_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print("Emotion model loaded successfully with input shape:", emotion_model.input_shape)
except Exception as e:
    print(f"Failed to load emotion model: {e}")
    emotion_model = None  # Fallback to avoid crashes

mtcnn = MTCNN()

# Emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Known face database
known_faces = {}

# Load student list
def load_students():
    import json
    with open('students.json', 'r', encoding='utf-8') as f:
        return json.load(f)

students = load_students()
student_ids = [student['sid'] for student in students]

# Load known face features
def load_known_faces(capture_dir, student_ids):
    for sID in student_ids:
        user_path = os.path.join(capture_dir, sID)
        if os.path.isdir(user_path):
            known_faces[sID] = []
            for img_file in os.listdir(user_path):
                img_path = os.path.join(user_path, img_file)
                try:
                    embedding = DeepFace.represent(img_path, model_name='Facenet', enforce_detection=False)[0]["embedding"]
                    known_faces[sID].append(embedding)
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
            return 0.25 * img_width < nose[0] < 0.75 * img_width
    except Exception:
        return False
    return False

# Manual capture function
def manual_capture(face_img, sID, capture_dir):
    user_dir = os.path.join(capture_dir, sID)
    os.makedirs(user_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{sID}_{timestamp}.jpg"
    filepath = os.path.join(user_dir, filename)
    cv2.imwrite(filepath, face_img)
    print(f"Saved face to: {filepath}")
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

# Predict emotion
def predict_emotion(face_img):
    if emotion_model is None:
        return "Unknown (Model not loaded)"
    try:
        if face_img is None or not isinstance(face_img, np.ndarray) or face_img.size == 0:
            return "Unknown"
        gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        processed_face = preprocess_image(gray_face)
        predictions = emotion_model.predict(processed_face, verbose=0)
        emotion_index = np.argmax(predictions)
        emotion_label = emotion_labels[emotion_index]
        confidence = np.max(predictions)
        return f"{emotion_label}: {confidence:.2f}"
    except Exception as e:
        print(f"Emotion prediction error: {e}")
        return "Unknown"

# Generator function for Flask
def generate_processed_frames(selected_student=None, manual_capture_trigger=False):
    load_known_faces(capture_dir, student_ids)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Camera not opened.")
        return

    frame_count = 0
    detected_faces = {}

    while True:
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

                if recognized_label:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, recognized_label, (x1, y1-25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(frame, emotion_label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                    if manual_capture_trigger and selected_student == recognized_label and is_frontal_face(face_img):
                        manual_capture(face_img, recognized_label, capture_dir)
                        known_faces[recognized_label].append(embedding)
                        manual_capture_trigger = False

                else:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
                    cv2.putText(frame, "Detected", (x1, y1-25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    cv2.putText(frame, emotion_label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                current_faces[f"face_{x1}_{y1}"] = (x1, y1, x2, y2)

        detected_faces = {k: v for k, v in detected_faces.items() if k in current_faces}

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()