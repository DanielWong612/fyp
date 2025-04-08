import cv2
import os
import numpy as np
from ultralytics import YOLO
from tensorflow.keras.models import load_model
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
import time
import mediapipe as mp

# === Load Models ===
# Load YOLO face detection model
model = YOLO("Test/Yolo/yolov10m-face.pt")

# Load emotion recognition model
emotion_model = load_model("/Users/sam/Documents/GitHub/fyp/Test/FER/model.h5")
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# === Initialize MediaPipe Face Detection ===
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

# === Storage Settings ===
base_dir = "detected_faces"
os.makedirs(base_dir, exist_ok=True)
reference_faces = {}  # Dictionary: {user_label: list of embedding vectors}
MAX_SAMPLES_PER_ID = 5  # Maximum number of embedding samples per user

# === YOLO Parameters ===
CONF_THRESHOLD = 0.25  # Confidence threshold for detection
IMG_SIZE = 640  # Image size for YOLO processing
LINE_THICKNESS = 1  # Thickness of bounding box lines
MAX_DET = 10  # Maximum number of detections per frame

# === Buffer Mechanism Parameters ===
DETECTION_INTERVAL = 3  # Run detection every 3 frames
frame_counter = 0
last_detections = []  # Store detections from the last detection frame

# === Tracking Parameters ===
face_tracking = {}  # Dictionary: {temp_id: (center_x, center_y)}
last_face_data = {}  # Dictionary: {temp_id: (embedding, face_img, gray_face, x1, y1, x2, y2)}
captured_faces = {}  # Dictionary: {temp_id: (user_label, emotion)}
next_temp_id = 0  # Temporary ID for tracking new faces
next_user_id = 1  # Incremental ID for new users (starts at user1)

# === Stability Detection Parameters ===
STABILITY_THRESHOLD = 0.01  # Threshold for nose position stability
STABLE_FRAMES = 30  # Number of frames to consider a face stable
face_stability_positions = {}  # Dictionary: {temp_id: [(x, y), ...]}
face_stability_counter = {}  # Dictionary: {temp_id: int}

# === FPS Control ===
TARGET_FPS = 30
FRAME_TIME = 1.0 / TARGET_FPS

# === Helper Functions ===

def preprocess_emotion(face_img, size=(48, 48)):
    """Preprocess face image for emotion recognition."""
    face_img = cv2.resize(face_img, size)
    face_img = np.stack([face_img] * 3, axis=-1).astype('float32') / 255.0
    return np.expand_dims(face_img, axis=0)

def extract_embedding(face_img):
    """Extract embedding vector from a face image."""
    face_resized = cv2.resize(face_img, (100, 100))
    face_normalized = face_resized.astype("float32") / 255.0
    embedding = face_normalized.flatten().reshape(1, -1)
    return embedding

def compare_faces_cosine(new_embedding, stored_embeddings):
    """Compare embeddings using cosine similarity."""
    similarities = [cosine_similarity(new_embedding, emb)[0][0] for emb in stored_embeddings]
    avg_similarity = np.mean(similarities)
    return avg_similarity > 0.95  # Similarity threshold

def find_match(embedding):
    """Find a matching user label in reference_faces."""
    for user_label, stored_embeddings in reference_faces.items():
        if compare_faces_cosine(embedding, stored_embeddings):
            return user_label
    return None

def save_face(face_img, user_label):
    """Save face image to the local directory."""
    folder_path = os.path.join(base_dir, user_label)
    os.makedirs(folder_path, exist_ok=True)
    filename = f"{user_label}_{datetime.now().strftime('%S')}.jpg"
    filepath = os.path.join(folder_path, filename)
    cv2.imwrite(filepath, face_img)
    print(f"Saved {filename} to {folder_path}/")

def predict_emotion(gray_face):
    """Predict emotion from a grayscale face image."""
    try:
        input_face = preprocess_emotion(gray_face)
        preds = emotion_model.predict(input_face, verbose=0)
        emotion = emotion_labels[np.argmax(preds)]
        return emotion
    except Exception:
        return "N/A"

def update_reference_faces(user_label, new_embedding):
    """Update reference embeddings for a user."""
    if user_label not in reference_faces:
        reference_faces[user_label] = [new_embedding]
    else:
        if len(reference_faces[user_label]) < MAX_SAMPLES_PER_ID:
            reference_faces[user_label].append(new_embedding)
        else:
            reference_faces[user_label].pop(0)  # Remove oldest embedding
            reference_faces[user_label].append(new_embedding)

def match_faces(current_boxes, frame):
    """Match detected faces to tracked faces."""
    global next_temp_id
    new_face_tracking = {}
    new_last_face_data = {}
    for box in current_boxes:
        x1, y1, x2, y2 = map(int, box[:4])
        face_img = frame[y1:y2, x1:x2]
        gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        embedding = extract_embedding(gray_face)
        center = ((x1 + x2) / 2, (y1 + y2) / 2)
        matched_id = None
        min_distance = float('inf')
        for temp_id, (prev_center_x, prev_center_y) in face_tracking.items():
            distance = np.sqrt((center[0] - prev_center_x)**2 + (center[1] - prev_center_y)**2)
            if distance < min_distance and distance < 100:  # Distance threshold
                min_distance = distance
                matched_id = temp_id
        if matched_id is None:
            matched_id = f"face_{next_temp_id}"
            next_temp_id += 1
        new_face_tracking[matched_id] = center
        new_last_face_data[matched_id] = (embedding, face_img, gray_face, x1, y1, x2, y2)
    face_tracking.clear()
    face_tracking.update(new_face_tracking)
    last_face_data.clear()
    last_face_data.update(new_last_face_data)
    return list(new_face_tracking.keys())

def check_stability(temp_id, frame):
    """Check face stability using MediaPipe (simplified version)."""
    # Process frame with MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(frame_rgb)
    
    if results.detections:
        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box
            h, w = frame.shape[:2]
            x1 = int(bbox.xmin * w)
            y1 = int(bbox.ymin * h)
            x2 = int((bbox.xmin + bbox.width) * w)
            y2 = int((bbox.ymin + bbox.height) * h)
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            # Assume this detection corresponds to temp_id (simplified matching)
            if temp_id not in face_stability_positions:
                face_stability_positions[temp_id] = []
            face_stability_positions[temp_id].append((center_x, center_y))
            
            if len(face_stability_positions[temp_id]) > STABLE_FRAMES:
                face_stability_positions[temp_id].pop(0)
            
            positions = face_stability_positions[temp_id]
            if len(positions) == STABLE_FRAMES:
                variances = np.var(positions, axis=0)
                if max(variances) < STABILITY_THRESHOLD:
                    face_stability_positions[temp_id] = []  # Reset after stability
                    return True
    return False

# === Start Webcam ===
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)

fps_start_time = time.time()
fps_counter = 0
fps_display = 0

# === Main Loop ===
while cap.isOpened():
    frame_start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        break

    frame_counter += 1
    current_detections = []

    # === Detection Phase (every DETECTION_INTERVAL frames) ===
    if frame_counter % DETECTION_INTERVAL == 0:
        results = model.predict(
            source=frame,
            conf=CONF_THRESHOLD,
            imgsz=IMG_SIZE,
            max_det=MAX_DET,
            verbose=False
        )
        boxes = results[0].boxes.xyxy.cpu().numpy() if results[0].boxes.xyxy is not None else []
        temp_ids = match_faces(boxes, frame)
        
        # Process each detected face
        for temp_id in temp_ids:
            if temp_id not in captured_faces:
                embedding = last_face_data[temp_id][0]
                matched_label = find_match(embedding)
                if matched_label is None:
                    # New face: capture immediately
                    user_label = f"user{next_user_id}"
                    next_user_id += 1
                    save_face(last_face_data[temp_id][1], user_label)
                    update_reference_faces(user_label, embedding)
                    captured_faces[temp_id] = (user_label, "N/A")
                else:
                    # Known face: assign label, emotion pending stability
                    captured_faces[temp_id] = (matched_label, "N/A")
        
        # Update current_detections with latest info
        for temp_id in temp_ids:
            x1, y1, x2, y2 = last_face_data[temp_id][3:7]
            if temp_id in captured_faces:
                label, emotion = captured_faces[temp_id]
            else:
                label = "Detecting"
                emotion = "N/A"
            current_detections.append([x1, y1, x2, y2, label, emotion, temp_id])
        last_detections = current_detections
    else:
        current_detections = last_detections

    # === Stability Check and Emotion Analysis ===
    for temp_id in face_tracking:
        if check_stability(temp_id, frame):
            if temp_id in captured_faces:
                user_label, _ = captured_faces[temp_id]
                gray_face = last_face_data[temp_id][2]
                emotion = predict_emotion(gray_face)
                captured_faces[temp_id] = (user_label, emotion)
                # Update current_detections with new emotion
                for det in current_detections:
                    if det[6] == temp_id:
                        det[5] = emotion

    # === Draw Detections on Frame ===
    for detection in current_detections:
        x1, y1, x2, y2, label, emotion, _ = detection
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), LINE_THICKNESS)
        display_text = f"{label} | {emotion}" if emotion != "N/A" else label
        cv2.putText(frame, display_text, (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), LINE_THICKNESS)

    # === Calculate and Display FPS ===
    fps_counter += 1
    elapsed_time = time.time() - fps_start_time
    if elapsed_time > 1.0:
        fps_display = fps_counter / elapsed_time
        fps_counter = 0
        fps_start_time = time.time()
    cv2.putText(frame, f"FPS: {fps_display:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # === Show Frame ===
    cv2.imshow("Classroom Monitor System", frame)

    # === Control FPS ===
    frame_end_time = time.time()
    frame_duration = frame_end_time - frame_start_time
    sleep_time = max(0, FRAME_TIME - frame_duration)
    time.sleep(sleep_time)

    # Exit on ESC key
    if cv2.waitKey(1) == 27:
        break

# === Cleanup ===
cap.release()
cv2.destroyAllWindows()