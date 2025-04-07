import cv2
import os
import numpy as np
from ultralytics import YOLO
from tensorflow.keras.models import load_model
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity

# === Load models ===
model = YOLO("Test/Yolo/yolov10m-face.pt")  # Use yolov8l-face.pt
emotion_model = load_model("/Users/sam/Documents/GitHub/fyp/Test/FER/model.h5")
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# === Storage settings ===
base_dir = "detected_faces"
os.makedirs(base_dir, exist_ok=True)
reference_faces = {}  # {label: list of embedding vectors}
MAX_SAMPLES_PER_ID = 5  # Maximum 5 samples per ID

# === YOLO parameters ===
CONF_THRESHOLD = 0.25
IMG_SIZE = 640  # Reduce image size to improve speed (from 1280 to 640)
LINE_THICKNESS = 1
MAX_DET = 100  # Reduce maximum detections (from 1000 to 100)

# === Buffer mechanism parameters ===
DETECTION_INTERVAL = 5  # Perform detection every 5 frames
frame_counter = 0
last_detections = []  # Store the last detection results

# === Emotion preprocessing ===
def preprocess_emotion(face_img, size=(48, 48)):
    face_img = cv2.resize(face_img, size)
    face_img = np.stack([face_img] * 3, axis=-1).astype('float32') / 255.0
    return np.expand_dims(face_img, axis=0)

# === Check if wearing a mask (simple method: check if the lower half has skin color) ===
def is_wearing_mask(face_img):
    h, w = face_img.shape[:2]
    lower_half = face_img[h//2:, :]  # Take the lower half
    hsv = cv2.cvtColor(lower_half, cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    skin_ratio = np.sum(mask) / (255 * mask.size)  # Ratio of skin pixels
    return skin_ratio < 0.3  # If skin ratio is low, assume a mask is worn

# === Face embedding extraction ===
def extract_embedding(face_img, wearing_mask=False):
    face_resized = cv2.resize(face_img, (100, 100))
    if wearing_mask:
        face_resized = face_resized[:50, :]  # Take the upper half
        face_resized = cv2.resize(face_resized, (100, 100))  # Resize again
    face_normalized = face_resized.astype("float32") / 255.0
    embedding = face_normalized.flatten().reshape(1, -1)
    return embedding

# === Cosine similarity comparison with multiple samples ===
def compare_faces_cosine(new_embedding, stored_embeddings):
    similarities = [cosine_similarity(new_embedding, emb)[0][0] for emb in stored_embeddings]
    avg_similarity = np.mean(similarities)
    return avg_similarity > 0.95  # Use average similarity, threshold set to 0.95

# === Check if an ID is contaminated ===
def check_id_contamination(label, embeddings, similarity_threshold=0.90):
    """
    Check if the reference embeddings for an ID are consistent.
    If the average similarity between embeddings is below the threshold, the ID is considered contaminated.
    """
    if len(embeddings) < 2:  # If there is only one embedding, cannot check
        return False

    # Calculate pairwise similarities
    similarities = []
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            sim = cosine_similarity(embeddings[i], embeddings[j])[0][0]
            similarities.append(sim)

    # Calculate average similarity and standard deviation
    avg_similarity = np.mean(similarities)
    std_similarity = np.std(similarities)

    # If average similarity is below threshold or standard deviation is too high, consider it contaminated
    if avg_similarity < similarity_threshold or std_similarity > 0.1:
        print(f"Warning: ID {label} may be contaminated! Avg similarity: {avg_similarity:.3f}, Std: {std_similarity:.3f}")
        return True
    return False

# === Update reference embeddings ===
def update_reference_faces(label, new_embedding):
    if label not in reference_faces:
        reference_faces[label] = [new_embedding]
    else:
        if len(reference_faces[label]) < MAX_SAMPLES_PER_ID:
            reference_faces[label].append(new_embedding)
        else:
            reference_faces[label].pop(0)  # Remove the oldest
            reference_faces[label].append(new_embedding)
        if check_id_contamination(label, reference_faces[label]):
            print(f"Clearing contaminated ID: {label}")
            reference_faces[label] = [new_embedding]  # Reset to the latest embedding

# === Auto-classify images to userX.0 folders ===
def save_face_with_folder(face_img, label):
    main_id = label.split('_')[0]
    folder_path = os.path.join(base_dir, main_id)
    os.makedirs(folder_path, exist_ok=True)
    filename = f"{label}.jpg"
    filepath = os.path.join(folder_path, filename)
    cv2.imwrite(filepath, face_img)
    print(f"Saved {filename} → {folder_path}/")
    return filepath

# === Start webcam ===
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_counter += 1
    current_detections = []

    # Perform detection every DETECTION_INTERVAL frames
    if frame_counter % DETECTION_INTERVAL == 0:
        results = model.predict(
            source=frame,
            conf=CONF_THRESHOLD,
            imgsz=IMG_SIZE,
            max_det=MAX_DET
        )

        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy() if result.boxes.xyxy is not None else []
            
            if len(boxes) > 0:
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = map(int, box[:4])
                    face_img = frame[y1:y2, x1:x2]
                    gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)

                    wearing_mask = is_wearing_mask(face_img)
                    new_embedding = extract_embedding(gray_face, wearing_mask)

                    matched_label = "Unknown"
                    for name, stored_embeddings in reference_faces.items():
                        if compare_faces_cosine(new_embedding, stored_embeddings):
                            matched_label = name
                            break

                    if matched_label == "Unknown":
                        label = f"user{i}.0_{datetime.now().strftime('%H%M%S')}"
                        update_reference_faces(label, new_embedding)
                        matched_label = label
                        save_face_with_folder(face_img, label)
                    else:
                        update_reference_faces(matched_label, new_embedding)

                    try:
                        input_face = preprocess_emotion(gray_face)
                        preds = emotion_model.predict(input_face, verbose=0)
                        emotion = emotion_labels[np.argmax(preds)]
                    except:
                        emotion = "N/A"

                    current_detections.append((x1, y1, x2, y2, matched_label, emotion))

        last_detections = current_detections  # Update the last detection results

    # Use the last detection results for drawing
    for detection in last_detections:
        x1, y1, x2, y2, matched_label, emotion = detection
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), LINE_THICKNESS)
        cv2.putText(frame, f"{matched_label} | {emotion}", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), LINE_THICKNESS)

    cv2.imshow("YOLO + Emotion + Auto-folder", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()