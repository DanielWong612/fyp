import cv2
import os
import numpy as np
from ultralytics import YOLO
from tensorflow.keras.models import load_model
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
# 可選：使用 DeepFace 進行特徵提取
# from deepface import DeepFace

# === Load models ===
model = YOLO("Test/Yolo/yolov8n-face.pt")
emotion_model = load_model("/Users/sam/Documents/GitHub/fyp/Test/FER/model.h5")
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# === Storage settings ===
base_dir = "detected_faces"
os.makedirs(base_dir, exist_ok=True)
reference_faces = {}  # {label: embedding vector}

# === YOLO parameters ===
CONF_THRESHOLD = 0.25
IMG_SIZE = 1280
LINE_THICKNESS = 1
MAX_DET = 1000

# === Emotion preprocessing ===
def preprocess_emotion(face_img, size=(48, 48)):
    face_img = cv2.resize(face_img, size)
    face_img = np.stack([face_img] * 3, axis=-1).astype('float32') / 255.0
    return np.expand_dims(face_img, axis=0)

# === Face embedding extraction ===
def extract_embedding(face_img):
    face_resized = cv2.resize(face_img, (100, 100))
    face_normalized = face_resized.astype("float32") / 255.0
    embedding = face_normalized.flatten().reshape(1, -1)
    return embedding

# 可選：使用 DeepFace 提取特徵（需安裝：pip install deepface）
# def extract_embedding_deepface(face_img):
#     embedding = DeepFace.represent(face_img, model_name='Facenet', detector_backend='opencv')[0]["embedding"]
#     return np.array(embedding).reshape(1, -1)

# === Cosine similarity comparison ===
def compare_faces_cosine(embedding1, embedding2):
    similarity = cosine_similarity(embedding1, embedding2)[0][0]
    return similarity > 0.95  # 提高閾值到 0.95，減少誤判

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

                # Face matching
                matched_label = "Unknown"
                new_embedding = extract_embedding(gray_face)
                # 如果使用 DeepFace：new_embedding = extract_embedding_deepface(face_img)

                for name, stored_embedding in reference_faces.items():
                    if compare_faces_cosine(stored_embedding, new_embedding):
                        matched_label = name
                        break

                # New face handling
                if matched_label == "Unknown":
                    label = f"user{i}.0_{datetime.now().strftime('%H%M%S')}"
                    reference_faces[label] = new_embedding
                    matched_label = label
                    save_face_with_folder(face_img, label)

                # Emotion recognition
                try:
                    input_face = preprocess_emotion(gray_face)
                    preds = emotion_model.predict(input_face, verbose=0)
                    emotion = emotion_labels[np.argmax(preds)]
                except:
                    emotion = "N/A"

                # Draw results
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), LINE_THICKNESS)
                cv2.putText(frame, f"{matched_label} | {emotion}", (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), LINE_THICKNESS)

    cv2.imshow("YOLO + Emotion + Auto-folder", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()