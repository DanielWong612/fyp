import cv2
import os
import numpy as np
from ultralytics import YOLO
from tensorflow.keras.models import load_model
from datetime import datetime

# === 載入模型 ===
model = YOLO("yolov8n.pt")
emotion_model = load_model("/Users/sam/Documents/GitHub/fyp/Test/FER/model.h5")
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# === 儲存設定 ===
os.makedirs("detected_faces", exist_ok=True)
reference_faces = {}  # 存已知人臉

# === 表情前處理 ===
def preprocess_emotion(face_img, size=(48, 48)):
    face_img = cv2.resize(face_img, size)
    face_img = np.stack([face_img] * 3, axis=-1).astype('float32') / 255.0
    return np.expand_dims(face_img, axis=0)

# === 簡單人臉比對 ===
def compare_faces(face1, face2):
    if face1.shape != face2.shape:
        face2 = cv2.resize(face2, (face1.shape[1], face1.shape[0]))
    diff = cv2.absdiff(face1, face2)
    return np.mean(diff) < 45  # threshold

# === 開始攝影機 ===
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(frame, persist=True)

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        ids = result.boxes.id

        for box, track_id in zip(boxes, ids):
            x1, y1, x2, y2 = map(int, box[:4])
            face_img = frame[y1:y2, x1:x2]
            gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)

            # 是否已存在
            matched_label = "Unknown"
            for name, ref_face in reference_faces.items():
                if compare_faces(ref_face, gray_face):
                    matched_label = name
                    break

            # 如果新臉：儲存並命名
            if matched_label == "Unknown":
                label = f"user_{track_id}_{datetime.now().strftime('%H%M%S')}"
                reference_faces[label] = gray_face
                matched_label = label
                path = f"detected_faces/{label}.jpg"
                cv2.imwrite(path, face_img)

            # 表情辨識
            try:
                input_face = preprocess_emotion(gray_face)
                preds = emotion_model.predict(input_face)
                emotion = emotion_labels[np.argmax(preds)]
            except:
                emotion = "N/A"

            # 畫框
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, f"{matched_label} | {emotion}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    cv2.imshow("YOLO + Emotion", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
