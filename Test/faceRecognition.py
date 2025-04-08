import cv2
import numpy as np
from ultralytics import YOLO
from deepface import DeepFace
from sklearn.metrics.pairwise import cosine_similarity
import os

# === 設置參數 ===
model_path = "Test/Yolo/yolov8n-face.pt"  # YOLO 人臉檢測模型路徑
database_dir = "face_database"  # 已知人臉數據庫目錄
similarity_threshold = 0.6      # 相似度閾值，用於判斷是否匹配

# === 加載 YOLO 模型 ===
model = YOLO(model_path)

# === 準備已知人臉特徵數據庫 ===
def load_known_faces(database_dir):
    """從指定目錄加載已知人臉圖像並提取特徵向量"""
    known_faces = {}
    for root, _, files in os.walk(database_dir):
        for file in files:
            if file.endswith(".jpg") or file.endswith(".png"):
                img_path = os.path.join(root, file)
                label = os.path.basename(root)  # 子目錄名稱作為標籤
                try:
                    embedding = DeepFace.represent(
                        img_path,
                        model_name="Facenet",
                        detector_backend="mtcnn"
                    )[0]["embedding"]
                    if label not in known_faces:
                        known_faces[label] = []
                    known_faces[label].append(embedding)
                except Exception as e:
                    print(f"處理 {img_path} 時出錯: {e}")
    return known_faces

# 加載已知人臉數據庫
known_faces = load_known_faces(database_dir)

# === 特徵比對函數 ===
def match_face(embedding, known_faces, threshold=similarity_threshold):
    """將檢測到的特徵向量與已知特徵進行比對"""
    for label, embeddings in known_faces.items():
        for known_embedding in embeddings:
            similarity = cosine_similarity([embedding], [known_embedding])[0][0]
            if similarity > threshold:
                return label
    return "Unknown"

# === 啟動攝像頭 ===
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 使用 YOLO 進行人臉檢測
    results = model.predict(source=frame, conf=0.25, imgsz=640)

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy() if result.boxes.xyxy is not None else []
        
        for box in boxes:
            x1, y1, x2, y2 = map(int, box[:4])  # 提取邊界框坐標
            face_img = frame[y1:y2, x1:x2]      # 裁剪人臉區域

            # 使用 DeepFace 進行特徵提取
            try:
                embedding = DeepFace.represent(
                    face_img, 
                    model_name='Facenet', 
                    enforce_detection=False
                )[0]["embedding"]
            except Exception as e:
                print(f"特徵提取錯誤: {e}")
                continue

            # 進行特徵比對
            label = match_face(embedding, known_faces)

            # 在圖像上標記結果
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # 顯示結果
    cv2.imshow("YOLO + DeepFace Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # 按 ESC 退出
        break

# === 釋放資源 ===
cap.release()
cv2.destroyAllWindows()