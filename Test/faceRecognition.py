import cv2
import numpy as np
from ultralytics import YOLO
from deepface import DeepFace
from sklearn.metrics.pairwise import cosine_similarity
import os
from datetime import datetime

# === 設置參數 ===
model_path = "Test/Yolo/yolov8n-face.pt"  # YOLO 人臉檢測模型路徑
capture_dir = "Test/face_database"        # 儲存捕獲人臉的目錄
delay_frames = 5                          # 延遲捕獲的幀數
similarity_threshold = 0.6                # 降低閾值以提高匹配成功率

# === 加載 YOLO 模型 ===
model = YOLO(model_path)

# === 已知人臉數據庫（初始為空） ===
known_faces = {}  # {label: embedding} 用於儲存捕獲的人臉特徵

# === 自動捕獲人臉 ===
def auto_capture(face_img, label, capture_dir):
    """將檢測到的人臉儲存到指定目錄並返回標籤"""
    os.makedirs(capture_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{label}_{timestamp}.jpg"
    filepath = os.path.join(capture_dir, filename)
    cv2.imwrite(filepath, face_img)
    print(f"已捕獲人臉並儲存至: {filepath}")
    return label

# === 識別人臉（與 known_faces 比對） ===
def recognize_face(embedding, known_faces, threshold=similarity_threshold):
    """嘗試將特徵與 known_faces 進行比對"""
    for label, known_embedding in known_faces.items():
        similarity = cosine_similarity([embedding], [known_embedding])[0][0]
        if similarity > threshold:
            return label
    return None

# === 追蹤人臉（基於特徵相似度） ===
def track_face(embedding, tracked_faces, threshold=similarity_threshold):
    """檢查當前特徵是否與已追蹤的人臉匹配"""
    for face_id, data in tracked_faces.items():
        tracked_embedding = data['embedding']
        similarity = cosine_similarity([embedding], [tracked_embedding])[0][0]
        if similarity > threshold:
            return face_id
    return None

# === 啟動攝像頭 ===
cap = cv2.VideoCapture(0)

# 用於儲存追蹤數據的緩衝區
tracked_faces = {}  # {face_id: {'embedding': embedding, 'frame_count': int, 'image': face_img, 'label': str}}

face_id_counter = 0  # 用於生成唯一 face_id

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 使用 YOLO 進行人臉檢測
    results = model.predict(source=frame, conf=0.25, imgsz=640)

    current_faces = {}  # 當前幀檢測到的人臉

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

            # 嘗試與 known_faces 進行比對
            recognized_label = recognize_face(embedding, known_faces)

            if recognized_label:
                # 如果識別成功，標記為該用戶
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, recognized_label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                continue  # 跳過追蹤和捕獲邏輯

            # 嘗試追蹤現有人臉
            face_id = track_face(embedding, tracked_faces)

            if face_id is None:
                # 新人臉，分配新 ID
                face_id = f"face_{face_id_counter}"
                face_id_counter += 1
                tracked_faces[face_id] = {
                    'embedding': embedding,
                    'frame_count': 1,
                    'image': face_img,
                    'label': "Unknown"
                }
            else:
                # 更新已追蹤人臉的數據
                tracked_faces[face_id]['frame_count'] += 1
                tracked_faces[face_id]['embedding'] = embedding  # 更新特徵
                tracked_faces[face_id]['image'] = face_img      # 更新影像

            current_faces[face_id] = (x1, y1, x2, y2)

            # 延遲捕獲邏輯
            if tracked_faces[face_id]['frame_count'] >= delay_frames:
                new_label = f"user_{len(known_faces) + 1}"
                auto_capture(tracked_faces[face_id]['image'], new_label, capture_dir)
                known_faces[new_label] = tracked_faces[face_id]['embedding']
                # 在圖像上標記結果
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, new_label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                # 移除已捕獲的人臉
                del tracked_faces[face_id]
            else:
                # 未達延遲幀數，顯示 "Detecting"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
                cv2.putText(frame, "Detecting", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    # 清理未出現在當前幀的追蹤數據
    tracked_faces = {k: v for k, v in tracked_faces.items() if k in current_faces}

    # 顯示結果
    cv2.imshow("YOLO + DeepFace Auto Capture", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # 按 ESC 退出
        break

# === 釋放資源 ===
cap.release()
cv2.destroyAllWindows()