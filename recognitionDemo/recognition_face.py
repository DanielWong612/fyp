import cv2
import numpy as np
from ultralytics import YOLO
from deepface import DeepFace
from sklearn.metrics.pairwise import cosine_similarity
from mtcnn import MTCNN
import os
from datetime import datetime
import time

# 設置參數
model_path = "recognitionDemo/Yolo/yolov8n-face.pt"  # YOLO 人臉檢測模型路徑
capture_dir = "recognitionDemo/face_database"        # 儲存人臉的根目錄
delay_frames = 5                          # 延遲捕獲的幀數（新用戶）
similarity_threshold = 0.6                # 識別閾值
high_similarity_threshold = 0.8           # 高相似度閾值，用於自動捕獲新特徵
capture_interval = 10                     # 自動捕獲新特徵的時間間隔（秒）
frontal_check_interval = 5                # 每隔幾幀檢查一次正面

# 加載模型
model = YOLO(model_path)
mtcnn = MTCNN()

# 已知人臉數據庫（初始為空）
known_faces = {}  # {label: [embedding1, embedding2, ...]}
last_capture_time = {}  # {label: timestamp}

# 從目錄加載已知人臉特徵
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
                    print(f"處理 {img_path} 時出錯: {e}")

# 初始加載已知人臉
load_known_faces(capture_dir)

# 判斷是否為正面人臉（新邏輯：鼻子兩側可見）
def is_frontal_face(face_img):
    try:
        detections = mtcnn.detect_faces(face_img)
        if detections:
            keypoints = detections[0]['keypoints']
            nose = keypoints['nose']
            img_width = face_img.shape[1]
            # 檢查鼻子是否位於圖像寬度的 25% 到 75% 之間
            if 0.25 * img_width < nose[0] < 0.75 * img_width:
                return True
    except Exception:
        return False
    return False

# 儲存人臉圖像
def auto_capture(face_img, label, capture_dir):
    user_dir = os.path.join(capture_dir, label)
    os.makedirs(user_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{label}_{timestamp}.jpg"
    filepath = os.path.join(user_dir, filename)
    cv2.imwrite(filepath, face_img)
    print(f"已儲存正面人臉至: {filepath}")
    return filepath

# 識別人臉（與 known_faces 比對）
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

# 追蹤人臉（基於特徵相似度）
def track_face(embedding, tracked_faces, threshold=similarity_threshold):
    for face_id, data in tracked_faces.items():
        tracked_embedding = data['embedding']
        similarity = cosine_similarity([embedding], [tracked_embedding])[0][0]
        if similarity > threshold:
            return face_id
    return None

# 啟動攝像頭
cap = cv2.VideoCapture(0)

# 用於儲存追蹤數據的緩衝區
tracked_faces = {}  # {face_id: {'embedding': embedding, 'frame_count': int, 'image': face_img, 'label': str, 'is_frontal': bool}}
face_id_counter = 0  # 用於生成唯一 face_id
frame_count = 0      # 用於控制正面檢查頻率
last_frontal_result = {}  # 儲存每個 face_id 的最近正面檢查結果

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # 使用 YOLO 檢測人臉（關閉日誌）
    results = model.predict(source=frame, conf=0.25, imgsz=640, verbose=False)

    current_faces = {}  # 當前幀檢測到的人臉

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy() if result.boxes.xyxy is not None else []
        
        for box in boxes:
            x1, y1, x2, y2 = map(int, box[:4])
            face_img = frame[y1:y2, x1:x2]

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
            recognized_label, similarity = recognize_face(embedding, known_faces)

            if recognized_label:
                # 如果識別成功，標記為該用戶
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, recognized_label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # 檢查是否為正面人臉並自動捕獲新特徵
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
                tracked_faces[face_id]['embedding'] = embedding
                tracked_faces[face_id]['image'] = face_img

            current_faces[face_id] = (x1, y1, x2, y2)

            # 延遲捕獲邏輯（新用戶）
            if tracked_faces[face_id]['frame_count'] >= delay_frames:
                # 檢查是否為正面人臉
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
                    cv2.putText(frame, new_label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    del tracked_faces[face_id]
                else:
                    # 非正面人臉，僅顯示檢測結果
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(frame, "Detected", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
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

# 釋放資源
cap.release()
cv2.destroyAllWindows()