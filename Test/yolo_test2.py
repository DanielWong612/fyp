import cv2
import os
import mediapipe as mp
from ultralytics import YOLO

# 创建存储文件夹
os.makedirs("unknown_faces", exist_ok=True)

# 初始化 MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

# 初始化 YOLOv8 模型（使用 yolov8n）
model = YOLO("yolov8n.pt")

# 初始化视频捕获（使用摄像头）
cap = cv2.VideoCapture(0)

# 已知 ID 集合，用于跟踪新面孔
tracked_ids = set()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 将帧转换为 RGB（MediaPipe 需要 RGB 格式）
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 使用 MediaPipe 检测人脸
    face_results = face_detection.process(rgb_frame)

    # 如果检测到人脸，则进行 YOLOv8 跟踪
    if face_results.detections:
        # 使用 YOLOv8 进行跟踪
        yolo_results = model.track(frame, persist=True)

        for result in yolo_results:
            boxes = result.boxes.xyxy.cpu().numpy()  # 边界框坐标
            ids = result.boxes.id  # 跟踪 ID

            if boxes is not None and ids is not None:
                for i, (box, track_id) in enumerate(zip(boxes, ids)):
                    x1, y1, x2, y2 = map(int, box[:4])
                    track_id = int(track_id)

                    # 检查是否是新检测到的对象
                    if track_id not in tracked_ids:
                        label = "[Unknown]"
                        tracked_ids.add(track_id)

                        # 裁剪并保存人脸
                        face_img = frame[y1:y2, x1:x2]
                        filename = f"unknown_faces/unknown_{track_id}_{len(tracked_ids)}.jpg"
                        cv2.imwrite(filename, face_img)
                        print(f"Saved new face: {filename}")
                    else:
                        label = f"ID_{track_id}"

                    # 绘制边界框和标签
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # 显示结果
    cv2.imshow("Face Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()
face_detection.close()