import cv2
import os
from ultralytics import YOLO

# 创建存储文件夹
os.makedirs("unknown_faces", exist_ok=True)

# 加载最新的 YOLOv8 模型（例如 yolov8n，最轻量版）
model = YOLO("yolov8n.pt")  # 可替换为 yolov8s.pt、yolov8m.pt 等更高精度模型

# 初始化视频捕获（使用摄像头）
cap = cv2.VideoCapture(0)

# 已知 ID 集合，用于跟踪新面孔
tracked_ids = set()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 使用 YOLOv8 进行检测和跟踪
    results = model.track(frame, persist=True)  # persist=True 启用跟踪

    # 处理检测结果
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  # 边界框坐标
        ids = result.boxes.id  # 跟踪 ID（如果有）

        if boxes is not None and ids is not None:
            for i, (box, track_id) in enumerate(zip(boxes, ids)):
                x1, y1, x2, y2 = map(int, box[:4])
                track_id = int(track_id)

                # 检查是否是新检测到的对象
                if track_id not in tracked_ids:
                    label = "[Unknown]"
                    tracked_ids.add(track_id)

                    # 裁剪并保存新面孔
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
    cv2.imshow("YOLOv8 Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()