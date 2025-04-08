import cv2
import numpy as np
from ultralytics import YOLO
from deepface import DeepFace
from sklearn.metrics.pairwise import cosine_similarity
from mtcnn import MTCNN
import os
from datetime import datetime
import time
import tensorflow as tf
from tensorflow.keras.models import load_model
from concurrent.futures import ThreadPoolExecutor

# Check GPU availability and configure memory growth
physical_devices = tf.config.list_physical_devices('GPU')
print("GPU Available:", physical_devices)
if physical_devices:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)

# Set parameters
yolo_model_path = "recognitionDemo/Yolo/yolov10n-face.pt"
emotion_model_path = "recognitionDemo/FER/model.h5"
capture_dir = "recognitionDemo/face_database"
delay_frames = 5
similarity_threshold = 0.6
high_similarity_threshold = 0.8
capture_interval = 10
frontal_check_interval = 5
emotion_cache_interval = 10

# Load models
yolo_model = YOLO(yolo_model_path)
emotion_model = load_model(emotion_model_path)
print("Emotion model input shape:", emotion_model.input_shape)  # Should be (None, 48, 48, 3)
mtcnn = MTCNN()

# Define emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Initialize dictionaries
known_faces = {}
last_capture_time = {}

def load_known_faces(capture_dir):
    """Load face embeddings from the database."""
    for user_dir in os.listdir(capture_dir):
        user_path = os.path.join(capture_dir, user_dir)
        if os.path.isdir(user_path):
            known_faces[user_dir] = []
            for img_file in os.listdir(user_path):
                img_path = os.path.join(user_path, img_file)
                try:
                    embedding = DeepFace.represent(img_path, model_name='Facenet', enforce_detection=False)[0]["embedding"]
                    known_faces[user_dir].append(embedding)
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")

def preprocess_image(face_image, target_size=(48, 48)):
    """Preprocess face image for emotion model (RGB, 48x48, normalized)."""
    if face_image.size == 0 or face_image.shape[0] == 0 or face_image.shape[1] == 0:
        raise ValueError("Invalid image: empty or zero-sized")
    face_image = cv2.resize(face_image, target_size)  # Keep RGB, resize to 48x48
    face_image = face_image.astype('float32') / 255.0  # Normalize to [0, 1]
    return np.expand_dims(face_image, axis=0)  # Shape: (1, 48, 48, 3)

def is_frontal_face(face_img):
    """Check if the face is frontal using MTCNN keypoints."""
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

def auto_capture(face_img, label, capture_dir):
    """Save face image to the database with a timestamp."""
    user_dir = os.path.join(capture_dir, label)
    os.makedirs(user_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{label}_{timestamp}.jpg"
    filepath = os.path.join(user_dir, filename)
    cv2.imwrite(filepath, face_img)
    print(f"Saved frontal face to: {filepath}")
    return filepath

def recognize_face(embedding, known_faces, threshold=similarity_threshold):
    """Recognize face by comparing embeddings."""
    best_label, best_similarity = None, 0
    for label, embeddings in known_faces.items():
        for known_embedding in embeddings:
            similarity = cosine_similarity([embedding], [known_embedding])[0][0]
            if similarity > best_similarity:
                best_similarity = similarity
                best_label = label if similarity > threshold else None
    return best_label, best_similarity

def track_face(embedding, tracked_faces, threshold=similarity_threshold):
    """Track face across frames."""
    for face_id, data in tracked_faces.items():
        similarity = cosine_similarity([embedding], [data['embedding']])[0][0]
        if similarity > threshold:
            return face_id
    return None

def predict_emotion_batch(batch_faces):
    """Predict emotions for a batch of faces."""
    try:
        if batch_faces.size == 0 or batch_faces.shape[0] == 0:
            raise ValueError("Empty batch provided")
        predictions = emotion_model.predict(batch_faces, batch_size=1, verbose=0)  # Limit batch size
        emotions = [f"{emotion_labels[np.argmax(pred)]}: {np.max(pred):.2f}" for pred in predictions]
        return emotions
    except Exception as e:
        print(f"Batch emotion prediction error: {e}")
        with tf.device('/CPU:0'):
            predictions = emotion_model.predict(batch_faces, batch_size=1, verbose=0)
            emotions = [f"{emotion_labels[np.argmax(pred)]}: {np.max(pred):.2f}" for pred in predictions]
        return emotions

def process_face(face_img):
    """Process a single face for embedding and emotion."""
    try:
        # Validate input image
        if face_img is None or face_img.size == 0 or face_img.shape[0] == 0 or face_img.shape[1] == 0:
            raise ValueError("Invalid face image: empty or zero-sized")
        if len(face_img.shape) != 3 or face_img.shape[2] != 3:
            raise ValueError(f"Invalid image format: expected RGB (shape with 3 channels), got {face_img.shape}")

        # Extract embedding
        embedding = DeepFace.represent(face_img, model_name='Facenet', enforce_detection=False)[0]["embedding"]
        
        # Preprocess for emotion model
        processed_face = preprocess_image(face_img)  # Shape: (1, 48, 48, 3)
        return embedding, processed_face
    except Exception as e:
        print(f"Error processing face: {str(e)}")
        return None, None

def main():
    """Main function for face and emotion recognition."""
    load_known_faces(capture_dir)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Camera not opened.")
        return

    window_name = "Face and Emotion Recognition"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    tracked_faces = {}
    face_id_counter = 0
    frame_count = 0
    last_frontal_result = {}
    executor = ThreadPoolExecutor(max_workers=4)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        results = yolo_model.predict(source=frame, conf=0.25, imgsz=640, verbose=False)
        current_faces = {}
        face_images = []
        boxes = []

        if results and results[0].boxes:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            for box in boxes:
                x1, y1, x2, y2 = map(int, box[:4])
                face_img = frame[y1:y2, x1:x2]
                if face_img.size > 0:
                    face_images.append(face_img)

        futures = [executor.submit(process_face, face_img) for face_img in face_images]
        results = [future.result() for future in futures]

        embeddings, processed_faces = [], []
        for res in results:
            embedding, processed_face = res
            if embedding is not None and processed_face is not None:
                embeddings.append(embedding)
                processed_faces.append(processed_face)

        emotions = predict_emotion_batch(np.vstack(processed_faces)) if processed_faces else []

        for i, (embedding, box) in enumerate(zip(embeddings, boxes)):
            x1, y1, x2, y2 = map(int, box[:4])
            face_img = face_images[i]
            emotion_label = emotions[i] if i < len(emotions) else "Unknown"

            recognized_label, similarity = recognize_face(embedding, known_faces)
            if recognized_label:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, recognized_label, (x1, y1-25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame, emotion_label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                is_frontal = last_frontal_result.get(recognized_label, False)
                if frame_count % frontal_check_interval == 0 or recognized_label not in last_frontal_result:
                    is_frontal = is_frontal_face(face_img)
                    last_frontal_result[recognized_label] = is_frontal

                if is_frontal:
                    current_time = time.time()
                    if (similarity > high_similarity_threshold and
                        (recognized_label not in last_capture_time or
                         current_time - last_capture_time[recognized_label] > capture_interval)):
                        auto_capture(face_img, recognized_label, capture_dir)
                        known_faces[recognized_label].append(embedding)
                        last_capture_time[recognized_label] = current_time
                continue

            face_id = track_face(embedding, tracked_faces)
            if face_id is None:
                face_id = f"face_{face_id_counter}"
                face_id_counter += 1
                tracked_faces[face_id] = {
                    'embedding': embedding, 'frame_count': 1, 'image': face_img,
                    'label': "Unknown", 'last_emotion': emotion_label, 'emotion_frame': frame_count
                }
            else:
                tracked_faces[face_id]['frame_count'] += 1
                tracked_faces[face_id]['embedding'] = embedding
                tracked_faces[face_id]['image'] = face_img
                if frame_count - tracked_faces[face_id]['emotion_frame'] >= emotion_cache_interval:
                    tracked_faces[face_id]['last_emotion'] = emotion_label
                    tracked_faces[face_id]['emotion_frame'] = frame_count
                else:
                    emotion_label = tracked_faces[face_id]['last_emotion']

            current_faces[face_id] = (x1, y1, x2, y2)

            if tracked_faces[face_id]['frame_count'] >= delay_frames:
                is_frontal = last_frontal_result.get(face_id, False)
                if frame_count % frontal_check_interval == 0 or face_id not in last_frontal_result:
                    is_frontal = is_frontal_face(tracked_faces[face_id]['image'])
                    last_frontal_result[face_id] = is_frontal

                if is_frontal:
                    new_label = f"user_{len(known_faces) + 1}"
                    auto_capture(tracked_faces[face_id]['image'], new_label, capture_dir)
                    known_faces[new_label] = [tracked_faces[face_id]['embedding']]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, new_label, (x1, y1-25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(frame, emotion_label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    del tracked_faces[face_id]
                else:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(frame, "Detected", (x1, y1-25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                    cv2.putText(frame, emotion_label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
                cv2.putText(frame, "Detecting", (x1, y1-25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                cv2.putText(frame, emotion_label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        tracked_faces = {k: v for k, v in tracked_faces.items() if k in current_faces}
        cv2.imshow(window_name, frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()