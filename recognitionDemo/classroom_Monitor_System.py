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

# Set parameters
yolo_model_path = "recognitionDemo/Yolo/yolov8n-face.pt"  # Path to YOLO face detection model
emotion_model_path = "recognitionDemo/FER/model.h5"  # Path to emotion recognition model
capture_dir = "recognitionDemo/face_database"  # Root directory for storing faces
delay_frames = 5  # Number of frames to delay capture (new users)
similarity_threshold = 0.6  # Recognition similarity threshold
high_similarity_threshold = 0.8  # High similarity threshold for auto-capturing new features
capture_interval = 10  # Time interval for auto-capturing new features (seconds)
frontal_check_interval = 5  # Check for frontal face every few frames

# Load models
yolo_model = YOLO(yolo_model_path)
emotion_model = load_model(emotion_model_path)
print("Emotion model input shape:", emotion_model.input_shape)  # Verify model input shape
mtcnn = MTCNN()

# Emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Known face database (initially empty)
known_faces = {}  # {label: [embedding1, embedding2, ...]}
last_capture_time = {}  # {label: timestamp}

# Mapping of user_X to student IDs
user_to_id = {
    "user_3": "22069999D",
    # Add more mappings as needed
    # "user_1": "22069998C",
    # "user_2": "22069997B"
}

# Load known face features from directory
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
                    print(f"Error processing {img_path}: {e}")

# Preprocess image for emotion recognition
def preprocess_image(face_image, target_size=(48, 48)):
    """Preprocess the image to fit the emotion recognition model"""
    face_image = cv2.resize(face_image, target_size)  # Resize to 48x48
    face_image = np.stack([face_image] * 3, axis=-1)  # Convert grayscale to RGB
    face_image = face_image.astype('float32') / 255.0  # Normalize to [0, 1]
    face_image = np.expand_dims(face_image, axis=0)  # Add batch dimension
    return face_image

# Check if the face is frontal
def is_frontal_face(face_img):
    try:
        detections = mtcnn.detect_faces(face_img)
        if detections:
            keypoints = detections[0]['keypoints']
            nose = keypoints['nose']
            img_width = face_img.shape[1]
            if 0.25 * img_width < nose[0] < 0.75 * img_width:
                return True
    except Exception as e:
        print(f"Error in frontal face check: {e}")
        return False
    return False

# Save face image
def auto_capture(face_img, label, capture_dir):
    user_dir = os.path.join(capture_dir, label)
    os.makedirs(user_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{label}_{timestamp}.jpg"
    filepath = os.path.join(user_dir, filename)
    cv2.imwrite(filepath, face_img)
    print(f"Saved frontal face to: {filepath}")
    return filepath

# Recognize face (returns student ID if mapped)
def recognize_face(embedding, known_faces, threshold=similarity_threshold):
    best_label = None
    best_similarity = 0
    for label, embeddings in known_faces.items():
        for known_embedding in embeddings:
            similarity = cosine_similarity([embedding], [known_embedding])[0][0]
            if similarity > best_similarity:
                best_similarity = similarity
                best_label = label if similarity > threshold else None
    if best_label and best_label in user_to_id:
        return user_to_id[best_label], best_similarity  # Return student ID
    return best_label, best_similarity

# Track face
def track_face(embedding, tracked_faces, threshold=similarity_threshold):
    for face_id, data in tracked_faces.items():
        tracked_embedding = data['embedding']
        similarity = cosine_similarity([embedding], [tracked_embedding])[0][0]
        if similarity > threshold:
            return face_id
    return None

# Predict emotion
@tf.function  # Use tf.function to reduce AutoGraph overhead
def predict_emotion(face_img):
    try:
        gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        processed_face = preprocess_image(gray_face)
        predictions = emotion_model.predict(processed_face, verbose=0)  # Disable progress bar
        emotion_index = np.argmax(predictions)
        emotion_label = emotion_labels[emotion_index]
        confidence = np.max(predictions)
        # Ensure the return value is a string
        return f"{emotion_label}: {confidence:.2f}"
    except Exception as e:
        print(f"Emotion prediction error: {e}")
        return "Unknown"

# Generator function to yield processed frames for Flask
def generate_processed_frames():
    # Initially load known faces
    load_known_faces(capture_dir)

    # Start the camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Camera not opened.")
        return

    # Tracking data buffer
    tracked_faces = {}  # {face_id: {'embedding': embedding, 'frame_count': int, 'image': face_img, 'label': str}}
    face_id_counter = 0  # Used to generate unique face_id
    frame_count = 0  # Control the frequency of frontal checks
    last_frontal_result = {}  # Store the most recent frontal check result for each face_id

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame from camera.")
            break

        frame_count += 1

        try:
            # Use YOLO to detect faces
            results = yolo_model.predict(source=frame, conf=0.25, imgsz=640, verbose=False)

            current_faces = {}  # Faces detected in the current frame

            for result in results:
                boxes = result.boxes.xyxy.cpu().numpy() if result.boxes.xyxy is not None else []
                
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box[:4])
                    face_img = frame[y1:y2, x1:x2]

                    # Extract features using DeepFace
                    try:
                        embedding = DeepFace.represent(
                            face_img,
                            model_name='Facenet',
                            enforce_detection=False
                        )[0]["embedding"]
                    except Exception as e:
                        print(f"Feature extraction error: {e}")
                        continue

                    # Predict emotion
                    emotion_label = predict_emotion(face_img)
                    # Ensure emotion_label is a string
                    if not isinstance(emotion_label, str):
                        print(f"Invalid emotion_label type: {type(emotion_label)}, value: {emotion_label}")
                        emotion_label = "Unknown"

                    # Attempt to recognize the face
                    recognized_label, similarity = recognize_face(embedding, known_faces)

                    if recognized_label:
                        # If recognition succeeds, display student ID (if mapped) and emotion
                        display_label = recognized_label if recognized_label in user_to_id.values() else f"Unknown ({recognized_label})"
                        # Ensure display_label is a string
                        if not isinstance(display_label, str):
                            print(f"Invalid display_label type: {type(display_label)}, value: {display_label}")
                            display_label = "Unknown"
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, display_label, (x1, y1-25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        cv2.putText(frame, emotion_label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                        # Check if it’s frontal and auto-capture new features
                        if frame_count % frontal_check_interval == 0 or recognized_label not in last_frontal_result:
                            is_frontal = is_frontal_face(face_img)
                            last_frontal_result[recognized_label] = is_frontal
                        else:
                            is_frontal = last_frontal_result.get(recognized_label, False)

                        if is_frontal:
                            current_time = time.time()
                            if similarity > high_similarity_threshold and (recognized_label not in last_capture_time or current_time - last_capture_time[recognized_label] > capture_interval):
                                # Use the original user_X label for saving
                                for user_label, student_id in user_to_id.items():
                                    if student_id == recognized_label:
                                        auto_capture(face_img, user_label, capture_dir)
                                        known_faces[user_label].append(embedding)
                                        last_capture_time[student_id] = current_time
                                        break
                        continue

                    # Attempt to track existing faces
                    face_id = track_face(embedding, tracked_faces)

                    if face_id is None:
                        # New face, assign a new ID
                        face_id = f"face_{face_id_counter}"
                        face_id_counter += 1
                        tracked_faces[face_id] = {
                            'embedding': embedding,
                            'frame_count': 1,
                            'image': face_img,
                            'label': "Unknown"
                        }
                    else:
                        # Update tracked face data
                        tracked_faces[face_id]['frame_count'] += 1
                        tracked_faces[face_id]['embedding'] = embedding
                        tracked_faces[face_id]['image'] = face_img

                    current_faces[face_id] = (x1, y1, x2, y2)

                    # Delay capture logic (new users)
                    if tracked_faces[face_id]['frame_count'] >= delay_frames:
                        # Check if it’s a frontal face
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
                            cv2.putText(frame, new_label, (x1, y1-25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                            cv2.putText(frame, emotion_label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                            del tracked_faces[face_id]
                        else:
                            # Non-frontal face, display detection result
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                            cv2.putText(frame, "Detected", (x1, y1-25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                            cv2.putText(frame, emotion_label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    else:
                        # Not yet reached delay frame count, display "Detecting"
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
                        cv2.putText(frame, "Detecting", (x1, y1-25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                        cv2.putText(frame, emotion_label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # Clean up tracking data for faces not in the current frame
            tracked_faces = {k: v for k, v in tracked_faces.items() if k in current_faces}

        except Exception as e:
            print(f"Error in frame processing: {e}")
            continue  # Continue processing the next frame even if an error occurs

        # Encode the frame as JPEG for streaming
        try:
            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        except Exception as e:
            print(f"Error encoding frame: {e}")
            continue

    # Release resources
    print("Releasing camera resources.")
    cap.release()

if __name__ == "__main__":
    # For standalone testing
    def main():
        load_known_faces(capture_dir)
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Camera not opened.")
            return
        window_name = "Face and Emotion Recognition"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        for frame in generate_processed_frames():
            try:
                frame = cv2.imdecode(np.frombuffer(frame.split(b'\r\n\r\n')[1], np.uint8), cv2.IMREAD_COLOR)
                cv2.imshow(window_name, frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
            except Exception as e:
                print(f"Error displaying frame in main: {e}")
                break
        cv2.destroyAllWindows()
    main()