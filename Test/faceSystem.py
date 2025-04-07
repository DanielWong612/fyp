import cv2
import os
import numpy as np
from ultralytics import YOLO
from tensorflow.keras.models import load_model
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

# === Load models ===
model = YOLO("Test/Yolo/yolov10m-face.pt")  # Use yolov10m-face.pt
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
DETECTION_INTERVAL = 1  # Perform detection every frame for better responsiveness
frame_counter = 0
last_detections = []  # Store the last detection results

# === Range-based detection parameters ===
RANGE_THRESHOLD = 50  # Maximum movement (in pixels) allowed from the initial position
face_initial_positions = {}  # {temp_id: (x1, y1, x2, y2)} to store initial position
face_ready_to_capture = {}  # {temp_id: bool} to track if face is ready to capture
captured_faces = {}  # {temp_id: (label, emotion)} to store captured face info
last_face_data = {}  # {temp_id: (face_img, gray_face, x1, y1, x2, y2)} to store face data for capture
face_tracking = {}  # {temp_id: (center_x, center_y)} to track faces across frames
next_temp_id = 0  # Incremental ID for tracking faces

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

# === Match current faces to previous faces for tracking ===
def match_faces(current_boxes):
    global next_temp_id
    new_face_tracking = {}
    new_initial_positions = {}
    new_ready_to_capture = {}
    new_face_data = {}
    new_captured_faces = {}

    # Calculate centers of current boxes
    current_centers = []
    for box in current_boxes:
        x1, y1, x2, y2 = map(int, box[:4])
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        current_centers.append((center_x, center_y, x1, y1, x2, y2))

    # Match current faces to previous faces based on proximity
    for center_x, center_y, x1, y1, x2, y2 in current_centers:
        matched_id = None
        min_distance = float('inf')

        # Find the closest previous face
        for temp_id, (prev_center_x, prev_center_y) in face_tracking.items():
            distance = np.sqrt((center_x - prev_center_x)**2 + (center_y - prev_center_y)**2)
            if distance < min_distance and distance < 100:  # Threshold for matching
                min_distance = distance
                matched_id = temp_id

        # If no match found, assign a new ID
        if matched_id is None:
            matched_id = f"face_{next_temp_id}"
            next_temp_id += 1

        # Update tracking and data
        new_face_tracking[matched_id] = (center_x, center_y)
        if matched_id in face_initial_positions:
            new_initial_positions[matched_id] = face_initial_positions[matched_id]
            new_ready_to_capture[matched_id] = face_ready_to_capture.get(matched_id, False)
        new_face_data[matched_id] = last_face_data.get(matched_id, (None, None, x1, y1, x2, y2))
        if matched_id in captured_faces:
            new_captured_faces[matched_id] = captured_faces[matched_id]

    # Update global tracking dictionaries
    face_tracking.clear()
    face_tracking.update(new_face_tracking)
    face_initial_positions.clear()
    face_initial_positions.update(new_initial_positions)
    face_ready_to_capture.clear()
    face_ready_to_capture.update(new_ready_to_capture)
    last_face_data.clear()
    last_face_data.update(new_face_data)
    captured_faces.clear()
    captured_faces.update(new_captured_faces)

    return list(new_face_tracking.keys())

# === Check if a face is within the capture range ===
def is_within_range(temp_id, new_position):
    """
    Check if the face stays within a certain range from its initial position.
    Returns True if within range, False otherwise.
    """
    # If this is the first detection, set the initial position
    if temp_id not in face_initial_positions:
        face_initial_positions[temp_id] = new_position
        return False  # Need at least two detections to compare

    # Get the initial position
    init_pos = face_initial_positions[temp_id]
    init_center = ((init_pos[0] + init_pos[2]) / 2, (init_pos[1] + init_pos[3]) / 2)

    # Calculate the center of the current position
    curr_center = ((new_position[0] + new_position[2]) / 2, (new_position[1] + new_position[3]) / 2)

    # Calculate movement from the initial position
    movement = np.sqrt((curr_center[0] - init_center[0])**2 + (curr_center[1] - init_center[1])**2)

    # If the movement is within the threshold, return True
    if movement < RANGE_THRESHOLD:
        return True
    else:
        # If the face moves out of range, reset the initial position
        face_initial_positions[temp_id] = new_position
        return False

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

        boxes = results[0].boxes.xyxy.cpu().numpy() if results[0].boxes.xyxy is not None else []
        print(f"Detected {len(boxes)} faces")  # Debug: Check if faces are detected

        # Match current faces to previous faces
        temp_ids = match_faces(boxes)

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box[:4])
            face_img = frame[y1:y2, x1:x2]
            gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)

            # Get the corresponding temp_id
            temp_id = temp_ids[i] if i < len(temp_ids) else f"face_{next_temp_id}"

            # Update face data
            last_face_data[temp_id] = (face_img, gray_face, x1, y1, x2, y2)

            # Check if the face has already been captured
            if temp_id in captured_faces:
                matched_label, emotion = captured_faces[temp_id]
                print(f"Displaying captured face {temp_id}: {matched_label} | {emotion}")  # Debug: Confirm label display
            else:
                # Check if the face is within the capture range
                if is_within_range(temp_id, (x1, y1, x2, y2)):
                    face_ready_to_capture[temp_id] = True
                    matched_label = "Ready"  # Indicate the face is ready to capture
                    emotion = "N/A"
                else:
                    face_ready_to_capture[temp_id] = False
                    matched_label = "Moving"
                    emotion = "N/A"

            # Store detection as a list instead of a tuple
            current_detections.append([x1, y1, x2, y2, matched_label, emotion, temp_id])

        last_detections = current_detections  # Update the last detection results

    # Check for keypress to capture
    key = cv2.waitKey(1)
    if key == ord('c'):  # Press 'c' to capture faces that are ready
        print(f"Attempting to capture faces. Ready faces: {face_ready_to_capture}")  # Debug: Check ready faces
        for temp_id in list(face_ready_to_capture.keys()):
            if face_ready_to_capture.get(temp_id, False) and temp_id not in captured_faces:
                face_img, gray_face, x1, y1, x2, y2 = last_face_data.get(temp_id, (None, None, None, None, None, None))
                if face_img is None or gray_face is None:
                    print(f"No face data for {temp_id}, skipping capture")  # Debug: Check face data
                    continue  # Skip if no face data is available

                wearing_mask = is_wearing_mask(face_img)
                new_embedding = extract_embedding(gray_face, wearing_mask)

                matched_label = "Unknown"
                for name, stored_embeddings in reference_faces.items():
                    if compare_faces_cosine(new_embedding, stored_embeddings):
                        matched_label = name
                        break

                if matched_label == "Unknown":
                    label = f"user{temp_id.split('_')[1]}.0_{datetime.now().strftime('%H%M%S')}"
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

                captured_faces[temp_id] = (matched_label, emotion)  # Store the label and emotion
                print(f"Captured face {temp_id} with label: {matched_label} | Emotion: {emotion}")  # Debug: Confirm capture

                # Update the detection label to reflect the captured state
                for detection in last_detections:
                    if detection[6] == temp_id:  # Match by temp_id
                        detection[4] = matched_label  # Update label
                        detection[5] = emotion  # Update emotion

    # Use the last detection results for drawing
    for detection in last_detections:
        x1, y1, x2, y2, matched_label, emotion, _ = detection
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), LINE_THICKNESS)
        cv2.putText(frame, f"{matched_label} | {emotion}", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), LINE_THICKNESS)

    cv2.imshow("YOLO + Emotion + Auto-folder", frame)
    if key == 27:  # Press ESC to exit
        break

cap.release()
cv2.destroyAllWindows()