import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model
from datetime import datetime

# Load the emotion recognition model
emotion_model_path = "/Users/sam/Documents/GitHub/fyp/Test/FER/model.h5"
emotion_model = load_model(emotion_model_path)

# Emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def plot_rectangle(image, faces, confidences, labels=None):
    for i in range(len(faces)):
        x1, y1, x2, y2 = faces[i]
        confidence = confidences[i]
        label = labels[i] if labels else f"Confidence: {confidence:.2f}"
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return image

def compare_faces(reference_face, new_face):
    if reference_face.shape != new_face.shape:
        new_face = cv2.resize(new_face, (reference_face.shape[1], reference_face.shape[0]))
    difference = cv2.absdiff(reference_face, new_face)
    return np.mean(difference) < 45

def preprocess_image(face_image, target_size=(48, 48)):
    face_image = cv2.resize(face_image, target_size)
    face_image = np.stack([face_image] * 3, axis=-1)
    face_image = face_image.astype('float32') / 255.0
    face_image = np.expand_dims(face_image, axis=0)
    return face_image

def save_new_face(face_image, directory_path):
    """儲存新的人臉到本地儲存"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"unknown_{timestamp}.jpg"
    filepath = os.path.join(directory_path, filename)
    cv2.imwrite(filepath, face_image)
    return filename

def detect_and_classify_faces(frame, face_net, reference_faces, storage_dir):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, scalefactor=1.0, size=(300, 300),
                                mean=(104.0, 177.0, 123.0), swapRB=False, crop=False)
    face_net.setInput(blob)
    detections = face_net.forward()

    faces = []
    confidences = []
    labels = []
    emotion_counter = {label: 0 for label in emotion_labels}
    new_faces = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * [w, h, w, h]
            (x1, y1, x2, y2) = box.astype("int")
            face = frame[y1:y2, x1:x2]

            if face.size == 0:
                continue

            gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            processed_face = preprocess_image(gray_face)
            predictions = emotion_model.predict(processed_face)
            emotion_index = np.argmax(predictions)
            emotion_label = emotion_labels[emotion_index]
            emotion_confidence = np.max(predictions)

            emotion_counter[emotion_label] += 1

            faces.append((x1, y1, x2, y2))
            confidences.append(confidence)
            
            matched = False
            for label, reference_face in reference_faces.items():
                if compare_faces(reference_face, gray_face):
                    labels.append(f"{label} | {emotion_label} ({emotion_confidence:.2f})")
                    matched = True
                    break
            
            if not matched:
                # 未匹配的人臉：儲存並添加到參考列表
                filename = save_new_face(face, storage_dir)
                reference_faces[filename.split('.')[0]] = gray_face
                labels.append(f"New Face Saved | {emotion_label} ({emotion_confidence:.2f})")
                new_faces.append(face)

    return faces, confidences, labels, emotion_counter, new_faces

def preprocess_face(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image {image_path}")
        return None
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade_path = "/Users/sam/Documents/GitHub/fyp/Test/haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) > 0:
        x, y, w, h = faces[0]
        return gray_image[y:y+h, x:x+w]
    print("No face detected.")
    return None

def load_reference_faces(directory_path):
    reference_faces = {}
    for file_name in os.listdir(directory_path):
        file_path = os.path.join(directory_path, file_name)
        face = preprocess_face(file_path)
        if face is not None:
            label = file_name.split('.')[0]
            reference_faces[label] = face
    return reference_faces

def main():
    # Load pre-trained DNN model
    model_path = "/Users/sam/Documents/GitHub/fyp/Test/DNN/res10_300x300_ssd_iter_140000.caffemodel"
    config_path = "/Users/sam/Documents/GitHub/fyp/Test/DNN/deploy.prototxt"
    face_net = cv2.dnn.readNetFromCaffe(config_path, model_path)

    # Load reference faces from a directory
    reference_faces_dir = "/Users/sam/Documents/GitHub/fyp/Test/known_faces/"
    reference_faces = load_reference_faces(reference_faces_dir)

    if not os.path.exists(reference_faces_dir):
        os.makedirs(reference_faces_dir)

    if not reference_faces:
        print("No reference faces loaded. Will start capturing new faces.")

    # Open camera
    capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        print("Error: Camera is not opened normally")
        return

    window_name = "Face Recognition and Storage System"

    while True:
        ret, frame = capture.read()
        if not ret:
            break

        # Detect faces and handle new face storage
        faces, confidences, labels, emotion_counter, new_faces = detect_and_classify_faces(
            frame, face_net, reference_faces, reference_faces_dir)

        # Draw rectangles and labels
        dets_image = plot_rectangle(frame, faces, confidences, labels)

        # Display people count
        cv2.putText(dets_image, f"People in Frame: {len(faces)}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
        cv2.putText(dets_image, f"People in Frame: {len(faces)}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Display emotion counter
        y_offset = 60
        for emotion, count in emotion_counter.items():
            cv2.putText(dets_image, f"{emotion}: {count}", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
            cv2.putText(dets_image, f"{emotion}: {count}", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y_offset += 30

        # Display the frame
        cv2.imshow(window_name, dets_image)

        # Exit on ESC key
        if cv2.waitKey(1) == 27:
            break

    capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()