import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the pre-trained emotion recognition model
emotion_model_path = "/Users/sam/Documents/GitHub/fyp/Test/FER/model.h5"
emotion_model = load_model(emotion_model_path)

# Emotion labels for the emotion recognition model
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Load DNN face detection model
face_model_path = "/Users/sam/Documents/GitHub/fyp/Test/DNN/res10_300x300_ssd_iter_140000.caffemodel"
face_config_path = "/Users/sam/Documents/GitHub/fyp/Test/DNN/deploy.prototxt"
face_net = cv2.dnn.readNetFromCaffe(face_config_path, face_model_path)

# Directory for storing face captures
captures_directory = "./face_captures"
os.makedirs(captures_directory, exist_ok=True)

# Helper to compare faces
def face_already_captured(new_face, captures_directory):
    for file_name in os.listdir(captures_directory):
        file_path = os.path.join(captures_directory, file_name)
        existing_face = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if existing_face is None:
            continue
        # Resize existing face to match the new face dimensions
        existing_face_resized = cv2.resize(existing_face, (new_face.shape[1], new_face.shape[0]))
        
        # Ensure both are grayscale
        if len(new_face.shape) != 2 or len(existing_face_resized.shape) != 2:
            continue
        
        # Calculate the difference
        difference = cv2.absdiff(existing_face_resized, new_face)
        if np.mean(difference) < 20:  # Threshold for similarity
            return file_name
    return None


# Detect faces and classify emotions
def detect_and_classify_faces(frame, face_index):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, scalefactor=1.0, size=(300, 300),
                                 mean=(104.0, 177.0, 123.0), swapRB=False, crop=False)
    face_net.setInput(blob)
    detections = face_net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:  # Confidence threshold
            box = detections[0, 0, i, 3:7] * [w, h, w, h]
            (x1, y1, x2, y2) = box.astype("int")
            face = frame[y1:y2, x1:x2]

            if face.size == 0:
                continue

            # Convert face to grayscale for saving
            gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

            # Check if face has already been captured
            existing_label = face_already_captured(gray_face, captures_directory)
            if existing_label is None:
                # Save the new face
                face_name = f"face_{face_index}.jpg"
                save_path = os.path.join(captures_directory, face_name)
                cv2.imwrite(save_path, gray_face)
                existing_label = face_name
                face_index += 1

            # Use the existing label for this face
            label_name = os.path.splitext(existing_label)[0]

            # Predict emotion
            face_resized = cv2.resize(gray_face, (48, 48))
            processed_face = np.stack([face_resized] * 3, axis=-1).astype('float32') / 255.0
            processed_face = np.expand_dims(processed_face, axis=0)
            predictions = emotion_model.predict(processed_face)
            emotion_index = np.argmax(predictions)
            emotion_label = emotion_labels[emotion_index]

            # Draw rectangle and label
            label = f"{label_name} - {emotion_label}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    return frame, face_index

# Main function for video processing
def main():
    video_capture = cv2.VideoCapture(0)  # Open webcam

    if not video_capture.isOpened():
        print("Error: Camera not opened.")
        return

    face_index = 0  # Initialize face index

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        # Detect and classify faces
        output_frame, face_index = detect_and_classify_faces(frame, face_index)

        # Display the frame
        cv2.imshow("Emotion Recognition with Face Management", output_frame)

        # Exit on ESC key
        if cv2.waitKey(1) == 27:
            break

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
