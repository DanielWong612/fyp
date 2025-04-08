import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the pre-trained emotion recognition model
emotion_model_path = "/Users/sam/Documents/GitHub/fyp/Test/FER/model.h5"
emotion_model = load_model(emotion_model_path)
print("Emotion model input shape:", emotion_model.input_shape)  # Print model input shape for verification

# Emotion labels for the emotion recognition model
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Load DNN face detection model
face_model_path = "/Users/sam/Documents/GitHub/fyp/Test/DNN/res10_300x300_ssd_iter_140000.caffemodel"
face_config_path = "/Users/sam/Documents/GitHub/fyp/Test/DNN/deploy.prototxt"
face_net = cv2.dnn.readNetFromCaffe(face_config_path, face_model_path)

# Preprocess image for emotion recognition
def preprocess_image(face_image, target_size=(48, 48)):
    """
    Preprocess the image for the emotion recognition model:
    - Converts to grayscale
    - Resizes to target size
    - Converts grayscale to RGB
    - Normalizes pixel values
    """
    face_image = cv2.resize(face_image, target_size)  # Resize to 48x48
    face_image = np.stack([face_image] * 3, axis=-1)  # Convert grayscale to RGB
    face_image = face_image.astype('float32') / 255.0  # Normalize to [0, 1]
    face_image = np.expand_dims(face_image, axis=0)  # Add batch dimension
    return face_image

# Detect faces and classify emotions
def detect_and_classify_faces(frame):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, scalefactor=1.0, size=(300, 300),
                                 mean=(104.0, 177.0, 123.0), swapRB=False, crop=False)
    face_net.setInput(blob)
    detections = face_net.forward()

    face_count = 0  # Counter for detected faces
    faces_data = []  # Store face rectangles and labels

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:  # Confidence threshold
            face_count += 1
            box = detections[0, 0, i, 3:7] * [w, h, w, h]
            (x1, y1, x2, y2) = box.astype("int")
            face = frame[y1:y2, x1:x2]

            if face.size == 0:
                continue

            # Preprocess the face and predict emotion
            gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
            processed_face = preprocess_image(gray_face)
            predictions = emotion_model.predict(processed_face, verbose=0)  # Set verbose=0 to suppress progress bar
            emotion_index = np.argmax(predictions)
            emotion_label = emotion_labels[emotion_index]
            confidence = np.max(predictions)

            # Add face data for drawing later
            faces_data.append(((x1, y1, x2, y2), f"{emotion_label}: {confidence:.2f}"))

    return frame, face_count, faces_data

# Main function for video processing
def main():
    video_capture = cv2.VideoCapture(0)  # Open webcam

    if not video_capture.isOpened():
        print("Error: Camera not opened.")
        return

    # Create a named window
    window_name = "Emotion Recognition"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)  # Window is resizable, fixed size setting removed

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        # Detect and classify faces, count total faces
        output_frame, face_count, faces_data = detect_and_classify_faces(frame)

        # Draw rectangles and labels for each detected face
        for (x1, y1, x2, y2), label in faces_data:
            cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Dynamically adjust label position
            if y1 - 10 > 10:  # If there is enough space above
                cv2.putText(output_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            else:  # Otherwise, display below
                cv2.putText(output_frame, label, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # Display total face count at the bottom of the frame
        h, w = output_frame.shape[:2]
        cv2.putText(output_frame, f"Total Faces: {face_count}", (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        # Display the frame
        cv2.imshow(window_name, output_frame)

        # Exit on ESC key
        if cv2.waitKey(1) == 27:
            break

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()