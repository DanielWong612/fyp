import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the pre-trained emotion recognition model
emotion_model_path = "recognitionDemo/FER/model.h5"
emotion_model = load_model(emotion_model_path)
print("Emotion model input shape:", emotion_model.input_shape)  # Print model input shape for verification

# Emotion labels for the emotion recognition model
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Load DNN face detection model
face_model_path = "recognitionDemo/DNN/res10_300x300_ssd_iter_140000.caffemodel"
face_config_path = "recognitionDemo/DNN/deploy.prototxt"
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
    emotion_counter = {label: 0 for label in emotion_labels}  # Initialize emotion counter
    unique_faces = set()  # Track unique faces to avoid duplicates

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

            # Track unique faces using a simple hash of the bounding box coordinates
            face_hash = f"{x1}_{y1}_{x2}_{y2}"
            if face_hash not in unique_faces:
                unique_faces.add(face_hash)
                # Increment the emotion counter
                if emotion_label in emotion_counter:
                    emotion_counter[emotion_label] += 1

            # Add face data for drawing later
            faces_data.append(((x1, y1, x2, y2), f"{emotion_label}: {confidence:.2f}"))

    return frame, face_count, faces_data, emotion_counter, len(unique_faces)

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

        # Detect and classify faces, count total faces, and get emotion counter
        output_frame, face_count, faces_data, emotion_counter = detect_and_classify_faces(frame)

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

        # Display the emotion counter in the top-right corner
        text_x = w - 150  # Position 150 pixels from the right edge
        text_y = 30  # Start 30 pixels from the top
        line_spacing = 20  # Space between lines

        # Calculate the dimensions of the counter area
        counter_height = (len(emotion_labels) + 2) * line_spacing + 10  # +2 for "Total Students" and "Total Faces in Frame"
        counter_width = 140  # Width of the counter area
        counter_x = w - 160  # Slightly more padding on the left
        counter_y = 20  # Slightly more padding on the top

        # Draw a semi-transparent white rectangle as the background
        overlay = output_frame.copy()
        cv2.rectangle(overlay, (counter_x, counter_y), 
                      (counter_x + counter_width, counter_y + counter_height), 
                      (255, 255, 255), -1)  # White rectangle
        alpha = 0.7  # Transparency factor
        cv2.addWeighted(overlay, alpha, output_frame, 1 - alpha, 0, output_frame)

        # Display "Total Students" and "Total Faces in Frame"
        total_faces_text = f"Total Faces: {face_count}"
        text_y += line_spacing
        cv2.putText(output_frame, total_faces_text, (text_x, text_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        text_y += line_spacing + 10  # Add extra spacing after counts

        # Display all emotions
        for i, (emotion, count) in enumerate(emotion_counter.items()):
            text = f"{emotion}: {count}"
            cv2.putText(output_frame, text, (text_x, text_y + i * line_spacing), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        # Display the frame
        cv2.imshow(window_name, output_frame)

        # Exit on ESC key
        if cv2.waitKey(1) == 27:
            break

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()