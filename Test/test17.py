import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model

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
        # Draw rectangle around face
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Display confidence score or label
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return image

def compare_faces(reference_face, new_face):
    """
    Compare the reference face with a newly detected face.
    Returns True if matched, False otherwise.
    """
    if reference_face.shape != new_face.shape:
        new_face = cv2.resize(new_face, (reference_face.shape[1], reference_face.shape[0]))
    difference = cv2.absdiff(reference_face, new_face)
    """
    <50: High tolerance for facial expressions but higher false positives
    <40: Good accuracy with some limitations on exaggerated expressions(1)
    <35: Strict match for facial differentiation but sensitive to expressions
    """
    return np.mean(difference) < 45  # Adjusted threshold for better matching

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

def detect_and_classify_faces(frame, face_net, reference_faces):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, scalefactor=1.0, size=(300, 300),
                                 mean=(104.0, 177.0, 123.0), swapRB=False, crop=False)
    face_net.setInput(blob)
    detections = face_net.forward()

    faces = []
    confidences = []
    labels = []
    emotion_counter = {label: 0 for label in emotion_labels}

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * [w, h, w, h]
            (x1, y1, x2, y2) = box.astype("int")
            face = frame[y1:y2, x1:x2]

            if face.size == 0:
                continue

            # Preprocess and classify emotion
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
                labels.append(f"Not Matched | {emotion_label} ({emotion_confidence:.2f})")

    return faces, confidences, labels, emotion_counter

def get_label_from_filename(image_path):
    """
    Extract the label (name) from the filename.
    Assumes filenames are in the format <label>.<extension>.
    """
    return image_path.split('/')[-1].split('.')[0]

def preprocess_face(image_path):
    """
    Preprocess the reference image to extract the face.
    Returns the grayscale face image or None if no face is detected.
    """
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
    """
    Load all reference faces from a directory.
    Returns a dictionary with labels as keys and face images as values.
    """
    reference_faces = {}
    for file_name in os.listdir(directory_path):
        file_path = os.path.join(directory_path, file_name)
        face = preprocess_face(file_path)
        if face is not None:
            label = get_label_from_filename(file_path)
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

    if not reference_faces:
        print("No reference faces loaded.")
        return

    # Open camera
    capture = cv2.VideoCapture(0)

    # Check if camera opens successfully
    if not capture.isOpened():
        print("Error: Camera is not opened normally")
        return

    window_name = "A classroom analysis system"

    while True:
        # Read the frame
        ret, frame = capture.read()

        # Check if the frame is read successfully
        if not ret:
            break

        # Detect faces and classify emotions
        faces, confidences, labels, emotion_counter = detect_and_classify_faces(frame, face_net, reference_faces)

        # Draw rectangles around detected faces
        dets_image = plot_rectangle(frame, faces, confidences, labels)

        # Display the number of people in the frame
        cv2.putText(dets_image, f"People in Frame: {len(faces)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)  # Black outline
        cv2.putText(dets_image, f"People in Frame: {len(faces)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)  # Text color

        # Display emotion counter in the top right corner
        y_offset = 60
        for emotion, count in emotion_counter.items():
            # Black outline
            cv2.putText(dets_image, f"{emotion}: {count}", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
            # Text color
            cv2.putText(dets_image, f"{emotion}: {count}", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y_offset += 30
        # Display the frame
        cv2.imshow(window_name, dets_image)

        # Exit the loop when the escape key (Esc) is pressed
        if cv2.waitKey(1) == 27:
            break

    # Release all resources
    capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
