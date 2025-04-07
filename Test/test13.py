import cv2
import numpy as np
import os

def plot_rectangle(image, faces, confidences, labels=None):
    for i in range(len(faces)):
        x1, y1, x2, y2 = faces[i]
        confidence = confidences[i]
        label = labels[i] if labels else f"Confidence: {confidence:.2f}"
        # Draw rectangle around face
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 255), 2)
        # Display confidence score or label
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
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
    <40: Good accuracy with some limitations on exaggerated expressions
    <35: Strict match for facial differentiation but sensitive to expressions
    """
    return np.mean(difference) < 40  # Adjusted threshold for better matching 

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
    net = cv2.dnn.readNetFromCaffe(config_path, model_path)

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

        # Prepare input blob for the DNN
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, scalefactor=1.0, size=(300, 300),
                                     mean=(104.0, 177.0, 123.0), swapRB=False, crop=False)

        # Perform forward pass to get face detections
        net.setInput(blob)
        detections = net.forward()

        faces = []
        confidences = []
        labels = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.7:  # Confidence threshold
                box = detections[0, 0, i, 3:7] * [w, h, w, h]
                (x1, y1, x2, y2) = box.astype("int")
                face = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
                faces.append((x1, y1, x2, y2))
                confidences.append(confidence)
                matched = False
                for label, reference_face in reference_faces.items():
                    if compare_faces(reference_face, face):
                        labels.append(label)
                        matched = True
                        break
                if not matched:
                    labels.append("Not Matched")

        # Draw rectangles around detected faces
        dets_image = plot_rectangle(frame, faces, confidences, labels)

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
