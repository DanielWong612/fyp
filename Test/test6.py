import cv2
import tensorflow as tf
import numpy as np

# Load the TensorFlow model
model = tf.keras.models.load_model("/Users/sam/Documents/GitHub/fyp/22067588d_additional/new_finalcustomedVGG16_model.h5")
model.load_weights("/Users/sam/Documents/GitHub/fyp/22067588d_additional/new_finalVGG16_weight.weights.h5")

# Define image preprocessing function
def preprocess_image(frame, target_size):
    """
    Preprocess the input frame: resize, normalize
    """
    # Resize to target size
    frame = cv2.resize(frame, target_size)
    # Normalize to [0, 1]
    frame = frame / 255.0
    # Add batch dimension
    frame = np.expand_dims(frame, axis=0)
    return frame

# Function to classify a cropped face
def classify_face(face_image):
    """Classify the given face image using the loaded model."""
    target_size = (64, 64)  # Update as per model input
    processed_image = preprocess_image(face_image, target_size)
    predictions = model.predict(processed_image)
    predicted_class = np.argmax(predictions)
    confidence = np.max(predictions)
    return predicted_class, confidence

# Draw bounding boxes and classifications
def plot_rectangle_with_classification(image, faces, confidences):
    class_labels = {
        0: "angry",
        1: "disgust",
        2: "fear",
        3: "happy",
        4: "neutral",
        5: "sad",
        6: "surprise"
    }

    for i in range(min(len(faces), len(confidences))):
        x1, y1, x2, y2 = faces[i]
        confidence = confidences[i]
        # Extract face region for classification
        face_image = image[y1:y2, x1:x2]
        try:
            predicted_class, class_confidence = classify_face(face_image)
            label = f"{class_labels.get(predicted_class, 'Unknown')}, Conf: {class_confidence:.2f}"
        except Exception:
            label = "Classification Failed"
        # Draw rectangle and label
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(image, f"{label}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return image

def main():
    total_faces_detected = 0  # Counter for total faces detected
    # Load pre-trained DNN model
    model_path = "/Users/sam/Documents/GitHub/fyp/Test/DNN/res10_300x300_ssd_iter_140000.caffemodel"
    config_path = "/Users/sam/Documents/GitHub/fyp/Test/DNN/deploy.prototxt"    
    net = cv2.dnn.readNetFromCaffe(config_path, model_path)

    # Open camera
    capture = cv2.VideoCapture(0)

    if not capture.isOpened():
        print("Error: Camera is not opened")
        return
    
    # Create a named window
    window_name = "Classroom Analysis System"

    # set fixed size
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    fixed_width, fixed_height = 1370, 850 # Set desired dimensions
    cv2.resizeWindow(window_name, fixed_width, fixed_height)

    while True:
        ret, frame = capture.read()
        if not ret:
            break

        sub_frames = []  # List to hold sub-frames for face detection
        height, width = frame.shape[:2]
        sub_regions = [
            (0, height // 2, 0, width // 2),  # Top-left
            (0, height // 2, width // 2, width),  # Top-right
            (height // 2, height, 0, width // 2),  # Bottom-left
            (height // 2, height, width // 2, width)  # Bottom-right
        ]

        for region in sub_regions:
            y_start, y_end, x_start, x_end = region
            sub_frame = frame[y_start:y_end, x_start:x_end]
            scaled_frame = cv2.resize(sub_frame, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)  # Scale sub-frame
            sub_frames.append((scaled_frame, (x_start, y_start)))  # Sub-frame and top-left corner coordinates
        ret, frame = capture.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, scalefactor=1.0, size=(300, 300),
                                     mean=(104.0, 177.0, 123.0), swapRB=False, crop=False)

        net.setInput(blob)
        detections = net.forward()

        faces = []
        confidences = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:  # Confidence threshold
                box = detections[0, 0, i, 3:7] * [w, h, w, h]
                (x1, y1, x2, y2) = box.astype("int")
                faces.append((x1, y1, x2, y2))
                confidences.append(confidence)
        total_faces_detected = len(faces)
        confidences = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:  # Confidence threshold
                box = detections[0, 0, i, 3:7] * [w, h, w, h]
                (x1, y1, x2, y2) = box.astype("int")
                faces.append((x1, y1, x2, y2))
                confidences.append(confidence)

        # Draw rectangles and classify detected faces
        dets_image = plot_rectangle_with_classification(frame, faces, confidences)

        # Display the frame
        cv2.putText(dets_image, f"Total Faces: {total_faces_detected}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.imshow(window_name, dets_image)

        if cv2.waitKey(1) == 27:  # Exit on ESC key
            break

    capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
