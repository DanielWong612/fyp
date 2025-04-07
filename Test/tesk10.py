import os
import cv2
import numpy as np

# Directory to store known faces
known_faces_dir = "./known_faces"
os.makedirs(known_faces_dir, exist_ok=True)

# Directory containing face captures
face_captures_dir = "./face_captures"

# Path to the Haar cascade file
face_cascade_path = "/Users/sam/Documents/GitHub/fyp/Test/haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(face_cascade_path)

# Helper to compare faces
def match_face(new_face, known_faces_dir):
    """
    Match a new face against known faces.
    Returns the file name of the matched face, or None if no match is found.
    """
    for file_name in os.listdir(known_faces_dir):
        file_path = os.path.join(known_faces_dir, file_name)
        known_face = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if known_face is None:
            continue
        # Resize known face to match the dimensions of the new face
        known_face_resized = cv2.resize(known_face, (new_face.shape[1], new_face.shape[0]))
        # Calculate the difference
        difference = cv2.absdiff(known_face_resized, new_face)
        if np.mean(difference) < 20:  # Threshold for similarity
            return file_name
    return None

def label_faces(image_path):
    """
    Detect faces in an image, match against known faces, and label or save new faces.
    """
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Image not found - {image_path}")
        return
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for i, (x, y, w, h) in enumerate(faces):
        face = gray_image[y:y+h, x:x+w]
        matched_face = match_face(face, known_faces_dir)

        if matched_face:
            label = os.path.splitext(matched_face)[0]  # Use the file name (without extension) as label
        else:
            # Assign a new label for unmatched face
            label = f"face_{len(os.listdir(known_faces_dir)) + 1}"
            save_path = os.path.join(known_faces_dir, f"{label}.jpg")
            cv2.imwrite(save_path, face)
            print(f"New face saved as {label}.jpg")

        # Draw rectangle and label on the original image
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # Display the labeled image
    cv2.imshow("Labeled Faces", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Process all images in the face captures directory
for file_name in os.listdir(face_captures_dir):
    file_path = os.path.join(face_captures_dir, file_name)
    print(f"Processing: {file_name}")
    label_faces(file_path)
