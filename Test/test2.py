# import libraries
import cv2

def plot_rectangle(image, faces, confidences):
    for i in range(len(faces)):
        x1, y1, x2, y2 = faces[i]
        confidence = confidences[i]
        # Draw rectangle around face
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        # Display confidence score
        cv2.putText(image, f"{confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    return image

def main():
    # Load pre-trained DNN model
    model_path = "/Users/sam/Documents/GitHub/fyp/Test/DNN/res10_300x300_ssd_iter_140000.caffemodel"
    config_path = "/Users/sam/Documents/GitHub/fyp/Test/DNN/deploy.prototxt"    
    net = cv2.dnn.readNetFromCaffe(config_path, model_path)

    # Open camera
    capture = cv2.VideoCapture(0)

    # Judge if camera is opened normally
    if not capture.isOpened():
        print("Error: Camera is not opened normally")
        return

    #Create a named window
    window_name = "A classroom analysis system"
    # Set fixed size
    # cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    # fixed_width, fixed_height = 640, 480  # Set desired dimensions
    # cv2.resizeWindow(window_name, fixed_width, fixed_height)

    # Looping if camera is opened normally
    while True:
        # Read the frame
        ret, frame = capture.read()

        # Check if the frame is read normally
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
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:  # Confidence threshold
                box = detections[0, 0, i, 3:7] * [w, h, w, h]
                (x1, y1, x2, y2) = box.astype("int")
                faces.append((x1, y1, x2, y2))
                confidences.append(confidence)

        # Draw rectangles around detected faces
        dets_image = plot_rectangle(frame, faces, confidences)

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
