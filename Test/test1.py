# import libraries
import cv2
import dlib

#loading face detector
def plot_rectangle(image, faces) :
    for face in faces:
        cv2.rectangle(image, (face.left(), face.top()), (face.right(), face.bottom()), (255,0,0), 4)
    return image

def main():
    # open camera
    capture = cv2.VideoCapture(0)

    # Judage if camera is opened normally
    if not capture.isOpened():
        print("Error: Camera is not opened normally")
        return

    # Create a named window and set fixed size
    window_name = "Frame"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    fixed_width, fixed_height = 640, 480  # Set desired dimensions
    cv2.resizeWindow(window_name, fixed_width, fixed_height)
    
    # Looping if camrea is opened narmally
    while True:
        # Read the frame
        ret, frame = capture.read()

        # Check if the frame is read normally
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #BGR to Gray

        # Load dlib detector
        detector = dlib.get_frontal_face_detector()
        detector_result = detector(gray, 0)  # `2` means upsample the image 2 times


        # check the frame
        dets_image = plot_rectangle(frame, detector_result)

        # print the frame
        cv2.imshow("Frame", dets_image)

        # Exit the loop when buttens are pressed
        if cv2.waitKey(1) == 27:
            break

    # Release all resources
    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()