import cv2
import tensorflow as tf
import numpy as np

# 加載模型和權重
model = tf.keras.models.load_model("/Users/sam/Documents/GitHub/fyp/22067588d_additional/new_finalcustomedVGG16_model.h5")
model.load_weights("/Users/sam/Documents/GitHub/fyp/22067588d_additional/new_finalVGG16_weight.weights.h5")

# Preprocess the input frame
def preprocess_frame(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    frame = cv2.resize(frame, (64, 64))            # Resize to 64x64
    frame = frame / 255.0                          # Normalize to [0, 1]
    frame = np.expand_dims(frame, axis=0)          # Add batch dimension
    return frame

def main():
    # Open camera
    capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        print("Error: Camera is not opened")
        return

    # Looping if camera is opened normally  
    while True:
        ret, frame = capture.read()
        if not ret:
            break

        # Preprocess the frame
        input_frame = preprocess_frame(frame)

        # Predict
        predictions = model.predict(input_frame)
        label = np.argmax(predictions)  # Get the predicted class index

        # Display predictions
        cv2.putText(frame, f"Prediction: {label}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Live Prediction", frame)

        # Exit on ESC key
        if cv2.waitKey(1) == 27:
            break

    capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()