import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the pre-trained model
model_path = "/Users/sam/Documents/GitHub/fyp/Test/FER/model.h5"  # Replace with the correct path
model = load_model(model_path)

# Define emotion labels (replace with actual labels from your dataset)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Preprocess the input image
def preprocess_image(image_path, target_size=(48, 48)):
    """
    Preprocesses the image to match the input requirements of the model.
    - Resizes the image
    - Converts to RGB if needed
    - Normalizes pixel values
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Read grayscale
    image = cv2.resize(image, target_size)  # Resize
    image = np.stack([image] * 3, axis=-1)  # Convert grayscale to RGB
    image = image.astype('float32') / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Predict emotion
def predict_emotion(image_path):
    image = preprocess_image(image_path)
    predictions = model.predict(image)
    predicted_label = emotion_labels[np.argmax(predictions)]
    return predicted_label

# Test the prediction
test_image_path = "/Users/sam/Documents/GitHub/fyp/Test/image.jpeg"  # Replace with the test image path
predicted_emotion = predict_emotion(test_image_path)
print(f"Predicted Emotion: {predicted_emotion}")
