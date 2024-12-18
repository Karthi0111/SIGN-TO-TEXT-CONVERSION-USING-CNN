import numpy as np
import cv2
from tensorflow.keras.models import load_model
from utils.config import model_path  # Ensure model_path is correct

from googletrans import Translator

# Load the trained model
model_path = 'C:\project\model\model.h5'  # Update this path to your model file
model = load_model(model_path)
translator = Translator()

# Initialize the camera (optional, remove if you are not using the camera)
cap = cv2.VideoCapture(0)

def predict_and_translate():
    # Capture a single frame from the camera
    ret, frame = cap.read()
    if not ret:
        return "Error capturing image"  # Return error if frame capture fails

    # Resize the frame only for model prediction
    img = cv2.resize(frame, (64, 64))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Get prediction
    prediction = model.predict(img)
    predicted_letter = chr(np.argmax(prediction) + ord('A'))

    # Translate the predicted letter
    cached_tamil = translator.translate(predicted_letter, dest='ta').text
    cached_hindi = translator.translate(predicted_letter, dest='hi').text

    # Return a formatted response
    return {
        'predicted_letter': predicted_letter,
        'tamil': cached_tamil,
        'hindi': cached_hindi
    }

# Note: Don't run the camera here; it should be handled in app.py

# Cleanup function (to release the camera)
def cleanup():
    cap.release()
