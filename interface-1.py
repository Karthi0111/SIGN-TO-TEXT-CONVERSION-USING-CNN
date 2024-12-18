# interface.py
import numpy as np
import cv2
import tkinter as tk
from tensorflow.keras.models import load_model
from googletrans import Translator
from PIL import Image, ImageTk  # For better image handling in tkinter
from utils.config import model_path  # Ensure model_path is correct

# Load the trained model
model = load_model(model_path)
translator = Translator()

# Initialize the camera
cap = cv2.VideoCapture(0)

# Variables to cache the current prediction and translations
current_letter = None
cached_tamil = None
cached_hindi = None

def predict_and_translate():
    global current_letter, cached_tamil, cached_hindi
    
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        window.after(5, predict_and_translate)
        return

    # Resize the frame only for model prediction
    img = cv2.resize(frame, (64, 64))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Get prediction and check if it has changed
    prediction = model.predict(img)
    predicted_letter = chr(np.argmax(prediction) + ord('A'))

    # Only translate if the prediction changed
    if predicted_letter != current_letter:
        current_letter = predicted_letter
        english_text.set(f"Prediction: {predicted_letter}")
        cached_tamil = translator.translate(predicted_letter, dest='ta').text
        cached_hindi = translator.translate(predicted_letter, dest='hi').text
        tamil_text.set(f"Tamil: {cached_tamil}")
        hindi_text.set(f"Hindi: {cached_hindi}")
    
    # Convert BGR to RGB for tkinter display
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_pil = Image.fromarray(frame_rgb)
    img_tk = ImageTk.PhotoImage(image=frame_pil)

    # Update the label with the new image
    video_label.config(image=img_tk)
    video_label.image = img_tk

    # Repeat function with reduced interval for better refresh rate
    window.after(1, predict_and_translate)

# Initialize the GUI window
window = tk.Tk()
window.title("Sign Language to Text with Translation")
window.geometry("800x600")

# Create video display and labels
video_label = tk.Label(window)
video_label.pack()

# Define text variables for displaying predictions and translations
english_text = tk.StringVar()
tamil_text = tk.StringVar()
hindi_text = tk.StringVar()

# Prediction and translation labels
tk.Label(window, textvariable=english_text, font=("Arial", 16)).pack()
tk.Label(window, textvariable=tamil_text, font=("Arial", 16), fg="blue").pack()
tk.Label(window, textvariable=hindi_text, font=("Arial", 16), fg="green").pack()

# Run the prediction and translation function
predict_and_translate()

# Run the GUI loop
window.mainloop()

# Release resources on close
cap.release()
cv2.destroyAllWindows()