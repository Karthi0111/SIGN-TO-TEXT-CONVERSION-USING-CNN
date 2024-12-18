#config
base_path = r"C:\project\data"
model_path=r'C:\project\model\model.h5'


#collect data
import cv2
import os

# Specify the directory path where you want to save the images
# For example, base_path = "C:/Users/YourUsername/Documents/sign_language_data"
base_path = r"C:\project\data"
os.makedirs(base_path, exist_ok=True)

def capture_data_for_alphabet(base_path):
    # Loop through each alphabet letter from A to Z
    for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        # Create a folder for each letter inside the base path
        letter_path = os.path.join(base_path, letter)
        os.makedirs(letter_path, exist_ok=True)

        cap = cv2.VideoCapture(0)  # Open the camera

        print(f"Collecting data for letter: {letter}")
        count = 0  # Image count for each letter
        while count < 400:  # Collect 200 images per alphabet
            ret, frame = cap.read()
            if not ret:
                break

            cv2.putText(frame, f"Collecting {letter} - Image {count + 1}/400", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            cv2.imshow("Capture", frame)
            
            # Save each frame to the specific letter folder
            cv2.imwrite(os.path.join(letter_path, f"{letter}_{count}.jpg"), frame)
            count += 1

            # Press 'q' to quit capturing for this letter
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    print("Data collection complete.")

#preprocessing
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(rescale=1.0/255, validation_split=0.2)

base_path=r"C:\project\data"

train_data = datagen.flow_from_directory(
    base_path, target_size=(64, 64), color_mode='rgb',
    class_mode='categorical', batch_size=32, subset='training'
)

val_data = datagen.flow_from_directory(
    base_path, target_size=(64, 64), color_mode='rgb',
    class_mode='categorical', batch_size=32, subset='validation'
)

#model
# model_training.py
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from utils.config import base_path, model_path  # Import paths

# Load data using ImageDataGenerator
datagen = ImageDataGenerator(rescale=1.0/255, validation_split=0.2)
train_data = datagen.flow_from_directory(
    base_path, target_size=(64, 64), color_mode='rgb',
    class_mode='categorical', batch_size=32, subset='training'
)
val_data = datagen.flow_from_directory(
    base_path, target_size=(64, 64), color_mode='rgb',
    class_mode='categorical', batch_size=32, subset='validation'
)

model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),  # Dropout for regularization
        Dense(26, activation='softmax')
    ])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model and capture the history
history = model.fit(train_data, validation_data=val_data, epochs=50)

# Save the model
model.save(model_path)
model.summary()

# Plotting the accuracy
plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()
plt.show()

print("Model training complete and saved.")


#interface
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from utils.config import model_path  # Ensure model_path is correct

from googletrans import Translator

# Load the trained model
model_path = 'path_to_your_model.h5'  # Update this path to your model file
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

