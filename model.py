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
