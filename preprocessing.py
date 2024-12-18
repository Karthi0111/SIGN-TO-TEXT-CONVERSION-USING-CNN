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
