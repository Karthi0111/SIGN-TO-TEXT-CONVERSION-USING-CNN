# real_time_prediction.py
import numpy as np
from tensorflow.keras.models import load_model
import cv2
from utils.config import model_path  # Load model path from config

model = load_model(model_path)

cap = cv2.VideoCapture(0)
def recognize_sign_language():
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        img = cv2.resize(frame, (64, 64))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)

        prediction = model.predict(img)
        predicted_letter = chr(np.argmax(prediction) + ord('A'))

        cv2.putText(frame, f"Prediction: {predicted_letter}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Sign Language to Text", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
