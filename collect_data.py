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
