import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import json
import os
import pyttsx3

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'ewaste_cnn.h5')
CLASS_PATH = os.path.join(MODEL_DIR, 'class_indices.json')
IMG_SIZE = (128, 128)

# Load model and class indices
model = load_model(MODEL_PATH)
with open(CLASS_PATH, 'r') as f:
    class_indices = json.load(f)
inv_class_map = {v: k for k, v in class_indices.items()}

# Text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)
engine.setProperty('volume', 1.0)

# Open webcam (DroidCam USB)
cap = cv2.VideoCapture(1)
min_confidence = 0.6  
last_decision = ""

if not cap.isOpened():
    print("Cannot open camera. Make sure DroidCam is running.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Convert to gray for simple motion/object detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, thresh = cv2.threshold(blur, 60, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Only predict if there is a significant object
    if contours:
        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) > 3000:  # ignore tiny background
            x, y, w, h = cv2.boundingRect(c)
            obj_img = frame[y:y+h, x:x+w]

            img_resized = cv2.resize(obj_img, IMG_SIZE)
            arr = img_to_array(img_resized) / 255.0
            arr = np.expand_dims(arr, axis=0)

            preds = model.predict(arr, verbose=0)[0]
            idx = np.argmax(preds)
            label = inv_class_map[idx]
            conf = np.max(preds)

            # Decision
            if conf >= min_confidence:
                decision = f"Recycle ♻️ ({label})"
                color = (0, 255, 0)
            else:
                decision = "Go to Waste ❌"
                color = (0, 0, 255)
        else:
            decision = "No object detected - Go to Waste ❌"
            color = (0, 0, 255)
    else:
        decision = "No object detected - Go to Waste ❌"
        color = (0, 0, 255)

    # Display decision on frame
    cv2.putText(frame, decision, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

    # Speak decision only if changed
    if decision != last_decision:
        engine.say(decision)
        engine.runAndWait()
        last_decision = decision

    cv2.imshow("E-Waste Detector", frame)

    # Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
