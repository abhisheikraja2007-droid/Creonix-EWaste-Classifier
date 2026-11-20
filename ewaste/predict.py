import os
import sys
import json
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# --- 1. Define Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'ewaste_cnn.h5')
CLASS_PATH = os.path.join(MODEL_DIR, 'class_indices.json')
IMG_SIZE = (128, 128)

# --- 2. Load Model and Class Mappings ---
try:
    model = load_model(MODEL_PATH)
    with open(CLASS_PATH, 'r') as f:
        class_indices = json.load(f)
    inv_class_map = {v: k for k, v in class_indices.items()}
    print("Model and class indices loaded successfully.")
except (OSError, FileNotFoundError) as e:
    print(f"Error: Could not load model or class indices. Have you trained the model yet? Details: {e}")
    sys.exit(1)

# --- 3. Prediction Function ---
def predict(image_path):
    """
    Makes a prediction on a single image.
    Args:
        image_path (str): The file path of the image to predict.
    Returns:
        tuple: A tuple containing the predicted label (str) and confidence (float).
    """
    try:
        img = load_img(image_path, target_size=IMG_SIZE)
        arr = img_to_array(img) / 255.0
        arr = np.expand_dims(arr, axis=0) # Add a batch dimension
        
        preds = model.predict(arr, verbose=0)
        idx = np.argmax(preds)
        label = inv_class_map[idx]
        conf = np.max(preds)
        
        return label, conf
    except Exception as e:
        print(f"Prediction failed for {image_path}. Error: {e}")
        return None, None

# --- 4. Command-Line Interface (CLI) ---
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path> [true_label]")
        sys.exit(1)

    image_path = sys.argv[1]
    true_label = sys.argv[2] if len(sys.argv) > 2 else None

    if not os.path.exists(image_path):
        print(f"Image not found at: {image_path}")
        sys.exit(1)

    label, conf = predict(image_path)

    if label and conf is not None:
        print(f"Prediction: {label} (confidence: {conf*100:.2f}%)")
        if true_label:
            print(f"True label: {true_label}")
            if label.lower() == true_label.lower():
                print("Prediction is CORRECT.")
            else:
                print("Prediction is INCORRECT.")
        if conf < 0.7:
            print("Warning: Low confidence in prediction. Consider using a clearer image or retraining the model.")
        # Print all class probabilities for debugging
        img = load_img(image_path, target_size=IMG_SIZE)
        arr = img_to_array(img) / 255.0
        arr = np.expand_dims(arr, axis=0)
        preds = model.predict(arr, verbose=0)[0]
        print("Class probabilities:")
        for idx, prob in enumerate(preds):
            cname = inv_class_map.get(idx, str(idx))
            print(f"  {cname}: {prob*100:.2f}%")
    else:
        print("Prediction failed.")