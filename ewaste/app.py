
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import json
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'ewaste_cnn.h5')
CLASS_PATH = os.path.join(MODEL_DIR, 'class_indices.json')
IMG_SIZE = (128, 128)

model = load_model(MODEL_PATH)
with open(CLASS_PATH, 'r') as f:
    class_indices = json.load(f)
inv_class_map = {v: k for k, v in class_indices.items()}


st.title("E-Waste Image Classifier")
st.write("Upload an image of e-waste. The model will predict the type and tell you if it should be recycled or scrapped. If the image is not e-waste, it will suggest to go to waste.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption="Uploaded Image", use_column_width=True)
    arr = img_to_array(img.resize(IMG_SIZE)) / 255.0
    arr = np.expand_dims(arr, axis=0)
    preds = model.predict(arr, verbose=0)[0]
    idx = np.argmax(preds)
    label = inv_class_map[idx]
    conf = np.max(preds)
    st.write(f"Prediction: **{label}**")
    st.write(f"Confidence: **{conf*100:.2f}%**")
    st.write("Class Probabilities:")
    for i, p in enumerate(preds):
        st.write(f"{inv_class_map[i]}: {p*100:.2f}%")

    # Decision logic
    ewaste_classes = set(inv_class_map.values())
    min_confidence = 0.5  # You can adjust this threshold
    if conf < min_confidence:
        st.warning("Non E-Waste detected or low confidence. Go to Waste ❌")
    else:
        st.success(f"This is a {label} and should be RECYCLED ♻️")
else:
    st.info("Please upload an image to get started.")


