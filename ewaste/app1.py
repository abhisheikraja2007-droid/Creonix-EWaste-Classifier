import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# App title
st.title("E-Waste Classifier")
st.write("Upload an e-waste image to see if it should be Recycled or Scrapped.")

# Load model
@st.cache_resource  # cache model so it doesn’t reload every run
def load_cnn_model():
    return load_model('ewaste_cnn.h5')

model = load_cnn_model()
image_size = 224
threshold = 0.5

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img_resized = img.resize((image_size, image_size))
    img_array = np.expand_dims(np.array(img_resized)/255.0, axis=0)

    # Predict
    preds = model.predict(img_array)
    confidence = np.max(preds)

    if confidence >= threshold:
        st.success(f"Good E-Waste ✅\nRecycle it ♻️\nConfidence: {confidence*100:.2f}%")
    else:
        st.error(f"Scrap E-Waste ❌\nConfidence: {confidence*100:.2f}%")
else:
    st.info("Please upload an image to get started.")
