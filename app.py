import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model("keras_model.h5")

# Class labels based on your training
class_names = ["color", "poke", "cut", "fold", "glue", "good"]

# Streamlit UI
st.set_page_config(page_title="Leather Anomaly Detector", layout="centered")
st.title("🧪 Leather Anomaly Detector")
st.write("Upload a leather sample image to detect the type of anomaly or quality.")

# File uploader
uploaded_file = st.file_uploader("📷 Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and display image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess image
    img = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize pixel values

    # Make prediction
    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions)
    predicted_label = class_names[predicted_index]
    confidence = np.max(predictions)

    # Display result
    st.markdown(f"### 🔍 Prediction: **{predicted_label}**")
    st.markdown(f"#### 🔢 Confidence: `{confidence * 100:.2f}%`")
