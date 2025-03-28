import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Load model
model = tf.keras.models.load_model("cats_vs_chickens_vs_dogs_efficientnet.h5")
class_names = ["Cat", "Chicken", "Golden Retriever"]

# Streamlit UI
st.title("🐱 vs. 🐔 vs. 🐶 Classifier")
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).resize((448, 448))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    st.image(img, caption="Uploaded Image", use_column_width=True)
    st.success(f"**Prediction:** {predicted_class} ({confidence:.2f}% confident)")