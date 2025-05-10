import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image

# Title
st.set_page_config(page_title="Masterpiece ID – ML Artist Classifier")
st.title("🎨 Masterpiece ID")
st.markdown("An AI-powered model that predicts the artist of a painting using deep learning.")

# Load model and labels
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("inception_artist_model_512_finetuned.h5")
    with open("class_names.json", "r") as f:
        class_names = json.load(f)
    return model, class_names

model, class_names = load_model()

# Image uploader
uploaded_file = st.file_uploader("Upload a painting image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Painting", use_column_width=True)

    # Preprocess
    image_resized = image.resize((512, 512))
    img_array = np.array(image_resized) / 255.0
    img_array = img_array[np.newaxis, ...]

    # Predict
    prediction = model.predict(img_array)[0]
    top_indices = prediction.argsort()[::-1][:3]
    st.subheader("🎯 Top 3 Predictions")
    for i in top_indices:
        st.write(f"**{class_names[i]}** — {prediction[i]*100:.2f}%")

    # Bar chart
    st.subheader("📊 Prediction Confidence")
    st.bar_chart({class_names[i]: prediction[i] for i in top_indices})
