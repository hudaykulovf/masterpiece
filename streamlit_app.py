import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image

st.set_page_config(page_title="Masterpiece ID – ML Artist Classifier", layout="centered")
st.title("🎨 Masterpiece ID")
st.markdown("An ML-powered app that predicts which of 10 famous artists painted your uploaded image.")

@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model("inception_artist_model_512_finetuned.h5")
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        st.stop()

    try:
        with open("class_names.json", "r") as f:
            class_names = json.load(f)
    except Exception as e:
        st.error(f"❌ Failed to load class labels: {e}")
        st.stop()

    return model, class_names

model, class_names = load_model()

uploaded_file = st.file_uploader("Upload a painting image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Painting", use_container_width=True)

    image_resized = image.resize((512, 512))
    img_array = np.array(image_resized) / 255.0
    img_array = img_array[np.newaxis, ...]

    prediction = model.predict(img_array)[0]
    top_indices = prediction.argsort()[::-1][:3]

    st.subheader("🎯 Top 3 Predictions")
    for i in top_indices:
        st.write(f"**{class_names[i]}** — {prediction[i]*100:.2f}%")

with st.sidebar:
    st.markdown("### ℹ️ About")
    st.info(
        "Masterpiece ID is an AI-powered ML project that uses a fine-tuned InceptionV3 model "
        "to classify paintings by 10 legendary artists. Built with TensorFlow and Streamlit."
    )
