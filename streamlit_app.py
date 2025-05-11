
import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image
import base64
from io import BytesIO
import matplotlib.cm as cm
import matplotlib.pyplot as plt

# PAGE SETUP
st.set_page_config(page_title="Masterpiece ID", layout="centered")

# CSS STYLING
st.markdown(
    '''
    <link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;700&display=swap" rel="stylesheet">
    <style>
    body, html {
        font-family: 'DM Sans', sans-serif;
    }
    .title {
        font-size: 36px;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0px;
    }
    .subtitle {
        font-size: 18px;
        text-align: center;
        color: #666;
        margin-top: 0px;
    }
    .stImage > img {
        border: 6px solid #ddd;
        border-radius: 4px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    </style>
    ''',
    unsafe_allow_html=True
)

st.markdown('<div class="title">🎨 Masterpiece ID</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">An ML-powered app that identifies the artist behind iconic artworks</div>', unsafe_allow_html=True)

# LOAD MODEL
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("inception_artist_model_512_finetuned.h5")
    with open("class_names.json", "r") as f:
        class_names = json.load(f)
    return model, class_names

model, class_names = load_model()

# GRAD-CAM UTILITIES
def make_gradcam_heatmap(img_array, model, last_conv_layer_name="mixed10", pred_index=None):
    grad_model = tf.keras.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_output = predictions[:, pred_index]
    grads = tape.gradient(class_output, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    return tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)

def plot_gradcam_with_legend(original_image, heatmap, alpha=0.5):
    heatmap_resized = Image.fromarray(np.uint8(255 * heatmap.numpy())).resize(original_image.size)
    heatmap_np = np.array(heatmap_resized) / 255.0
    colored_heatmap = cm.jet(heatmap_np)[:, :, :3]
    colored_heatmap_img = Image.fromarray(np.uint8(colored_heatmap * 255))
    blended = Image.blend(original_image, colored_heatmap_img, alpha=alpha)
    fig, ax = plt.subplots()
    ax.imshow(blended)
    ax.axis('off')
    sm = plt.cm.ScalarMappable(cmap='jet')
    sm.set_array(heatmap_np)
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Relevance to Prediction', rotation=270, labelpad=15)
    st.pyplot(fig)

# IMAGE TO BASE64
def image_to_base64(img):
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

# IMAGE UPLOAD AND PREDICTION
uploaded_file = st.file_uploader("Upload the artwork", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    thumb = image.resize((500, 500))
    st.image(thumb, caption="Uploaded Painting", use_column_width=True)

    image_resized = image.resize((512, 512))
    img_array = np.array(image_resized) / 255.0
    img_array = img_array[np.newaxis, ...]

    prediction = model.predict(img_array)[0]
    top_indices = prediction.argsort()[::-1][:3]

    for idx, i in enumerate(top_indices):
        name = class_names[i].replace("_", " ")
        conf = prediction[i] * 100
        if idx == 0:
            st.markdown(f"### 🎯 {name} — {conf:.2f}%")
            st.markdown("### 🔥 Model Focus (Grad-CAM)")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Original Image**")
    st.image(image_resized, use_column_width=True)

with col2:
    st.markdown("**Grad-CAM Heatmap**")
    heatmap = make_gradcam_heatmap(img_array, model, pred_index=i)
    heatmap_resized = Image.fromarray(np.uint8(255 * heatmap.numpy())).resize(image_resized.size)
    heatmap_np = np.array(heatmap_resized) / 255.0
    colored_heatmap = cm.jet(heatmap_np)[:, :, :3]
    colored_heatmap_img = Image.fromarray(np.uint8(colored_heatmap * 255))
    blended = Image.blend(image_resized, colored_heatmap_img, alpha=0.5)

    fig, ax = plt.subplots()
    ax.imshow(blended)
    ax.axis('off')
    sm = plt.cm.ScalarMappable(cmap='jet')
    sm.set_array(heatmap_np)
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Relevance to Prediction', rotation=270, labelpad=15)
    st.pyplot(fig)
        else:
            st.markdown(f"- {name} — {conf:.2f}%")

# SIDEBAR
with st.sidebar:
    st.markdown("## 🖼️ About")
    st.info("Masterpiece ID uses a fine-tuned InceptionV3 model trained on 1,000+ paintings from 10 legendary artists. Built with TensorFlow + Streamlit.")
    st.markdown("### Supported Artists")
    for name in class_names:
        st.markdown(f"- {name.replace('_', ' ')}")
