import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from io import BytesIO

# ---------- Page Setup ----------
st.set_page_config(page_title="Masterpiece ID", layout="centered")
st.title("🎨 Masterpiece ID")
st.markdown("An AI-powered app that identifies the artist behind iconic artworks using deep learning.")

# ---------- Load Model ----------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("inception_artist_model_512_finetuned.h5")
    with open("class_names.json", "r") as f:
        class_names = json.load(f)
    return model, class_names

model, class_names = load_model()

# ---------- Grad-CAM Functions ----------
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

# ---------- Image Upload ----------
uploaded_file = st.file_uploader("Upload a painting", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    image_resized = image.resize((512, 512))
    img_array = np.array(image_resized) / 255.0
    img_array = img_array[np.newaxis, ...]

    # ---------- Prediction ----------
    predictions = model.predict(img_array)[0]
    top_indices = predictions.argsort()[::-1][:3]
    top_label = class_names[top_indices[0]].replace("_", " ")
    top_conf = predictions[top_indices[0]] * 100

    st.markdown(f"### 🧠 Top Prediction: **{top_label}** ({top_conf:.2f}%)")
    st.markdown("Other likely artists:")
    for i in top_indices[1:]:
        st.markdown(f"- {class_names[i].replace('_', ' ')} — {predictions[i]*100:.2f}%")

    # ---------- Grad-CAM Visualization ----------
    st.markdown("### 🔥 Model Focus Visualization")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Original Image**")
        st.image(image_resized, use_column_width=True)

    with col2:
        st.markdown("**Grad-CAM Heatmap**")
        heatmap = make_gradcam_heatmap(img_array, model, pred_index=top_indices[0])
        heatmap_resized = Image.fromarray(np.uint8(255 * heatmap.numpy())).resize(image_resized.size)
        heatmap_np = np.array(heatmap_resized) / 255.0
        colored_heatmap = cm.jet(heatmap_np)[:, :, :3]
        colored_heatmap_img = Image.fromarray(np.uint8(colored_heatmap * 255))
        blended = Image.blend(image_resized, colored_heatmap_img, alpha=0.5)

        # Plot with legend
        fig, ax = plt.subplots()
        ax.imshow(blended)
        ax.axis('off')
        sm = plt.cm.ScalarMappable(cmap='jet')
        sm.set_array(heatmap_np)
        cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Relevance to Prediction", rotation=270, labelpad=15)
        st.pyplot(fig)
