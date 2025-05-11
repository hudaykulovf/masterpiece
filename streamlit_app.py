import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image
import base64
from io import BytesIO
import matplotlib.cm as cm
import matplotlib.pyplot as plt

# ---------- PAGE SETUP ----------
st.set_page_config(page_title="Masterpiece ID", layout="centered")

# ---------- CSS STYLING ----------
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;700&display=swap" rel="stylesheet">
<style>
body, html {
    font-family: 'DM Sans', sans-serif;
}
.title {
    font-size: 36px;
    font-family: 'DM Sans', sans-serif;
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
p {
    font-size: 18px;
    font-family: 'DM Sans', sans-serif;
    text-align: center;
    margin: 6px 0;
}
.stImage > img {
    border: 6px solid #ddd;
    border-radius: 4px;
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">🎨 Masterpiece ID</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">An ML-powered app that identifies the artist behind iconic artworks</div>', unsafe_allow_html=True)

# ---------- LOAD MODEL ----------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("inception_artist_model_512_finetuned.h5")
    with open("class_names.json", "r") as f:
        class_names = json.load(f)
    return model, class_names

model, class_names = load_model()

# ---------- GRAD-CAM UTILS ----------
def make_gradcam_heatmap(img_array, model, last_conv_layer_name="mixed10", pred_index=None):
    grad_model = tf.keras.models.Model(
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
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def overlay_heatmap_on_image(original_image, heatmap, alpha=0.5):
    heatmap_resized = Image.fromarray(np.uint8(255 * heatmap)).resize(original_image.size)
    heatmap_np = np.array(heatmap_resized) / 255.0
    colored_heatmap = cm.jet(heatmap_np)[:, :, :3]
    colored_heatmap_img = Image.fromarray(np.uint8(colored_heatmap * 255))
    return Image.blend(original_image, colored_heatmap_img, alpha=alpha)

# ---------- ARTIST INFO ----------
artist_info = {
    "Vincent_van_Gogh": {"who": "Dutch Post-Impressionist painter", "period_style": "Late 19th century – Post-Impressionism", "examples": ["Starry Night"]},
    "Pablo_Picasso": {"who": "Spanish painter and Cubism co-founder", "period_style": "Early 20th century – Cubism, Surrealism", "examples": ["Guernica"]},
    "Claude_Monet": {"who": "French painter and founder of Impressionism", "period_style": "Late 19th to early 20th century – Impressionism", "examples": ["Water Lilies"]},
    "Salvador_Dali": {"who": "Spanish Surrealist known for dreamlike visuals", "period_style": "20th century – Surrealism", "examples": ["The Persistence of Memory"]},
    "Rembrandt": {"who": "Dutch Baroque painter and printmaker", "period_style": "17th century – Baroque", "examples": ["The Night Watch"]},
    "Marc_Chagall": {"who": "French modernist with folkloric dreamlike themes", "period_style": "20th century – Expressionism, Surrealism", "examples": ["I and the Village"]},
    "Paul_Gauguin": {"who": "French Post-Impressionist focused on Tahitian themes", "period_style": "Late 19th century – Symbolism", "examples": ["Tahitian Women on the Beach"]},
    "Albrecht_Durer": {"who": "German Renaissance painter and engraver", "period_style": "15th–16th century – Northern Renaissance", "examples": ["Melencolia I"]},
    "Henri_Matisse": {"who": "French Fauvist known for bold color and form", "period_style": "Early 20th century – Fauvism, Modernism", "examples": ["The Dance"]},
    "Nicholas_Roerich": {"who": "Russian painter and mystic of Himalayan scenes", "period_style": "20th century – Symbolism, Spiritual Art", "examples": ["The Himalayas"]}
}

# ---------- UTILS ----------
def image_to_base64(img):
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

# ---------- IMAGE UPLOAD ----------
uploaded_file = st.file_uploader("Upload the artwork", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    thumb = image.resize((500, 500))
    st.markdown(f"""
    <div style="text-align:center;">
        <img src="data:image/png;base64,{image_to_base64(thumb)}" style="max-width:550px; border:8px solid #ddd; border-radius:6px; box-shadow:0 4px 8px rgba(0,0,0,0.1);" />
        <p style="font-size:12px; color:gray;">Uploaded Painting</p>
    </div>
    """, unsafe_allow_html=True)

    # ---------- PREDICTION ----------
    image_resized = image.resize((512, 512))
    img_array = np.array(image_resized) / 255.0
    img_array = img_array[np.newaxis, ...]

    prediction = model.predict(img_array)[0]
    top_indices = prediction.argsort()[::-1][:3]

    for idx, i in enumerate(top_indices):
        name = class_names[i].replace("_", " ")
        conf = prediction[i] * 100
        info = artist_info.get(class_names[i], {})
        who = info.get('who', '')
        style = info.get('period_style', '')
        example = ', '.join(info.get('examples', []))

        if idx == 0:
            st.markdown(f"""
            <div style="text-align:center; margin-top:20px; font-family: 'Inter', sans-serif;">
                <p style="font-size:22px; margin-bottom:4px;">🎯 <b>{name}</b> — {conf:.2f}%</p>
                <p style="font-size:16px; color:#777;">{who}</p>
                <p style="font-size:16px; color:#777;">{style}</p>
                <p style="font-size:16px; color:#777;">Famous Work: {example}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="text-align:center; margin-top:10px; font-family: 'Inter', sans-serif;">
                <p style="font-size:18px; color:#444;">• {name} — {conf:.2f}%</p>
            </div>
            """, unsafe_allow_html=True)

    # ---------- GRAD-CAM VISUALIZATION ----------
    st.markdown("""---""")
    st.subheader("🔍 Grad-CAM: What the model focused on")
    heatmap = make_gradcam_heatmap(img_array, model, pred_index=top_indices[0])
    gradcam_image = overlay_heatmap_on_image(image.resize((512, 512)), heatmap)
    st.image(gradcam_image, caption="Model Attention via Grad-CAM", use_column_width=True)

# ---------- SIDEBAR ----------
with st.sidebar:
    st.markdown("## 🖼️ About")
    st.info("Masterpiece ID uses a fine-tuned InceptionV3 model trained on 1,000+ paintings from 10 legendary artists. Built with TensorFlow + Streamlit.")
    st.markdown("### Supported Artists")
    for name in class_names:
        st.markdown(f"- {name.replace('_', ' ')}")
