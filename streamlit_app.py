import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image

# Page setup
st.set_page_config(page_title="Masterpiece ID – AI Artist Classifier", layout="centered")

# Minimalist CSS with Helvetica or fallback sans-serif
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap" rel="stylesheet">

<style>
body, html {
    font-family: 'Inter', 'Helvetica', sans-serif;
}
.title {
    font-size: 36px;
    font-family: 'Inter', 'Helvetica', sans-serif;
    text-align: center;
    margin-bottom: 0px;
}
.subtitle {
    font-size: 18px;
    font-style: italic;
    text-align: center;
    color: #666;
    margin-top: 0px;
}
.artist-label {
    font-size: 22px;
    text-align: center;
    padding: 10px 0;
    border-top: 1px solid #eee;
    border-bottom: 1px solid #eee;
    margin-top: 20px;
    font-family: 'Inter', 'Helvetica', sans-serif;
}
.stImage > img {
    border: 6px solid #ddd;
    border-radius: 4px;
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">🎨 Masterpiece ID</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-powered app that predicts the artist behind your painting</div>', unsafe_allow_html=True)

# Load model and class names
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("inception_artist_model_512_finetuned.h5")
    with open("class_names.json", "r") as f:
        class_names = json.load(f)
    return model, class_names

model, class_names = load_model()

# Concise artist info
artist_info = {
    "Vincent_van_Gogh": {
        "who": "Dutch Post-Impressionist painter",
        "period_style": "Late 19th century – Post-Impressionism",
        "examples": ["Starry Night"]
    },
    "Pablo_Picasso": {
        "who": "Spanish painter and Cubism co-founder",
        "period_style": "Early 20th century – Cubism, Surrealism",
        "examples": ["Guernica"]
    },
    "Claude_Monet": {
        "who": "French painter and founder of Impressionism",
        "period_style": "Late 19th to early 20th century – Impressionism",
        "examples": ["Water Lilies"]
    },
    "Salvador_Dali": {
        "who": "Spanish Surrealist known for dreamlike visuals",
        "period_style": "20th century – Surrealism",
        "examples": ["The Persistence of Memory"]
    },
    "Rembrandt": {
        "who": "Dutch Baroque painter and printmaker",
        "period_style": "17th century – Baroque",
        "examples": ["The Night Watch"]
    },
    "Marc_Chagall": {
        "who": "French modernist with folkloric dreamlike themes",
        "period_style": "20th century – Expressionism, Surrealism",
        "examples": ["I and the Village"]
    },
    "Paul_Gauguin": {
        "who": "French Post-Impressionist focused on Tahitian themes",
        "period_style": "Late 19th century – Symbolism",
        "examples": ["Tahitian Women on the Beach"]
    },
    "Albrecht_Durer": {
        "who": "German Renaissance painter and engraver",
        "period_style": "15th–16th century – Northern Renaissance",
        "examples": ["Melencolia I"]
    },
    "Henri_Matisse": {
        "who": "French Fauvist known for bold color and form",
        "period_style": "Early 20th century – Fauvism, Modernism",
        "examples": ["The Dance"]
    },
    "Nicholas_Roerich": {
        "who": "Russian painter and mystic of Himalayan scenes",
        "period_style": "20th century – Symbolism, Spiritual Art",
        "examples": ["The Himalayas"]
    }
}

# Upload painting
uploaded_file = st.file_uploader("Upload a painting", type=["jpg", "jpeg", "png"])

# Predict and get top-3
image_resized = image.resize((512, 512))
img_array = np.array(image_resized) / 255.0
img_array = img_array[np.newaxis, ...]
prediction = model.predict(img_array)[0]
top_indices = prediction.argsort()[::-1][:3]

# Unified layout (no dividers, same font)
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
            <p style="font-size:18px; margin-bottom:4px;">🎯 <b>{name}</b> — {conf:.2f}%</p>
            <p style="font-size:16px; color:#555;">{who}</p>
            <p style="font-size:15px; font-style:italic; color:#777;">{style}</p>
            <p style="font-size:15px; font-style:italic; color:#999;">Famous Work: {example}</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="text-align:center; margin-top:10px; font-family: 'Inter', sans-serif;">
            <


# Sidebar
with st.sidebar:
    st.markdown("## 🖼️ About")
    st.info("Masterpiece ID uses a fine-tuned InceptionV3 model trained on 1,000+ paintings from 10 legendary artists. Built with TensorFlow + Streamlit.")
    st.markdown("### Supported Artists")
    for name in class_names:
        st.markdown(f"- {name.replace('_', ' ')}")
