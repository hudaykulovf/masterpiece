import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image

# Page setup
st.set_page_config(page_title="Masterpiece ID – AI Artist Classifier", layout="centered")

# Custom styling
st.markdown("""
<style>
.title { font-size: 36px; font-family: 'Georgia'; text-align: center; margin-bottom: 0px; }
.subtitle { font-size: 18px; font-style: italic; text-align: center; color: #666; margin-top: 0px; }
.artist-label { font-size: 22px; text-align: center; padding: 10px 0; border-top: 1px solid #eee; border-bottom: 1px solid #eee; margin-top: 20px; font-family: 'Georgia'; }
.stImage > img { border: 6px solid #ddd; border-radius: 4px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">🎨 Masterpiece ID</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-powered gallery that identifies the artist behind the painting</div>', unsafe_allow_html=True)

# Load model and class names
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("inception_artist_model_512_finetuned.h5")
    with open("class_names.json", "r") as f:
        class_names = json.load(f)
    return model, class_names

model, class_names = load_model()

# Artist info dictionary
artist_info = {
    "Vincent_van_Gogh": {
        "who": "Dutch Post-Impressionist painter",
        "period_style": "Late 19th century – Post-Impressionism",
        "famous_for": "Expressive brushwork, swirling forms, and emotional intensity",
        "examples": ["Starry Night", "Sunflowers"]
    },
    "Pablo_Picasso": {
        "who": "Spanish painter, sculptor, and co-founder of Cubism",
        "period_style": "Early to mid 20th century – Cubism, Surrealism",
        "famous_for": "Abstracted forms, emotional range, and groundbreaking styles",
        "examples": ["Guernica", "Les Demoiselles d'Avignon"]
    },
    "Claude_Monet": {
        "who": "French painter and founder of the Impressionist movement",
        "period_style": "Late 19th to early 20th century – Impressionism",
        "famous_for": "Light, color, and fluid brushwork in nature scenes",
        "examples": ["Water Lilies", "Impression, Sunrise"]
    },
    "Salvador_Dali": {
        "who": "Spanish surrealist known for dreamlike visuals and eccentricity",
        "period_style": "20th century – Surrealism",
        "famous_for": "Melting clocks, bizarre dreamscapes, and hyperrealism",
        "examples": ["The Persistence of Memory", "The Elephants"]
    },
    "Rembrandt": {
        "who": "Dutch Baroque painter and printmaker",
        "period_style": "17th century – Baroque",
        "famous_for": "Dramatic lighting, portraiture, and religious themes",
        "examples": ["The Night Watch", "Self-Portrait with Two Circles"]
    },
    "Marc_Chagall": {
        "who": "Belarusian-French modernist painter",
        "period_style": "20th century – Expressionism, Surrealism",
        "famous_for": "Dreamlike, colorful scenes drawn from folklore and religion",
        "examples": ["I and the Village", "The Birthday"]
    },
    "Paul_Gauguin": {
        "who": "French Post-Impressionist painter",
        "period_style": "Late 19th century – Symbolism, Primitivism",
        "famous_for": "Tahitian scenes, bold color, and symbolic content",
        "examples": ["Where Do We Come From?", "Tahitian Women on the Beach"]
    },
    "Albrecht_Durer": {
        "who": "German Renaissance painter and printmaker",
        "period_style": "15th–16th century – Northern Renaissance",
        "famous_for": "Engravings, religious works, and anatomical precision",
        "examples": ["Melencolia I", "Young Hare"]
    },
    "Henri_Matisse": {
        "who": "French artist known for Fauvism",
        "period_style": "Early 20th century – Fauvism, Modernism",
        "famous_for": "Vibrant color, flattened form, and expressive decoration",
        "examples": ["The Dance", "Woman with a Hat"]
    },
    "Nicholas_Roerich": {
        "who": "Russian painter, philosopher, and mystic",
        "period_style": "20th century – Symbolism, Spiritual Art",
        "famous_for": "Mystical Himalayan landscapes and spiritual themes",
        "examples": ["The Himalayas", "St. Mercurius"]
    }
}

# Upload image
uploaded_file = st.file_uploader("Upload a painting", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Painting", use_container_width=True)

    image_resized = image.resize((512, 512))
    img_array = np.array(image_resized) / 255.0
    img_array = img_array[np.newaxis, ...]

    prediction = model.predict(img_array)[0]
    top_indices = prediction.argsort()[::-1][:3]

    st.markdown('<div class="artist-label">🧠 Predicted Artist(s):</div>', unsafe_allow_html=True)

    for i in top_indices:
        name = class_names[i]
        confidence = prediction[i] * 100
        info = artist_info.get(name, {})

        st.markdown(f"<div class='artist-label'><strong>{name.replace('_', ' ')}</strong> — {confidence:.2f}%</div>", unsafe_allow_html=True)
        st.markdown(f"**Who:** {info.get('who', 'N/A')}")
        st.markdown(f"**Period & Style:** {info.get('period_style', 'N/A')}")
        st.markdown(f"**Famous For:** {info.get('famous_for', 'N/A')}")
        st.markdown(f"**Notable Works:** _{', '.join(info.get('examples', []))}_")

# Sidebar
with st.sidebar:
    st.markdown("## 🖼️ About")
    st.info("Masterpiece ID uses a fine-tuned InceptionV3 model trained on 1,000+ paintings from 10 legendary artists. Built with TensorFlow + Streamlit.")
    st.markdown("### Supported Artists")
    for name in class_names:
        st.markdown(f"- {name.replace('_', ' ')}")
