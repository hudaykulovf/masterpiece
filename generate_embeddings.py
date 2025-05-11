import os
import numpy as np
import tensorflow as tf
from PIL import Image
from tqdm import tqdm
import json

# Parameters
IMAGE_SIZE = (512, 512)
MODEL_PATH = "inception_artist_model_512_finetuned.h5"
DATA_DIR = "train_dataset_by_artist/"  # structure: train_dataset_by_artist/ArtistName/*.jpg

# Load model and remove last dense layer
base_model = tf.keras.models.load_model(MODEL_PATH)
feature_model = tf.keras.Model(inputs=base_model.input,
                               outputs=base_model.get_layer("avg_pool").output)

embeddings = []
labels = []
label_names = []

# Create class name to index mapping
class_dirs = sorted(os.listdir(DATA_DIR))
class_names = [d.replace(" ", "_") for d in class_dirs]
label_map = {name: idx for idx, name in enumerate(class_names)}

for artist in class_dirs:
    folder = os.path.join(DATA_DIR, artist)
    for filename in tqdm(os.listdir(folder), desc=f"Processing {artist}"):
        img_path = os.path.join(folder, filename)
        try:
            image = Image.open(img_path).convert("RGB").resize(IMAGE_SIZE)
            img_array = np.array(image) / 255.0
            img_array = img_array[np.newaxis, ...]
            features = feature_model.predict(img_array)
            embeddings.append(features[0])
            labels.append(label_map[artist.replace(" ", "_")])
            label_names.append(artist.replace(" ", "_"))
        except Exception as e:
            print(f"Error with {img_path}: {e}")

# Save to disk
np.save("embeddings.npy", np.array(embeddings))
np.save("labels.npy", np.array(labels))
with open("label_names.json", "w") as f:
    json.dump(label_names, f)
