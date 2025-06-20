import os
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import pathlib
import requests

# Download model from external URL (Google Drive, Dropbox, etc.)
MODEL_URL = "https://drive.google.com/uc?id=1C3SR061sGok7yYZCLhENCWAIR6cQ6hWG"
MODEL_PATH = "models/best_model.h5"

os.makedirs("models", exist_ok=True)

if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model..."):
        r = requests.get(MODEL_URL)
        with open(MODEL_PATH, 'wb') as f:
            f.write(r.content)
        st.success("Model downloaded!")

model = tf.keras.models.load_model(MODEL_PATH)
