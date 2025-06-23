import os
import csv
import time
import random
from PIL import Image
from datetime import datetime
from fpdf import FPDF
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import pathlib
import requests

# Download model from external URL (Google Drive, Dropbox, etc.)
MODEL_URL = "https://your-model-link.com/best_model.h5"
MODEL_PATH = "models/best_model.h5"

os.makedirs("models", exist_ok=True)

if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model..."):
        r = requests.get(MODEL_URL)
        with open(MODEL_PATH, 'wb') as f:
            f.write(r.content)
        st.success("Model downloaded!")

model = tf.keras.models.load_model(MODEL_PATH)
