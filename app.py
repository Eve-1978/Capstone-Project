import streamlit as st
import tensorflow as tf   # or keras
from PIL import Image
import numpy as np
import pathlib

MODEL_PATH = pathlib.Path("models/best_model.h5")
model = tf.keras.models.load_model(MODEL_PATH)

st.title("Solar Streamlit")

uploaded = st.file_uploader("Upload a panel image", type=["jpg","png","jpeg"])
if uploaded:
    img = Image.open(uploaded).convert("RGB").resize((256,256))
    arr = np.expand_dims(np.asarray(img)/255.0, 0)
    pred = model.predict(arr)[0][0]
    st.write(f"Predicted dust level : **{pred:.2%}**")

