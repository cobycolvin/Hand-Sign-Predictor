"""Week 3 Streamlit app for Sign Language prediction."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.app.inference import (
    load_classical_artifact,
    load_neural_checkpoint,
    predict_classical,
    predict_neural,
    preprocess_uploaded_image,
)

st.set_page_config(page_title="Hand Sign Predictor", page_icon="🤟")
st.title("🤟 Hand Sign Predictor")
st.caption("Upload a hand sign image and get predicted letter + confidence.")

model_type = st.selectbox("Model type", ["classical", "neural"])
model_source = st.radio("Model source", ["Path on disk", "Upload file"], horizontal=True)

default_model = "models/classical/svm_baseline.joblib" if model_type == "classical" else "models/neural/week2_mlp.pt"
model_path = ""
uploaded_model_file = None

if model_source == "Path on disk":
    model_path = st.text_input("Model path", value=default_model)
else:
    model_exts = ["joblib"] if model_type == "classical" else ["pt", "pth"]
    uploaded_model_file = st.file_uploader("Upload model file", type=model_exts)

uploaded_file = st.file_uploader("Upload image", type=["png", "jpg", "jpeg"])
submit = st.button("Submit")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded image", width=220)

if submit:
    if uploaded_file is None:
        st.error("Please upload an image before submitting.")
        st.stop()

    image = Image.open(uploaded_file)
    vector = preprocess_uploaded_image(image)

    if model_source == "Path on disk":
        model_file = Path(model_path)
        if not model_file.exists():
            st.error(f"Model file not found: {model_file}")
            st.stop()
        model_input = model_file
    else:
        if uploaded_model_file is None:
            st.error("Please upload a model file.")
            st.stop()
        uploaded_model_file.seek(0)
        model_input = uploaded_model_file

    try:
        if model_type == "classical":
            artifact = load_classical_artifact(model_input)
            letter, confidence = predict_classical(artifact, vector)
        else:
            checkpoint = load_neural_checkpoint(model_input)
            letter, confidence = predict_neural(checkpoint, vector)
    except Exception as exc:
        st.error(f"Failed to run inference: {exc}")
        st.stop()

    st.success(f"Prediction: **{letter}**")
    st.info(f"Confidence: **{confidence:.2%}**")
else:
    st.caption("Set model/image inputs, then click Submit to run prediction.")

st.markdown("---")
st.markdown(
    """
### Notes
- Expected input: a single hand-sign image (PNG/JPG/JPEG).
- Preprocessing: grayscale → resize 28x28 → normalize to [0,1].
- Model can be loaded from disk path or uploaded directly in the app.
"""
)
