"""Week 3 Streamlit app for Sign Language prediction."""

from __future__ import annotations

from pathlib import Path

import streamlit as st
from PIL import Image

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
default_model = "models/classical/svm_baseline.joblib" if model_type == "classical" else "models/neural/week2_mlp.pt"
model_path = st.text_input("Model path", value=default_model)

uploaded_file = st.file_uploader("Upload image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded image", width=220)

    vector = preprocess_uploaded_image(image)

    model_file = Path(model_path)
    if not model_file.exists():
        st.error(f"Model file not found: {model_file}")
    else:
        if model_type == "classical":
            artifact = load_classical_artifact(model_file)
            letter, confidence = predict_classical(artifact, vector)
        else:
            checkpoint = load_neural_checkpoint(model_file)
            letter, confidence = predict_neural(checkpoint, vector)

        st.success(f"Prediction: **{letter}**")
        st.info(f"Confidence: **{confidence:.2%}**")

st.markdown("---")
st.markdown(
    """
### Notes
- Expected input: a single hand-sign image.
- Preprocessing: grayscale → resize 28x28 → normalize to [0,1].
- Ensure the selected model path points to a trained artifact.
"""
)
