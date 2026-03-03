from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path so `import src...` works when run via Streamlit
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # .../Hand-Sign-Predictor
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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
st.caption("Upload a hand sign image, tune preprocessing, then click Predict.")

# --- Model selection ---
model_type = st.selectbox("Model type", ["classical", "neural"])

model_source = st.radio(
    "Model source",
    ["Path on disk", "Upload file"],
    horizontal=True,
)

default_model = (
    "models/classical/svm_baseline.joblib"
    if model_type == "classical"
    else "models/neural/week2_mlp.pt"
)

model_path = ""
uploaded_model_file = None

if model_source == "Path on disk":
    model_path = st.text_input("Model path", value=default_model)
else:
    model_exts = ["joblib"] if model_type == "classical" else ["pt", "pth"]
    uploaded_model_file = st.file_uploader("Upload model file", type=model_exts)

st.markdown("---")

# --- Upload image ---
uploaded_file = st.file_uploader("Upload image", type=["png", "jpg", "jpeg"])

# Persist last result so it doesn't disappear on reruns
if "last_result" not in st.session_state:
    st.session_state.last_result = None

if uploaded_file is None:
    st.info("Upload an image to begin.")
else:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded image", width=240)

    # --- Preprocessing controls ---
    st.subheader("Preprocessing (tune first, then click Predict)")
    mirror = st.checkbox("Mirror (flip left/right)", value=False)
    auto_invert = st.checkbox("Auto-invert (match MNIST style)", value=True)
    auto_crop = st.checkbox("Auto-crop hand", value=True)
    threshold = st.slider("Crop threshold", 0, 255, 30)

    vector = preprocess_uploaded_image(
        image,
        mirror=mirror,
        auto_invert=auto_invert,
        auto_crop=auto_crop,
        threshold=threshold,
    )

    # Show what the model sees
    debug_img = (vector.reshape(28, 28) * 255).astype("uint8")
    st.image(
        Image.fromarray(debug_img).resize((280, 280)),
        caption="Model input (28×28)",
        width=280,
    )

    st.markdown("---")

    # Resolve model input (either path or uploaded file)
    model_input = None
    predict_disabled = False

    if model_source == "Path on disk":
        model_file = Path(model_path)
        if not model_file.exists():
            st.warning(f"Model file not found: {model_file}")
            predict_disabled = True
        else:
            model_input = model_file
    else:
        if uploaded_model_file is None:
            st.warning("Upload a model file to enable prediction.")
            predict_disabled = True
        else:
            uploaded_model_file.seek(0)
            model_input = uploaded_model_file

    # --- Predict button ---
    if st.button("Predict", type="primary", disabled=predict_disabled):
        try:
            if model_type == "classical":
                artifact = load_classical_artifact(model_input)
                letter, confidence = predict_classical(artifact, vector)
            else:
                checkpoint = load_neural_checkpoint(model_input)
                letter, confidence = predict_neural(checkpoint, vector)

            st.session_state.last_result = (letter, confidence)
        except Exception as exc:
            st.error(f"Failed to run inference: {exc}")
            st.session_state.last_result = None

    # Show last result so it doesn't vanish when you tweak options
    if st.session_state.last_result is not None:
        letter, confidence = st.session_state.last_result
        st.success(f"Prediction: **{letter}**")
        st.info(f"Confidence: **{confidence:.2%}**")

st.markdown("---")
st.markdown(
    """
### Notes
- Sign Language MNIST expects a tight, high-contrast 28×28 hand image.
- If the 28×28 preview doesn’t resemble the sign, the prediction will usually be wrong.
- You can load a model from disk or upload a model file directly in the app.
"""
)