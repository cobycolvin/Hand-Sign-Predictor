"""Shared inference helpers for Streamlit app."""

from __future__ import annotations

from pathlib import Path
from typing import BinaryIO
from typing import Any

import joblib
import numpy as np
import torch
from PIL import Image, ImageOps

from src.neural.model import SignMLP
from src.utils.label_map import label_to_letter


def preprocess_uploaded_image(
    uploaded_image: Image.Image,
    *,
    auto_crop: bool = True,
    auto_invert: bool = True,
    mirror: bool = False,
    threshold: int = 30,
) -> np.ndarray:
    """Convert user image to flattened 28x28 normalized vector with optional preprocessing.

    Designed to reduce domain shift between real uploads and Sign Language MNIST:
    - autocontrast
    - optional auto-invert (based on corner brightness)
    - optional crop around foreground pixels
    - pad to square to avoid stretching
    - resize to 28x28 using NEAREST (preserves crisp pixel boundaries)
    """
    # Grayscale + contrast
    img = ImageOps.autocontrast(uploaded_image.convert("L"))
    arr = np.array(img, dtype=np.uint8)

    # Auto-invert if the background is bright (check corners)
    if auto_invert:
        corners = np.concatenate(
            [
                arr[:5, :5].ravel(),
                arr[:5, -5:].ravel(),
                arr[-5:, :5].ravel(),
                arr[-5:, -5:].ravel(),
            ]
        )
        if corners.mean() > 127:
            arr = 255 - arr

    # Crop around "ink" (foreground)
    mask = arr > int(threshold)
    if auto_crop and mask.any():
        ys, xs = np.where(mask)
        arr = arr[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]

    # Pad to square so resize doesn't stretch
    h, w = arr.shape
    s = max(h, w)
    padded = np.zeros((s, s), dtype=np.uint8)
    y0 = (s - h) // 2
    x0 = (s - w) // 2
    padded[y0 : y0 + h, x0 : x0 + w] = arr

    img = Image.fromarray(padded, mode="L")

    if mirror:
        img = ImageOps.mirror(img)

    # Resize with NEAREST (avoid blur vs default resampling)
    try:
        resample = Image.Resampling.NEAREST  # Pillow >= 10
    except AttributeError:
        resample = Image.NEAREST  # older Pillow

    img = img.resize((28, 28), resample=resample)

    vec = np.array(img, dtype=np.float32).reshape(1, -1) / 255.0
    return vec


def load_classical_artifact(model_source: str | Path | BinaryIO) -> dict[str, Any]:
    return joblib.load(model_source)


def predict_classical(model_artifact: dict[str, Any], vector: np.ndarray) -> tuple[str, float]:
    model = model_artifact["model"]
    probs = model.predict_proba(vector)[0]
    classes = model.classes_
    idx = int(np.argmax(probs))
    label = int(classes[idx])
    return label_to_letter(label), float(probs[idx])


def load_neural_checkpoint(model_source: str | Path | BinaryIO) -> dict[str, Any]:
    checkpoint = torch.load(model_source, map_location="cpu")
    model = SignMLP(
        input_dim=784,
        hidden_dims=checkpoint["hidden_dims"],
        num_classes=len(checkpoint["labels"]),
        activation=checkpoint["activation"],
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    checkpoint["model"] = model
    return checkpoint


def predict_neural(checkpoint: dict[str, Any], vector: np.ndarray) -> tuple[str, float]:
    model = checkpoint["model"]
    labels = checkpoint["labels"]
    with torch.no_grad():
        logits = model(torch.tensor(vector, dtype=torch.float32))
        probs = torch.softmax(logits, dim=1).numpy()[0]
    idx = int(np.argmax(probs))
    label = int(labels[idx])
    return label_to_letter(label), float(probs[idx])