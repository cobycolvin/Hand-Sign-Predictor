"""Shared inference helpers for Streamlit app."""

from __future__ import annotations

from pathlib import Path
from typing import BinaryIO
from typing import Any

import joblib
import numpy as np
import torch
from PIL import Image

from src.neural.model import SignMLP
from src.utils.label_map import label_to_letter


def preprocess_uploaded_image(uploaded_image: Image.Image) -> np.ndarray:
    """Convert user image to flattened 28x28 normalized vector."""
    image = uploaded_image.convert("L").resize((28, 28))
    arr = np.array(image, dtype=np.float32).reshape(1, -1) / 255.0
    return arr


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
