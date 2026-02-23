"""Data loading and preprocessing helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class DatasetSplit:
    """Container for model-ready arrays."""

    x: np.ndarray
    y: np.ndarray


def load_sign_mnist_csv(csv_path: str | Path, normalize: bool = True) -> DatasetSplit:
    """Load Sign Language MNIST CSV into features/labels arrays."""
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    if "label" not in df.columns:
        raise ValueError(f"Missing required 'label' column in {csv_path}")

    y = df["label"].to_numpy(dtype=np.int64)
    x = df.drop(columns=["label"]).to_numpy(dtype=np.float32)

    if normalize:
        x = x / 255.0

    return DatasetSplit(x=x, y=y)


def pixels_to_image_vector(pixel_values: np.ndarray, normalize: bool = True) -> np.ndarray:
    """Prepare a single flattened 28x28 image vector for inference."""
    vector = pixel_values.astype(np.float32).reshape(1, -1)
    return vector / 255.0 if normalize else vector
