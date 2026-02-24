# Random Forest chosen over SVM because:
# - No feature scaling sensitivity
# - Native predict_proba for confidence scores
# - More stable baseline for deployment


"""
P2 – Classical ML baseline (Week 1)
Random Forest control model for Sign Language MNIST.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline

from src.data.loaders import load_sign_mnist_csv
from src.utils.label_map import get_label_to_letter


def train_random_forest(
    train_csv: str | Path,
    test_csv: str | Path,
    n_estimators: int = 300,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Train RandomForest baseline and return model artifact + metrics.
    """
    train = load_sign_mnist_csv(train_csv, normalize=True)
    test = load_sign_mnist_csv(test_csv, normalize=True)

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        n_jobs=-1,
        random_state=random_state,
    )

    pipeline = Pipeline([("model", model)])
    pipeline.fit(train.x, train.y)

    preds = pipeline.predict(test.x)
    accuracy = accuracy_score(test.y, preds)

    artifact = {
        "model": pipeline,
        "label_to_letter": get_label_to_letter(),
        "config": {
            "model_type": "random_forest",
            "n_estimators": n_estimators,
            "random_state": random_state,
        },
        "metrics": {
            "accuracy": accuracy,
        },
    }

    return artifact


def save_model(artifact: Dict[str, Any], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output_path) 
