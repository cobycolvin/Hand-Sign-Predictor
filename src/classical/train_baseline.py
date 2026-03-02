"""Train classical baseline model (Week 1 control)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.data.loaders import load_sign_mnist_csv
from src.utils.label_map import get_label_to_letter


def build_model(model_type: str, random_state: int) -> Pipeline:
    """Construct baseline sklearn pipeline."""
    if model_type == "svm":
        model = SVC(kernel="rbf", C=5.0, gamma="scale", probability=True, random_state=random_state)
    elif model_type == "rf":
        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            n_jobs=-1,
            random_state=random_state,
        )
    else:
        raise ValueError("model_type must be one of: svm, rf")

    return Pipeline([("scaler", StandardScaler()), ("model", model)])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Sign Language baseline model")
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--test_csv", required=True)
    parser.add_argument("--model_type", choices=["svm", "rf"], default="svm")
    parser.add_argument("--model_out", required=True)
    parser.add_argument("--metrics_out", default="reports/metrics/week1_train_metrics.json")
    parser.add_argument("--random_state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    train = load_sign_mnist_csv(args.train_csv, normalize=True)
    test = load_sign_mnist_csv(args.test_csv, normalize=True)

    pipeline = build_model(model_type=args.model_type, random_state=args.random_state)
    pipeline.fit(train.x, train.y)

    preds = pipeline.predict(test.x)
    accuracy = accuracy_score(test.y, preds)

    artifact = {
        "model": pipeline,
        "label_to_letter": get_label_to_letter(),
        "config": {
            "model_type": args.model_type,
            "random_state": args.random_state,
        },
    }

    model_path = Path(args.model_out)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, model_path)

    metrics = {
        "accuracy": accuracy,
        "model_type": args.model_type,
        "train_samples": int(train.x.shape[0]),
        "test_samples": int(test.x.shape[0]),
    }
    metrics_path = Path(args.metrics_out)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Saved model to: {model_path}")
    print(f"Saved metrics to: {metrics_path}")
    print(f"Test accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()
