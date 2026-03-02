"""Evaluate classical model and produce confusion matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.data.loaders import load_sign_mnist_csv
from src.utils.label_map import get_sorted_labels, label_to_letter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Sign Language baseline model")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--test_csv", required=True)
    parser.add_argument("--confusion_out", default="reports/figures/week1_confusion_matrix.png")
    parser.add_argument("--metrics_out", default="reports/metrics/week1_eval_metrics.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    artifact = joblib.load(args.model_path)
    model = artifact["model"]

    test = load_sign_mnist_csv(args.test_csv, normalize=True)
    preds = model.predict(test.x)

    labels = get_sorted_labels()
    letter_ticks = [label_to_letter(idx) for idx in labels]

    cm = confusion_matrix(test.y, preds, labels=labels)
    report = classification_report(test.y, preds, labels=labels, output_dict=True, zero_division=0)
    accuracy = accuracy_score(test.y, preds)

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(cm, annot=False, cmap="Blues", xticklabels=letter_ticks, yticklabels=letter_ticks, ax=ax)
    ax.set_title("Week 1 Baseline Confusion Matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    plt.tight_layout()

    confusion_path = Path(args.confusion_out)
    confusion_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(confusion_path, dpi=200)
    plt.close(fig)

    metrics = {
        "accuracy": accuracy,
        "classification_report": report,
        "labels": labels,
        "letters": letter_ticks,
    }
    metrics_path = Path(args.metrics_out)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Saved confusion matrix to: {confusion_path}")
    print(f"Saved metrics to: {metrics_path}")
    print(f"Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()
