"""Train MLP experiment model with optional W&B tracking (Week 2)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import accuracy_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.data.loaders import load_sign_mnist_csv
from src.neural.model import SignMLP
from src.utils.label_map import get_sorted_labels


def parse_hidden_dims(raw: str) -> list[int]:
    return [int(v.strip()) for v in raw.split(",") if v.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Week 2 MLP")
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--test_csv", required=True)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--hidden_dims", type=str, default="256,128")
    parser.add_argument("--activation", choices=["relu", "tanh", "gelu"], default="relu")
    parser.add_argument("--model_out", default="models/neural/week2_mlp.pt")
    parser.add_argument("--metrics_out", default="reports/metrics/week2_mlp_metrics.json")
    parser.add_argument("--curve_out", default="reports/figures/week2_loss_curves.png")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", default="hand-sign-predictor")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    hidden_dims = parse_hidden_dims(args.hidden_dims)

    train = load_sign_mnist_csv(args.train_csv, normalize=True)
    test = load_sign_mnist_csv(args.test_csv, normalize=True)

    labels = get_sorted_labels()
    label_to_idx = {label: idx for idx, label in enumerate(labels)}

    y_train_idx = torch.tensor([label_to_idx[v] for v in train.y], dtype=torch.long)
    y_test_idx = torch.tensor([label_to_idx[v] for v in test.y], dtype=torch.long)
    x_train = torch.tensor(train.x, dtype=torch.float32)
    x_test = torch.tensor(test.x, dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(x_train, y_train_idx), batch_size=args.batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SignMLP(input_dim=784, hidden_dims=hidden_dims, num_classes=len(labels), activation=args.activation).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    run = None
    if args.use_wandb:
        import wandb

        run = wandb.init(
            project=args.wandb_project,
            config={
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "hidden_dims": hidden_dims,
                "activation": args.activation,
            },
        )

    train_losses: list[float] = []
    val_losses: list[float] = []
    val_accs: list[float] = []

    for epoch in range(args.epochs):
        model.train()
        batch_losses: list[float] = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())

        train_loss = sum(batch_losses) / max(1, len(batch_losses))

        model.eval()
        with torch.no_grad():
            logits_test = model(x_test.to(device))
            val_loss = criterion(logits_test, y_test_idx.to(device)).item()
            pred_idx = logits_test.argmax(dim=1).cpu().numpy()
            val_acc = accuracy_score(y_test_idx.numpy(), pred_idx)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        if run is not None:
            run.log({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss, "val_acc": val_acc})

        print(
            f"Epoch {epoch + 1}/{args.epochs} - train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

    model_path = Path(args.model_out)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "hidden_dims": hidden_dims,
            "activation": args.activation,
            "labels": labels,
        },
        model_path,
    )

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(train_losses, label="train_loss")
    ax1.plot(val_losses, label="val_loss")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.plot(val_accs, color="green", linestyle="--", label="val_acc")
    ax2.set_ylabel("accuracy")
    ax2.legend(loc="upper right")
    plt.title("Week 2 MLP Training Curves")
    plt.tight_layout()

    curve_path = Path(args.curve_out)
    curve_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(curve_path, dpi=200)
    plt.close(fig)

    metrics = {
        "best_val_acc": max(val_accs) if val_accs else None,
        "final_val_acc": val_accs[-1] if val_accs else None,
        "train_loss_last": train_losses[-1] if train_losses else None,
        "val_loss_last": val_losses[-1] if val_losses else None,
        "epochs": args.epochs,
        "hidden_dims": hidden_dims,
        "activation": args.activation,
        "wandb_run_url": run.url if run is not None else None,
    }
    metrics_path = Path(args.metrics_out)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    if run is not None:
        run.finish()

    print(f"Saved model to: {model_path}")
    print(f"Saved metrics to: {metrics_path}")
    print(f"Saved training curves to: {curve_path}")


if __name__ == "__main__":
    main()
