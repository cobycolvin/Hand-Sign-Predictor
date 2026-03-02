"""Convert custom image folders into Sign-MNIST-like CSV for training expansion."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from PIL import Image

from src.utils.label_map import get_label_to_letter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare custom image dataset as CSV")
    parser.add_argument("--input_dir", required=True, help="Folder with letter subfolders, e.g. data/raw/custom_images/A")
    parser.add_argument("--output_csv", default="data/processed/custom_asl_28x28.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)

    letter_to_label = {letter: label for label, letter in get_label_to_letter().items()}

    rows: list[dict[str, int]] = []
    for letter_dir in sorted(input_dir.iterdir()):
        if not letter_dir.is_dir():
            continue
        letter = letter_dir.name.upper()
        if letter not in letter_to_label:
            continue

        label = letter_to_label[letter]
        for img_path in sorted(letter_dir.glob("*")):
            if img_path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                continue
            image = Image.open(img_path).convert("L").resize((28, 28))
            pixels = list(image.getdata())

            row = {"label": label}
            for idx, value in enumerate(pixels, start=1):
                row[f"pixel{idx}"] = int(value)
            rows.append(row)

    if not rows:
        raise ValueError("No valid image files found. Check folder structure and file types.")

    df = pd.DataFrame(rows)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    print(f"Saved {len(df)} rows to {output_csv}")


if __name__ == "__main__":
    main()
