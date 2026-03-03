from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # repo root
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.label_map import label_to_letter

def main() -> None:
    csv_path = Path("data/raw/sign_mnist_test.csv")
    out_dir = Path("data/raw/app_test_images_28")

    per_label = 25  # export this many per class

    if not csv_path.exists():
        raise FileNotFoundError(f"Missing: {csv_path}")

    df = pd.read_csv(csv_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    for label, group in df.groupby("label"):
        label = int(label)
        letter = label_to_letter(label)

        folder = out_dir / letter
        folder.mkdir(parents=True, exist_ok=True)

        samples = group.sample(n=min(per_label, len(group)), random_state=0)
        for i, (_, row) in enumerate(samples.iterrows()):
            img28 = row.drop("label").to_numpy(dtype=np.uint8).reshape(28, 28)
            Image.fromarray(img28, mode="L").save(folder / f"{letter}_{label}_{i:02d}.png")

    print(f"Saved 28x28 images to: {out_dir.resolve()}")

if __name__ == "__main__":
    main()