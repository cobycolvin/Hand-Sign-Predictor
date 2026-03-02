"""Label mapping utilities for Sign Language MNIST."""

from __future__ import annotations

from typing import Dict, List

# Sign Language MNIST contains 24 classes (static signs), commonly excluding J and Z.
# Dataset labels are encoded as integer indices where 9 and 25 are skipped.
LABEL_TO_LETTER: Dict[int, str] = {
    0: "A",
    1: "B",
    2: "C",
    3: "D",
    4: "E",
    5: "F",
    6: "G",
    7: "H",
    8: "I",
    10: "K",
    11: "L",
    12: "M",
    13: "N",
    14: "O",
    15: "P",
    16: "Q",
    17: "R",
    18: "S",
    19: "T",
    20: "U",
    21: "V",
    22: "W",
    23: "X",
    24: "Y",
}


def get_label_to_letter() -> Dict[int, str]:
    """Return immutable-style copy of index -> letter mapping."""
    return dict(LABEL_TO_LETTER)


def get_sorted_labels() -> List[int]:
    """Return labels in ascending order for deterministic reporting."""
    return sorted(LABEL_TO_LETTER.keys())


def label_to_letter(label: int) -> str:
    """Map numeric label to display letter (fallback to numeric string)."""
    return LABEL_TO_LETTER.get(label, str(label))
