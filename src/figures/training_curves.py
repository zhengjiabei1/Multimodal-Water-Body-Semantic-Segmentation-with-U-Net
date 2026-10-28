"""Plot validation loss curves for the four input configurations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLORS = {
    "A": "#3A6B8C",
    "B": "#D98C2E",
    "C": "#C44E52",
    "D": "#7F7F7F",
}
LABELS = {
    "A": "A: RGB",
    "B": "B: RGB + NIR",
    "C": "C: RGB + NIR + MNDWI",
    "D": "D: All channels",
}


def smooth(values: list[float], window: int = 4) -> np.ndarray:
    if len(values) < window:
        return np.asarray(values)
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--history", type=Path, default=Path("results/training_history.json")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("docs/figures/training_curves.png")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    history = json.loads(args.history.read_text())

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for scheme, label in LABELS.items():
        if scheme not in history:
            continue
        values = history[scheme]["val_loss"]
        smoothed = smooth(values)
        epochs = np.arange(len(values) - len(smoothed) + 1, len(values) + 1)
        ax.plot(
            epochs,
            smoothed,
            color=COLORS[scheme],
            linewidth=1.7,
            label=label,
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation loss")
    ax.set_title("Validation loss during training", fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200, bbox_inches="tight")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
