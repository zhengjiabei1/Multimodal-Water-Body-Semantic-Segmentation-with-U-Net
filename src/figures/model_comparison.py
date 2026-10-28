"""Compare segmentation performance across input configurations and baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLORS = {
    "A: RGB": "#3A6B8C",
    "B: RGB + NIR": "#D98C2E",
    "C: RGB + NIR + MNDWI": "#C44E52",
    "D: All channels": "#7F7F7F",
}
METRICS = ("Precision", "Recall", "IoU", "F1", "Kappa")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models", type=Path, default=Path("results/model_comparison.json")
    )
    parser.add_argument(
        "--indices", type=Path, default=Path("results/spectral_index.json")
    )
    parser.add_argument(
        "--forest", type=Path, default=Path("results/random_forest.json")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("docs/figures/model_comparison.png")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    models = load_json(args.models)
    indices = load_json(args.indices)
    forest = load_json(args.forest)

    fig, (left, right) = plt.subplots(
        1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [1.35, 1]}
    )

    # (a) Detailed metrics for the four U-Net input configurations.
    names = list(models)
    x = np.arange(len(METRICS))
    width = 0.8 / max(len(names), 1)
    for offset, name in enumerate(names):
        values = [models[name][metric] for metric in METRICS]
        left.bar(
            x + offset * width - 0.4 + width / 2,
            values,
            width,
            label=name,
            color=COLORS.get(name, "#999999"),
            edgecolor="white",
            linewidth=0.5,
        )

    left.set_xticks(x)
    left.set_xticklabels(METRICS)
    left.set_ylim(0.5, 1.0)
    left.set_ylabel("Score")
    left.set_title("(a) U-Net input configurations", fontweight="bold")
    left.grid(axis="y", linestyle="--", alpha=0.25)
    left.legend(
        frameon=False,
        fontsize=8.5,
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 1.0),
    )
    left.spines["top"].set_visible(False)
    left.spines["right"].set_visible(False)

    # (b) F1-score of index thresholds, random forest and the best U-Net.
    methods = []
    scores = []
    colors = []
    for name, result in indices.items():
        methods.append(f"{name} threshold")
        scores.append(result["F1"])
        colors.append("#B0B0B0")
    if forest:
        methods.append("Random forest")
        scores.append(forest["F1"])
        colors.append("#8C8C8C")
    best_unet = max(models, key=lambda key: models[key]["F1"]) if models else None
    if best_unet:
        methods.append(f"U-Net {best_unet.split(':')[0]}")
        scores.append(models[best_unet]["F1"])
        colors.append("#C44E52")

    right.barh(methods[::-1], scores[::-1], color=colors[::-1], height=0.55)
    for position, score in enumerate(scores[::-1]):
        right.text(score + 0.005, position, f"{score:.3f}", va="center", fontsize=9)
    right.set_xlim(0.0, 0.95)
    right.set_xlabel("F1-score")
    right.set_title("(b) Method comparison", fontweight="bold")
    right.grid(axis="x", linestyle="--", alpha=0.25)
    right.spines["top"].set_visible(False)
    right.spines["right"].set_visible(False)

    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200, bbox_inches="tight")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
