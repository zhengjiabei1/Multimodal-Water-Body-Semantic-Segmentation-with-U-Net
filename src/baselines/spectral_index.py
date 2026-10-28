"""Spectral water index baselines (NDWI and MNDWI) with threshold search."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from ..metrics import compute_metrics
except ImportError:  # Allows `python src/baselines/spectral_index.py`.
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from metrics import compute_metrics


def water_index(images: np.ndarray, first: int, second: int) -> np.ndarray:
    a = images[:, first]
    b = images[:, second]
    return (a - b) / (a + b + 1e-8)


def best_threshold(
    index: np.ndarray, labels: np.ndarray
) -> tuple[float, dict[str, float]]:
    best_score = -1.0
    best = (0.0, {})
    for threshold in np.arange(-0.5, 0.65, 0.05):
        prediction = (index > threshold).astype(np.float32)
        metrics = compute_metrics(prediction, labels)
        if metrics["IoU"] > best_score:
            best_score = metrics["IoU"]
            best = (float(threshold), metrics)
    return best


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path, default=Path("data/processed/water_dataset.npz")
    )
    parser.add_argument("--output", type=Path, default=Path("results/spectral_index.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = np.load(args.data)
    x_val, y_val = data["X_val"], data["y_val"]
    x_test, y_test = data["X_test"], data["y_test"]

    # Channel order: B2, B3, B4, B8, B11, NDWI, MNDWI.
    indices = {
        "NDWI": water_index(x_val, 1, 3),
        "MNDWI": water_index(x_val, 1, 4),
    }
    test_indices = {
        "NDWI": water_index(x_test, 1, 3),
        "MNDWI": water_index(x_test, 1, 4),
    }

    results = {}
    for name in indices:
        threshold, _ = best_threshold(indices[name], y_val)
        threshold = round(threshold, 2)
        prediction = (test_indices[name] > threshold).astype(np.float32)
        metrics = compute_metrics(prediction, y_test)
        results[name] = {"threshold": threshold, **metrics}
        print(
            f"{name:<6} threshold={threshold:+.2f}  "
            f"F1={metrics['F1']:.4f}  IoU={metrics['IoU']:.4f}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2))
    print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
