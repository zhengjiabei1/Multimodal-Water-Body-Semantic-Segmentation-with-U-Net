"""Random forest baseline using per-pixel RGB + NIR reflectance."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier

try:
    from ..metrics import compute_metrics
except ImportError:  # Allows `python src/baselines/random_forest.py`.
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from metrics import compute_metrics


CHANNELS = (2, 1, 0, 3)  # B4, B3, B2, B8


def flatten_patches(images: np.ndarray) -> np.ndarray:
    selected = images[:, list(CHANNELS)]
    return selected.transpose(0, 2, 3, 1).reshape(-1, len(CHANNELS))


def sample_pixels(
    features: np.ndarray, labels: np.ndarray, limit: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    valid = ~np.isnan(features).any(axis=1)
    features, labels = features[valid], labels[valid]
    if len(features) <= limit:
        return features, labels

    rng = np.random.default_rng(seed)
    water = np.flatnonzero(labels == 1)
    non_water = np.flatnonzero(labels == 0)
    n_water = min(len(water), int(limit * labels.mean()))
    n_non_water = limit - n_water
    indices = np.concatenate(
        [
            rng.choice(water, n_water, replace=False),
            rng.choice(non_water, n_non_water, replace=False),
        ]
    )
    rng.shuffle(indices)
    return features[indices], labels[indices]


def predict_patches(
    model: RandomForestClassifier, images: np.ndarray
) -> np.ndarray:
    predictions = []
    for start in range(0, len(images), 10):
        batch = images[start : start + 10]
        features = flatten_patches(batch)
        valid = ~np.isnan(features).any(axis=1)
        output = np.zeros(len(features), dtype=np.float32)
        if valid.any():
            output[valid] = model.predict(features[valid])
        predictions.append(output.reshape(len(batch), *batch.shape[-2:]))
    return np.concatenate(predictions, axis=0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path, default=Path("data/processed/water_dataset.npz")
    )
    parser.add_argument("--max-pixels", type=int, default=200_000)
    parser.add_argument("--trees", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("results/random_forest.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = np.load(args.data)

    train_features = np.concatenate(
        [flatten_patches(data["X_train"]), flatten_patches(data["X_val"])]
    )
    train_labels = np.concatenate(
        [data["y_train"].reshape(-1), data["y_val"].reshape(-1)]
    )
    train_features, train_labels = sample_pixels(
        train_features, train_labels, args.max_pixels, args.seed
    )
    print(
        f"Training pixels: {len(train_features):,} "
        f"(water {train_labels.mean() * 100:.1f}%)"
    )

    model = RandomForestClassifier(
        n_estimators=args.trees,
        max_depth=15,
        min_samples_split=50,
        min_samples_leaf=10,
        class_weight="balanced",
        n_jobs=-1,
        random_state=args.seed,
        verbose=1,
    )
    start = time.time()
    model.fit(train_features, train_labels)
    print(f"Training time: {time.time() - start:.1f}s")

    prediction = predict_patches(model, data["X_test"])
    metrics = compute_metrics(prediction, data["y_test"])
    print(
        f"Random forest: F1={metrics['F1']:.4f}  IoU={metrics['IoU']:.4f}  "
        f"Precision={metrics['Precision']:.4f}  Recall={metrics['Recall']:.4f}"
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2))
    print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
