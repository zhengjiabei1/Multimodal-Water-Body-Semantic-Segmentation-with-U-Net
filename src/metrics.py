"""Segmentation metrics shared by evaluation scripts."""

from __future__ import annotations

import numpy as np


def compute_metrics(
    prediction: np.ndarray, label: np.ndarray, threshold: float = 0.5
) -> dict[str, float]:
    """Compute pixel-wise binary classification metrics.

    Args:
        prediction: Probability map or binary mask.
        label: Binary reference mask.
        threshold: Values above this threshold are classified as water.
    """
    flat_prediction = np.asarray(prediction).reshape(-1)
    flat_label = np.asarray(label).reshape(-1)
    valid = ~np.isnan(flat_prediction)
    predicted = flat_prediction[valid] > threshold
    target = flat_label[valid] > 0.5

    tp = np.logical_and(predicted, target).sum()
    fp = np.logical_and(predicted, ~target).sum()
    fn = np.logical_and(~predicted, target).sum()
    tn = np.logical_and(~predicted, ~target).sum()

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0

    return {
        "Accuracy": float(accuracy),
        "Precision": float(precision),
        "Recall": float(recall),
        "IoU": float(iou),
        "F1": float(f1),
    }
