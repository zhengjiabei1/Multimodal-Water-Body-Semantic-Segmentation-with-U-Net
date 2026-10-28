"""Evaluate trained U-Net models and water-index post-processing strategies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    from .metrics import compute_metrics
    from .schemes import INPUT_SCHEMES, SCHEME_LABELS
    from .unet import UNet
except ImportError:  # Allows `python src/evaluate.py`.
    from metrics import compute_metrics
    from schemes import INPUT_SCHEMES, SCHEME_LABELS
    from unet import UNet


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(path: Path, in_channels: int, device: torch.device) -> UNet:
    model = UNet(in_channels=in_channels).to(device)
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state)
    model.eval()
    return model


def predict_dataset(
    model: UNet,
    images: np.ndarray,
    channels: tuple[int, ...],
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    """Return per-pixel water probabilities for the whole split."""
    selected = np.ascontiguousarray(images[:, list(channels)])
    outputs = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(selected), batch_size):
            batch = torch.from_numpy(selected[start : start + batch_size]).to(device)
            outputs.append(model(batch).cpu().numpy()[:, 0])
    return np.concatenate(outputs, axis=0)


def normalize(array: np.ndarray) -> np.ndarray:
    finite = array[np.isfinite(array)]
    return (array - finite.min()) / (finite.max() - finite.min() + 1e-8)


def apply_postprocessing(
    probabilities: np.ndarray,
    mndwi: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    alpha: float,
    confidence: float,
) -> dict[str, dict[str, float]]:
    """Compare three ways of combining network output with the MNDWI index."""
    mndwi_scaled = normalize(mndwi)
    binary = (probabilities > 0.5).astype(np.float32)
    results: dict[str, dict[str, float]] = {}

    filtered = (binary * (mndwi > threshold)).astype(np.float32)
    results["Threshold filtering"] = compute_metrics(filtered, labels)

    fused = alpha * probabilities + (1 - alpha) * mndwi_scaled
    results["Probability fusion"] = compute_metrics((fused > 0.5).astype(float), labels)

    certainty = np.abs(probabilities - 0.5) * 2
    refined = binary.copy()
    uncertain = certainty < confidence
    mixed = 0.5 * probabilities + 0.5 * mndwi_scaled
    refined[uncertain] = (mixed[uncertain] > 0.5).astype(np.float32)
    results["Confidence-guided refinement"] = compute_metrics(refined, labels)

    return results


def print_table(title: str, results: dict[str, dict[str, float]]) -> None:
    print(f"\n{title}")
    print(f"{'Method':<32}{'Acc':>8}{'Prec':>8}{'Rec':>8}{'IoU':>8}{'F1':>8}")
    print("-" * 72)
    for name, metrics in results.items():
        print(
            f"{name:<32}{metrics['Accuracy']:>8.4f}{metrics['Precision']:>8.4f}"
            f"{metrics['Recall']:>8.4f}{metrics['IoU']:>8.4f}{metrics['F1']:>8.4f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path, default=Path("data/processed/water_dataset.npz")
    )
    parser.add_argument(
        "--checkpoints", type=Path, default=Path("checkpoints")
    )
    parser.add_argument(
        "--schemes",
        nargs="+",
        choices=tuple(INPUT_SCHEMES),
        default=list(INPUT_SCHEMES),
    )
    parser.add_argument("--post-scheme", choices=tuple(INPUT_SCHEMES), default="C")
    parser.add_argument("--mndwi-threshold", type=float, default=0.05)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--confidence", type=float, default=0.3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    print(f"Device: {device}")

    data = np.load(args.data)
    images, labels = data["X_test"], data["y_test"]
    mndwi = images[:, 6]
    print(f"Test patches: {len(images)}")

    model_results: dict[str, dict[str, float]] = {}
    probabilities: dict[str, np.ndarray] = {}

    for scheme_id in args.schemes:
        scheme = INPUT_SCHEMES[scheme_id]
        checkpoint = args.checkpoints / f"unet_{scheme_id}.pth"
        if not checkpoint.exists():
            print(f"Skipping {scheme_id}: {checkpoint} not found.")
            continue

        model = load_model(checkpoint, len(scheme["channels"]), device)
        probability = predict_dataset(
            model, images, scheme["channels"], device, args.batch_size
        )
        probabilities[scheme_id] = probability
        metrics = compute_metrics(probability, labels)
        model_results[SCHEME_LABELS[scheme_id]] = metrics

    if not model_results:
        raise SystemExit("No checkpoints were evaluated.")

    print_table("Model comparison", model_results)

    post_results = {}
    if args.post_scheme in probabilities:
        post_results = {
            "No post-processing": model_results[
                SCHEME_LABELS[args.post_scheme]
            ],
            **apply_postprocessing(
                probabilities[args.post_scheme],
                mndwi,
                labels,
                args.mndwi_threshold,
                args.alpha,
                args.confidence,
            ),
        }
        print_table(
            f"Post-processing (U-Net {args.post_scheme})", post_results
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "model_comparison.json").write_text(
        json.dumps(model_results, indent=2)
    )
    (args.output_dir / "postprocessing.json").write_text(
        json.dumps(post_results, indent=2)
    )
    print(f"\nResults written to {args.output_dir}")


if __name__ == "__main__":
    main()
