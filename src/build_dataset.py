"""Build the patch dataset used for training and evaluation.

The input is a multichannel raster (Sentinel-2 bands and water indices) together
with a binary water mask. The raster is cut into overlapping patches, invalid
patches are discarded, and the result is stored as a compressed NumPy archive
with train/validation/test splits.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import rasterio


def load_rasters(image_path: Path, label_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read the image stack and the binary label mask."""
    with rasterio.open(image_path) as src:
        image = src.read().astype(np.float32)
        height, width = src.height, src.width
        bands = src.count
    with rasterio.open(label_path) as src:
        label = src.read(1).astype(np.float32)
        label = (label > 0.5).astype(np.float32)

    if label.shape != (height, width):
        raise ValueError(
            f"Image and label grids differ: {(height, width)} vs {label.shape}"
        )

    print(f"Image: {height} x {width} pixels, {bands} channels")
    print(f"Water pixels: {label.mean() * 100:.2f}%")
    return image, label


def extract_patches(
    image: np.ndarray,
    label: np.ndarray,
    patch_size: int,
    stride: int,
    column_range: tuple[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Cut the rasters into patches, skipping patches that contain NaNs."""
    height, width = label.shape
    col_start, col_end = column_range if column_range else (0, width)

    patches, targets = [], []
    for row in range(0, height - patch_size + 1, stride):
        for col in range(col_start, col_end - patch_size + 1, stride):
            patch = image[:, row : row + patch_size, col : col + patch_size]
            if np.isnan(patch).any():
                continue
            patches.append(patch)
            targets.append(label[row : row + patch_size, col : col + patch_size])

    if not patches:
        raise RuntimeError("No valid patches were extracted from the given rasters.")

    return (
        np.asarray(patches, dtype=np.float32),
        np.asarray(targets, dtype=np.float32),
    )


def random_split(
    images: np.ndarray, labels: np.ndarray, seed: int
) -> tuple[np.ndarray, ...]:
    """Shuffle patches and split them into 70% / 15% / 15% subsets."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(images))
    n_train = int(len(images) * 0.70)
    n_val = int(len(images) * 0.15)

    train = order[:n_train]
    val = order[n_train : n_train + n_val]
    test = order[n_train + n_val :]
    return images[train], labels[train], images[val], labels[val], images[test], labels[test]


def spatial_split(
    image: np.ndarray, label: np.ndarray, patch_size: int, stride: int
) -> tuple[np.ndarray, ...]:
    """Split the scene into contiguous column blocks (70% / 15% / 15%).

    Unlike a random patch split, neighbouring patches never appear in different
    subsets, which gives a stricter estimate of spatial generalisation.
    """
    width = label.shape[1]
    train_end = int(width * 0.70)
    val_end = int(width * 0.85)

    if min(train_end, val_end - train_end, width - val_end) < patch_size:
        raise ValueError(
            "The raster is too narrow for a spatial split with "
            f"patch size {patch_size}."
        )

    print(f"Train columns: 0-{train_end}")
    print(f"Validation columns: {train_end}-{val_end}")
    print(f"Test columns: {val_end}-{width}")

    train = extract_patches(image, label, patch_size, stride, (0, train_end))
    val = extract_patches(image, label, patch_size, stride, (train_end, val_end))
    test = extract_patches(image, label, patch_size, stride, (val_end, width))
    return (*train, *val, *test)


def report_split(name: str, images: np.ndarray, labels: np.ndarray) -> None:
    print(
        f"{name:<11} patches={len(images):4d}  water={labels.mean() * 100:5.2f}%"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image",
        type=Path,
        default=Path("data/raw/Leshan_AllBands.tif"),
        help="Multichannel image stack.",
    )
    parser.add_argument(
        "--label",
        type=Path,
        default=Path("data/raw/Leshan_JRC_WaterLabel.tif"),
        help="Binary water mask.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/water_dataset.npz"),
        help="Output archive.",
    )
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument(
        "--split",
        choices=("random", "spatial"),
        default="random",
        help="Random patch split or contiguous spatial blocks.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image, label = load_rasters(args.image, args.label)

    if args.split == "random":
        patches = extract_patches(image, label, args.patch_size, args.stride)
        print(f"Valid patches: {len(patches[0])}")
        splits = random_split(*patches, seed=args.seed)
    else:
        splits = spatial_split(image, label, args.patch_size, args.stride)

    x_train, y_train, x_val, y_val, x_test, y_test = splits
    report_split("Train", x_train, y_train)
    report_split("Validation", x_val, y_val)
    report_split("Test", x_test, y_test)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        X_train=x_train,
        y_train=y_train,
        X_val=x_val,
        y_val=y_val,
        X_test=x_test,
        y_test=y_test,
    )
    size_mb = args.output.stat().st_size / 1024**2
    print(f"Saved {args.output} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
