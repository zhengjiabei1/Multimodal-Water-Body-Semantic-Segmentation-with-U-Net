"""Run sliding-window inference over a full satellite scene."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import rasterio
import torch

try:
    from .schemes import INPUT_SCHEMES
    from .unet import UNet
except ImportError:  # Allows `python src/infer.py`.
    from schemes import INPUT_SCHEMES
    from unet import UNet


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def tile_positions(length: int, tile: int, stride: int) -> list[int]:
    positions = list(range(0, length - tile + 1, stride))
    if positions[-1] != length - tile:
        positions.append(length - tile)
    return positions


def predict_scene(
    model: UNet,
    image: np.ndarray,
    channels: tuple[int, ...],
    tile: int,
    stride: int,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """Average predictions from overlapping tiles to avoid seam artefacts."""
    _, height, width = image.shape
    pad_h = max(tile, ((height + tile - 1) // tile) * tile)
    pad_w = max(tile, ((width + tile - 1) // tile) * tile)
    padded = np.pad(
        image, ((0, 0), (0, pad_h - height), (0, pad_w - width)), mode="constant"
    )

    probability = np.zeros((pad_h, pad_w), dtype=np.float32)
    counts = np.zeros((pad_h, pad_w), dtype=np.float32)

    rows = tile_positions(pad_h, tile, stride)
    cols = tile_positions(pad_w, tile, stride)
    windows = [(row, col) for row in rows for col in cols]
    print(f"Tiles: {len(windows)} ({len(rows)} x {len(cols)})")

    model.eval()
    with torch.no_grad():
        for start in range(0, len(windows), batch_size):
            batch_windows = windows[start : start + batch_size]
            batch = np.stack(
                [
                    padded[list(channels), row : row + tile, col : col + tile]
                    for row, col in batch_windows
                ]
            )
            inputs = torch.from_numpy(batch).to(device)
            outputs = model(inputs).cpu().numpy()[:, 0]
            for (row, col), output in zip(batch_windows, outputs):
                probability[row : row + tile, col : col + tile] += output
                counts[row : row + tile, col : col + tile] += 1
            if start % (batch_size * 20) == 0:
                print(f"  {start}/{len(windows)} tiles")

    probability /= np.maximum(counts, 1)
    return probability[:height, :width]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image", type=Path, default=Path("data/raw/Leshan_AllBands.tif")
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("checkpoints/unet_C.pth")
    )
    parser.add_argument(
        "--scheme",
        choices=tuple(INPUT_SCHEMES),
        default=None,
        help="Input configuration. Read from the checkpoint when available.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    print(f"Device: {device}")

    checkpoint = torch.load(
        args.checkpoint, map_location=device, weights_only=True
    )
    state = checkpoint.get("model_state_dict", checkpoint)
    scheme_id = args.scheme or checkpoint.get("scheme")
    if scheme_id not in INPUT_SCHEMES:
        # Older checkpoints store only the state dict; recover the number of
        # input channels from the first encoder convolution.
        for key, value in state.items():
            if key.endswith(("encoder.0.conv.0.weight", "encoders.0.block.0.weight")):
                in_channels = value.shape[1]
                scheme_id = next(
                    (
                        key
                        for key, config in INPUT_SCHEMES.items()
                        if len(config["channels"]) == in_channels
                    ),
                    None,
                )
                break
    if scheme_id not in INPUT_SCHEMES:
        raise SystemExit(
            "The scheme could not be determined. Please pass --scheme explicitly."
        )
    scheme = INPUT_SCHEMES[scheme_id]
    print(f"Scheme {scheme_id}: {scheme['name']}")

    model = UNet(in_channels=len(scheme["channels"])).to(device)
    model.load_state_dict(state)
    model.eval()

    with rasterio.open(args.image) as src:
        image = src.read().astype(np.float32)
        profile = src.profile
    image = np.nan_to_num(image, nan=0.0)
    print(f"Image: {image.shape[1]} x {image.shape[2]}, {image.shape[0]} channels")

    probability = predict_scene(
        model,
        image,
        scheme["channels"],
        args.tile_size,
        args.stride,
        args.batch_size,
        device,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    profile.update(dtype="float32", count=1, compress="lzw")
    prob_path = args.output_dir / f"prob_{scheme_id}.tif"
    with rasterio.open(prob_path, "w", **profile) as dst:
        dst.write(probability, 1)

    mask_path = args.output_dir / f"water_{scheme_id}.tif"
    with rasterio.open(mask_path, "w", **profile) as dst:
        dst.write((probability > args.threshold).astype(np.float32), 1)

    print(f"Probability map: {prob_path}")
    print(f"Binary mask: {mask_path}")


if __name__ == "__main__":
    main()
