"""Create qualitative comparison panels for representative sub-regions."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.windows import Window

REGIONS = (
    {"name": "Main channel and confluence", "x": 1800, "y": 2100},
    {"name": "Fragmented water bodies", "x": 6300, "y": 4500},
    {"name": "Built-up area", "x": 4800, "y": 4800},
)
SCHEMES = ("A", "B", "C")
COL_TITLES = (
    "Original image",
    "False colour composite",
    "Reference mask",
    "U-Net A\n(RGB)",
    "U-Net B\n(+NIR)",
    "U-Net C\n(+NIR+MNDWI)",
)


def read_window(
    path: Path, window: Window, indexes: list[int] | None = None
) -> np.ndarray:
    with rasterio.open(path) as src:
        return src.read(indexes=indexes, window=window)


def overlay_mask(background: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Draw water pixels in blue over a false colour composite."""
    output = background.copy()
    output[mask > 0.5] = (0.03, 0.10, 0.55)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image", type=Path, default=Path("data/raw/Leshan_RGB.tif")
    )
    parser.add_argument(
        "--stack",
        type=Path,
        default=Path("data/raw/Leshan_AllBands.tif"),
        help="Seven-channel stack used for the false colour composite.",
    )
    parser.add_argument(
        "--label",
        type=Path,
        default=Path("data/raw/Leshan_JRC_WaterLabel.tif"),
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("outputs"),
        help="Directory containing water_A.tif, water_B.tif and water_C.tif.",
    )
    parser.add_argument("--patch-size", type=int, default=300)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/qualitative_examples.png"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    size = args.patch_size

    fig, axes = plt.subplots(
        len(REGIONS), 6, figsize=(19, 4.3 * len(REGIONS))
    )

    for row, region in enumerate(REGIONS):
        window = Window(region["x"], region["y"], size, size)

        rgb = read_window(args.image, window)
        rgb = np.transpose(rgb[:3], (1, 2, 0))
        axes[row, 0].imshow(np.clip(rgb * 3, 0, 1))

        # Standard false colour composite: NIR, red and green.
        stack = read_window(args.stack, window, [4, 3, 2])
        false_colour = np.transpose(np.clip(stack * 3, 0, 1), (1, 2, 0))
        axes[row, 1].imshow(false_colour)

        reference = read_window(args.label, window)[0]
        axes[row, 2].imshow(reference > 0.5, cmap="gray", vmin=0, vmax=1)

        for column, scheme in enumerate(SCHEMES, start=3):
            path = args.predictions / f"water_{scheme}.tif"
            prediction = read_window(path, window)[0]
            axes[row, column].imshow(overlay_mask(false_colour, prediction))

        for column in range(6):
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
            if row == 0:
                axes[row, column].set_title(COL_TITLES[column], fontsize=11)

        row_label = region["name"]
        axes[row, 0].text(
            -0.08,
            0.5,
            row_label,
            transform=axes[row, 0].transAxes,
            rotation=90,
            ha="right",
            va="center",
            fontsize=11,
            fontweight="bold",
        )

    fig.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200, bbox_inches="tight")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
