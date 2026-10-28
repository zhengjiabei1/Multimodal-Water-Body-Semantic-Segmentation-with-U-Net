"""Draw an English version of the study technical route."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle

NAVY = "#0b1736"
HEADER = "#0e1c3d"
BODY = "#d9dbe6"
EDGE = "#4d5f7d"
ARROW = "#7d8a9a"
DASH = "#264a8a"

plt.rcParams["font.family"] = "Arial"


def draw_box(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str | None = None,
    lines: tuple[str, ...] = (),
    title_fraction: float = 0.36,
    body_size: float = 6.6,
) -> None:
    """Draw a light box with an optional dark header."""
    if title:
        header_height = h * title_fraction
        body_height = h - header_height
        ax.add_patch(
            Rectangle(
                (x, y), w, body_height, facecolor=BODY, edgecolor=EDGE, linewidth=0.8
            )
        )
        ax.add_patch(
            Rectangle(
                (x, y + body_height),
                w,
                header_height,
                facecolor=HEADER,
                edgecolor=EDGE,
                linewidth=0.8,
            )
        )
        ax.text(
            x + w / 2,
            y + body_height + header_height / 2,
            title,
            ha="center",
            va="center",
            fontsize=7.0,
            fontweight="bold",
            color="white",
        )
        if lines:
            ax.text(
                x + w / 2,
                y + body_height / 2,
                "\n".join(lines),
                ha="center",
                va="center",
                fontsize=body_size,
                color="#111111",
                linespacing=1.25,
            )
    else:
        ax.add_patch(
            Rectangle((x, y), w, h, facecolor=BODY, edgecolor=EDGE, linewidth=0.8)
        )
        if lines:
            ax.text(
                x + w / 2,
                y + h / 2,
                "\n".join(lines),
                ha="center",
                va="center",
                fontsize=body_size,
                color="#111111",
                linespacing=1.25,
            )


def arrow(ax, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(
            arrowstyle="-|>", color=ARROW, linewidth=1.3, shrinkA=0, shrinkB=0
        ),
    )


def band(ax, number: int, top: float, bottom: float) -> None:
    """Draw the step marker and the dashed outline for one stage."""
    height = top - bottom
    ax.add_patch(
        FancyBboxPatch(
            (1.4, bottom),
            7.2,
            height,
            boxstyle="round,pad=0,rounding_size=0.5",
            facecolor=NAVY,
            edgecolor="none",
        )
    )
    center = (top + bottom) / 2
    ax.add_patch(Circle((5.0, center), 1.55, fill=False, edgecolor="white", linewidth=1.2))
    ax.text(
        5.0,
        center,
        str(number),
        ha="center",
        va="center",
        color="white",
        fontsize=10,
        fontweight="bold",
    )
    ax.add_patch(
        Rectangle(
            (11.0, bottom),
            86.5,
            height,
            fill=False,
            edgecolor=DASH,
            linewidth=1.1,
            linestyle=(0, (3, 2)),
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("docs/figures/technical_route.png")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fig, ax = plt.subplots(figsize=(9.0, 14.4))
    ax.set_xlim(0, 100)
    ax.set_ylim(22, 160)
    ax.axis("off")

    # Band boundaries: (top, bottom).
    bands = [
        (158.0, 140.0),
        (138.0, 116.0),
        (113.0, 96.0),
        (93.0, 82.0),
        (79.0, 62.0),
        (59.0, 46.0),
        (43.0, 24.0),
    ]
    for number, (top, bottom) in enumerate(bands, start=1):
        band(ax, number, top, bottom)

    # ---------------------------------------------------------------- stage 1
    top, bottom = bands[0]
    y = bottom + 2.0
    h = top - bottom - 4.0
    items = [
        ("Study area", ("Min-Dadu-Qingyi", "river confluence")),
        ("Data source", ("Sentinel-2 MSI", "Level-2A")),
        ("Time period", ("August-September", "2025")),
        ("Scene selection", ("Cloud cover", "< 10%")),
        ("Selection result", ("5 scenes",)),
    ]
    x_positions = [12.5, 29.5, 46.5, 63.5, 80.5]
    width = 14.0
    for x, (title, lines) in zip(x_positions, items):
        draw_box(ax, x, y, width, h, title, lines)
    for x in x_positions[:-1]:
        arrow(ax, (x + width, y + h / 2), (x + width + 3.0, y + h / 2))

    # ---------------------------------------------------------------- stage 2
    top, bottom = bands[1]
    # Upper row: cloud masking and mosaic.
    row_a_y = top - 9.0
    row_a_h = 7.0
    row_a = [
        (12.5, 20.0, "Cloud and shadow masking"),
        (36.5, 20.0, "Sort by cloud cover"),
        (60.5, 20.0, "Scene mosaicking"),
        (84.5, 11.5, "Composite\nimage"),
    ]
    for x, w, label in row_a:
        draw_box(ax, x, row_a_y, w, row_a_h, None, tuple(label.split("\n")), body_size=6.5)
    for (x, w, _), (nx, _, _) in zip(row_a[:-1], row_a[1:]):
        arrow(ax, (x + w, row_a_y + row_a_h / 2), (nx, row_a_y + row_a_h / 2))

    # "Common preprocessing" tag.
    tag_y = top - 13.0
    ax.add_patch(
        Rectangle((62.0, tag_y), 17.0, 3.0, facecolor=HEADER, edgecolor="none")
    )
    ax.text(
        70.5,
        tag_y + 1.5,
        "Common preprocessing",
        ha="center",
        va="center",
        color="white",
        fontsize=6.3,
        fontweight="bold",
    )
    arrow(ax, (70.5, row_a_y), (70.5, tag_y + 3.0))

    # Lower row: common preprocessing operations.
    row_b_y = bottom + 2.0
    row_b_h = 6.5
    row_b = [
        (12.5, 16.0, "Band selection"),
        (32.5, 16.0, "Spatial clipping"),
        (52.5, 20.0, "Resolution\nharmonisation"),
        (76.5, 19.5, "Normalisation"),
    ]
    for x, w, label in row_b:
        draw_box(ax, x, row_b_y, w, row_b_h, None, tuple(label.split("\n")), body_size=6.5)
    for (x, w, _), (nx, _, _) in zip(row_b[:-1], row_b[1:]):
        arrow(ax, (x + w, row_b_y + row_b_h / 2), (nx, row_b_y + row_b_h / 2))
    arrow(ax, (70.5, tag_y), (70.5, row_b_y + row_b_h))

    # ---------------------------------------------------------------- stage 3
    top, bottom = bands[2]
    y = bottom + 1.5
    h = top - bottom - 3.0
    draw_box(ax, 12.5, y, 21.0, h, "Water index calculation", ("NDWI    MNDWI",))
    draw_box(
        ax,
        38.5,
        y,
        23.0,
        h,
        "Index enhancement and\ninitial segmentation",
        ("Initial water mask",),
    )
    draw_box(
        ax,
        66.5,
        y,
        16.0,
        h,
        "Manual correction",
        ("High-resolution", "basemap", "Visual interpretation"),
        body_size=6.0,
    )
    draw_box(ax, 85.5, y, 11.0, h, "Final labels", ("Binary mask",))
    for x, w in [(33.5, 5.0), (61.5, 5.0), (82.5, 3.0)]:
        arrow(ax, (x, y + h / 2), (x + w, y + h / 2))

    # ---------------------------------------------------------------- stage 4
    top, bottom = bands[3]
    y = bottom + 2.0
    h = top - bottom - 4.0
    for x, label in [
        (12.5, "Patch extraction"),
        (35.0, "Data augmentation"),
        (57.5, "Dataset split"),
        (79.0, "Dataset output"),
    ]:
        draw_box(ax, x, y, 16.5, h, None, (label,), body_size=7.0)
    for x in (29.0, 51.5, 74.0):
        arrow(ax, (x, y + h / 2), (x + 6.0, y + h / 2))

    # ---------------------------------------------------------------- stage 5
    top, bottom = bands[4]
    y = bottom + 1.5
    h = top - bottom - 3.0
    draw_box(
        ax,
        12.5,
        y,
        28.0,
        h,
        "Input feature schemes",
        ("Scheme 1: RGB", "Scheme 2: RGB + NIR", "Scheme 3: RGB + NIR + MNDWI", "Scheme 4: All channels"),
        body_size=6.2,
    )
    draw_box(
        ax,
        45.0,
        y,
        22.0,
        h,
        "U-Net",
        ("Encoder-decoder", "Skip connections"),
    )
    draw_box(
        ax,
        72.0,
        y,
        25.0,
        h,
        "Loss and training",
        ("BCE + Dice", "Adam"),
    )
    arrow(ax, (40.5, y + h / 2), (45.0, y + h / 2))
    arrow(ax, (67.0, y + h / 2), (72.0, y + h / 2))

    # ---------------------------------------------------------------- stage 6
    top, bottom = bands[5]
    y = bottom + 1.5
    h = top - bottom - 3.0
    draw_box(
        ax,
        12.5,
        y,
        26.0,
        h,
        "Model inference",
        ("Test imagery", "Water probability map"),
    )
    draw_box(ax, 44.0, y, 52.0, h, None, ("Final water prediction map",), body_size=7.5)
    arrow(ax, (38.5, y + h / 2), (44.0, y + h / 2))

    # ---------------------------------------------------------------- stage 7
    top, bottom = bands[6]
    y = bottom + 1.5
    h = top - bottom - 3.0
    blocks = [
        (12.0, 15.0, "Accuracy metrics", ("OA    PA", "UA", "IoU", "F1-score")),
        (28.5, 15.0, "Quantitative\ncomparison", ("Compare", "input schemes")),
        (45.0, 15.0, "Visual analysis", ("Representative", "areas", "Detail comparison")),
        (61.5, 17.0, "Error analysis", ("Building/terrain", "shadows", "Water-land", "Small water bodies")),
        (80.0, 16.0, "Conclusion and\noutlook", ("Summary of findings", "Limitations and", "improvements")),
    ]
    for x, w, title, lines in blocks:
        draw_box(ax, x, y, w, h, title, lines, body_size=5.6)
    for (x, w, _, _), (nx, _, _, _) in zip(blocks[:-1], blocks[1:]):
        arrow(ax, (x + w, y + h / 2), (nx, y + h / 2))

    # Vertical connectors between stages.
    for (_, bottom), (next_top, _) in zip(bands[:-1], bands[1:]):
        arrow(ax, (54.5, bottom), (54.5, next_top))

    fig.tight_layout(pad=0.1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220, bbox_inches="tight", facecolor="white")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
