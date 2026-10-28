"""Train a U-Net for water body segmentation under a given input configuration."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

try:
    from .schemes import BAND_NAMES, INPUT_SCHEMES, SCHEME_LABELS
    from .unet import UNet
except ImportError:  # Allows `python src/train.py`.
    from schemes import BAND_NAMES, INPUT_SCHEMES, SCHEME_LABELS
    from unet import UNet


class DiceLoss(nn.Module):
    """Soft Dice loss for binary segmentation."""

    def __init__(self, smooth: float = 1e-6) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        prediction = prediction.reshape(-1)
        target = target.reshape(-1)
        intersection = (prediction * target).sum()
        dice = (2 * intersection + self.smooth) / (
            prediction.sum() + target.sum() + self.smooth
        )
        return 1 - dice


class PatchDataset(Dataset):
    """Tensor dataset with geometric and spectral augmentation for training."""

    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        channels: tuple[int, ...],
        augment: bool,
    ) -> None:
        self.images = images
        self.labels = labels
        self.channels = channels
        self.augment = augment
        # Positions of raw reflectance bands within the selected channels.
        self.raw_positions = [i for i, c in enumerate(channels) if c < 5]

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = self.images[index][list(self.channels)]
        label = self.labels[index]

        if self.augment:
            if torch.rand(1).item() > 0.5:
                image = np.flip(image, axis=2).copy()
                label = np.flip(label, axis=1).copy()
            if torch.rand(1).item() > 0.5:
                image = np.flip(image, axis=1).copy()
                label = np.flip(label, axis=0).copy()
            if torch.rand(1).item() > 0.5:
                rotations = torch.randint(1, 4, (1,)).item()
                image = np.rot90(image, rotations, axes=(1, 2)).copy()
                label = np.rot90(label, rotations).copy()
            if self.raw_positions and torch.rand(1).item() > 0.3:
                factor = 1 + (torch.rand(1).item() - 0.5) * 0.2
                for position in self.raw_positions:
                    image[position] = np.clip(image[position] * factor, 0, 1)
            if self.raw_positions and torch.rand(1).item() > 0.5:
                position = random.choice(self.raw_positions)
                image[position] = np.clip(
                    image[position] + torch.randn(1).item() * 0.02, 0, 1
                )

        return (
            torch.from_numpy(image.astype(np.float32)),
            torch.from_numpy(label.astype(np.float32)).unsqueeze(0),
        )


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion_bce: nn.Module,
    criterion_dice: nn.Module,
    device: torch.device,
    optimizer: optim.Optimizer | None = None,
) -> float:
    training = optimizer is not None
    model.train() if training else model.eval()
    total_loss = 0.0
    context = torch.enable_grad() if training else torch.no_grad()

    with context:
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            predictions = model(images)
            loss = criterion_bce(predictions, labels) + criterion_dice(
                predictions, labels
            )
            if training:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total_loss += loss.item()

    return total_loss / max(len(loader), 1)


def save_history(path: Path, scheme: str, history: dict[str, list[float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = {}
    if path.exists():
        existing = json.loads(path.read_text())
    existing[scheme] = history
    path.write_text(json.dumps(existing, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/processed/water_dataset.npz"),
        help="Patch dataset produced by build_dataset.py.",
    )
    parser.add_argument("--scheme", choices=tuple(INPUT_SCHEMES), default="C")
    parser.add_argument("--output", type=Path, default=None, help="Checkpoint path.")
    parser.add_argument(
        "--history", type=Path, default=Path("results/training_history.json")
    )
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience-es", type=int, default=10)
    parser.add_argument("--patience-lr", type=int, default=5)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device", default="auto", help="auto, cpu, cuda, cuda:0 or mps."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    scheme = INPUT_SCHEMES[args.scheme]
    channels = scheme["channels"]
    device = resolve_device(args.device)
    print(f"Device: {device}")
    print(f"Scheme {args.scheme}: {scheme['name']} ({scheme['description']})")
    print("Input channels: " + ", ".join(BAND_NAMES[c] for c in channels))

    data = np.load(args.data)
    train_set = PatchDataset(data["X_train"], data["y_train"], channels, augment=True)
    val_set = PatchDataset(data["X_val"], data["y_val"], channels, augment=False)
    print(
        f"Train patches: {len(train_set)} | Validation patches: {len(val_set)}"
    )

    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(train_set, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_set, shuffle=False, **loader_kwargs)

    model = UNet(in_channels=len(channels)).to(device)
    parameter_count = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {parameter_count / 1e6:.1f}M")

    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=args.patience_lr, min_lr=1e-6
    )
    criterion_bce = nn.BCELoss()
    criterion_dice = DiceLoss()

    output = args.output or Path(f"checkpoints/unet_{args.scheme}.pth")
    output.parent.mkdir(parents=True, exist_ok=True)

    best_loss = float("inf")
    stale_epochs = 0
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(
            model, train_loader, criterion_bce, criterion_dice, device, optimizer
        )
        val_loss = run_epoch(
            model, val_loader, criterion_bce, criterion_dice, device
        )
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        scheduler.step(val_loss)

        if epoch % args.print_every == 0 or epoch == 1:
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch:3d}/{args.epochs} | train {train_loss:.4f} | "
                f"val {val_loss:.4f} | lr {lr:.2e}"
            )

        if val_loss < best_loss - 1e-5:
            best_loss = val_loss
            stale_epochs = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "scheme": args.scheme,
                    "in_channels": len(channels),
                    "epoch": epoch,
                    "val_loss": val_loss,
                },
                output,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience_es:
                print(f"Early stopping at epoch {epoch}.")
                break

    save_history(args.history, args.scheme, history)
    print(f"Best validation loss: {best_loss:.4f}")
    print(f"Checkpoint: {output}")
    print(f"History: {args.history}")


if __name__ == "__main__":
    main()
