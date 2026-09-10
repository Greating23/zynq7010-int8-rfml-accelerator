from __future__ import annotations

import argparse
import csv

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from .dataset import dataset_from_config
from .model import model_from_config
from .utils import ensure_dir, load_config, resolve_device, set_seed


@torch.no_grad()
def evaluate_loader(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0
    criterion = nn.CrossEntropyLoss()
    for inputs, labels, _ in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        logits = model(inputs)
        loss = criterion(logits, labels)
        loss_sum += float(loss.item()) * labels.numel()
        correct += int((logits.argmax(dim=1) == labels).sum().item())
        total += labels.numel()
    return loss_sum / max(total, 1), correct / max(total, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg["seed"]))
    output_dir = ensure_dir(cfg["output_dir"])
    checkpoint_dir = ensure_dir(output_dir / "checkpoints")

    device = resolve_device(str(cfg["train"]["device"]))
    train_ds = dataset_from_config(cfg, "train", seed_offset=0)
    val_ds = dataset_from_config(cfg, "val", seed_offset=10_000_000)

    batch_size = int(cfg["train"]["batch_size"])
    num_workers = int(cfg["train"]["num_workers"])
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    model = model_from_config(cfg).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=float(cfg["train"]["learning_rate"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )
    criterion = nn.CrossEntropyLoss()

    best_val = -1.0
    patience = int(cfg["train"]["patience"])
    stale_epochs = 0
    history_path = output_dir / "history.csv"

    with open(history_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["epoch", "train_loss", "train_acc", "val_loss", "val_acc"],
        )
        writer.writeheader()

        for epoch in range(1, int(cfg["train"]["epochs"]) + 1):
            model.train()
            total = 0
            correct = 0
            loss_sum = 0.0

            for inputs, labels, _ in train_loader:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                logits = model(inputs)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

                loss_sum += float(loss.item()) * labels.numel()
                correct += int((logits.argmax(dim=1) == labels).sum().item())
                total += labels.numel()

            train_loss = loss_sum / max(total, 1)
            train_acc = correct / max(total, 1)
            val_loss, val_acc = evaluate_loader(model, val_loader, device)

            writer.writerow(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                }
            )
            handle.flush()
            print(
                f"epoch={epoch:03d} train_loss={train_loss:.4f} "
                f"train_acc={train_acc:.4f} val_loss={val_loss:.4f} "
                f"val_acc={val_acc:.4f}"
            )

            if val_acc > best_val:
                best_val = val_acc
                stale_epochs = 0
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "classes": cfg["data"]["classes"],
                        "config": cfg,
                        "best_val_acc": best_val,
                    },
                    checkpoint_dir / "best.pt",
                )
            else:
                stale_epochs += 1
                if stale_epochs >= patience:
                    print(f"Early stopping after {epoch} epochs.")
                    break

    print(f"Best validation accuracy: {best_val:.4f}")
    print(f"Checkpoint: {checkpoint_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
