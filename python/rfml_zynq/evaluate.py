from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from .dataset import dataset_from_config
from .model import model_from_config
from .utils import ensure_dir, load_config, resolve_device, set_seed


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--snr-bin-db", type=float, default=2.0)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg["seed"]))
    device = resolve_device(str(cfg["train"]["device"]))
    output_dir = ensure_dir(Path(cfg["output_dir"]) / "evaluation")

    model = model_from_config(cfg)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()

    dataset = dataset_from_config(cfg, "test", seed_offset=20_000_000)
    loader = DataLoader(
        dataset,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["train"]["num_workers"]),
    )

    labels_all: list[int] = []
    preds_all: list[int] = []
    snrs_all: list[float] = []

    for inputs, labels, snrs in loader:
        logits = model(inputs.to(device))
        preds = logits.argmax(dim=1).cpu()
        labels_all.extend(labels.tolist())
        preds_all.extend(preds.tolist())
        snrs_all.extend(snrs.tolist())

    classes = list(cfg["data"]["classes"])
    report = classification_report(
        labels_all,
        preds_all,
        target_names=classes,
        digits=4,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).transpose().to_csv(output_dir / "classification_report.csv")

    cm = confusion_matrix(labels_all, preds_all, labels=range(len(classes)))
    np.savetxt(output_dir / "confusion_matrix.csv", cm, delimiter=",", fmt="%d")

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111)
    image = ax.imshow(cm)
    fig.colorbar(image, ax=ax)
    ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix")
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=160)
    plt.close(fig)

    bins: dict[float, list[bool]] = defaultdict(list)
    step = float(args.snr_bin_db)
    for label, pred, snr in zip(labels_all, preds_all, snrs_all):
        center = round(snr / step) * step
        bins[center].append(label == pred)

    rows = [
        {
            "snr_db": snr,
            "accuracy": float(np.mean(correct)),
            "count": len(correct),
        }
        for snr, correct in sorted(bins.items())
    ]
    pd.DataFrame(rows).to_csv(output_dir / "accuracy_by_snr.csv", index=False)

    accuracy = float(np.mean(np.array(labels_all) == np.array(preds_all)))
    print(f"Test accuracy: {accuracy:.4f}")
    print(f"Results written to: {output_dir}")


if __name__ == "__main__":
    main()
