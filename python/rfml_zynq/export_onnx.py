from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .model import model_from_config
from .utils import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default="outputs/model.onnx")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model = model_from_config(cfg)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    frame_length = int(cfg["data"]["frame_length"])
    dummy = torch.zeros(1, 2, frame_length, dtype=torch.float32)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy,
        output,
        input_names=["iq"],
        output_names=["logits"],
        dynamic_axes={"iq": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )
    print(f"Exported ONNX model: {output}")


if __name__ == "__main__":
    main()
