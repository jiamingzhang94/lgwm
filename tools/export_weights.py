"""Export a local checkpoint without optimizer state or private training paths."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "weights/lgwm-online")
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / "weights"):
        parser.error("Output must stay inside this release workspace's weights/ directory")
    if output.exists():
        parser.error("Refusing to overwrite an existing export")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True, mmap=True)
    cfg = checkpoint["cfg"]["model"]
    if cfg.get("coord_posenc", "shared") != "shared":
        raise ValueError("Only the core shared-coordinate model is supported")
    weights = {k: v.detach().contiguous() for k, v in checkpoint["model"].items()
               if k.startswith(("encoder.", "action_encoder.", "predictor."))}
    if {k.split(".")[0] for k in weights} != {"encoder", "action_encoder", "predictor"}:
        raise ValueError("Checkpoint is missing a runtime component")
    config = {
        "format": "lgwm-inference-v1",
        "step": int(checkpoint["step"]), "observed_encoder": "online",
        "model": {k: cfg[k] for k in ["encoder", "encoder_hw", "pred_depth", "pred_heads"]},
        "text_model": "sentence-transformers/all-MiniLM-L6-v2",
        "text_field": "canonical action text, not task instruction",
        "pooling": "mean spatial tokens, then L2 normalize each screen vector",
        "source_checkpoint_sha256": sha256(args.checkpoint),
        "license": "Apache-2.0",
    }
    output.mkdir(parents=True)
    save_file(weights, str(output / "model.safetensors"), metadata={"format": config["format"]})
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    digest = sha256(output / "model.safetensors")
    (output / "SHA256SUMS").write_text(f"{digest}  model.safetensors\n")
    print(json.dumps({"output": str(output), "step": config["step"],
                      "state_elements": sum(v.numel() for v in weights.values()),
                      "bytes": (output / "model.safetensors").stat().st_size,
                      "sha256": digest}, indent=2))


if __name__ == "__main__":
    main()
