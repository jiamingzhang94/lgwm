"""Export a completed Stage A encoder as the initialization for main training."""
import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from lgwm.data.transitions import sha256_file

ROOT = Path(__file__).resolve().parents[1]


def check_complete(state):
    if state.get("complete") is not True:
        raise ValueError("Stage A checkpoint is incomplete; finish Stage A training before export")
    if (int(state['step']) <= 0 or int(state['frames']) <= 0 or float(state['epochs']) <= 0):
        raise ValueError("Invalid Stage A training metadata")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Completed tools/train_stage_a.py encoder checkpoint")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/stage_a")
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / "outputs") or output.exists():
        parser.error("Output must be a new directory inside outputs/")
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True, mmap=True)
    digest = sha256_file(args.checkpoint)
    check_complete(state)
    tensors = {k: v.detach().contiguous() for k, v in state["vit"].items()}
    import timm
    encoder = timm.create_model(state["encoder_name"], pretrained=False, num_classes=0,
                                img_size=(448, 224), global_pool="")
    encoder.load_state_dict(tensors, strict=True)
    if any(not torch.isfinite(t).all() for t in tensors.values() if t.is_floating_point()):
        raise ValueError("Non-finite Stage A weights")
    output.mkdir(parents=True)
    save_file(tensors, str(output / "encoder.safetensors"), metadata={"format": "lgwm-stage-a-init-v1"})
    config = {"format": "lgwm-stage-a-init-v1",
              "encoder": state["encoder_name"], "stage_a_step": int(state["step"]), "epochs": float(state["epochs"]),
              "unique_frames": int(state["frames"]), "load_into": "LGWM.encoder.vit (strict)",
              "source_checkpoint_sha256": digest}
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (output / "SHA256SUMS").write_text(f"{sha256_file(output / 'encoder.safetensors')}  encoder.safetensors\n")
    print(json.dumps(config, indent=2))


if __name__ == "__main__":
    main()
