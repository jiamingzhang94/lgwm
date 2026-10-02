"""Encode every transition of a benchmark manifest with the released weights."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from lgwm.benchmark import read_jsonl
from lgwm.data.transitions import TextTable, find_index, sha256_file
from lgwm.features import TransitionItems, extract, load_tables, manifest_keys
from lgwm.inference import Verifier

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=ROOT / "weights/lgwm-online")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--index-dir", type=Path, action="append", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--text-table", type=Path, default=ROOT / "data/text_table.npz")
    parser.add_argument("--output", type=Path, required=True, help="new directory inside outputs/")
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--storage", choices=("float16", "float32"), default="float16")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / "outputs") or output.exists():
        parser.error("Output must be a new directory inside outputs/")

    records = read_jsonl(args.manifest)
    splits = {r[k]["split"] for r in records for k in ("base", "obs", "clean")}
    if len(splits) != 1:
        raise ValueError(f"Manifest mixes splits: {splits}")
    split = splits.pop()
    tables, index_paths = load_tables(records, args.index_dir, split, find_index)
    keys = manifest_keys(records)
    text = TextTable(args.text_table)
    items = TransitionItems(keys, tables, text, args.data_root.resolve())

    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    model = Verifier.from_pretrained(args.weights, device=args.device)
    arrays = extract(model.encoder, model.action_encoder, model.predictor, items, device=args.device,
                     batch_size=args.batch_size, workers=args.workers, precision=args.precision,
                     storage=np.dtype(args.storage).type)
    output.mkdir(parents=True)
    for name, value in arrays.items():
        np.save(output / f"{name}.npy", value)
    config = json.loads((args.weights / "config.json").read_text())
    meta = {
        "keys": [f"{s}|{t}" for s, t in keys], "n": len(keys), "dim": int(model.encoder.dim),
        "observed_encoder": config["observed_encoder"], "weights_sha256": sha256_file(args.weights / "model.safetensors"),
        "precision": args.precision, "storage": args.storage, "batch_size": args.batch_size,
        "torch": torch.__version__, "device": torch.cuda.get_device_name() if args.device.startswith("cuda") else "cpu",
        "manifest": args.manifest.name, "manifest_sha256": sha256_file(args.manifest), "split": split,
        "text_table_sha256": text.sha256, "index_sha256": {s: sha256_file(p) for s, p in index_paths.items()},
    }
    (output / "keys.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(json.dumps({k: v for k, v in meta.items() if k != "keys"}, indent=2))


if __name__ == "__main__":
    main()
