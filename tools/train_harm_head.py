"""Fit the linear residual harm head on the supervised split and export it."""
import argparse
import json
from pathlib import Path

import numpy as np
import sklearn
import torch
from safetensors.torch import save_file

from lgwm.benchmark import read_jsonl
from lgwm.data.transitions import sha256_file
from lgwm.evaluation import fit_harm_head
from lgwm.features import readout_inputs

ROOT = Path(__file__).resolve().parents[1]


def load_features(directory):
    meta = json.loads((directory / "keys.json").read_text())
    keys = [tuple(k.split("|", 1)) for k in meta["keys"]]
    arrays = {n: np.load(directory / f"{n}.npy") for n in ("pool_zhat", "pool_zt1", "pool_action")}
    return keys, arrays, meta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "benchmark/rswt_bench_train.jsonl")
    parser.add_argument("--features", type=Path, required=True, help="output of extract_features.py for the manifest")
    parser.add_argument("--output", type=Path, default=ROOT / "weights/lgwm-harm-head")
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / "weights") or output.exists():
        parser.error("Output must be a new directory inside weights/")
    records = read_jsonl(args.manifest)
    keys, arrays, meta = load_features(args.features)
    if meta["manifest_sha256"] != sha256_file(args.manifest):
        raise ValueError("Features were extracted for a different manifest")
    _, features = readout_inputs(records, keys, arrays)
    head = fit_harm_head(features, [r["cls"] for r in records])
    output.mkdir(parents=True)
    save_file({k: torch.as_tensor(np.atleast_1d(v), dtype=torch.float64) for k, v in head.items()},
              str(output / "head.safetensors"), metadata={"format": "lgwm-harm-head-v1"})
    config = {
        "format": "lgwm-harm-head-v1",
        "features": "concat(unit(mean z_obs) - unit(mean z_hat), mean valid action tokens)", "dim": features.shape[1],
        "positive_class": "1_hijack", "negative_classes": "2_paired_safe, 3_legit_cred, 4_surprising, 5_benign_misplace",
        "model": "StandardScaler + LogisticRegression(C=1.0, lbfgs, max_iter=3000), no class reweighting",
        "score": "sigmoid(((x - mean) / scale) @ coef + intercept), x cast to float32 first",
        "training_items": len(records), "training_positives": int(sum(r["cls"] == "1_hijack" for r in records)),
        "training_manifest_sha256": meta["manifest_sha256"], "world_model_weights_sha256": meta["weights_sha256"],
        "observed_encoder": meta["observed_encoder"], "feature_precision": meta["precision"],
        "feature_storage": meta["storage"], "sklearn": sklearn.__version__,
        "license": "Apache-2.0",
    }
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (output / "SHA256SUMS").write_text(f"{sha256_file(output / 'head.safetensors')}  head.safetensors\n")
    print(json.dumps(config, indent=2))


if __name__ == "__main__":
    main()
