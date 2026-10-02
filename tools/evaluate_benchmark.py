"""Score RSWT-Bench and report the core metrics with donor-paired CIs."""
import argparse
import json
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file

from lgwm.benchmark import check_manifest, donor_bootstrap, donor_pairs, paper_table, read_jsonl
from lgwm.data.transitions import sha256_file
from lgwm.evaluation import predict_harm
from lgwm.features import readout_inputs

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "benchmark/rswt_bench_test.jsonl")
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--harm-head", type=Path, default=ROOT / "weights/lgwm-harm-head")
    parser.add_argument("--n-boot", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--output", type=Path, required=True, help="new JSON file inside outputs/")
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / "outputs") or output.exists():
        parser.error("Output must be a new file inside outputs/")

    records = read_jsonl(args.manifest)
    check_manifest(records, test_split=True)
    meta = json.loads((args.features / "keys.json").read_text())
    if meta["manifest_sha256"] != sha256_file(args.manifest):
        raise ValueError("Features were extracted for a different manifest")
    keys = [tuple(k.split("|", 1)) for k in meta["keys"]]
    arrays = {n: np.load(args.features / f"{n}.npy") for n in ("pool_zhat", "pool_zt1", "pool_action")}
    integrity, residual = readout_inputs(records, keys, arrays)
    head_config = json.loads((args.harm_head / "config.json").read_text())
    if head_config["world_model_weights_sha256"] != meta["weights_sha256"]:
        raise ValueError("Harm head was fit on features from different world-model weights")
    head = {k: (v if v.size > 1 else v.item()) for k, v in load_file(str(args.harm_head / "head.safetensors")).items()}
    harm = predict_harm(residual, head)

    classes, sids = [r["cls"] for r in records], [r["sid"] for r in records]
    pairs = donor_pairs(records)
    by_sid = {"integrity": dict(zip(sids, integrity)), "harm": dict(zip(sids, harm))}
    result = {
        "integrity": {**paper_table(classes, integrity),
                      "donor_paired": donor_bootstrap(by_sid["integrity"], pairs, n_boot=args.n_boot, seed=args.seed)},
        "harm_head": {**paper_table(classes, harm),
                      "donor_paired": donor_bootstrap(by_sid["harm"], pairs, n_boot=args.n_boot, seed=args.seed,
                                                      reference=by_sid["integrity"])},
        "provenance": {"manifest_sha256": meta["manifest_sha256"], "features": {k: v for k, v in meta.items() if k != "keys"},
                       "harm_head_sha256": sha256_file(args.harm_head / "head.safetensors"),
                       "protocol": "cross-split: head fit on the 6,786-item supervised split only"},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    output.with_name(output.stem + "_scores.json").write_text(json.dumps(
        {"sid": sids, "cls": classes, "scores": {"integrity": integrity.tolist(), "harm": harm.tolist()}}) + "\n")
    for name in ("integrity", "harm_head"):
        r = result[name]
        print(f"{name:10s}", {k: round(v, 4) for k, v in r["paper_columns"].items()},
              "CI", [round(x, 4) for x in r["donor_paired"]["rswt_ci95"]],
              "pairs", r["donor_paired"]["pairs_ordered"], "/", r["donor_paired"]["n_pairs"])


if __name__ == "__main__":
    main()
