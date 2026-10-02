"""Train the core LGWM world model (single GPU or ``torchrun --nproc_per_node=N``)."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
_CACHE_ROOT = Path(__file__).resolve().parents[1] / "outputs/cache"
os.environ.setdefault("HF_HOME", str(_CACHE_ROOT / "huggingface"))
os.environ.setdefault("TORCH_HOME", str(_CACHE_ROOT / "torch"))
import torch
import torch.distributed as dist
import yaml

from lgwm.data.dataset import collate
from lgwm.data.transitions import MixedSources, PortableTransitions, QuotaSampler, TextTable, find_index
from lgwm.train.metrics import _collate_window
from lgwm.train.trainer import Trainer, is_main

ROOT = Path(__file__).resolve().parents[1]


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/lgwm_main.yaml")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--index-dir", type=Path, action="append", help="replaces data.index_dirs")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-at", type=int, default=None, help="stop early without changing the schedule")
    parser.add_argument("--batch-size", type=int, help="global batch size")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--log-every", type=int)
    parser.add_argument("--eval-every", type=int)
    parser.add_argument("--ckpt-every", type=int)
    parser.add_argument("--init-encoder", help="Completed Stage A encoder safetensors path")
    parser.add_argument("--dist-backend", default="nccl", help="gloo allows DDP tests with several ranks on one GPU")
    args = parser.parse_args()
    if Path(args.run_name).name != args.run_name or args.run_name in {".", ".."}:
        parser.error("run-name must be a directory name")
    cfg = yaml.safe_load(args.config.read_text())
    deviations = {}

    def override(section, key, value, label, given=None):
        given = value is not None if given is None else given
        if given and value != cfg[section][key]:
            deviations[label] = {"paper": cfg[section][key], "run": value}
            cfg[section][key] = value

    override("train", "batch_size", args.batch_size, "train.batch_size")
    for key, value in (("every", args.log_every), ("eval_every", args.eval_every), ("ckpt_every", args.ckpt_every)):
        if value is not None:
            cfg["log"][key] = value
    if args.workers is not None:
        cfg["data"]["workers"] = args.workers
    if args.index_dir:
        cfg["data"]["index_dirs"] = [str(p) for p in args.index_dir]
    if args.init_encoder is not None:
        override("model", "init_encoder", args.init_encoder, "model.init_encoder")
    if not cfg["model"].get("init_encoder") or cfg["model"]["init_encoder"] == "none":
        parser.error("The core recipe requires a completed Stage A encoder")
    cfg["model"]["init_encoder"] = str(resolve(cfg["model"]["init_encoder"]))
    if not Path(cfg["model"]["init_encoder"]).is_file():
        parser.error("Missing Stage A encoder; run tools/train_stage_a.py and tools/export_stage_a_init.py first")
    if not cfg["data"].get("excluded_apps"):
        parser.error("The core recipe requires its fixed app exclusion list")
    path = resolve(cfg["data"]["excluded_apps"])
    if not path.is_file():
        parser.error(f"Missing required app exclusion list: {path}")
    excluded = set(json.loads(path.read_text())["apps"])
    if not excluded:
        parser.error("The fixed app exclusion list must not be empty")

    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    local = int(os.environ.get("LOCAL_RANK", 0))
    if world > 1:
        dist.init_process_group(args.dist_backend)
    device = torch.device("cuda", local % torch.cuda.device_count())
    torch.cuda.set_device(device)
    seed = cfg["train"]["seed"]
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    if cfg["train"]["batch_size"] % world:
        parser.error("Global batch size must be divisible by the number of processes")
    per_gpu = cfg["train"]["batch_size"] // world

    data, data_root = cfg["data"], args.data_root.resolve()
    index_dirs = [resolve(d) for d in data["index_dirs"]]
    text = TextTable(resolve(data["text_table"]))
    sources = data["sources"]
    tau_c = [float(s["tau_c"]) for s in sources]

    def build(horizon, split="train", keep=None):
        parts = {}
        for i, s in enumerate(sources):
            if split == "validation" and s.get("holdout"):
                index, part_keep = find_index(index_dirs, s["name"], "train"), "holdout"
            elif split == "validation":
                index, part_keep = find_index(index_dirs, s["name"], "validation"), "all"
            else:
                index, part_keep = find_index(index_dirs, s["name"], "train"), "train" if s.get("holdout") else "all"
            parts[s["name"]] = PortableTransitions(
                index, data_root, text, holdout=resolve(s["holdout"]) if s.get("holdout") else None,
                keep=part_keep, horizon=horizon, excluded_apps=excluded if split == "train" else None, src_id=i)
        return MixedSources(parts)

    single = build(1)
    windows = {int(h): build(int(h)) for h in cfg["rollout"]["horizon"]}
    held_out = build(1, split="validation")
    if is_main():
        print("usable transitions:", dict(zip(single.names, single.lens)), "total", len(single), flush=True)
        print("held-out:", dict(zip(held_out.names, held_out.lens)), flush=True)
    samples = cfg["train"]["steps"] * cfg["train"]["batch_size"]

    def make_loader(key, start_batches):
        dataset = single if key == "single" else windows[key]
        sampler = QuotaSampler(dataset, {s["name"]: s["quota"] for s in sources}, samples, rank=rank, world=world,
                               seed=seed, start=start_batches * per_gpu)
        workers = data["workers"] if key == "single" else max(2, data["workers"] // 2)
        return torch.utils.data.DataLoader(
            dataset, batch_size=per_gpu, sampler=sampler, num_workers=workers, pin_memory=True, drop_last=True,
            collate_fn=collate if key == "single" else _collate_window, persistent_workers=workers > 0,
            prefetch_factor=data["prefetch"] if workers > 0 else None)

    eval_size = min(len(held_out), cfg["log"]["eval_batches"] * cfg["log"]["eval_batch_size"])
    subset = np.random.default_rng(seed).choice(len(held_out), size=eval_size, replace=False).tolist()
    eval_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(held_out, subset),
                                              batch_size=cfg["log"]["eval_batch_size"], shuffle=False,
                                              num_workers=min(8, data["workers"]), collate_fn=collate)

    run_dir = ROOT / "outputs/runs" / args.run_name
    if is_main():
        run_dir.mkdir(parents=True, exist_ok=args.resume)
        record = {**cfg, "deviations": deviations, "world_size": world, "data_root": str(data_root),
                  "usable_transitions": dict(zip(single.names, single.lens)),
                  "text_table_sha256": text.sha256}
        name = "config.yaml" if not args.resume else f"config_resume_{len(list(run_dir.glob('config*.yaml')))}.yaml"
        (run_dir / name).write_text(yaml.safe_dump(record, sort_keys=False, allow_unicode=True))
    if world > 1:
        dist.barrier()
    trainer = Trainer(cfg, make_loader, eval_loader, device, run_dir, tau_c)
    if args.resume and not trainer.resume():
        raise SystemExit("--resume given but no checkpoint found")
    trainer.fit(stop_at=args.stop_at)
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
