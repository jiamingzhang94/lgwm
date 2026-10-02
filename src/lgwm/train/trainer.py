"""Training loop for the LGWM world model."""
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from ..data.mask import changed_mask
from ..models.lgwm import LGWM, TargetEncoder
from .losses import latent_loss
from .metrics import copy_score

ROLLOUT_SEED = 2027 * 1_000_003


def is_main():
    return not dist.is_initialized() or dist.get_rank() == 0


def cosine(step, total, base, final, warmup):
    if step < warmup:
        return base * (step + 1) / max(warmup, 1)
    t = (step - warmup) / max(total - warmup, 1)
    return final + 0.5 * (base - final) * (1 + math.cos(math.pi * min(t, 1.0)))


def param_groups(model, weight_decay, llrd=0.75):
    """Optimizer parameter groups: no weight decay on 1-D tensors."""
    vit, groups, seen = model.encoder.vit, [], set()

    def add(params, scale, name):
        params = [p for p in params if p.requires_grad and id(p) not in seen]
        seen.update(id(p) for p in params)
        decay, flat = [p for p in params if p.ndim > 1], [p for p in params if p.ndim <= 1]
        if decay:
            groups.append({"params": decay, "lr_scale": scale, "weight_decay": weight_decay, "name": name})
        if flat:
            groups.append({"params": flat, "lr_scale": scale, "weight_decay": 0.0, "name": name + "_nd"})

    depth = len(vit.blocks)
    layer0 = list(vit.patch_embed.parameters())
    layer0 += [t for t in (getattr(vit, a, None) for a in ("cls_token", "pos_embed", "reg_token")) if t is not None]
    add(layer0, llrd ** (depth + 1), "layer0")
    for i, block in enumerate(vit.blocks):
        add(block.parameters(), llrd ** (depth - i), f"blk{i}")
    add(model.encoder.parameters(), 1.0, "enc_rest")
    for module, name in ((model.predictor, "predictor"), (model.action_encoder, "action_enc"), (model.invdyn, "invdyn")):
        add(module.parameters(), 1.0, name)
    return groups


def rollout_choice(step, cfg):
    """(use_window, horizon, teacher-forcing draws), a pure function of the global step."""
    r = cfg["rollout"]
    rng = np.random.default_rng(ROLLOUT_SEED + step)
    p = r["p"] if step >= r["enable_after"] else 0.0
    if not rng.random() < p:
        return False, 1, None
    horizon = int(r["horizon"][rng.integers(len(r["horizon"]))])
    return True, horizon, rng.random(horizon).tolist()


def rng_state(device):
    """Global RNG states as tensors and plain values, so checkpoints load with weights_only=True."""
    _, keys, pos, has_gauss, cached = np.random.get_state()
    return {"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state(device),
            "numpy": {"keys": torch.from_numpy(keys.astype(np.int64)), "pos": int(pos),
                      "has_gauss": int(has_gauss), "cached": float(cached)},
            "python": random.getstate()}


def set_rng_state(state, device):
    torch.set_rng_state(state["torch"].cpu())
    torch.cuda.set_rng_state(state["cuda"].cpu(), device)
    n = state["numpy"]
    np.random.set_state(("MT19937", n["keys"].cpu().numpy().astype(np.uint32), n["pos"], n["has_gauss"], n["cached"]))
    random.setstate(state["python"])


def consumed_batches(step, cfg):
    """Batches drawn from each loader before ``step``; replays the step-derived choices."""
    counts = {"single": 0, **{int(h): 0 for h in cfg["rollout"]["horizon"]}}
    for s in range(step):
        use, horizon, _ = rollout_choice(s, cfg)
        counts[horizon if use else "single"] += 1
    return counts


class Trainer:
    def __init__(self, cfg, make_loader, eval_loader, device, run_dir, tau_c):
        self.cfg, self.device, self.run_dir = cfg, device, Path(run_dir)
        self.make_loader, self.eval_loader = make_loader, eval_loader
        self.tau_c = torch.tensor(tau_c, dtype=torch.float32, device=device)
        m = cfg["model"]
        model = LGWM(m["encoder"], m.get("pretrained", True) and not m.get("init_encoder"), m["pred_depth"], m["pred_heads"],
                     encoder_hw=tuple(m["encoder_hw"])).to(device)
        if m.get("init_encoder"):
            from safetensors.torch import load_file
            model.encoder.vit.load_state_dict(load_file(m["init_encoder"]), strict=True)
            model.target = TargetEncoder(model.encoder).to(device)
        self.dynamic_ckpt = bool(cfg["train"].get("grad_ckpt_dynamic", True))
        model.predictor.grad_ckpt = bool(cfg["train"].get("grad_ckpt", True)) and not self.dynamic_ckpt
        self.model = model
        self.net = (DistributedDataParallel(model, device_ids=[device.index], find_unused_parameters=True)
                    if dist.is_initialized() else model)
        t = cfg["train"]
        self.opt = torch.optim.AdamW(param_groups(model, t["wd"]), lr=t["lr"], betas=tuple(t["betas"]),
                                     weight_decay=t["wd"])
        self.step, self.skipped = 0, 0
        (self.run_dir / "ckpt").mkdir(parents=True, exist_ok=True)
        self.loaders, self.iters = {}, {}

    def state(self):
        return {"step": self.step, "model": self.model.state_dict(), "opt": self.opt.state_dict(),
                "cfg": self.cfg, "skipped": self.skipped, "rng": rng_state(self.device)}

    def save(self, tag):
        if not is_main():
            return
        path = self.run_dir / "ckpt" / f"{tag}.pt"
        torch.save(self.state(), path)
        for old in sorted((self.run_dir / "ckpt").glob("step*.pt"))[:-self.cfg["log"].get("keep", 3)]:
            old.unlink()
        (self.run_dir / "latest.txt").write_text(path.name + "\n")

    def resume(self, path=None):
        if path is None:
            found = sorted((self.run_dir / "ckpt").glob("step*.pt"))
            if not found:
                return False
            path = found[-1]
        state = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state["model"], strict=True)
        self.opt.load_state_dict(state["opt"])
        self.step, self.skipped = state["step"], state.get("skipped", 0)
        set_rng_state(state["rng"], self.device)
        self.log({"event": "resumed", "checkpoint": Path(path).name})
        return True

    def next_batch(self, key):
        if key not in self.iters:
            start = consumed_batches(self.step, self.cfg)[key]
            self.loaders[key] = self.make_loader(key, start)
            self.iters[key] = iter(self.loaders[key])
        return next(self.iters[key])

    def learning_rate(self):
        t = self.cfg["train"]
        return cosine(self.step, t["steps"], t["lr"], t["lr_final"], t["warmup"])

    def ema_momentum(self):
        start, end = self.cfg["model"]["ema"]
        progress = self.step / max(self.cfg["train"]["steps"], 1)
        return end - (end - start) * 0.5 * (1 + math.cos(math.pi * progress))

    def teacher_forcing(self):
        r = self.cfg["rollout"]
        if self.step < r["enable_after"]:
            return 1.0
        progress = (self.step - r["enable_after"]) / max(self.cfg["train"]["steps"] - r["enable_after"], 1)
        start, end = r["tf_anneal"]
        return start + (end - start) * min(progress, 1.0)

    def train_step(self):
        use_window, horizon, draws = rollout_choice(self.step, self.cfg)
        if use_window:
            batch = {k: v.to(self.device, non_blocking=True) for k, v in self.next_batch(horizon).items()}
            mode = dict(mode="window", horizon=horizon, tf_prob=self.teacher_forcing(), rollout_rand=draws)
            fingerprint = batch["frames"][:, 0].float().mean()
        else:
            batch = self.next_batch("single").to(self.device)
            mode = dict(mode="single")
            fingerprint = batch.img_t.float().mean()
        if self.dynamic_ckpt:
            self.model.predictor.grad_ckpt = use_window
        loss_cfg = self.cfg["loss"]
        with torch.autocast("cuda", torch.bfloat16):
            loss, stats = self.net(batch, tau_c=self.tau_c, lambda_hi=float(loss_cfg["w_changed"]),
                                   invdyn_w=float(loss_cfg["inv_dyn_weight"]),
                                   vicreg_var_w=float(loss_cfg["vicreg_var"]),
                                   vicreg_cov_w=float(loss_cfg["vicreg_cov"]), **mode)
        if not torch.isfinite(loss):
            self.opt.zero_grad(set_to_none=True)
            self.skipped += 1
            return {"event": "nonfinite_loss"}
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg["train"]["grad_clip"])
        if not torch.isfinite(norm):
            self.opt.zero_grad(set_to_none=True)
            self.skipped += 1
            return {"event": "nonfinite_grad"}
        self.opt.step()
        self.opt.zero_grad(set_to_none=True)
        self.model.target.update(self.model.encoder, self.ema_momentum())
        return {**{k: float(v) for k, v in stats.items()}, "loss": float(loss), "grad_norm": float(norm),
                "window": float(use_window), "data_fp": float(fingerprint)}

    @torch.no_grad()
    def evaluate(self):
        """Held-out L_pred, identity-copy reference and copy score (monitoring only)."""
        if not is_main():
            return {}
        self.model.eval()
        lam, rows = float(self.cfg["loss"]["w_changed"]), []
        for i, batch in enumerate(self.eval_loader):
            if i >= self.cfg["log"].get("eval_batches", 8):
                break
            batch = batch.to(self.device)
            with torch.autocast("cuda", torch.bfloat16):
                z = self.model.encoder(batch.img_t)
                target = self.model.target(batch.img_t1)
                predicted = self.model.predict(z, batch)
            changed = changed_mask(batch.img_t, batch.img_t1, self.tau_c, batch.src_id)
            rows.append({"val_L_pred": latent_loss(predicted, target, changed, lam)[0].item(),
                         "val_copy_L": latent_loss(z, target, changed, lam)[0].item(),
                         "val_copy_score": copy_score(predicted, z, target, changed),
                         "val_pooled_std": float(predicted.float().mean(1).std(0).mean())})
        self.model.train()
        out = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]} if rows else {}
        if out:
            out["val_L_pred/copy_L"] = out["val_L_pred"] / out["val_copy_L"]
        return out

    def log(self, record):
        if not is_main():
            return
        record = {"step": self.step, **record}
        with (self.run_dir / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)

    def fit(self, stop_at=None):
        cfg, total = self.cfg, self.cfg["train"]["steps"]
        stop_at = min(stop_at or total, total)
        self.model.train()
        buffer, t0, s0 = [], time.time(), self.step
        while self.step < stop_at:
            lr = self.learning_rate()
            for group in self.opt.param_groups:
                group["lr"] = lr * group["lr_scale"]
            buffer.append(self.train_step())
            self.step += 1
            if self.step % cfg["log"]["every"] == 0:
                keys = {k for b in buffer for k in b if k != "event"}
                record = {k: float(np.nanmean([b[k] for b in buffer if k in b])) for k in sorted(keys)}
                record.update(lr=lr, ema=self.ema_momentum(), skipped=self.skipped,
                              it_s=(self.step - s0) / (time.time() - t0))
                self.log(record)
                buffer, t0, s0 = [], time.time(), self.step
            if self.step % cfg["log"]["eval_every"] == 0:
                self.log({"event": "eval", **self.evaluate()})
                if dist.is_initialized():
                    dist.barrier()
            if self.step % cfg["log"]["ckpt_every"] == 0:
                self.save(f"step{self.step:06d}")
        if self.step >= total:
            if self.step % cfg["log"]["eval_every"]:
                self.log({"event": "eval", **self.evaluate()})
            self.save("final")
