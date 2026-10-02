"""LGWM model: encoder, EMA target encoder, action encoder, predictor and inverse-dynamics head."""
from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

GRID_W, GRID_H = 16, 32
N_TOKENS = GRID_W * GRID_H
N_ACTION_TYPES = 11
N_DIRECTIONS = 4
MAX_ACTION_TOKENS = 4


def sincos_1d(v: torch.Tensor, dim: int) -> torch.Tensor:
    """1D sin-cos embedding: [...] -> [..., dim]; dim must be even."""
    half = dim // 2
    omega = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=v.device, dtype=torch.float32) / half)
    out = v.float().unsqueeze(-1) * omega
    return torch.cat([out.sin(), out.cos()], dim=-1)


def sincos_2d(xy: torch.Tensor, dim: int) -> torch.Tensor:
    """2D sin-cos embedding of normalized xy: [...,2] -> [..., dim]."""
    return torch.cat([sincos_1d(xy[..., 0], dim // 2), sincos_1d(xy[..., 1], dim // 2)], dim=-1)


def patch_centers(device=None) -> torch.Tensor:
    """Patch-centre coordinates [512, 2] in row-major order."""
    row = torch.arange(GRID_H, device=device).float()
    col = torch.arange(GRID_W, device=device).float()
    y = ((row + 0.5) / GRID_H).view(GRID_H, 1).expand(GRID_H, GRID_W)
    x = ((col + 0.5) / GRID_W).view(1, GRID_W).expand(GRID_H, GRID_W)
    return torch.stack([x.reshape(-1), y.reshape(-1)], dim=-1)


class Encoder(nn.Module):
    """ViT screen encoder returning layer-normalized patch tokens."""

    def __init__(self, name: str = "vit_small_patch14_dinov2.lvd142m", pretrained: bool = True,
                 hw: tuple[int, int] = (448, 224)):
        """`hw` is the pixel size fed to the ViT."""
        super().__init__()
        import timm
        self.hw = tuple(hw)
        self.vit = timm.create_model(
            name, pretrained=pretrained, num_classes=0, img_size=self.hw, global_pool="")
        self.dim = self.vit.embed_dim
        self.n_prefix = getattr(self.vit, "num_prefix_tokens", 1)
        self.ln = nn.LayerNorm(self.dim, elementwise_affine=False)
        cfg = self.vit.default_cfg
        self.register_buffer("norm_mean", torch.tensor(cfg["mean"]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("norm_std", torch.tensor(cfg["std"]).view(1, 3, 1, 1), persistent=False)
        self.pretrained_tag = f"{name} | mean={cfg['mean']} std={cfg['std']} interp={cfg.get('interpolation')}"

    def forward(self, img_u8: torch.Tensor) -> torch.Tensor:
        x = img_u8.float().div_(255.0)
        if x.shape[-2:] != self.hw:
            x = F.interpolate(x, size=self.hw, mode="bicubic", align_corners=False, antialias=True)
        x = (x - self.norm_mean) / self.norm_std
        z = self.vit.forward_features(x)[:, self.n_prefix:]
        assert z.shape[1] == N_TOKENS, f"token count {z.shape[1]} != {N_TOKENS}"
        return self.ln(z)


class TargetEncoder(nn.Module):
    """EMA copy of the encoder; never receives gradients."""

    def __init__(self, encoder: Encoder):
        super().__init__()
        self.enc = copy.deepcopy(encoder)
        for p in self.enc.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, img_u8: torch.Tensor) -> torch.Tensor:
        return self.enc(img_u8)

    @torch.no_grad()
    def update(self, online: Encoder, tau: float) -> None:
        for pe, po in zip(self.enc.parameters(), online.parameters()):
            pe.mul_(tau).add_(po.detach(), alpha=1 - tau)
        for be, bo in zip(self.enc.buffers(), online.buffers()):
            be.copy_(bo)


class ActionEncoder(nn.Module):
    """Encode an action into one to four tokens plus a padding mask."""

    def __init__(self, dim: int, text_dim: int = 384, coord_posenc: str = "shared"):
        super().__init__()
        self.dim = dim
        if coord_posenc != "shared":
            raise ValueError("This release contains only the shared-coordinate core model")
        self.coord_posenc = coord_posenc
        self.type_emb = nn.Embedding(N_ACTION_TYPES, dim)
        self.dir_emb = nn.Embedding(N_DIRECTIONS + 1, dim)
        self.mlp_c = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.text_proj = nn.Linear(text_dim, dim)
        self.flag = nn.Parameter(torch.zeros(3, dim))
        nn.init.normal_(self.flag, std=0.02)

    def forward(self, b) -> tuple[torch.Tensor, torch.Tensor]:
        B, D, dev = b.a_type.shape[0], self.dim, b.a_type.device
        toks = torch.zeros(B, MAX_ACTION_TOKENS, D, device=dev)
        mask = torch.ones(B, MAX_ACTION_TOKENS, dtype=torch.bool, device=dev)

        d = torch.where(b.a_dir >= 0, b.a_dir, torch.full_like(b.a_dir, N_DIRECTIONS))
        toks[:, 0] = self.type_emb(b.a_type) + self.dir_emb(d)
        mask[:, 0] = False

        embed = lambda c: sincos_2d(c, D)
        c1 = self.mlp_c(embed(b.a_coord)) + self.flag[0]
        toks[:, 1] = torch.where(b.a_has_coord.unsqueeze(-1), c1, torch.zeros_like(c1))
        mask[:, 1] = ~b.a_has_coord

        c2 = self.mlp_c(embed(b.a_coord2)) + self.flag[1]
        toks[:, 2] = torch.where(b.a_has_coord2.unsqueeze(-1), c2, torch.zeros_like(c2))
        mask[:, 2] = ~b.a_has_coord2

        tx = self.text_proj(b.a_text) + self.flag[2]
        toks[:, 3] = torch.where(b.a_has_text.unsqueeze(-1), tx, torch.zeros_like(tx))
        mask[:, 3] = ~b.a_has_text
        return toks, mask


class Predictor(nn.Module):
    """Self-attention transformer that predicts the 512 next-screen tokens."""

    def __init__(self, dim: int, depth: int = 12, heads: int = 12):
        super().__init__()
        self.dim = dim
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=dim * 4, dropout=0.0,
            activation="gelu", batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(layer, num_layers=depth)
        self.query_bias = nn.Parameter(torch.zeros(N_TOKENS, dim))
        self.src = nn.Parameter(torch.zeros(3, dim))
        nn.init.normal_(self.query_bias, std=0.02)
        nn.init.normal_(self.src, std=0.02)
        self.register_buffer("patch_pos", torch.zeros(N_TOKENS, dim), persistent=False)
        self._pos_ready = False
        self.ln_out = nn.LayerNorm(dim, elementwise_affine=False)
        self.grad_ckpt = False

    def _pos(self, device, dtype):
        if not self._pos_ready:
            self.patch_pos = sincos_2d(patch_centers(device), self.dim).to(dtype)
            self._pos_ready = True
        return self.patch_pos

    def forward(self, z_ctx: torch.Tensor, a_toks: torch.Tensor,
                a_mask: torch.Tensor) -> torch.Tensor:
        B = z_ctx.shape[0]
        pos = self._pos(z_ctx.device, z_ctx.dtype)
        q = (self.query_bias + pos + self.src[2]).unsqueeze(0).expand(B, -1, -1)
        seq = torch.cat([z_ctx + pos + self.src[0], a_toks + self.src[1], q], dim=1)
        zeros = torch.zeros(B, N_TOKENS, dtype=torch.bool, device=z_ctx.device)
        kpm = torch.cat([zeros, a_mask, zeros], dim=1)
        if self.grad_ckpt and self.training:
            from torch.utils.checkpoint import checkpoint
            out = seq
            for layer in self.tf.layers:
                out = checkpoint(layer, out, src_key_padding_mask=kpm, use_reentrant=False)
            if self.tf.norm is not None:
                out = self.tf.norm(out)
        else:
            out = self.tf(seq, src_key_padding_mask=kpm)
        return self.ln_out(out[:, -N_TOKENS:, :])


class InvDynHead(nn.Module):
    """Inverse-dynamics head that recovers the action from current and next-screen tokens."""

    def __init__(self, dim: int):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(2 * dim, dim), nn.GELU(), nn.Linear(dim, dim), nn.GELU())
        self.type_head = nn.Linear(dim, N_ACTION_TYPES)
        self.coord_head = nn.Linear(dim, N_TOKENS)
        self.text_head = nn.Linear(dim, 1)

    def forward(self, z_t: torch.Tensor, z_t1: torch.Tensor):
        h = self.trunk(torch.cat([z_t.mean(1), z_t1.mean(1)], dim=-1))
        return self.type_head(h), self.coord_head(h), self.text_head(h).squeeze(-1)


class LGWM(nn.Module):
    def __init__(self, encoder_name: str = "vit_small_patch14_dinov2.lvd142m",
                 pretrained: bool = True,
                 pred_depth: int = 12, pred_heads: int = 6, text_dim: int = 384,
                 coord_posenc: str = "shared", encoder_hw: tuple[int, int] = (448, 224)):
        super().__init__()
        self.encoder = Encoder(encoder_name, pretrained, encoder_hw)
        d = self.encoder.dim
        self.target = TargetEncoder(self.encoder)
        self.action_encoder = ActionEncoder(d, text_dim, coord_posenc)
        self.predictor = Predictor(d, pred_depth, pred_heads)
        self.invdyn = InvDynHead(d)
        self.dim = d

    def predict(self, z_ctx, b):
        return self.predictor(z_ctx, *self.action_encoder(b))

    def forward(self, batch, *, tau_c, lambda_hi: float, invdyn_w: float,
                mode: str = "single", horizon: int = 1, tf_prob: float = 1.0,
                rollout_rand=None,
                vicreg_var_w: float = 1.0, vicreg_cov_w: float = 0.01):
        """Full training-step forward pass."""
        import numpy as np

        from ..data.mask import changed_mask
        from ..train.losses import invdyn_loss, latent_loss, vicreg_reg
        from ..train.metrics import _slice_action, copy_score, coord_top1

        if mode == "single":
            b = batch
            chg = changed_mask(b.img_t, b.img_t1, tau_c, b.src_id)
            z = self.encoder(b.img_t)
            zb = self.target(b.img_t1)
            a_toks_for_nce, a_msk_nce = self.action_encoder(b)
            zh = self.predictor(z, a_toks_for_nce, a_msk_nce)
            L_pred, _ = latent_loss(zh, zb, chg, lambda_hi)
            tl, cl, xl = self.invdyn(z, zb)
            L_inv, invp = invdyn_loss(tl, cl, xl, b)
            L_vic, vicp = vicreg_reg(zh.mean(1), vicreg_var_w, vicreg_cov_w)
            loss = L_pred + invdyn_w * L_inv + L_vic
            with torch.no_grad():
                stats = {
                    "L_pred": L_pred.detach(), "L_inv": L_inv.detach(),
                    "copy_score": copy_score(zh, z, zb, chg),
                    "changed_ratio": chg.mean().detach(),
                    "invdyn_type_acc": (tl.argmax(-1) == b.a_type).float().mean().detach(),
                    "coord_top1": coord_top1(cl, b),
                    **invp, **vicp,
                }
            return loss, stats

        frames = batch["frames"]
        z = self.encoder(frames[:, 0])
        for h in range(horizon):
            a_tok, a_msk = _slice_action(self, batch, h, frames.device)
            zh = self.predictor(z, a_tok, a_msk)
            zb = self.target(frames[:, h + 1])
            if h == horizon - 1:
                chg = changed_mask(frames[:, h], frames[:, h + 1], tau_c,
                                   batch["src_id"][:, h].to(frames.device))
                L_pred, _ = latent_loss(zh, zb, chg, lambda_hi)
            if h + 1 < horizon:
                feed_pred = (rollout_rand[h] if rollout_rand is not None else np.random.rand()) > tf_prob
                z = zh if feed_pred else zb
        L_vic, vicp = vicreg_reg(zh.mean(1), vicreg_var_w, vicreg_cov_w)
        loss = L_pred + L_vic
        with torch.no_grad():
            stats = {"L_pred": L_pred.detach(), "rollout_H": float(horizon),
                     "copy_score": copy_score(zh, z, zb, chg), **vicp}
        return loss, stats
