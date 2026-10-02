"""Training losses."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from ..models.lgwm import GRID_H, GRID_W, N_TOKENS


def latent_loss(z_hat: torch.Tensor, z_bar: torch.Tensor, chg: torch.Tensor,
                lambda_hi: float = 8.0):
    """Latent regression weighted toward changed patches; returns (scalar, per-patch [B,N])."""
    w = 1.0 + (lambda_hi - 1.0) * chg.float()
    l = F.smooth_l1_loss(z_hat.float(), z_bar.float(), reduction="none").mean(-1)
    per_sample = (w * l).sum(-1) / w.sum(-1).clamp_min(1e-6)
    return per_sample.mean(), l


def gaussian_on_grid(coord: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    """Gaussian soft labels over patches for normalized coordinates: [B,2] -> [B,N]."""
    dev = coord.device
    col = torch.arange(GRID_W, device=dev).float() + 0.5
    row = torch.arange(GRID_H, device=dev).float() + 0.5
    cx = coord[:, 0:1] * GRID_W
    cy = coord[:, 1:2] * GRID_H
    dx = (col.view(1, 1, GRID_W) - cx.view(-1, 1, 1)) ** 2
    dy = (row.view(1, GRID_H, 1) - cy.view(-1, 1, 1)) ** 2
    g = torch.exp(-(dx + dy) / (2 * sigma ** 2)).reshape(-1, N_TOKENS)
    return g / g.sum(-1, keepdim=True).clamp_min(1e-8)


def invdyn_loss(type_logits, coord_logits, text_logits, b):
    """Inverse-dynamics loss: action type, coordinate heatmap and has-text terms."""
    L = F.cross_entropy(type_logits.float(), b.a_type)
    parts = {"invdyn_type": L.detach()}
    if b.a_has_coord.any():
        tgt = gaussian_on_grid(b.a_coord[b.a_has_coord])
        lp = F.log_softmax(coord_logits[b.a_has_coord].float(), dim=-1)
        Lc = -(tgt * lp).sum(-1).mean()
        L = L + Lc
        parts["invdyn_coord"] = Lc.detach()
    Lt = F.binary_cross_entropy_with_logits(text_logits.float(), b.a_has_text.float())
    parts["invdyn_text"] = Lt.detach()
    return L + Lt, parts


def vicreg_reg(p: torch.Tensor, w_var: float = 1.0, w_cov: float = 0.01):
    """VICReg variance and covariance terms on pooled predictions [B,D]."""
    p = p.float()
    std = p.std(dim=0)
    L_v = F.relu(1.0 - std).mean()
    pc = p - p.mean(0, keepdim=True)
    cov = (pc.T @ pc) / max(p.shape[0] - 1, 1)
    off = cov - torch.diag_embed(torch.diagonal(cov))
    L_c = off.pow(2).sum() / p.shape[1]
    return w_var * L_v + w_cov * L_c, {"vicreg_var": L_v.detach(), "vicreg_cov": L_c.detach(),
                                       "pooled_std": std.mean().detach()}
