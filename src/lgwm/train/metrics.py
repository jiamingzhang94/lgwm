"""Training-only pooling and rollout helpers; auxiliary analyses excluded."""
from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F
from ..models.lgwm import GRID_W, N_TOKENS

def copy_score(z_hat, z_t, z_bar, chg) -> float:
    """Copy score: cos(pred, current) - cos(pred, target), averaged over changed patches."""
    a = F.cosine_similarity(z_hat.float(), z_t.float(), dim=-1)
    b = F.cosine_similarity(z_hat.float(), z_bar.float(), dim=-1)
    d = (a - b)
    m = chg.float()
    n = m.sum()
    return float((d * m).sum() / n) if n > 0 else float("nan")


def coord_top1(coord_logits, b, tol_patch: int = 1) -> float:
    """Inverse-dynamics coordinate top-1 accuracy within one patch."""
    if not b.a_has_coord.any():
        return float("nan")
    pred = coord_logits[b.a_has_coord].argmax(-1)
    px, py = pred % GRID_W, pred // GRID_W
    c = b.a_coord[b.a_has_coord]
    tx = (c[:, 0] * GRID_W).long().clamp(0, GRID_W - 1)
    ty = (c[:, 1] * (N_TOKENS // GRID_W)).long().clamp(0, N_TOKENS // GRID_W - 1)
    hit = (px - tx).abs().le(tol_patch) & (py - ty).abs().le(tol_patch)
    return float(hit.float().mean())


def _collate_window(items):
    import torch as T
    out = {"frames": T.from_numpy(np.stack([i["frames"] for i in items])).permute(0, 1, 4, 2, 3).contiguous()}
    for k in items[0]:
        if k != "frames":
            out[k] = T.from_numpy(np.stack([i[k] for i in items]))
    return out


def _slice_action(model, w, k, device):
    from ..data.dataset import Batch
    b = Batch(
        img_t=torch.empty(0), img_t1=torch.empty(0),
        a_type=w["a_type"][:, k].long().to(device),
        a_coord=w["a_coord"][:, k].float().to(device),
        a_has_coord=w["a_has_coord"][:, k].bool().to(device),
        a_coord2=w["a_coord2"][:, k].float().to(device),
        a_has_coord2=w["a_has_coord2"][:, k].bool().to(device),
        a_text=w["a_text"][:, k].float().to(device),
        a_has_text=w["a_has_text"][:, k].bool().to(device),
        a_dir=w["a_dir"][:, k].long().to(device),
        src_id=w["src_id"][:, k].long().to(device),
    )
    return model.action_encoder(b)

