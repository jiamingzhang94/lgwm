"""Per-patch change masks between consecutive screenshots, computed in CIELAB space."""
from __future__ import annotations

import numpy as np

GRID_W, GRID_H = 16, 32
PATCH = 16


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Convert [...,3] uint8 sRGB to float32 CIELAB (D65 white point)."""
    x = rgb.astype(np.float32) / 255.0
    m = x > 0.04045
    x = np.where(m, ((x + 0.055) / 1.055) ** 2.4, x / 12.92)
    mat = np.array([[0.4124564, 0.3575761, 0.1804375],
                    [0.2126729, 0.7151522, 0.0721750],
                    [0.0193339, 0.1191920, 0.9503041]], dtype=np.float32)
    xyz = x @ mat.T
    xyz /= np.array([0.95047, 1.0, 1.08883], dtype=np.float32)
    e, k = 216.0 / 24389.0, 24389.0 / 27.0
    f = np.where(xyz > e, np.cbrt(xyz), (k * xyz + 16.0) / 116.0)
    fx, fy, fz = f[..., 0], f[..., 1], f[..., 2]
    return np.stack([116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)], axis=-1)


def patch_lab(img: np.ndarray) -> np.ndarray:
    """Mean LAB colour of each patch: [512, 256, 3] uint8 -> [32, 16, 3] float32."""
    h, w = img.shape[:2]
    assert (h, w) == (GRID_H * PATCH, GRID_W * PATCH), f"unexpected image shape {img.shape}"
    lab = rgb_to_lab(img)
    return lab.reshape(GRID_H, PATCH, GRID_W, PATCH, 3).mean(axis=(1, 3))


def patch_delta(img_t: np.ndarray, img_t1: np.ndarray) -> np.ndarray:
    """Per-patch mean absolute LAB difference -> [32, 16] float32."""
    a, b = patch_lab(img_t), patch_lab(img_t1)
    return np.abs(a - b).mean(axis=-1)


def changed_mask(img_t: np.ndarray, img_t1: np.ndarray, tau_c: float) -> np.ndarray:
    """Boolean [32, 16] mask; True marks a changed patch."""
    return patch_delta(img_t, img_t1) >= tau_c


def changed_ratio(delta: np.ndarray, tau_c: float) -> float:
    return float((delta >= tau_c).mean())


_XYZ_MAT = [[0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041]]
_WHITE = [0.95047, 1.0, 1.08883]


def rgb_to_lab_torch(rgb_u8):
    """[B,3,H,W] uint8 -> [B,3,H,W] float32 CIELAB。"""
    import torch
    x = rgb_u8.float() / 255.0
    x = torch.where(x > 0.04045, ((x + 0.055) / 1.055) ** 2.4, x / 12.92)
    mat = torch.tensor(_XYZ_MAT, dtype=x.dtype, device=x.device)
    xyz = torch.einsum("ij,bjhw->bihw", mat, x)
    xyz = xyz / torch.tensor(_WHITE, dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
    e, k = 216.0 / 24389.0, 24389.0 / 27.0
    f = torch.where(xyz > e, xyz.clamp_min(1e-8).pow(1.0 / 3.0), (k * xyz + 16.0) / 116.0)
    fx, fy, fz = f[:, 0], f[:, 1], f[:, 2]
    return torch.stack([116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)], dim=1)


def patch_delta_torch(img_t_u8, img_t1_u8):
    """Batched per-patch LAB difference: two [B,3,512,256] uint8 -> [B, 512] float32, row-major."""
    import torch
    b = img_t_u8.shape[0]
    lab = rgb_to_lab_torch(torch.cat([img_t_u8, img_t1_u8], dim=0))
    lab = lab.view(2 * b, 3, GRID_H, PATCH, GRID_W, PATCH).mean(dim=(3, 5))
    a, c = lab[:b], lab[b:]
    return (a - c).abs().mean(dim=1).reshape(b, GRID_H * GRID_W)


def changed_mask(img_t_u8, img_t1_u8, tau_c, src_id=None):
    """Change mask [B,512] float; `tau_c` is a scalar or a per-source threshold vector."""
    import torch
    d = patch_delta_torch(img_t_u8, img_t1_u8)
    if torch.is_tensor(tau_c) and tau_c.numel() > 1:
        assert src_id is not None, "per-source tau_c requires src_id in the batch"
        t = tau_c.to(d.device)[src_id.long()].unsqueeze(-1)
    else:
        t = tau_c
    return (d >= t).float()


def calibrate_tau(deltas: np.ndarray, target_median_ratio: float = 0.08) -> float:
    """Find the tau_c whose median changed-patch ratio over [N, 32, 16] deltas equals the target."""
    lo, hi = 0.0, float(deltas.max()) + 1e-6
    for _ in range(60):
        mid = (lo + hi) / 2
        med = float(np.median((deltas >= mid).mean(axis=(1, 2))))
        if med > target_median_ratio:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2
