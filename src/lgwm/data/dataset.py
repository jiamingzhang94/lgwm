"""Canonical tensor batch and collation for the portable data loaders."""
from dataclasses import dataclass
import numpy as np
import torch

IMG_H, IMG_W = 512, 256
N_ACTION_TYPES = 11
N_DIRECTIONS = 4

@dataclass
class Batch:
    img_t: torch.Tensor
    img_t1: torch.Tensor
    a_type: torch.Tensor
    a_coord: torch.Tensor
    a_has_coord: torch.Tensor
    a_coord2: torch.Tensor
    a_has_coord2: torch.Tensor
    a_text: torch.Tensor
    a_has_text: torch.Tensor
    a_dir: torch.Tensor
    src_id: torch.Tensor

    def to(self, device, non_blocking=True):
        for f in self.__dataclass_fields__:
            setattr(self, f, getattr(self, f).to(device, non_blocking=non_blocking))
        return self

def collate(items: list[dict]) -> Batch:
    def st(k, dt=None):
        a = np.stack([it[k] for it in items])
        t = torch.from_numpy(a)
        return t.to(dt) if dt else t
    return Batch(
        img_t=st("img_t").permute(0, 3, 1, 2).contiguous(),
        img_t1=st("img_t1").permute(0, 3, 1, 2).contiguous(),
        a_type=st("a_type", torch.long),
        a_coord=st("a_coord", torch.float32),
        a_has_coord=st("a_has_coord", torch.bool),
        a_coord2=st("a_coord2", torch.float32),
        a_has_coord2=st("a_has_coord2", torch.bool),
        a_text=st("a_text", torch.float32),
        a_has_text=st("a_has_text", torch.bool),
        a_dir=st("a_dir", torch.long),
        src_id=st("src_id", torch.long),
    )
