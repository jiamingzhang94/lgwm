"""GUI-domain encoder initialization used by the main training run."""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pyarrow.parquet as pq
from lgwm.models.lgwm import GRID_H, GRID_W, N_TOKENS, patch_centers, sincos_2d
from lgwm.data.transitions import ShardReader, decode
from lgwm.data.paths import input_path


class FrameDataset(torch.utils.data.Dataset):
    def __init__(self, indexes, data_root):
        self.data_root=data_root
        seen={}
        for path in sorted(indexes, key=str):
            t=pq.read_table(path,columns=['shard','off_t','size_t','off_t1','size_t1']).to_pydict()
            for sh,o0,s0,o1,s1 in zip(t['shard'],t['off_t'],t['size_t'],t['off_t1'],t['size_t1'],strict=True):
                seen.setdefault((sh,o0),s0);seen.setdefault((sh,o1),s1)
        self.items=[(sh,offset,size) for (sh,offset),size in seen.items() if size>=1000]
        self.reader=None
    def __len__(self):return len(self.items)
    def __getitem__(self,index):
        if self.reader is None:self.reader=ShardReader(self.data_root)
        shard,offset,size=self.items[index]
        return decode(self.reader.read(shard,offset,size))


def collate_frames(items):
    return torch.from_numpy(np.stack(items)).permute(0,3,1,2).contiguous()


def sample_block_mask(rng, n_blocks=4, scale=(0.15, 0.2), ar=(0.75, 1.5)) -> np.ndarray:
    """Sample an I-JEPA style block mask [512] bool; True marks masked target patches."""
    m = np.zeros((GRID_H, GRID_W), dtype=bool)
    for _ in range(n_blocks):
        s = rng.uniform(*scale) * GRID_H * GRID_W
        a = rng.uniform(*ar)
        h = max(1, min(GRID_H, int(round(math.sqrt(s / a)))))
        w = max(1, min(GRID_W, int(round(math.sqrt(s * a)))))
        top = rng.integers(0, GRID_H - h + 1)
        left = rng.integers(0, GRID_W - w + 1)
        m[top:top + h, left:left + w] = True
    if m.all():
        m[0, 0] = False
    if not m.any():
        m[0, 0] = True
    return m.reshape(-1)


class StageAModel(nn.Module):
    def __init__(self, encoder_name: str, hw=(448, 224), pred_depth=6, pred_width=384, pretrained=True):
        super().__init__()
        import copy as _c
        import timm
        self.hw = tuple(hw)
        self.vit = timm.create_model(encoder_name, pretrained=pretrained, num_classes=0,
                                     img_size=self.hw, global_pool="")
        cfg = self.vit.default_cfg
        self.register_buffer("nm", torch.tensor(cfg["mean"]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("ns", torch.tensor(cfg["std"]).view(1, 3, 1, 1), persistent=False)
        self.pretrained_tag = f"{encoder_name} | mean={cfg['mean']} std={cfg['std']}"
        self.d = self.vit.embed_dim
        self.n_prefix = self.vit.num_prefix_tokens
        self.ln = nn.LayerNorm(self.d, elementwise_affine=False)
        self.target = _c.deepcopy(self.vit)
        for p in self.target.parameters():
            p.requires_grad_(False)
        layer = nn.TransformerEncoderLayer(pred_width, 6, pred_width * 4, dropout=0.0,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.pred = nn.TransformerEncoder(layer, pred_depth)
        self.pin = nn.Linear(self.d, pred_width)
        self.pout = nn.Linear(pred_width, self.d)
        self.mask_tok = nn.Parameter(torch.zeros(1, 1, pred_width))
        nn.init.normal_(self.mask_tok, std=0.02)
        self.register_buffer("ppos", sincos_2d(patch_centers(), pred_width), persistent=False)

    def _prep(self, x_u8):
        x = x_u8.float().div_(255.0)
        if x.shape[-2:] != self.hw:
            x = F.interpolate(x, size=self.hw, mode="bicubic", align_corners=False, antialias=True)
        return (x - self.nm) / self.ns

    def _encode_visible(self, x, vis_mask):
        """Encode only the visible tokens."""
        v = self.vit
        z = v.patch_embed(x)
        z = z + v.pos_embed[:, self.n_prefix:]
        B, N, D = z.shape
        keep = vis_mask.sum(1)[0].item()
        idx = vis_mask.nonzero(as_tuple=False)[:, 1].view(B, keep)
        z = torch.gather(z, 1, idx.unsqueeze(-1).expand(-1, -1, D))
        pre = v.pos_embed[:, : self.n_prefix] + v.cls_token.expand(B, -1, -1) \
            if self.n_prefix else None
        if pre is not None:
            z = torch.cat([pre, z], dim=1)
        z = v.norm_pre(z) if hasattr(v, "norm_pre") else z
        for blk in v.blocks:
            z = blk(z)
        z = v.norm(z)
        return z[:, self.n_prefix:]

    @torch.no_grad()
    def _target(self, x):
        v = self.target
        z = v.patch_embed(x) + v.pos_embed[:, self.n_prefix:]
        B = z.shape[0]
        if self.n_prefix:
            z = torch.cat([v.pos_embed[:, : self.n_prefix] + v.cls_token.expand(B, -1, -1), z], 1)
        for blk in v.blocks:
            z = blk(z)
        return self.ln(v.norm(z)[:, self.n_prefix:])

    def forward(self, x_u8, mask):
        """Masked-prediction loss; mask is [B,512] bool with True for masked patches."""
        x = self._prep(x_u8)
        vis = ~mask
        zc = self._encode_visible(x, vis)
        zt = self._target(x)
        B, V, _ = zc.shape
        M = int(mask.sum(1)[0].item())
        vis_idx = vis.nonzero(as_tuple=False)[:, 1].view(B, V)
        msk_idx = mask.nonzero(as_tuple=False)[:, 1].view(B, M)
        pw = self.mask_tok.shape[-1]
        ctx = self.pin(zc) + self.ppos[vis_idx]
        q = self.mask_tok.expand(B, M, pw) + self.ppos[msk_idx]
        out = self.pred(torch.cat([ctx, q], 1))[:, V:]
        pred = self.pout(out)
        tgt = torch.gather(zt, 1, msk_idx.unsqueeze(-1).expand(-1, -1, zt.shape[-1]))
        return F.smooth_l1_loss(pred.float(), tgt.float())

    @torch.no_grad()
    def ema(self, tau):
        for pt, po in zip(self.target.parameters(), self.vit.parameters()):
            pt.mul_(tau).add_(po.detach(), alpha=1 - tau)
