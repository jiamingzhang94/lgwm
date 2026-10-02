"""Pooled readout inputs for benchmark transitions, computed from images."""
import numpy as np
import torch

from .data.dataset import collate
from .data.transitions import IndexTable, ShardReader, decode


class TransitionItems(torch.utils.data.Dataset):
    """Unique (source, transition ID) pairs resolved through relocatable indexes."""

    def __init__(self, keys, tables, text_table, data_root):
        self.keys, self.tables, self.data_root = keys, tables, data_root
        self.text_ids = {s: text_table.ids(t.text) for s, t in tables.items()}
        self.text_emb, self.reader = text_table.emb, None

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, j):
        if self.reader is None:
            self.reader = ShardReader(self.data_root)
        source, transition = self.keys[j]
        table = self.tables[source]
        row = table.row_of[transition]
        return {"img_t": decode(table.frame(self.reader, row, "t")),
                "img_t1": decode(table.frame(self.reader, row, "t1")),
                **table.action(row, self.text_ids[source], self.text_emb), "_j": np.int64(j)}


def _collate(items):
    js = torch.tensor([int(item.pop("_j")) for item in items])
    return collate(items), js


@torch.no_grad()
def extract(encoder, action_encoder, predictor, items, *, observed_encoder=None, device="cuda",
            batch_size=128, workers=8, precision="bf16", storage=np.float16):
    """Return pool_zhat, pool_zt1 and pool_action arrays aligned with ``items.keys``."""
    observed_encoder = observed_encoder or encoder
    n, dim = len(items), encoder.dim
    out = {name: np.zeros((n, dim), dtype=storage) for name in ("pool_zhat", "pool_zt1", "pool_action")}
    loader = torch.utils.data.DataLoader(items, batch_size=batch_size, shuffle=False, num_workers=workers,
                                         collate_fn=_collate, pin_memory=device != "cpu")
    enabled = precision == "bf16"
    with torch.autocast("cuda" if str(device).startswith("cuda") else "cpu", dtype=torch.bfloat16, enabled=enabled):
        for batch, js in loader:
            batch = batch.to(device)
            z_t = encoder(batch.img_t)
            tokens, pad = action_encoder(batch)
            z_hat = predictor(z_t, tokens, pad)
            z_t1 = observed_encoder(batch.img_t1)
            valid = (~pad).unsqueeze(-1).float()
            pooled_action = (tokens * valid).sum(1) / valid.sum(1).clamp(min=1.0)
            index = js.numpy()
            out["pool_zhat"][index] = z_hat.mean(1).float().cpu().numpy().astype(storage)
            out["pool_zt1"][index] = z_t1.mean(1).float().cpu().numpy().astype(storage)
            out["pool_action"][index] = pooled_action.float().cpu().numpy().astype(storage)
    return out


def manifest_keys(records):
    """Sorted unique (source, transition ID) pairs referenced by base/obs/clean."""
    keys = {(r[k]["transition_id"].split("/", 1)[0], r[k]["transition_id"])
            for r in records for k in ("base", "obs", "clean")}
    return sorted(keys)


def load_tables(records, index_dirs, split, find_index):
    sources = sorted({r[k]["transition_id"].split("/", 1)[0] for r in records for k in ("base", "obs", "clean")})
    paths = {s: find_index(index_dirs, s, split) for s in sources}
    return {s: IndexTable(p) for s, p in paths.items()}, paths


def readout_inputs(records, keys, arrays):
    """Integrity scores and residual-head features."""
    lookup = {transition: i for i, (_, transition) in enumerate(keys)}

    def unit(x):
        x = x.astype(np.float32)
        return x / np.clip(np.linalg.norm(x, axis=-1, keepdims=True), 1e-6, None)

    base = np.array([lookup[r["base"]["transition_id"]] for r in records])
    obs = np.array([lookup[r["obs"]["transition_id"]] for r in records])
    predicted, observed = unit(arrays["pool_zhat"])[base], unit(arrays["pool_zt1"])[obs]
    action = arrays["pool_action"][base].astype(np.float32)
    integrity = 1.0 - (predicted * observed).sum(1)
    residual = np.concatenate([observed - predicted, action], axis=1)
    if not (np.isfinite(integrity).all() and np.isfinite(residual).all()):
        raise ValueError("Non-finite readout inputs")
    return integrity, residual
