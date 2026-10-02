"""Portable transition access on top of the relocatable index."""
import hashlib
import io
import json
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image

from .paths import input_path

IMG_H, IMG_W = 512, 256
BLANK_BELOW = 1000
COLUMNS = ["shard", "transition_id", "episode_id", "seg_key", "step", "n_steps", "off_t", "size_t",
           "off_t1", "size_t1", "action_type", "x", "y", "x2", "y2", "direction", "text", "app"]


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def find_index(index_dirs, source, split):
    """Exactly one completed ``<source>_<split>.parquet`` below the given directories."""
    name = f"{source}_{split}.parquet"
    found = sorted({p.resolve() for d in index_dirs for p in Path(d).rglob(name) if p.is_file()})
    if len(found) != 1:
        raise FileNotFoundError(f"Expected one {name} under {list(map(str, index_dirs))}, found {found}")
    return found[0]


def decode(blob):
    with Image.open(io.BytesIO(blob)) as image:
        array = np.asarray(image.convert("RGB"))
    if array.shape != (IMG_H, IMG_W, 3):
        raise ValueError(f"Unexpected frame shape {array.shape}")
    return array


class TextTable:
    """Frozen MiniLM action-text table; row 0 is the all-zero "no text" entry."""

    def __init__(self, path, allow_pickle=False):
        with np.load(path, allow_pickle=allow_pickle) as table:
            vocab, self.emb = table["vocab"].tolist(), table["emb"].astype(np.float32)
        if (not vocab or not all(isinstance(t, str) for t in vocab)
                or len(set(vocab)) != len(vocab) or self.emb.shape != (len(vocab), 384)
                or not np.isfinite(self.emb).all()):
            raise ValueError("Text table requires unique strings and finite [N,384] embeddings")
        if vocab[0] != "" or np.any(self.emb[0]) or len(vocab) != len(self.emb):
            raise ValueError("Text table must reserve an all-zero row 0 for missing text")
        self.lookup = {text: i for i, text in enumerate(vocab)}
        self.sha256 = sha256_file(path)

    def ids(self, texts):
        texts = [t or "" for t in texts]
        missing = {t for t in texts if t not in self.lookup}
        if missing:
            raise ValueError(
                f"Text table is missing {len(missing)} nonempty action texts."
            )
        return np.array([self.lookup[t] for t in texts], dtype=np.int64)


class ShardReader:
    """Per-process file handles; forked DataLoader workers must not share offsets."""

    def __init__(self, data_root):
        self.data_root, self.handles, self.pid = Path(data_root), {}, os.getpid()

    def read(self, shard, offset, size):
        if os.getpid() != self.pid:
            self.handles, self.pid = {}, os.getpid()
        handle = self.handles.get(shard)
        if handle is None:
            handle = self.handles[shard] = open(input_path(self.data_root, shard), "rb", buffering=0)
        handle.seek(int(offset))
        return handle.read(int(size))


class IndexTable:
    """Column arrays of one relocatable index, addressable by row or stable ID."""

    _cache = {}

    @classmethod
    def cached(cls, path):
        """One parsed table per index file and process (rollout windows reuse it)."""
        key = str(Path(path).resolve())
        if key not in cls._cache:
            cls._cache[key] = cls(path)
        return cls._cache[key]

    def __init__(self, path):
        self.path = Path(path)
        table = pq.read_table(self.path, columns=COLUMNS).to_pydict()
        self.n = len(table["shard"])
        self.shards = sorted(set(table["shard"]))
        shard_id = {s: i for i, s in enumerate(self.shards)}
        self.shard_id = np.array([shard_id[s] for s in table["shard"]], dtype=np.int32)
        self.transition_id = table["transition_id"]
        self.episode_id = table["episode_id"]
        self.seg_key = table["seg_key"]
        self.app = table["app"]
        self.text = table["text"]
        for name in ("step", "n_steps", "off_t", "size_t", "off_t1", "size_t1", "action_type", "direction"):
            setattr(self, name, np.asarray(table[name], dtype=np.int64))
        for name in ("x", "y", "x2", "y2"):
            setattr(self, name, np.array([np.nan if v is None else v for v in table[name]], dtype=np.float32))
        self.row_of = {t: i for i, t in enumerate(self.transition_id)}
        if len(self.row_of) != self.n:
            raise ValueError(f"Duplicate transition IDs in {self.path}")

    def frame(self, reader, row, which):
        offset, size = (self.off_t, self.size_t) if which == "t" else (self.off_t1, self.size_t1)
        return reader.read(self.shards[self.shard_id[row]], offset[row], size[row])

    def action(self, row, text_ids, text_emb, src_id=0):
        """Build the canonical action fields for one transition."""
        x, y, x2, y2 = self.x[row], self.y[row], self.x2[row], self.y2[row]
        has, has2, tid = not np.isnan(x), not np.isnan(x2), int(text_ids[row])
        return {
            "a_type": self.action_type[row],
            "a_coord": np.array([x if has else 0.0, y if has else 0.0], np.float32),
            "a_has_coord": has,
            "a_coord2": np.array([x2 if has2 else 0.0, y2 if has2 else 0.0], np.float32),
            "a_has_coord2": has2,
            "a_text": text_emb[tid],
            "a_has_text": bool(tid),
            "a_dir": self.direction[row],
            "src_id": np.int64(src_id),
        }



class PortableTransitions(torch.utils.data.Dataset):
    """One item = transition (frame t, action t) -> frame t+1, or an H-step window."""

    def __init__(self, index_path, data_root, text_table, *, holdout=None, keep="train", horizon=1,
                 excluded_apps=None, blank_below=BLANK_BELOW, src_id=0):
        if keep not in {"train", "holdout", "all"}:
            raise ValueError(keep)
        self.table, self.data_root, self.horizon, self.src_id = IndexTable.cached(index_path), data_root, horizon, src_id
        t, n = self.table, self.table.n
        select = np.ones(n, dtype=bool)
        if holdout is not None and keep != "all":
            held = set(json.loads(Path(holdout).read_text())["holdout_seg_keys"])
            inside = np.fromiter((k in held for k in t.seg_key), bool, n)
            select = inside if keep == "holdout" else ~inside
        blank = t.size_t1 < blank_below
        if blank_below > 0:
            select &= ~blank
        if excluded_apps:
            excluded = set(excluded_apps)
            select &= ~np.fromiter((a in excluded for a in t.app), bool, n)
        if horizon > 1:
            ok = t.step + horizon <= t.n_steps
            for h in range(horizon):
                clear = np.ones(n, dtype=bool)
                clear[: n - h] = ~blank[h:] if h else ~blank
                ok &= clear
            select &= ok
        self.rows = np.flatnonzero(select)
        self.text_ids, self.text_emb = text_table.ids(t.text), text_table.emb
        self.reader = None

    def __len__(self):
        return len(self.rows)

    def _reader(self):
        if self.reader is None:
            self.reader = ShardReader(self.data_root)
        return self.reader

    def __getitem__(self, j):
        i, t, reader = int(self.rows[j]), self.table, self._reader()
        if self.horizon == 1:
            return {"img_t": decode(t.frame(reader, i, "t")), "img_t1": decode(t.frame(reader, i, "t1")),
                    **t.action(i, self.text_ids, self.text_emb, self.src_id)}
        rows = [i + h for h in range(self.horizon)]
        if any(t.seg_key[r] != t.seg_key[i] for r in rows):
            raise RuntimeError("Rollout window crosses a segment boundary")
        frames = [decode(t.frame(reader, i, "t"))] + [decode(t.frame(reader, r, "t1")) for r in rows]
        actions = [t.action(r, self.text_ids, self.text_emb, self.src_id) for r in rows]
        return {"frames": np.stack(frames), **{k: np.stack([a[k] for a in actions]) for k in actions[0]}}


class MixedSources(torch.utils.data.Dataset):
    """Concatenation whose source IDs are bound by name to the per-source thresholds."""

    def __init__(self, parts):
        self.names, self.parts = list(parts), list(parts.values())
        self.lens = [len(p) for p in self.parts]
        self.offsets = np.cumsum([0] + self.lens)

    def __len__(self):
        return int(self.offsets[-1])

    def __getitem__(self, j):
        k = int(np.searchsorted(self.offsets, j, side="right") - 1)
        return self.parts[k][j - int(self.offsets[k])]


class QuotaSampler(torch.utils.data.Sampler):
    """Per-source quota sampler (with replacement, per-rank stream) with an exact resume offset."""

    def __init__(self, dataset, quotas, num_samples, *, rank=0, world=1, seed=2027, start=0):
        if set(quotas) != set(dataset.names) or abs(sum(quotas.values()) - 1) > 1e-6:
            raise ValueError("Quotas must cover every source and sum to one")
        weights = np.zeros(len(dataset))
        for i, name in enumerate(dataset.names):
            if dataset.lens[i] == 0:
                raise ValueError(f"Source {name} has no usable transitions")
            weights[dataset.offsets[i]:dataset.offsets[i + 1]] = quotas[name] / dataset.lens[i]
        self.p = weights / weights.sum()
        self.n, self.rank, self.seed, self.start = num_samples // world, rank, seed, start

    def __len__(self):
        return self.n - self.start

    def __iter__(self):
        draws = np.random.default_rng(self.seed + self.rank).choice(len(self.p), size=self.n, replace=True, p=self.p)
        yield from draws[self.start:].tolist()
