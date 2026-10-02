"""RSWT-Bench manifests and donor-paired statistics."""
import json
from pathlib import Path

import numpy as np

from .evaluation import CLASSES, metrics

POSITIVE, LEGITIMATE = CLASSES[0], CLASSES[2]
MANIFEST_FORMAT = "rswt-bench-v1"
FRAME_KEYS = ("base", "obs", "clean")


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_jsonl(path, records):
    Path(path).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))


def check_manifest(records, *, test_split):
    """Validate the benchmark structure; raises on any violation."""
    sids = [r["sid"] for r in records]
    if len(set(sids)) != len(sids):
        raise ValueError("Duplicate benchmark sample IDs")
    if {r["cls"] for r in records} != set(CLASSES):
        raise ValueError("Benchmark must contain all five categories")
    for r in records:
        if r.get("format") != MANIFEST_FORMAT:
            raise ValueError(f"Unexpected manifest format in {r['sid']}")
        for key in FRAME_KEYS:
            if set(r[key]) != {"transition_id", "split"}:
                raise ValueError(f"{r['sid']}.{key} must hold transition_id and split only")
    if test_split:
        hijack = [r["donor_id"] for r in records if r["cls"] == POSITIVE]
        legit = [r["donor_id"] for r in records if r["cls"] == LEGITIMATE]
        if len(set(hijack)) != len(hijack) or set(hijack) != set(legit) or len(legit) != len(set(legit)):
            raise ValueError("Test donors must pair hijacked and legitimate occurrences one-to-one")


def donor_pairs(records):
    """[(hijacked sid, legitimate sid)] ordered by donor_id, the bootstrap unit."""
    hijack, legit = {}, {}
    for r in records:
        if r["cls"] == POSITIVE:
            hijack[r["donor_id"]] = r["sid"]
        elif r["cls"] == LEGITIMATE:
            legit[r["donor_id"]] = r["sid"]
    if set(hijack) != set(legit):
        raise ValueError("Donor pairing is incomplete")
    return [(hijack[d], legit[d]) for d in sorted(hijack)]


def _auc(pos, neg):
    """Mid-rank AUC, identical to sklearn's roc_auc_score for binary labels."""
    values = np.r_[pos, neg]
    order = values.argsort(kind="mergesort")
    ranks = np.empty(len(values))
    ranks[order] = np.arange(1, len(values) + 1)
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inverse, ranks)
    ranks = (sums / counts)[inverse]
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def _weighted_auc(a, b, weights):
    """AUC for each row of donor multiplicities; equivalent to resampling donors."""
    numerator = np.zeros(len(weights))
    below = np.zeros(len(weights))
    for value in np.unique(np.r_[a, b]):
        pos, neg = np.flatnonzero(a == value), np.flatnonzero(b == value)
        wp = weights[:, pos].sum(axis=1) if len(pos) else 0.0
        wn = weights[:, neg].sum(axis=1) if len(neg) else 0.0
        numerator += wp * (below + 0.5 * wn)
        below += wn
    return numerator / float(a.size * a.size)


def donor_bootstrap(scores, pairs, *, n_boot=200_000, seed=2027, batch=2000, reference=None):
    """RSWT, donor-pair accuracy and 95% intervals; optional paired difference vs reference."""
    a = np.array([scores[h] for h, _ in pairs], dtype=np.float64)
    b = np.array([scores[c] for _, c in pairs], dtype=np.float64)
    k, rng = len(a), np.random.default_rng(seed)
    wins = (a > b) + 0.5 * (a == b)
    aucs, accs, deltas = np.empty(n_boot), np.empty(n_boot), np.empty(n_boot)
    if reference is not None:
        ra = np.array([reference[h] for h, _ in pairs], dtype=np.float64)
        rb = np.array([reference[c] for _, c in pairs], dtype=np.float64)
    for start in range(0, n_boot, batch):
        stop = min(start + batch, n_boot)
        weights = rng.multinomial(k, np.full(k, 1.0 / k), size=stop - start)
        aucs[start:stop] = _weighted_auc(a, b, weights)
        accs[start:stop] = weights @ wins / k
        if reference is not None:
            deltas[start:stop] = aucs[start:stop] - _weighted_auc(ra, rb, weights)
    result = {
        "rswt": _auc(a, b), "rswt_ci95": np.percentile(aucs, [2.5, 97.5]).tolist(),
        "pair_accuracy": float(wins.mean()), "pairs_ordered": float(wins.sum()),
        "pair_accuracy_ci95": np.percentile(accs, [2.5, 97.5]).tolist(),
        "n_pairs": k, "n_boot": n_boot, "seed": seed, "unit": "donor pair",
    }
    if reference is not None:
        result.update({
            "delta_vs_reference": float(deltas.mean()),
            "delta_ci95": np.percentile(deltas, [2.5, 97.5]).tolist(),
            "p_two_sided": float(2 * min((deltas <= 0).mean(), (deltas >= 0).mean())),
        })
    return result


def paper_table(classes, scores):
    """Core metrics, including the RSWT (1v3), Swap (5v2) and Cred (5v3) columns."""
    result = metrics(classes, scores)
    pairwise = result["pairwise"]
    result["paper_columns"] = {"RSWT": pairwise["1v3"], "Swap": pairwise["5v2"],
                               "Cred": pairwise["5v3"], "Harm_AUC": pairwise["1v5"]}
    return result
