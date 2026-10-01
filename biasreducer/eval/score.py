"""Pairwise preference accuracy of a head set on a benchmark."""
from __future__ import annotations

import numpy as np

from .extract import BenchmarkHidden


def pairwise_accuracy(bench: BenchmarkHidden, heads: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Mean over heads [n_fits, d] of the fraction of pairs whose benchmark-preferred
    response scores higher (the bias term cancels in pairwise comparisons)."""
    m = bench.pair_mask() if mask is None else mask
    Hc, Hr = bench.H[bench.pair_c[m]], bench.H[bench.pair_r[m]]
    return float(np.mean([(Hc @ h > Hr @ h).mean() for h in np.atleast_2d(heads)]))


def accuracy_report(bench: BenchmarkHidden, heads: np.ndarray) -> dict:
    per_group = {g: pairwise_accuracy(bench, heads, bench.pair_mask(g)) for g in bench.groups}
    return dict(accuracy=pairwise_accuracy(bench, heads),
                macro=float(np.mean(list(per_group.values()))), per_group=per_group)
