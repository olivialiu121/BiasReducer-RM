"""BiasReducer-S and BiasReducer-M reward heads from the selection order.

Heads are built per trained dictionary (one C_a per seed); results are averaged
over seeds downstream. A head set is an array [n_fits, d].
"""
from __future__ import annotations

import numpy as np

from ..edit.bank import EditBank
from ..edit.head import edit_head


def apply_edits(w: np.ndarray, bank: EditBank, edits: list[tuple[str, float]]) -> np.ndarray:
    """Sequentially apply (attribute, lambda) edits; returns heads [n_fits, d]."""
    heads = [np.asarray(w, np.float64).copy() for _ in range(bank.n_fits)]
    for a, lam in edits:
        heads = [edit_head(h, bank.C[a][i], lam) for i, h in enumerate(heads)]
    return np.stack(heads)


def biasreducer_s(w: np.ndarray, bank: EditBank, order: list[str]) -> tuple[list[tuple[str, float]], np.ndarray]:
    a1 = order[0]
    edits = [(a1, bank.lam[a1])]
    return edits, apply_edits(w, bank, edits)


def biasreducer_m(w: np.ndarray, bank: EditBank, order: list[str], acc_fn, grid: list[float],
            budget: float, k_max: int) -> tuple[list[tuple[str, float]], np.ndarray]:
    """Traverse attributes in selection order; for each, search lambda downward from
    lambda*_a on the grid until the cumulative head satisfies the source budget.
    If no value is admissible, up to two adjustment steps reduce the largest
    selected lambda by one grid level and retry. Stops after k_max accepted edits."""
    base_acc = bank.base_val_acc
    n = bank.n_fits
    chosen: list[str] = []
    lams: list[float] = []
    heads = [np.asarray(w, np.float64).copy() for _ in range(n)]

    def rebuild():
        hs = [np.asarray(w, np.float64).copy() for _ in range(n)]
        for a, l in zip(chosen, lams):
            hs = [edit_head(h, bank.C[a][i], l) for i, h in enumerate(hs)]
        return hs

    def admissible(hs) -> bool:
        return base_acc - float(np.mean([acc_fn(h) for h in hs])) <= budget

    def try_add(a, base):
        start = grid.index(bank.lam[a])
        for lam in [grid[j] for j in range(start, -1, -1)]:
            cand = [edit_head(h, bank.C[a][i], lam) for i, h in enumerate(base)]
            if admissible(cand):
                return lam, cand
        return None, None

    for a in order:
        if len(chosen) >= k_max:
            break
        lam, cand = try_add(a, heads)
        adjust = 0
        while lam is None and adjust < 2 and chosen:
            j = max(range(len(chosen)), key=lambda k: lams[k])
            gi = grid.index(lams[j])
            if gi == 0:
                break
            lams[j] = grid[gi - 1]
            heads = rebuild()
            adjust += 1
            lam, cand = try_add(a, heads)
        if lam is None:
            continue
        heads = cand
        chosen.append(a)
        lams.append(lam)
    return list(zip(chosen, lams)), np.stack(heads)
