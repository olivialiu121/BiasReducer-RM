"""Edit-selection signals on a new dataset D (only responses, hidden states and
original rewards; no preference labels):

    G_a(D) = |Cov(w_r^T h, z_pre,a(h))|          reward dependence on attribute a
    I_a(D) = E |h^T w_a(lambda*_a) - h^T w_r|     impact of the attribute-specific edit
    B_a(D) = rank_G(a) + rank_I(a)                rank sum; smaller = higher priority
"""
from __future__ import annotations

import numpy as np

from ..edit.bank import EditBank
from ..edit.head import edit_head
from ..representation.model import Dictionary


def reward_dependence(H: np.ndarray, r: np.ndarray, dics: list[Dictionary], attributes: list[str]) -> dict[str, float]:
    rc = r - r.mean()
    zs = [dic.encode_pre(H).astype(np.float64) for dic in dics]
    return {a: float(np.mean([abs(float(np.cov(rc, z[:, dic.atom(a)])[0, 1]))
                              for dic, z in zip(dics, zs)])) for a in attributes}


def edit_impact(H: np.ndarray, r: np.ndarray, w: np.ndarray, bank: EditBank, attributes: list[str]) -> dict[str, float]:
    return {a: float(np.mean([np.abs(H @ edit_head(w, C, bank.lam[a]) - (H @ w)).mean()
                              for C in bank.C[a]])) for a in attributes}


def rank_sum_order(G: dict[str, float], I: dict[str, float], attributes: list[str]) -> tuple[list[str], dict[str, int]]:
    """Attributes sorted by increasing rank sum; ties broken by attribute order."""
    rank_G = {a: i for i, a in enumerate(sorted(attributes, key=lambda a: -G[a]))}
    rank_I = {a: i for i, a in enumerate(sorted(attributes, key=lambda a: -I[a]))}
    B = {a: rank_G[a] + rank_I[a] for a in attributes}
    order = sorted(attributes, key=lambda a: (B[a], attributes.index(a)))
    return order, B


def selection_signals(H: np.ndarray, w: np.ndarray, b: float, dics: list[Dictionary], bank: EditBank) -> dict:
    """All signals for one dataset: H [n_responses, d] pooled, pairing-blind."""
    H = np.asarray(H, np.float64)
    r = H @ w + b
    elig = bank.eligible
    G = reward_dependence(H, r, dics, elig)
    I = edit_impact(H, r, w, bank, elig)
    order, B = rank_sum_order(G, I, elig)
    return dict(dependence=G, impact=I, rank_sum=B, order=order)
