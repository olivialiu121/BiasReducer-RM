"""The reward-head edit:  w_a(lambda) = w - lambda (w^T C_a)/(C_a^T C_a + eps) C_a."""
from __future__ import annotations

import numpy as np

EPS = 1e-12


def edit_head(w: np.ndarray, C: np.ndarray, lam: float) -> np.ndarray:
    return w - lam * (float(w @ C) / (float(C @ C) + EPS)) * C
