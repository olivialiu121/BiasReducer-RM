"""Shared rewrite-artifact direction g: the top principal direction of all pooled
controlled-rewrite differences, oriented along their mean. It initializes the
artifact coordinate and is projected out of artifact-adjusted streams."""
from __future__ import annotations

import numpy as np


def artifact_direction(pooled: np.ndarray) -> np.ndarray:
    _, _, Vt = np.linalg.svd(np.asarray(pooled, np.float32), full_matrices=False)
    g = Vt[0]
    if float(pooled.mean(0) @ g) < 0:
        g = -g
    return g.astype(np.float32)
