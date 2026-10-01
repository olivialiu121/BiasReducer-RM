"""Assemble the supervision tensors used to train the representation:

* standardized behavioral differences Delta y_a on the source-train pairs,
* intervention rows (delta, target coordinate, expected sign) per attributes.yaml,
* the pooled rewrite differences used by the artifact coordinate.
"""
from __future__ import annotations

import numpy as np

from ..config import Paths
from .extract import load_rewrite_deltas


def standardized_dy(attrs_path, train_idx: np.ndarray, attributes: list[str]) -> np.ndarray:
    """Delta y for the train rows, standardized column-wise over those rows."""
    z = np.load(attrs_path, allow_pickle=False)
    names = [str(a) for a in z["attributes"]]
    cols = [names.index(a) for a in attributes]
    Dy = (z["Y_chosen"][train_idx][:, cols] - z["Y_rejected"][train_idx][:, cols]).astype(np.float32)
    return (Dy - Dy.mean(0)) / (Dy.std(0) + 1e-8)


def _load_all(paths: Paths, model: str, rewrites: list[str]) -> dict:
    return {r: load_rewrite_deltas(paths.intervention_cache(model, r)) for r in rewrites}


def artifact_pool(paths: Paths, model: str, rewrites: list[str]) -> np.ndarray:
    """All forward and reverse rewrite differences, pooled over rewrite rules."""
    parts = []
    for r in rewrites:
        d = load_rewrite_deltas(paths.intervention_cache(model, r))
        parts.append(d["d_fwd"])
        if len(d["d_anti"]):
            parts.append(d["d_anti"])
    return np.concatenate(parts).astype(np.float32)


def intervention_rows(paths: Paths, model: str, attributes_cfg: dict, atom_of: dict[str, int],
                      g: np.ndarray):
    """(delta [M, d], atom [M], sign [M]) over every attribute with intervention
    streams. `g` is the unit artifact direction used for artifact-adjusted streams."""
    gs = g / (np.linalg.norm(g) + 1e-12)
    cache, deltas, atoms, signs = {}, [], [], []
    for name, spec in attributes_cfg["attributes"].items():
        for st in spec.get("interventions", []):
            if st["rewrite"] not in cache:
                cache[st["rewrite"]] = load_rewrite_deltas(paths.intervention_cache(model, st["rewrite"]))
            D = {"fwd": "d_fwd", "anti": "d_anti", "control": "d_control"}[st["side"]]
            D = cache[st["rewrite"]][D]
            if not len(D):
                continue
            if st.get("artifact_adjusted"):
                D = D - np.outer(D @ gs, gs)
            deltas.append(D.astype(np.float32))
            atoms.append(np.full(len(D), atom_of[name], np.int64))
            signs.append(np.full(len(D), float(st["sign"]), np.float32))
    return np.concatenate(deltas), np.concatenate(atoms), np.concatenate(signs)
