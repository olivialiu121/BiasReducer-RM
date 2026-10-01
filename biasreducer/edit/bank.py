"""Edit bank B = {(a, C_a, lambda*_a)}: covariance directions estimated on natural
source-train responses and per-attribute edit amounts chosen under the
source-validation accuracy budget. One C_a per trained dictionary;
lambda*_a is shared across seeds (chosen on seed-averaged statistics).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..representation.model import Dictionary
from ..rm.cache import SourceCache
from .head import edit_head


def natural_responses(cache: SourceCache, n: int) -> np.ndarray:
    """A fixed sample of n source-train responses: n/2 pairs drawn without
    replacement (seed 0), chosen responses followed by rejected responses."""
    tr = cache.train_idx
    take = np.random.RandomState(0).choice(len(tr), size=min(n // 2, len(tr)), replace=False)
    idx = tr[np.sort(take)]
    return np.concatenate([np.asarray(cache.H_chosen[idx], np.float32),
                           np.asarray(cache.H_rejected[idx], np.float32)])


def source_val_accuracy(cache: SourceCache):
    """acc(w): preference accuracy of head w on the source validation split."""
    va = cache.val_idx
    D = np.asarray(cache.H_chosen[va], np.float64) - np.asarray(cache.H_rejected[va], np.float64)

    def acc(w: np.ndarray) -> float:
        return float((D @ np.asarray(w, np.float64) > 0).mean())
    return acc


def covariance_directions(dic: Dictionary, H: np.ndarray) -> dict[str, np.ndarray]:
    """C_a = Cov_h(h, z_post,a(h)) for every attribute, on natural responses H."""
    H64 = H.astype(np.float64)
    Hc = H64 - H64.mean(0, keepdims=True)
    z = dic.encode(H).astype(np.float64)
    out = {}
    for a in dic.attributes:
        zc = z[:, dic.atom(a)] - z[:, dic.atom(a)].mean()
        out[a] = (Hc.T @ zc) / len(H)
    return out


def calibrate(w: np.ndarray, b: float, C: list[np.ndarray], H_agg: np.ndarray, acc_fn,
              base_acc: float, grid: list[float], budget: float) -> dict | None:
    """lambda* = argmax_lambda Agg(lambda) s.t. base_acc - Acc_src(w(lambda)) <= budget,
    with Agg and Acc averaged over the per-seed heads. None if nothing is admissible."""
    r_base = H_agg @ w + b
    best = None
    for lam in grid:
        heads = [edit_head(w, c, lam) for c in C]
        agg = float(np.mean([np.abs(H_agg @ h + b - r_base).mean() for h in heads]))
        acc = float(np.mean([acc_fn(h) for h in heads]))
        if base_acc - acc <= budget and (best is None or agg > best["agg"]):
            best = dict(lam=lam, agg=agg, val_acc=acc)
    return best


class EditBank:
    def __init__(self, attributes: list[str], C: dict[str, list[np.ndarray]],
                 lam: dict[str, float], base_val_acc: float, calibration: dict):
        self.attributes = attributes          # fixed attribute order
        self.C = C                            # attribute -> [C_a per seed]
        self.lam = lam                        # attribute -> lambda* (NaN if ineligible)
        self.base_val_acc = base_val_acc
        self.calibration = calibration

    @property
    def n_fits(self) -> int:
        return len(next(iter(self.C.values())))

    @property
    def eligible(self) -> list[str]:
        return [a for a in self.attributes if np.isfinite(self.lam[a])]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {f"C__{a}__{i}": c for a in self.attributes for i, c in enumerate(self.C[a])}
        np.savez(path, **arrays, attributes=np.array(self.attributes), n_fits=self.n_fits,
                 lam=np.array([self.lam[a] for a in self.attributes]), base_val_acc=self.base_val_acc)
        path.with_suffix(".json").write_text(json.dumps(
            dict(base_val_acc=self.base_val_acc, calibration=self.calibration), indent=2))

    @classmethod
    def load(cls, path: Path) -> "EditBank":
        z = np.load(path, allow_pickle=False)
        attrs = [str(a) for a in z["attributes"]]
        n = int(z["n_fits"])
        C = {a: [z[f"C__{a}__{i}"] for i in range(n)] for a in attrs}
        lam = {a: float(l) for a, l in zip(attrs, z["lam"])}
        cal = json.loads(path.with_suffix(".json").read_text())["calibration"] \
            if path.with_suffix(".json").exists() else {}
        return cls(attrs, C, lam, float(z["base_val_acc"]), cal)


def build_bank(dics: list[Dictionary], cache: SourceCache, edit_cfg: dict, verbose: bool = True) -> EditBank:
    w, b = cache.w_r.astype(np.float64), cache.b_r
    H = natural_responses(cache, int(edit_cfg["n_natural_responses"]))
    H_agg = H[:int(edit_cfg["n_aggressiveness"])].astype(np.float64)
    acc_fn = source_val_accuracy(cache)
    base_acc = acc_fn(w)
    attributes = dics[0].attributes
    C = {a: [] for a in attributes}
    for dic in dics:
        for a, c in covariance_directions(dic, H).items():
            C[a].append(c)
    lam, cal = {}, {}
    for a in attributes:
        best = calibrate(w, b, C[a], H_agg, acc_fn, base_acc, edit_cfg["lambda_grid"], edit_cfg["source_budget"])
        lam[a] = best["lam"] if best else float("nan")
        cal[a] = dict(eligible=best is not None, **(best or {}))
        if verbose:
            print(f"  [bank] {a:20s} " + (f"lambda*={best['lam']}  val_acc={best['val_acc']:.4f}" if best else "ineligible"))
    return EditBank(attributes, C, lam, base_acc, cal)
