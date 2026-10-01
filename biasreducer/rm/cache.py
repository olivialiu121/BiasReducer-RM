"""Source hidden-state cache: h(x+), h(x-) for every source preference pair, the
reward head (w_r, b_r), the prompt-level train/validation split and the pair
texts. The preference contrast Delta h = h(x+) - h(x-) is computed on read.
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np

from ..data.pairs import Pair, prompt_level_split
from .loader import RewardModel

INVARIANT_RTOL = {"float32": 0.02, "bfloat16": 0.06, "float16": 0.15}


class SourceCache:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.meta = json.loads((self.path / "meta.json").read_text())
        split = json.loads((self.path / "split.json").read_text())
        self.train_idx = np.asarray(split["train_idx"], dtype=np.int64)
        self.val_idx = np.asarray(split["val_idx"], dtype=np.int64)

    @property
    def d(self) -> int:
        return int(self.meta["d"])

    @property
    def H_chosen(self) -> np.ndarray:
        return np.load(self.path / "H_chosen.npy", mmap_mode="r")

    @property
    def H_rejected(self) -> np.ndarray:
        return np.load(self.path / "H_rejected.npy", mmap_mode="r")

    def dh(self, idx=None) -> np.ndarray:
        """Preference contrasts Delta h [n, d] for row indices `idx` (all rows if None)."""
        Hc, Hr = self.H_chosen, self.H_rejected
        if idx is None:
            return np.asarray(Hc, np.float32) - np.asarray(Hr, np.float32)
        return np.asarray(Hc[idx], np.float32) - np.asarray(Hr[idx], np.float32)

    @property
    def w_r(self) -> np.ndarray:
        return np.load(self.path / "w_r.npy")

    @property
    def b_r(self) -> float:
        return float(np.load(self.path / "b_r.npy"))

    def pairs(self) -> list[Pair]:
        with gzip.open(self.path / "pairs.jsonl.gz", "rt") as f:
            return [Pair(**json.loads(line)) for line in f]


def build_source_cache(rm: RewardModel, pairs: list[Pair], out: Path, val_frac: float,
                       seed: int, meta_extra: dict | None = None, verbose: bool = True) -> Path:
    """Forward every pair through the RM and write the cache. The head invariant
    w_r^T h + b_r == emitted reward is checked on every batch."""
    out.mkdir(parents=True, exist_ok=True)
    N, d = len(pairs), rm.d
    Hc = np.lib.format.open_memmap(out / "H_chosen.npy", mode="w+", dtype=np.float32, shape=(N, d))
    Hr = np.lib.format.open_memmap(out / "H_rejected.npy", mode="w+", dtype=np.float32, shape=(N, d))
    bs, worst = rm.batch_size, 0.0
    for s in range(0, N, bs):
        batch = pairs[s:s + bs]
        hc, rc = rm.hidden_and_rewards([(p.prompt, p.chosen) for p in batch])
        hr, _ = rm.hidden_and_rewards([(p.prompt, p.rejected) for p in batch])
        assert np.isfinite(hc).all() and np.isfinite(hr).all(), f"non-finite hidden states at {s}"
        Hc[s:s + len(batch)], Hr[s:s + len(batch)] = hc, hr
        rel = float((np.abs(hc @ rm.w_r + rm.b_r - rc) / (np.abs(rc) + 1.0)).max())
        worst = max(worst, rel)
        if verbose and (s // bs) % 50 == 0:
            print(f"  [source] {s + len(batch)}/{N} invariant rel-err <= {worst:.2e}", flush=True)
    Hc.flush(); Hr.flush()
    rtol = INVARIANT_RTOL.get(rm.cfg.get("forward_dtype", "float32"), 0.05)
    assert worst < rtol, f"head invariant violated ({worst:.3f} >= {rtol}); pooling does not match the head"
    np.save(out / "w_r.npy", rm.w_r.astype(np.float32))
    np.save(out / "b_r.npy", np.asarray(rm.b_r, dtype=np.float32))
    train_idx, val_idx = prompt_level_split(pairs, val_frac, seed)
    (out / "split.json").write_text(json.dumps({"train_idx": train_idx, "val_idx": val_idx}))
    with gzip.open(out / "pairs.jsonl.gz", "wt") as f:
        for p in pairs:
            f.write(json.dumps(dict(prompt=p.prompt, chosen=p.chosen, rejected=p.rejected)) + "\n")
    (out / "meta.json").write_text(json.dumps(dict(
        model=rm.cfg["name"], hf_id=rm.cfg["hf_id"], d=d, N=N, seed=seed,
        n_train=len(train_idx), n_val=len(val_idx), max_invariant_rel_err=worst,
        **(meta_extra or {})), indent=2))
    if verbose:
        print(f"[source] wrote {out}: {N} pairs, d={d}, train {len(train_idx)} / val {len(val_idx)}")
    return out
