"""Hidden states of every benchmark response under one RM, with the unit / pair
structure of the pack."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..rm.loader import RewardModel


def extract_benchmark(rm: RewardModel, units: list[dict], out: Path, verbose: bool = True) -> Path:
    items, resp_unit, pair_c, pair_r, pair_unit = [], [], [], [], []
    for ui, u in enumerate(units):
        off = len(items)
        items += [(u["prompt"], t) for t in u["responses"]]
        resp_unit += [ui] * len(u["responses"])
        for c, r in u["pairs"]:
            pair_c.append(off + c); pair_r.append(off + r); pair_unit.append(ui)
    H = rm.hidden(items, verbose, out.parent.name)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, H=H, resp_unit=np.asarray(resp_unit, np.int64),
             pair_c=np.asarray(pair_c, np.int64), pair_r=np.asarray(pair_r, np.int64),
             pair_unit=np.asarray(pair_unit, np.int64),
             unit_group=np.array([u["group"] for u in units]),
             unit_uid=np.array([u["uid"] for u in units]))
    if verbose:
        print(f"[benchmark] wrote {out}: {len(units)} units, {len(items)} responses, {len(pair_c)} pairs")
    return out


class BenchmarkHidden:
    def __init__(self, path: Path):
        z = np.load(path, allow_pickle=False)
        self.H = z["H"].astype(np.float64)
        self.resp_unit, self.pair_c, self.pair_r = z["resp_unit"], z["pair_c"], z["pair_r"]
        self.pair_unit, self.unit_group = z["pair_unit"], z["unit_group"]
        self.groups = list(dict.fromkeys(self.unit_group.tolist()))

    def responses(self, group: str | None = None) -> np.ndarray:
        """Pooled response hidden states (all units, or one group)."""
        if group is None:
            return self.H
        units = np.where(self.unit_group == group)[0]
        return self.H[np.isin(self.resp_unit, units)]

    def pair_mask(self, group: str | None = None) -> np.ndarray:
        if group is None:
            return np.ones(len(self.pair_c), bool)
        units = np.where(self.unit_group == group)[0]
        return np.isin(self.pair_unit, units)
