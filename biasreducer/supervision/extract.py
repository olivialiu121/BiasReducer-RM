"""Hidden states of controlled rewrites. For each rewrite rule the RM is run on the
original, the rewritten and (where present) the matched-control response; the
cache stores h(x_orig), h(x_edit) and h(x_control) row-aligned with the pairs.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..rm.loader import RewardModel


def extract_rewrite(rm: RewardModel, rows: list[dict], out: Path, verbose: bool = True) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    tag = rows[0]["rewrite"] if rows else out.stem
    H_orig = rm.hidden([(r["prompt"], r["original"]) for r in rows], verbose, f"{tag} orig")
    H_edit = rm.hidden([(r["prompt"], r["edited"]) for r in rows], verbose, f"{tag} edit")
    ctl = [i for i, r in enumerate(rows) if r.get("control")]
    H_ctl = rm.hidden([(rows[i]["prompt"], rows[i]["control"]) for i in ctl], verbose, f"{tag} ctl")
    np.savez(out, H_orig=H_orig, H_edit=H_edit, H_control=H_ctl,
             control_rows=np.asarray(ctl, np.int64),
             side=np.array([r["side"] for r in rows]),
             pair_id=np.array([r["pair_id"] for r in rows]))
    if verbose:
        print(f"[rewrite] wrote {out}: {len(rows)} pairs, {len(ctl)} controls", flush=True)
    return out


def load_rewrite_deltas(path: Path) -> dict:
    """{'d_fwd': delta of forward pairs, 'd_anti': delta of reverse pairs,
    'd_control': h(edit) - h(control) for forward pairs with a control}."""
    z = np.load(path, allow_pickle=False)
    delta = (z["H_edit"] - z["H_orig"]).astype(np.float32)
    side = z["side"]
    d = z["H_orig"].shape[1]
    out = dict(d_fwd=delta[side == "fwd"] if (side == "fwd").any() else np.zeros((0, d), np.float32),
               d_anti=delta[side == "anti"] if (side == "anti").any() else np.zeros((0, d), np.float32))
    ctl = z["control_rows"]
    out["d_control"] = ((z["H_edit"][ctl] - z["H_control"]).astype(np.float32)
                        if len(ctl) else np.zeros((0, d), np.float32))
    return out
