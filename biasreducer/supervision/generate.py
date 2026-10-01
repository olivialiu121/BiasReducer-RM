"""Generate controlled rewrite pairs for every rewrite rule.

For each rule: walk a deterministically shuffled pool, apply the forward rewrite
(and the reverse rewrite where the rule has one), QC-filter, keep n pairs, at
most `max_pairs_per_prompt` per prompt. Rules with a matched control emit a
control rewrite alongside every forward pair. One JSONL row per kept pair.
"""
from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from .pool import PoolItem
from .qc import NED_LO, NED_LO_ANTI, qc_pair, structure_ok
from .transforms import REWRITES


def sample_rewrite(rewrite: str, side: str, pool: list[PoolItem], gen_cfg: dict,
                   prompt_quota: dict) -> tuple[list[dict], dict]:
    """One rewrite rule x one side ("fwd" uses .apply, "anti" uses .anti). Anti
    pairs iterate the pool densest-first in the attribute being removed."""
    mod = REWRITES[rewrite]
    fn = mod.apply if side == "fwd" else getattr(mod, "anti", None)
    assert fn is not None, f"{rewrite} has no {side} rewrite"
    has_control = side == "fwd" and getattr(mod, "HAS_CONTROL", False)
    ned_lo = NED_LO_ANTI if side == "anti" else NED_LO
    n_target = gen_cfg["n_pairs_per_rewrite"]
    max_attempts = int(gen_cfg["max_attempts_factor"]) * n_target
    seed = gen_cfg["seed"]
    rng = random.Random(f"{seed}:{rewrite}:{side}")

    order = list(range(len(pool)))
    rng.shuffle(order)
    if side == "anti" and getattr(mod, "anti_density", None) is not None:
        order.sort(key=lambda i: -mod.anti_density(pool[i].response))

    kept, attempts, n_controls, fail = [], 0, 0, Counter()
    for idx in order:
        if len(kept) >= n_target or attempts >= max_attempts:
            break
        item = pool[idx]
        if prompt_quota[item.prompt] >= gen_cfg["max_pairs_per_prompt"]:
            continue
        attempts += 1
        control = None
        if has_control:
            ctl_rng = random.Random(f"ctl:{seed}:{rewrite}:{item.pair_idx}:{item.side}")
            r = mod.apply_with_control(item.response, rng, ctl_rng)
            edited, control = r if r is not None else (None, None)
        else:
            edited = fn(item.response, rng)
        if edited is None:
            fail["transform-none"] += 1
            continue
        ok, why, ned = qc_pair(item.response, edited, ned_lo=ned_lo)
        if not ok:
            fail[why] += 1
            continue
        if control is not None:
            c_ok, c_why = structure_ok(control)
            if not c_ok or control.strip() == item.response.strip():
                fail[f"control-{c_why or 'no-op'}"] += 1
                control = None
            else:
                n_controls += 1
        prompt_quota[item.prompt] += 1
        row = dict(rewrite=rewrite, side=side, pair_id=f"{rewrite}:{side}:{len(kept)}",
                   src_pair_idx=item.pair_idx, src_side=item.side,
                   prompt=item.prompt, original=item.response, edited=edited,
                   ned=round(ned, 4), n_words_orig=item.n_words)
        if control is not None:
            row["control"] = control
        kept.append(row)
    stats = dict(side=side, kept=len(kept), target=n_target, attempts=attempts,
                 n_controls=n_controls, fail_counts=dict(fail))
    return kept, stats


def generate_all(pool: list[PoolItem], rewrites: list[str], gen_cfg: dict,
                 out_path_for, verbose: bool = True) -> dict:
    """Write <rewrite>.jsonl for every rule; returns the yield statistics."""
    yields = {}
    for rewrite in rewrites:
        quota = defaultdict(int)
        rows, st = sample_rewrite(rewrite, "fwd", pool, gen_cfg, quota)
        yields[rewrite] = [st]
        if getattr(REWRITES[rewrite], "HAS_ANTI", False):
            anti_rows, st2 = sample_rewrite(rewrite, "anti", pool, gen_cfg, quota)
            rows += anti_rows
            yields[rewrite].append(st2)
        out = Path(out_path_for(rewrite))
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        if verbose:
            for st in yields[rewrite]:
                print(f"[rewrite] {rewrite:20s} {st['side']:4s} kept {st['kept']:4d}/{st['target']} "
                      f"attempts {st['attempts']:5d} controls {st['n_controls']}", flush=True)
    return yields


def load_rewrite_pairs(path: str | Path) -> list[dict]:
    p = Path(path)
    assert p.exists(), f"no rewrite pairs at {p}"
    return [json.loads(ln) for ln in p.read_text().splitlines() if ln.strip()]
