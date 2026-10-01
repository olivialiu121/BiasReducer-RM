"""sycophancy_surface: prepend an opener and/or append a closer
from the template bank.
Anti-pair: strip naturally occurring openers/closers.

Matched control: neutral opener/closer of matched word count at the same
placement, chosen from a separate rng (edit stream unchanged vs `apply`).
"""
from __future__ import annotations

import re

from ._text import load_bank

AXIS = "sycophancy_surface"
HAS_ANTI = True
HAS_CONTROL = True


def _nearest(bank_by_wc: dict, wc: int, ctl_rng) -> str:
    for k in sorted(bank_by_wc, key=lambda k: abs(int(k) - wc)):
        if bank_by_wc[k]:
            return ctl_rng.choice(bank_by_wc[k])
    raise RuntimeError("empty neutral bank")


def _run(text: str, rng, ctl_rng):
    bank = load_bank("sycophancy")
    neutral = load_bank("neutral") if ctl_rng is not None else None
    mode = rng.choice(["both", "both", "opener", "closer"])
    body = text.rstrip()
    out = ctl = body
    if mode in ("both", "opener"):
        op = rng.choice(bank["openers"])
        out = f"{op} {out}"
        if ctl_rng is not None:
            ctl = f"{_nearest(neutral['openers'], len(op.split()), ctl_rng)} {ctl}"
    if mode in ("both", "closer"):
        cl = rng.choice(bank["closers"])
        newline = rng.random() < 0.5
        out = f"{out}\n\n{cl}" if newline else f"{out} {cl}"
        if ctl_rng is not None:
            ncl = _nearest(neutral["closers"], len(cl.split()), ctl_rng)
            ctl = f"{ctl}\n\n{ncl}" if newline else f"{ctl} {ncl}"
    return out, (ctl if ctl_rng is not None else None)


def apply(text: str, rng) -> str | None:
    return _run(text, rng, None)[0]


def apply_with_control(text: str, rng, ctl_rng):
    return _run(text, rng, ctl_rng)


def anti_density(text: str) -> float:
    """does the response naturally open/close sycophantically."""
    bank = load_bank("sycophancy")
    head, tail = text.strip()[:200], text.strip()[-200:]
    n = sum(bool(re.search(p, head)) for p in bank["anti_opener_patterns"])
    n += sum(bool(re.search(p, tail)) for p in bank["anti_closer_patterns"])
    return float(n)


def anti(text: str, rng) -> str | None:
    bank = load_bank("sycophancy")
    removed = 0
    out = text.strip()
    for pat in bank["anti_opener_patterns"]:
        new, n = re.subn(pat, "", out, count=1)
        if n:
            out, removed = new.lstrip(), removed + n
    for pat in bank["anti_closer_patterns"]:
        new, n = re.subn(pat, "", out, count=1)
        if n:
            out, removed = new.rstrip(), removed + n
    if removed < 1:
        return None
    return out.strip() or None
