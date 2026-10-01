"""Behavioral measurements y_a(x): deterministic text statistics of the seven
attributes (definitions in templates/behavioral.json; lexicons in the other
banks). Positive Delta y_a = y_a(x+) - y_a(x-) means the attribute is more
prominent in the preferred response.
"""
from __future__ import annotations

import re

import numpy as np

from ..data.pairs import Pair
from .transforms._text import load_bank

ATTRIBUTES = ["length", "verbosity_matched", "hedging", "sycophancy_surface",
              "format_bullets", "confidence", "emoji_excl"]


def _wb(term: str) -> re.Pattern:
    return re.compile(r"(?<![\w'])" + re.escape(term.strip()) + r"(?![\w])", re.I)


class Measurements:
    """Compiled attribute computers, exactly per templates/behavioral.json."""

    def __init__(self):
        spec = load_bank("behavioral")
        hed, syc, conf = load_bank("hedges"), load_bank("sycophancy"), load_bank("confidence")
        self.hedge_pats = [_wb(t) for t in hed["anti_lead_ins"] + hed["anti_adverbs"]]
        self.op_pats = [re.compile(p) for p in syc["anti_opener_patterns"]]
        self.cl_pats = [re.compile(p) for p in syc["anti_closer_patterns"]]
        self.strong = [_wb(f) for f, _ in conf["strong_to_weak"]]
        self.weak = [_wb(f) for f, _ in conf["weaken_to_strong"]]
        self.bullet = re.compile(spec["format_bullets"]["regex_bullet"])
        self.bold = re.compile(r"\*\*[^*\n]+\*\*")
        self.emoji = re.compile(spec["emoji_excl"]["regex"])
        self.ngram = spec["verbosity_matched"]["ngram"]

    def compute(self, text: str) -> dict[str, float]:
        words = text.split()
        n = max(len(words), 1)
        toks = [w.lower() for w in words]
        grams = [tuple(toks[i:i + self.ngram]) for i in range(len(toks) - self.ngram + 1)]
        rep = 0.0 if len(toks) < 20 or not grams else 1.0 - len(set(grams)) / len(grams)
        head, tail = text[:200], text[-200:]
        lines = [ln for ln in text.splitlines() if ln.strip()] or [""]
        return {
            "length": float(n),
            "verbosity_matched": rep,
            "hedging": 100.0 * sum(len(p.findall(text)) for p in self.hedge_pats) / n,
            "sycophancy_surface": float(sum(bool(p.search(head)) for p in self.op_pats)
                                        + sum(bool(p.search(tail)) for p in self.cl_pats)),
            "format_bullets": (sum(bool(self.bullet.match(ln)) for ln in lines)
                               + len(self.bold.findall(text))) / len(lines),
            "confidence": 100.0 * (sum(len(p.findall(text)) for p in self.strong)
                                   - sum(len(p.findall(text)) for p in self.weak)) / n,
            "emoji_excl": 100.0 * len(self.emoji.findall(text)) / n,
        }


def measure_pairs(pairs: list[Pair], verbose: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """(Y_chosen, Y_rejected) [N, len(ATTRIBUTES)] float32."""
    M = Measurements()
    Yc = np.zeros((len(pairs), len(ATTRIBUTES)), np.float32)
    Yr = np.zeros_like(Yc)
    for i, p in enumerate(pairs):
        yc, yr = M.compute(p.chosen), M.compute(p.rejected)
        Yc[i] = [yc[a] for a in ATTRIBUTES]
        Yr[i] = [yr[a] for a in ATTRIBUTES]
        if verbose and i % 20000 == 0:
            print(f"  [behavioral] {i}/{len(pairs)}", flush=True)
    return Yc, Yr
