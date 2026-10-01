"""Source/benchmark decontamination: a source pair is dropped when its prompt
matches a benchmark prompt after normalization, or shares at least one 13-gram
with a benchmark prompt.
"""
from __future__ import annotations

import re
import unicodedata

NGRAM = 13


def normalize(s: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", s).strip().lower())


def ngrams(tokens: list[str], n: int = NGRAM) -> set[tuple[str, ...]]:
    return {tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


class PromptIndex:
    def __init__(self, benchmark_prompts):
        self.exact, self.grams = set(), set()
        for p in benchmark_prompts:
            n = normalize(p)
            self.exact.add(n)
            self.grams |= ngrams(n.split())

    def collides(self, prompt: str) -> bool:
        n = normalize(prompt)
        return n in self.exact or bool(ngrams(n.split()) & self.grams)


def keep_indices(source_prompts, benchmark_prompts) -> list[int]:
    """Indices of source prompts that do not collide with any benchmark prompt."""
    index = PromptIndex(benchmark_prompts)
    return [i for i, p in enumerate(source_prompts) if not index.collides(p)]
