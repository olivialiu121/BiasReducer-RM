"""Pool of source-train responses eligible for controlled rewrites."""
from __future__ import annotations

import re
from dataclasses import dataclass

from ..rm.cache import SourceCache
from .transforms._text import word_count


@dataclass(frozen=True)
class PoolItem:
    pair_idx: int          # row of the source cache
    side: str              # "chosen" | "rejected"
    prompt: str
    response: str
    n_words: int


_FENCE = re.compile(r"```|~~~")
_TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$", re.MULTILINE)
_SPECIAL_TOK = re.compile(r"<\|")
_CODE_LINE = re.compile(
    r"^\s*(?:import\s+\w|from\s+\S+\s+import|def\s+\w+\(|class\s+\w+[(:]|"
    r"#include|console\.|print\(|</?\w+>|\w+\s*=\s*\w+[.(])|[{};]\s*$")


def _english_ish(text: str, min_ascii_frac: float = 0.85) -> bool:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False
    return sum(c.isascii() for c in letters) / len(letters) >= min_ascii_frac


def eligible(response: str, min_words: int, max_words: int) -> bool:
    """Plain prose or simple markdown in English, within the word budget: no code
    fences, tables, chat-template tokens, or code-shaped lines."""
    n = word_count(response)
    if not (min_words <= n <= max_words):
        return False
    if _FENCE.search(response) or _TABLE_ROW.search(response) or _SPECIAL_TOK.search(response):
        return False
    if sum(bool(_CODE_LINE.match(ln)) for ln in response.splitlines()) >= 2:
        return False
    return _english_ish(response)


def build_pool(cache: SourceCache, min_words: int, max_words: int) -> list[PoolItem]:
    """Every eligible chosen and rejected response of the source TRAIN split."""
    pairs = cache.pairs()
    train = set(int(i) for i in cache.train_idx)
    items = []
    for i in sorted(train):
        p = pairs[i]
        for side, resp in (("chosen", p.chosen), ("rejected", p.rejected)):
            if eligible(resp, min_words, max_words):
                items.append(PoolItem(i, side, p.prompt, resp, word_count(resp)))
    return items
