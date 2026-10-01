"""length_contract: drop the last k sentences at paragraph
boundaries so that ~20–30% of tokens are removed. Confounded with content loss
by design. No anti-pair."""
from __future__ import annotations

from ._text import (ends_terminal, rebuild_block, split_blocks,
                    split_sentences, word_count)

AXIS = "length_contract"
HAS_ANTI = False

LO, HI = 0.20, 0.30


def apply(text: str, rng) -> str | None:
    total = word_count(text)
    if total == 0:
        return None
    blocks = split_blocks(text)
    removed = 0

    def frac() -> float:
        return removed / total

    while frac() < LO:
        # find the last content block; only prose blocks may be trimmed
        ci = max((i for i, (k, _) in enumerate(blocks) if k in ("prose", "other")),
                 default=None)
        if ci is None or blocks[ci][0] != "prose":
            return None  # would have to cut inside a markdown structure
        sents = split_sentences(blocks[ci][1])
        if not sents:
            return None
        # drop trailing sentences of this paragraph one at a time
        while sents and frac() < LO:
            removed += word_count(sents.pop())
        if frac() > HI:
            return None  # sentences too chunky to land in the 20–30% window
        if not sents:
            # paragraph fully consumed -> drop block and its leading separator
            del blocks[ci]
            if ci > 0 and blocks[ci - 1][0] == "sep":
                del blocks[ci - 1]
        else:
            # prefer a paragraph-aligned cut if the rest also fits the budget
            rest = sum(word_count(s) for s in sents)
            if frac() >= LO and removed + rest <= HI * total and rng.random() < 0.5:
                removed += rest
                del blocks[ci]
                if ci > 0 and blocks[ci - 1][0] == "sep":
                    del blocks[ci - 1]
            else:
                blocks[ci] = ("prose", rebuild_block(sents))

    if not (LO <= frac() <= HI):
        return None
    out = "".join(t for _, t in blocks).rstrip()
    if not out or not ends_terminal(out):
        return None
    return out
