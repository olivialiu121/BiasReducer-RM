"""length_expand: append a restatement of 1–2 of the response's own
sentences, or duplicate the concluding paragraph with a connective. Pure length;
content fixed. No anti-pair."""
from __future__ import annotations

from ._text import (eligible_sentences, load_bank, lower_first, split_blocks,
                    word_count)

AXIS = "length_expand"
HAS_ANTI = False


def apply(text: str, rng) -> str | None:
    bank = load_bank("connectives")
    blocks = split_blocks(text)
    mode = rng.choice(["restate", "restate", "dup_paragraph"])

    if mode == "restate":
        sents = [s for _, _, s in eligible_sentences(blocks, min_words=8)]
        if not sents:
            return None
        k = min(rng.choice([1, 2]), len(sents))
        chosen = rng.sample(sents, k)
        pieces = [f"{rng.choice(bank['restate'])} {lower_first(s, text)}" for s in chosen]
        return text.rstrip() + "\n\n" + " ".join(pieces)

    prose = [t for kind, t in blocks if kind == "prose" and word_count(t) >= 20]
    if not prose:
        return None
    return text.rstrip() + "\n\n" + rng.choice(bank["recap"]) + " " + prose[-1].strip()
