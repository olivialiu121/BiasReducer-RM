"""confidence: modal-verb lexicon swap toward assertiveness
("might be"→"is", "could"→"will", "perhaps"→∅, …). Only emit a pair if ≥3
swaps fired. Anti-pair: the inverse (strong→weak) lexicon. Lexicons live in
templates/confidence.json."""
from __future__ import annotations

from ._text import apply_lexicon, load_bank

AXIS = "confidence"
HAS_ANTI = True

MIN_FIRES = 3


def apply(text: str, rng) -> str | None:
    lex = load_bank("confidence")["weaken_to_strong"]
    out, fires = apply_lexicon(text, lex)
    if fires < MIN_FIRES:
        return None
    return out


def anti_density(text: str) -> float:
    """how much strong/assertive lexicon the response carries."""
    import re
    lex = load_bank("confidence")["strong_to_weak"]
    low = text.lower()
    return float(sum(len(re.findall(r"(?<![\w'])" + re.escape(find.strip().lower()), low))
                     for find, _ in lex))


def anti(text: str, rng) -> str | None:
    lex = load_bank("confidence")["strong_to_weak"]
    out, fires = apply_lexicon(text, lex)
    if fires < MIN_FIRES:
        return None
    return out
