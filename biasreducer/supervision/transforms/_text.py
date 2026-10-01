"""Shared text machinery for the rule-based rewrites.

Block model: a response is split into alternating (kind, text) blocks where
kind is one of {"prose", "other", "sep"}. Separators are preserved verbatim so an
untouched response reconstructs exactly; every character difference between the
original and the rewritten text is an intended edit, which the pair QC relies on.

Template banks are the versioned JSON files in templates/.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "templates"

assert TEMPLATE_DIR.is_dir(), (
    f"template banks missing: {TEMPLATE_DIR}"
)


@lru_cache(maxsize=None)
def load_bank(name: str) -> dict:
    p = TEMPLATE_DIR / f"{name}.json"
    assert p.exists(), (
        f"template bank {name!r} not found at {p}"
    )
    return json.loads(p.read_text())


# --------------------------------------------------------------------------- #
# block / sentence structure
# --------------------------------------------------------------------------- #
_LIST_LINE = re.compile(r"^\s*(?:[-*+]\s|\d+[.)]\s|#{1,6}\s|>\s?|\|)")
_SEP_SPLIT = re.compile(r"(\n[ \t]*\n+)")


def split_blocks(text: str) -> list[tuple[str, str]]:
    """Split into (kind, text) blocks; ''.join(t for _, t in blocks) == text."""
    blocks: list[tuple[str, str]] = []
    for piece in _SEP_SPLIT.split(text):
        if piece == "":
            continue
        if _SEP_SPLIT.fullmatch(piece):
            blocks.append(("sep", piece))
        else:
            blocks.append(("prose" if is_prose(piece) else "other", piece))
    return blocks


def join_blocks(blocks: list[tuple[str, str]]) -> str:
    out = "".join(t for _, t in blocks)
    # collapse a dangling separator left by a removed trailing block
    return out.rstrip() if out else out


def is_prose(block: str) -> bool:
    """True if no line of the block is a markdown structure (list/header/quote/
    table). Inline code (`x`) is allowed; sentences containing backticks are
    simply never chosen as insertion targets."""
    lines = [ln for ln in block.splitlines() if ln.strip()]
    if not lines:
        return False
    return not any(_LIST_LINE.match(ln) for ln in lines)


_ABBREV = {"e.g", "i.e", "etc", "vs", "cf", "al", "dr", "mr", "mrs", "ms",
           "prof", "st", "no", "fig", "eq", "sec", "approx", "dept", "inc"}
_SENT_SPLIT = re.compile(r"(?<=[.!?])[ \t]+(?=[\"'(\[]?[A-Z])")


def split_sentences(par: str) -> list[str]:
    """Sentence list for one prose paragraph; ' '.join(...) ≈ par (whitespace-
    normalized within the paragraph — callers rebuild the block from these)."""
    parts = _SENT_SPLIT.split(par.strip())
    merged: list[str] = []
    for p in parts:
        if merged:
            m = re.search(r"([A-Za-z][A-Za-z.]*)\.$", merged[-1])
            w = m.group(1).lower().rstrip(".") if m else ""
            if w in _ABBREV or (len(w) == 1 and w.isalpha()):
                merged[-1] = merged[-1] + " " + p
                continue
        merged.append(p)
    return merged


def word_count(text: str) -> int:
    """Whitespace word count: the token proxy used by the word-count filter and the length budgets."""
    return len(text.split())


def ends_terminal(sentence: str) -> bool:
    return bool(re.search(r"[.!?…][\"')\]]*\s*$", sentence))


# --------------------------------------------------------------------------- #
# safe lower-casing for template insertion
# --------------------------------------------------------------------------- #
# Sentence-initial words that can be safely lower-cased after "It may be that …"
# etc. Anything else (proper nouns, acronyms, "I") keeps its capitalization —
# a capitalized proper noun after a connective is grammatical; "The" is not.
_SAFE_LOWER = {
    "The", "This", "That", "These", "Those", "There", "Here", "It", "They",
    "You", "We", "He", "She", "One", "People", "Some", "Many", "Most", "All",
    "Both", "Each", "Every", "Any", "No", "None", "A", "An", "Your", "Their",
    "Our", "Its", "His", "Her", "My", "If", "When", "While", "Although",
    "Though", "Because", "Since", "As", "In", "On", "At", "For", "To", "With",
    "By", "From", "Over", "Under", "During", "After", "Before", "Once",
    "However", "Also", "Additionally", "Furthermore", "Moreover", "Finally",
    "First", "Second", "Third", "Next", "Then", "Now", "Today", "Generally",
    "Usually", "Often", "Sometimes", "Typically", "Ultimately", "Overall",
    "Remember", "Note", "Consider", "Keep", "Make", "Try", "Use", "Using",
    "Doing", "Being", "Having", "Getting", "Not", "Even", "Just", "Only",
}


def lower_first(sentence: str, context: str = "") -> str:
    """Lower-case the sentence's first word for insertion after a connective
    ("It may be that …"). Keep capitalization only when the word is plausibly
    a proper noun: 'I'/'I'm', an acronym / internal-capital word, a word in
    the safe-starter list stays lowerable, and otherwise a word that also
    appears capitalized MID-sentence in `context` (proper-noun signal)."""
    m = re.match(r"([\"'(\[]?)(\w+)(.*)", sentence, flags=re.DOTALL)
    if not m:
        return sentence
    pre, word, rest = m.groups()
    if word == "I" or word.startswith("I'"):
        return sentence
    if len(word) > 1 and any(c.isupper() for c in word[1:]):
        return sentence  # acronym / CamelCase
    if word[0].islower():
        return sentence
    if word not in _SAFE_LOWER and context:
        # capitalized occurrence after a lowercase word => likely proper noun
        if re.search(r"[a-z,;:]\s+" + re.escape(word) + r"\b", context):
            return sentence
    return f"{pre}{word.lower()}{rest}"


def sentence_has_code(sentence: str) -> bool:
    return "`" in sentence


def eligible_sentences(blocks, min_words: int = 5):
    """Yield (block_idx, sent_idx, sentence) for prose sentences that are safe
    insertion targets: no inline code, long enough, terminally punctuated."""
    for bi, (kind, btext) in enumerate(blocks):
        if kind != "prose":
            continue
        for si, s in enumerate(split_sentences(btext)):
            if sentence_has_code(s) or word_count(s) < min_words or not ends_terminal(s):
                continue
            yield bi, si, s


def rebuild_block(sentences: list[str]) -> str:
    return " ".join(sentences)


# --------------------------------------------------------------------------- #
# case-aware lexicon swaps (confidence axis; shared with hedging-anti)
# --------------------------------------------------------------------------- #
def _sub_case_variant(text: str, find: str, repl: str) -> tuple[str, int]:
    """One case variant: replace `find` (word-boundary) with `repl`; when `find`
    is deleted ('' repl) at a sentence start, re-capitalize the next word."""
    # entries ending in a space carry their own right boundary; otherwise
    # require a non-word char so "could" never matches inside "couldn't"
    tail = "" if find.endswith(" ") else r"(?![\w])"
    pat = re.compile(r"(?<![\w'])" + re.escape(find) + tail)
    if repl != "":
        return pat.subn(repl, text)
    # deletion: absorb it, then fix capitalization where the deletion exposed a
    # sentence start ("Perhaps the answer" -> "The answer").
    n_total = 0
    out = text
    while True:
        m = pat.search(out)
        if not m:
            break
        start, end = m.span()
        was_sentence_start = start == 0 or re.search(r"[.!?:]\s+$|\n\s*$", out[:start])
        rest = out[end:]
        if was_sentence_start and find[:1].isupper():
            rest = rest[:1].upper() + rest[1:]
        out = out[:start] + rest
        n_total += 1
    return out, n_total


def apply_lexicon(text: str, entries: list[list[str]]) -> tuple[str, int]:
    """Apply [find, replace] entries longest-first, both lowercase and
    Capitalized variants, only OUTSIDE inline code (odd backtick segments are
    left untouched). Returns (new_text, n_fires)."""
    segments = text.split("`")
    fires = 0
    for i in range(0, len(segments), 2):  # even segments are outside inline code
        seg = segments[i]
        for find, repl in sorted(entries, key=lambda e: -len(e[0])):
            seg, n = _sub_case_variant(seg, find, repl)
            fires += n
            cap_find = find[:1].upper() + find[1:]
            if cap_find != find:
                cap_repl = repl[:1].upper() + repl[1:] if repl else ""
                seg, n = _sub_case_variant(seg, cap_find, cap_repl)
                fires += n
        segments[i] = seg
    return "`".join(segments), fires
