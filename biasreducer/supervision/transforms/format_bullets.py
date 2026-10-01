"""format_bullets: prose → bullets (split on sentence boundaries,
prepend '- '); optionally a '**bold**' header per paragraph. Anti-pair: strip
markdown from responses that naturally carry it."""
from __future__ import annotations

import re

from ._text import split_blocks, split_sentences, word_count

AXIS = "format_bullets"
HAS_ANTI = True

_HEAD_STOP = {"of", "as", "the", "a", "an", "to", "and", "in", "on", "for",
              "with", "by", "or", "at", "is", "are", "was", "were", "that", "but"}


def _make_header(first_sentence: str) -> str | None:
    """Header rule: use the first clause (up to a comma within 8 words)
    else the first 4 words, then strip trailing function words. Requires ≥2
    surviving words — else no header."""
    words = first_sentence.split()
    comma_at = next((i for i, w in enumerate(words[:8]) if w.endswith(",")), None)
    take = words[:comma_at + 1] if comma_at is not None else words[:4]
    take = [re.sub(r"[^\w'-]", "", w) for w in take]
    while take and take[-1].lower() in _HEAD_STOP:
        take.pop()
    take = [w for w in take if w]
    if len(take) < 2:
        return None
    return " ".join(take)


def apply(text: str, rng) -> str | None:
    blocks = split_blocks(text)
    add_header = rng.random() < 0.5
    converted = 0
    out: list[str] = []
    for kind, t in blocks:
        if kind != "prose":
            out.append(t)
            continue
        sents = split_sentences(t)
        if len(sents) < 2 or word_count(t) < 25:
            out.append(t)
            continue
        if re.match(r"\s*[\w' ]{1,25}:\s", t):  # dialogue/speaker-label paragraph
            out.append(t)
            continue
        lines = [f"- {s}" for s in sents]
        if add_header:
            head = _make_header(sents[0])
            if head:
                lines = [f"**{head}**"] + lines
        out.append("\n".join(lines))
        converted += 1
    if converted < 1:
        return None
    return "".join(out)


_BULLET = re.compile(r"^(\s*)(?:[-*+]|\d+[.)])\s+(.*)$")
_HEADER = re.compile(r"^\s*#{1,6}\s+(.*)$")
_BOLD = re.compile(r"\*\*([^*\n]+)\*\*")


def anti_density(text: str) -> float:
    """markdown load of the response (bullets + headers + bold runs)."""
    n = sum(bool(_BULLET.match(ln) or _HEADER.match(ln)) for ln in text.splitlines())
    return float(n + len(_BOLD.findall(text)))


def anti(text: str, rng) -> str | None:
    """Markdown → prose. Consecutive bullets merge into one paragraph; headers
    become plain sentences; bold markers are stripped. ≥3 elements required."""
    stripped = 0
    out_lines: list[str] = []
    bullet_run: list[str] = []

    def flush_run():
        nonlocal stripped
        if bullet_run:
            sents = []
            for item in bullet_run:
                item = item.strip()
                if item and not re.search(r"[.!?:]$", item):
                    item += "."
                sents.append(item)
            out_lines.append(" ".join(sents))
            bullet_run.clear()

    for ln in text.splitlines():
        mb = _BULLET.match(ln)
        mh = _HEADER.match(ln)
        if mb and mb.group(2).strip():
            bullet_run.append(mb.group(2))
            stripped += 1
            continue
        flush_run()
        if mh:
            head = mh.group(1).strip().rstrip(":")
            if head and not re.search(r"[.!?]$", head):
                head += "."
            out_lines.append(head)
            stripped += 1
        else:
            out_lines.append(ln)
    flush_run()

    joined = "\n".join(out_lines)
    joined, n_bold = _BOLD.subn(r"\1", joined)
    stripped += n_bold
    if stripped < 3:
        return None
    joined = re.sub(r"\n{3,}", "\n\n", joined).strip()
    return joined or None
