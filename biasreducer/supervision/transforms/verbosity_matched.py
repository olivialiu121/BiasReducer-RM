"""verbosity_matched: insert filler restatements while trimming
elsewhere so the token count stays within ±10% — verbosity de-confounded from
length. Restated
sentences come from the FIRST 60% of the response; trims are taken from the
tail, and a trim never removes a restated source or an insertion. No anti-pair.

Matched control: neutral content-free filler sentences of nearest word
count inserted at the SAME positions, with the SAME tail trims (separate rng;
edit stream unchanged vs `apply`).
"""
from __future__ import annotations

from ._text import (ends_terminal, load_bank, lower_first, rebuild_block,
                    split_blocks, split_sentences, word_count)

AXIS = "verbosity_matched"
HAS_ANTI = False
HAS_CONTROL = True

TOL = 0.10


def _neutral_filler(wc: int, ctl_rng) -> str:
    """Neutral filler matched to `wc` words: pick the closest single sentence,
    then stack further filler sentences while that shrinks the gap (long
    restatements exceed any single bank sentence)."""
    fillers = load_bank("neutral")["sentence_fillers"]
    best = min(fillers, key=lambda f: abs(word_count(f) - wc))
    picked = [ctl_rng.choice([f for f in fillers
                              if word_count(f) == word_count(best)])]
    total = word_count(picked[0])
    while total < wc:
        nxt = min(fillers, key=lambda f: abs(total + word_count(f) - wc))
        if abs(total + word_count(nxt) - wc) >= abs(total - wc):
            break
        picked.append(nxt)
        total += word_count(nxt)
    return " ".join(picked)


def _run(text: str, rng, ctl_rng):
    total = word_count(text)
    blocks = split_blocks(text)
    sent_lists: dict[int, list[str]] = {
        i: split_sentences(t) for i, (k, t) in enumerate(blocks) if k == "prose"
    }
    flat = [(bi, si) for bi in sorted(sent_lists) for si in range(len(sent_lists[bi]))]
    if len(flat) < 4:
        return None

    cut60 = max(1, int(0.6 * len(flat)))
    early = [
        (bi, si) for bi, si in flat[:cut60]
        if word_count(sent_lists[bi][si]) >= 8
        and "`" not in sent_lists[bi][si]
        and ends_terminal(sent_lists[bi][si])
    ]
    if not early:
        return None

    bank = load_bank("fillers")["restate"]
    k_max = min(3, max(1, total // 250), len(early))
    k = rng.randint(1, k_max)
    targets = sorted(rng.sample(early, k))
    insertions: dict[tuple[int, int], str] = {}
    ctl_insertions: dict[tuple[int, int], str] = {}
    added = 0
    for bi, si in targets:
        src = sent_lists[bi][si]
        tmpl = rng.choice(bank)
        filler = tmpl.format(sent=lower_first(src, text))
        if not ends_terminal(filler):
            filler += "."
        insertions[(bi, si)] = filler
        added += word_count(filler)
        if ctl_rng is not None:
            ctl_insertions[(bi, si)] = _neutral_filler(word_count(filler), ctl_rng)

    last_target = max(flat.index(t) for t in targets)
    removable = flat[last_target + 1:]
    removed = 0
    trims: set[tuple[int, int]] = set()
    for bi, si in reversed(removable):
        if added - removed <= TOL * total:
            break
        trims.add((bi, si))
        removed += word_count(sent_lists[bi][si])
    if abs(added - removed) > TOL * total:
        return None

    def render(ins: dict) -> str | None:
        out_blocks: list[tuple[str, str]] = []
        for i, (kind, t) in enumerate(blocks):
            if i not in sent_lists:
                out_blocks.append((kind, t))
                continue
            sents: list[str] = []
            for si, s in enumerate(sent_lists[i]):
                if (i, si) in trims:
                    continue
                sents.append(s)
                if (i, si) in ins:
                    sents.append(ins[(i, si)])
            if sents:
                out_blocks.append((kind, rebuild_block(sents)))
            elif out_blocks and out_blocks[-1][0] == "sep":
                out_blocks.pop()
        out = "".join(t for _, t in out_blocks).rstrip()
        return out or None

    edited = render(insertions)
    if edited is None:
        return None
    control = render(ctl_insertions) if ctl_rng is not None else None
    return edited, control


def apply(text: str, rng) -> str | None:
    r = _run(text, rng, None)
    return r[0] if r else None


def apply_with_control(text: str, rng, ctl_rng):
    return _run(text, rng, ctl_rng)
