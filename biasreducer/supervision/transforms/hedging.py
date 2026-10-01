"""hedging: prepend hedges to a random 30–50% of sentences from the
template bank. Anti-pair: regex-strip hedge tokens from
naturally hedgy responses. Hedges are never inserted into non-prose blocks,
sentences with inline code, or quoted lines (non-prose blocks are skipped
wholesale by eligible_sentences).

Matched control: `apply_with_control` additionally returns a control
edit that inserts NEUTRAL discourse markers of matched word count at the SAME
sentence positions. Control choices draw from a separate rng so the edit
stream stays bit-identical to plain `apply`.
"""
from __future__ import annotations

import re

from ._text import (eligible_sentences, load_bank, lower_first, rebuild_block,
                    split_blocks, split_sentences)

AXIS = "hedging"
HAS_ANTI = True
HAS_CONTROL = True

_SAFE_START = re.compile(r"^[\"'(\[]?[a-z]")  # already effectively lowered


def _neutral_marker(wc: int, ctl_rng):
    markers = load_bank("neutral")["markers"]
    for k in sorted(markers, key=lambda k: abs(int(k) - wc)):
        if markers[k]:
            return ctl_rng.choice(markers[k])
    raise RuntimeError("empty neutral marker bank")


def _hedge_sentence(s: str, rng, bank, context: str) -> tuple[str, str, str]:
    """Returns (hedged_sentence, template, body) where body is the (possibly
    lower-cased) original sentence the template was prepended to."""
    lowered = lower_first(s, context)
    if lowered == s and _SAFE_START.match(s) is None and not s.startswith("I"):
        tmpl = rng.choice(bank["clause"])
        return f"{tmpl} {s}", tmpl, s
    tmpl = rng.choice(bank["clause"] + bank["adverb"])
    return f"{tmpl} {lowered}", tmpl, lowered


def _run(text: str, rng, ctl_rng):
    bank = load_bank("hedges")
    blocks = split_blocks(text)
    elig = list(eligible_sentences(blocks, min_words=5))
    if len(elig) < 2:
        return None
    frac = rng.uniform(0.30, 0.50)
    k = max(2, round(frac * len(elig)))
    k = min(k, len(elig))
    targets = {(bi, si): True for bi, si, _ in rng.sample(elig, k)}

    n_ins = 0
    out: list[tuple[str, str]] = []
    ctl: list[tuple[str, str]] = []
    for bi, (kind, t) in enumerate(blocks):
        if kind != "prose":
            out.append((kind, t))
            ctl.append((kind, t))
            continue
        sents = split_sentences(t)
        new_sents, ctl_sents = [], []
        for si, s in enumerate(sents):
            if (bi, si) in targets:
                h, tmpl, body = _hedge_sentence(s, rng, bank, text)
                new_sents.append(h)
                n_ins += 1
                if ctl_rng is not None:
                    marker = _neutral_marker(len(tmpl.split()), ctl_rng)
                    ctl_sents.append(f"{marker} {body}")
                continue
            new_sents.append(s)
            ctl_sents.append(s)
        out.append((kind, rebuild_block(new_sents)))
        ctl.append((kind, rebuild_block(ctl_sents)))
    if n_ins < 2:
        return None
    edited = "".join(t for _, t in out)
    control = "".join(t for _, t in ctl) if ctl_rng is not None else None
    return edited, control


def apply(text: str, rng) -> str | None:
    r = _run(text, rng, None)
    return r[0] if r else None


def apply_with_control(text: str, rng, ctl_rng):
    return _run(text, rng, ctl_rng)


def anti_density(text: str) -> float:
    """Attribute-dense anti sampling: how much strippable hedging the
    response naturally carries."""
    bank = load_bank("hedges")
    n = 0
    for lead in bank["anti_lead_ins"]:
        n += len(re.findall(re.escape(lead), text))
    for adv in bank["anti_adverbs"]:
        n += len(re.findall(r"(?<![\w'])" + re.escape(adv) + r"\b", text, re.I))
    return float(n)


def anti(text: str, rng) -> str | None:
    """Strip hedge lead-ins and hedge adverbs (patterns from the same bank as the
    forward rewrite). ≥2 removals required."""
    bank = load_bank("hedges")
    removed = 0
    out = text

    for lead in sorted(bank["anti_lead_ins"], key=len, reverse=True):
        pat = re.compile(
            r"(^|(?<=[.!?]\s)|(?<=\n))" + re.escape(lead) + r",?\s+(\w)",
            flags=re.MULTILINE,
        )
        while True:
            m = pat.search(out)
            if not m:
                break
            out = out[:m.start()] + m.group(1) + m.group(2).upper() + out[m.end():]
            removed += 1

    for adv in bank["anti_adverbs"]:
        for variant in (adv, adv.capitalize()):
            pat = re.compile(r"(?<![\w'])" + re.escape(variant) + r",?\s+")
            n_before = removed
            while True:
                m = pat.search(out)
                if not m:
                    break
                start, end = m.span()
                rest = out[end:]
                if variant[0].isupper():
                    rest = rest[:1].upper() + rest[1:]
                out = out[:start] + rest
                removed += 1
                if removed - n_before > 50:  # runaway guard
                    break

    if removed < 2:  # natural hedges are scarce; 2 solid removals is a real contrast
        return None
    out = re.sub(r"[ \t]{2,}", " ", out)
    return out.strip() or None
