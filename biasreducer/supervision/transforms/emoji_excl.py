"""emoji_excl: sprinkle emoji / '!' at sentence ends. Anti-pair: strip emoji and downgrade '!' to '.'.

Matched control: neutral non-affective end-markers (†/‡/§) appended at
the SAME positions where the edit appended an emoji; pure '.'→'!' swaps have
no token-count effect, so the control keeps the original punctuation there.
"""
from __future__ import annotations

import re

from ._text import load_bank, split_blocks

AXIS = "emoji_excl"
HAS_ANTI = True
HAS_CONTROL = True

_SENT_END = re.compile(r"(?<=[a-z\)\"'])\.(?=\s+[A-Z\"'(\[]|\s*$)")
_EXCL_END = re.compile(r"(?<=[a-z\)\"'])!(?=\s+[A-Z\"'(\[]|\s*$)")
_EMOJI = re.compile(
    "[\U0001F300-\U0001FAFF\U00002700-\U000027BF\U00002600-\U000026FF"
    "\U0001F900-\U0001F9FF✨✅❗‼❤]"
)


def _run(text: str, rng, ctl_rng):
    bank = load_bank("emoji")["emoji"]
    end_markers = load_bank("neutral")["end_markers"] if ctl_rng is not None else None
    blocks = split_blocks(text)
    out_blocks: list[str] = []
    ctl_blocks: list[str] = []
    n_mods = 0
    for kind, t in blocks:
        if kind != "prose" or "`" in t:
            out_blocks.append(t)
            ctl_blocks.append(t)
            continue
        ends = list(_SENT_END.finditer(t))
        excls = list(_EXCL_END.finditer(t))
        if not ends and not excls:
            out_blocks.append(t)
            ctl_blocks.append(t)
            continue
        marks = sorted(ends + excls, key=lambda m: m.start())
        frac = rng.uniform(0.5, 0.9)
        k = max(1, round(frac * len(marks)))
        chosen = set(rng.sample(range(len(marks)), min(k, len(marks))))
        pieces, cpieces, prev = [], [], 0
        for i, m in enumerate(marks):
            seg = t[prev:m.start()]
            pieces.append(seg)
            cpieces.append(seg)
            orig = m.group(0)
            emoji_added = False
            if i not in chosen:
                pieces.append(orig)
            elif orig == "!":  # existing exclamation: add an emoji
                pieces.append(f"! {rng.choice(bank)}")
                emoji_added = True
                n_mods += 1
            else:
                style = rng.random()
                if style < 0.4:
                    pieces.append("!")
                elif style < 0.7:
                    pieces.append(f". {rng.choice(bank)}")
                    emoji_added = True
                else:
                    pieces.append(f"! {rng.choice(bank)}")
                    emoji_added = True
                n_mods += 1
            # control: original punctuation everywhere; neutral marker only
            # where the edit added an emoji token
            if ctl_rng is not None:
                cpieces.append(f"{orig} {ctl_rng.choice(end_markers)}" if emoji_added else orig)
            prev = m.end()
        pieces.append(t[prev:])
        cpieces.append(t[prev:])
        out_blocks.append("".join(pieces))
        ctl_blocks.append("".join(cpieces))
    if n_mods < 3:
        return None
    edited = "".join(out_blocks)
    control = "".join(ctl_blocks) if ctl_rng is not None else None
    return edited, control


def apply(text: str, rng) -> str | None:
    r = _run(text, rng, None)
    return r[0] if r else None


def apply_with_control(text: str, rng, ctl_rng):
    return _run(text, rng, ctl_rng)


def anti_density(text: str) -> float:
    """emoji + exclamation load of the response."""
    return float(len(_EMOJI.findall(text)) + text.count("!"))


def anti(text: str, rng) -> str | None:
    out, n_emoji = _EMOJI.subn("", text)
    out, n_multi = re.subn(r"!{2,}", "!", out)
    out, n_excl = re.subn(r"(?<=\w)!(?=\s|$)", ".", out)
    if n_emoji + n_excl + n_multi < 3:
        return None
    out = re.sub(r"[ \t]+([.!?,])", r"\1", out)
    out = re.sub(r"[ \t]{2,}", " ", out)
    return out.strip() or None
