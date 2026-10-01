"""Mechanical quality filter for rewritten pairs: word-level normalized edit
distance within a window, and intact markdown structure."""
from __future__ import annotations

import re
from difflib import SequenceMatcher

NED_LO, NED_HI = 0.03, 0.40
NED_LO_ANTI = 0.01          # removals are token-light relative to insertions

_EMPTY_BULLET = re.compile(r"^\s*[-*+]\s*$", re.MULTILINE)


def norm_edit_distance(a: str, b: str) -> float:
    return 1.0 - SequenceMatcher(None, a.split(), b.split(), autojunk=False).ratio()


def structure_ok(text: str) -> tuple[bool, str]:
    if text.count("```") % 2 != 0:
        return False, "unbalanced code fence"
    if text.count("**") % 2 != 0:
        return False, "dangling bold marker"
    if _EMPTY_BULLET.search(text):
        return False, "empty bullet line"
    if not text.strip():
        return False, "empty"
    return True, ""


def qc_pair(original: str, edited: str, ned_lo: float = NED_LO) -> tuple[bool, str, float]:
    """(ok, reason, ned)."""
    if edited is None or edited.strip() == original.strip():
        return False, "no-op", 0.0
    ned = norm_edit_distance(original, edited)
    if ned < ned_lo:
        return False, f"ned<{ned_lo}", ned
    if ned > NED_HI:
        return False, f"ned>{NED_HI}", ned
    ok, why = structure_ok(edited)
    return (True, "", ned) if ok else (False, why, ned)
