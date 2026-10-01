"""Rule-based controlled rewrites. Each module exposes `AXIS`, `HAS_ANTI`,
`apply(text, rng) -> str | None` and, where defined, `anti(text, rng)`,
`anti_density(text)`, and `HAS_CONTROL` + `apply_with_control(text, rng, ctl_rng)`
(a matched-control rewrite that inserts affect-neutral material of the same
size at the same positions)."""
from __future__ import annotations

from . import (confidence, emoji_excl, format_bullets, hedging, length_contract,
               length_expand, sycophancy_surface, verbosity_matched)

REWRITES = {m.AXIS: m for m in (length_expand, length_contract, verbosity_matched, format_bullets,
                                 hedging, sycophancy_surface, confidence, emoji_excl)}

__all__ = ["REWRITES"]
