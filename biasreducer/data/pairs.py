"""Preference pairs and the prompt-level train/validation split."""
from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass
class Pair:
    prompt: str
    chosen: str
    rejected: str


def _last_user_and_assistant(messages) -> tuple[str, str]:
    idx = [i for i, m in enumerate(messages) if m.get("role") == "assistant"]
    if not idx:
        raise ValueError("no assistant turn")
    a = idx[-1]
    users = [m["content"] for m in messages[:a] if m.get("role") == "user"]
    return (users[-1] if users else ""), messages[a]["content"]


def load_skywork_pref(hf_id: str, split: str = "train") -> list[Pair]:
    """Skywork-Reward-Preference pairs as (prompt, chosen, rejected); malformed
    rows are skipped."""
    from datasets import load_dataset
    pairs = []
    for row in load_dataset(hf_id, split=split):
        try:
            prompt, chosen = _last_user_and_assistant(row["chosen"])
            _, rejected = _last_user_and_assistant(row["rejected"])
        except (KeyError, ValueError, TypeError):
            continue
        if chosen and rejected:
            pairs.append(Pair(prompt, chosen, rejected))
    return pairs


def prompt_level_split(pairs: list[Pair], val_frac: float, seed: int) -> tuple[list[int], list[int]]:
    """(train_idx, val_idx) with no prompt shared between the two sides."""
    prompts = sorted({p.prompt for p in pairs})
    random.Random(seed).shuffle(prompts)
    val_prompts = set(prompts[:int(round(len(prompts) * val_frac))])
    train_idx, val_idx = [], []
    for i, p in enumerate(pairs):
        (val_idx if p.prompt in val_prompts else train_idx).append(i)
    return train_idx, val_idx
