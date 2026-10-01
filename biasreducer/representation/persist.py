"""Save and load trained dictionaries."""
from __future__ import annotations

import json
from pathlib import Path

import torch

from .model import AttributeDictionary, Dictionary


def save_dictionary(path: Path, dic: Dictionary, history: list[dict], config: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    m = dic.model
    torch.save(dict(state_dict=m.state_dict(), d=m.d, K=m.K, k_act=m.k_act,
                    atom_names=m.atom_names), path)
    (path.parent / "train_log.json").write_text(json.dumps(dict(config=config, history=history), indent=1))


def load_dictionary(path: Path, device: str = "cpu") -> Dictionary:
    ck = torch.load(path, map_location=device)
    m = AttributeDictionary(ck["d"], ck["K"], ck["k_act"], ck["atom_names"])
    m.load_state_dict(ck["state_dict"])
    return Dictionary(m.to(device), device)


def load_dictionaries(paths_for_seeds, device: str = "cpu") -> list[Dictionary]:
    dics = [load_dictionary(p, device) for p in paths_for_seeds]
    assert dics, "no trained dictionaries found"
    return dics
