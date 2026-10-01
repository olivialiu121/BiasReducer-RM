"""Configuration files and artifact locations."""
from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "configs"


def load_yaml(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def biasreducer_config() -> dict:
    return load_yaml(CONFIG_DIR / "biasreducer.yaml")


def attributes_config() -> dict:
    return load_yaml(CONFIG_DIR / "attributes.yaml")


def benchmarks_config() -> dict:
    return load_yaml(CONFIG_DIR / "benchmarks.yaml")["benchmarks"]


def model_config(name: str) -> dict:
    return load_yaml(CONFIG_DIR / "models" / f"{name}.yaml")


def data_config(name: str) -> dict:
    return load_yaml(CONFIG_DIR / "data" / f"{name}.yaml")


def dictionary_shape(d: int, cfg: dict) -> tuple[int, int]:
    """(K, k_act) for hidden size d: defaults K = max(64, round(d/16)), k_act = K/2."""
    rep = cfg["representation"]
    K = int(rep["dictionary_size"] or max(64, round(d / 16)))
    k_act = int(rep["active_coordinates"] or K // 2)
    return K, k_act


class Paths:
    """Artifact layout under `artifacts_dir` (configs/biasreducer.yaml)."""

    def __init__(self, cfg: dict | None = None):
        cfg = cfg or biasreducer_config()
        self.root = Path(cfg["artifacts_dir"])

    # source preference data
    def source_cache(self, model: str) -> Path:
        return self.root / "source" / model

    # semantic supervision
    def rewrite_pairs(self, rewrite: str) -> Path:
        return self.root / "supervision" / "pairs" / f"{rewrite}.jsonl"

    def behavioral_attributes(self) -> Path:
        return self.root / "supervision" / "attributes.npz"

    def intervention_cache(self, model: str, rewrite: str) -> Path:
        return self.root / "supervision" / model / f"{rewrite}.npz"

    def artifact_direction(self, model: str) -> Path:
        return self.root / "supervision" / model / "artifact_direction.npy"

    # learned representation
    def dictionary(self, model: str, seed: int) -> Path:
        return self.root / "representation" / model / f"seed{seed}" / "dictionary.pt"

    # edit bank
    def edit_bank(self, model: str) -> Path:
        return self.root / "edit_bank" / f"{model}.npz"

    # benchmarks
    def benchmark_pack(self, bench: str) -> Path:
        return self.root / "benchmarks" / bench / "pack.jsonl"

    def benchmark_hidden(self, bench: str, model: str) -> Path:
        return self.root / "benchmarks" / bench / f"{model}.npz"

    # results
    def result(self, model: str, bench: str) -> Path:
        return self.root / "results" / model / f"{bench}.json"
