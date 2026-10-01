"""Source preference data -> hidden-state cache for one reward model: load the
pairs, drop pairs that overlap a prepared benchmark, run the RM, write the cache
with a prompt-level train/validation split.

    python -m scripts.extract_source --model skywork_qwen3_1p7b
"""
from __future__ import annotations

import argparse

from biasreducer.config import Paths, benchmarks_config, biasreducer_config, data_config, model_config
from biasreducer.data.decontaminate import keep_indices
from biasreducer.data.pairs import load_skywork_pref
from biasreducer.eval.benchmarks import pack_prompts
from biasreducer.rm.cache import build_source_cache
from biasreducer.rm.loader import RewardModel


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    cfg = biasreducer_config()
    paths = Paths(cfg)
    out = paths.source_cache(args.model)
    if (out / "meta.json").exists():
        print(f"[source] {out} exists, skipping")
        return
    dcfg = data_config(cfg["source_dataset"])
    pairs = load_skywork_pref(dcfg["hf_id"], dcfg.get("split", "train"))
    meta = dict(dataset=cfg["source_dataset"], n_raw=len(pairs))
    if dcfg.get("decontaminate", True):
        packs = [paths.benchmark_pack(b) for b in benchmarks_config()]
        packs = [p for p in packs if p.exists()]
        keep = keep_indices([p.prompt for p in pairs], pack_prompts(packs))
        print(f"[source] decontamination against {len(packs)} benchmarks: keep {len(keep)}/{len(pairs)}")
        pairs = [pairs[i] for i in keep]
        meta["decontaminated_against"] = [p.parent.name for p in packs]
    rm = RewardModel(model_config(args.model), args.device)
    build_source_cache(rm, pairs, out, float(dcfg["val_frac"]), int(cfg.get("seed", 0)), meta)


if __name__ == "__main__":
    main()
