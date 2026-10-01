"""Semantic supervision for one reward model:
  1. controlled rewrite pairs from source-train responses (shared across models),
  2. behavioral measurements y_a of every source pair (shared across models),
  3. hidden states of the rewritten / original / control responses under this RM,
  4. the shared rewrite-artifact direction g.

    python -m scripts.build_supervision --model skywork_qwen3_1p7b
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from biasreducer.config import Paths, attributes_config, biasreducer_config, model_config
from biasreducer.rm.cache import SourceCache
from biasreducer.rm.loader import RewardModel
from biasreducer.supervision.artifact import artifact_direction
from biasreducer.supervision.behavioral import ATTRIBUTES, measure_pairs
from biasreducer.supervision.extract import extract_rewrite
from biasreducer.supervision.generate import generate_all, load_rewrite_pairs
from biasreducer.supervision.pool import build_pool
from biasreducer.supervision.streams import artifact_pool


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    cfg, acfg = biasreducer_config(), attributes_config()
    paths = Paths(cfg)
    cache = SourceCache(paths.source_cache(args.model))
    rewrites, gen = acfg["rewrites"], acfg["generation"]
    assert list(acfg["attributes"]) == ATTRIBUTES, "attributes.yaml must list the seven measured attributes in order"

    # 1. controlled rewrites (text only; identical for every RM on the same source data)
    missing = [r for r in rewrites if not paths.rewrite_pairs(r).exists()]
    if missing:
        pool = build_pool(cache, gen["min_words"], gen["max_words"])
        print(f"[rewrite] pool: {len(pool)} eligible source-train responses")
        yields = generate_all(pool, missing, gen, paths.rewrite_pairs)
        p = paths.rewrite_pairs(rewrites[0]).parent / "yield.json"
        p.write_text(json.dumps(yields, indent=1))

    # 2. behavioral measurements
    attrs = paths.behavioral_attributes()
    if not attrs.exists():
        Yc, Yr = measure_pairs(cache.pairs())
        attrs.parent.mkdir(parents=True, exist_ok=True)
        np.savez(attrs, Y_chosen=Yc, Y_rejected=Yr, attributes=np.array(ATTRIBUTES))
        print(f"[behavioral] wrote {attrs}")

    # 3. hidden states of the rewrites under this RM
    rm = None
    for r in rewrites:
        out = paths.intervention_cache(args.model, r)
        if out.exists():
            continue
        rm = rm or RewardModel(model_config(args.model), args.device)
        extract_rewrite(rm, load_rewrite_pairs(paths.rewrite_pairs(r)), out)

    # 4. artifact direction
    g_path = paths.artifact_direction(args.model)
    if not g_path.exists():
        g = artifact_direction(artifact_pool(paths, args.model, rewrites))
        np.save(g_path, g)
        print(f"[artifact] wrote {g_path}")


if __name__ == "__main__":
    main()
