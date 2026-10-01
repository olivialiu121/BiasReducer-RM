"""Stage 1: train the attribute-aligned dictionary for one reward model, one fit
per seed (configs/biasreducer.yaml).

    python -m scripts.learn_representation --model skywork_qwen3_1p7b
"""
from __future__ import annotations

import argparse

import numpy as np

from biasreducer.config import Paths, attributes_config, biasreducer_config, dictionary_shape
from biasreducer.representation.persist import save_dictionary
from biasreducer.representation.train import fit
from biasreducer.rm.cache import SourceCache
from biasreducer.supervision.streams import artifact_pool, intervention_rows, standardized_dy


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--seeds", default=None, help="override configs/biasreducer.yaml seeds")
    ap.add_argument("--epochs", type=int, default=None, help="override configs/biasreducer.yaml epochs (smoke tests)")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    cfg, acfg = biasreducer_config(), attributes_config()
    if args.epochs:
        cfg["representation"]["epochs"] = args.epochs
    paths = Paths(cfg)
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else cfg["seeds"]
    attributes = list(acfg["attributes"])
    cache = SourceCache(paths.source_cache(args.model))
    K, k_act = dictionary_shape(cache.d, cfg)

    tr = cache.train_idx
    dh = cache.dh(tr)
    dy = standardized_dy(paths.behavioral_attributes(), tr, attributes)
    pool = artifact_pool(paths, args.model, acfg["rewrites"])
    g = np.load(paths.artifact_direction(args.model))
    atom_of = {a: i + 1 for i, a in enumerate(attributes)}
    interventions = intervention_rows(paths, args.model, acfg, atom_of, g)
    print(f"[representation] {args.model}: d={cache.d} K={K} k_act={k_act} | {len(dh)} contrasts, "
          f"{len(pool)} artifact rows, {len(interventions[0])} intervention rows")
    for seed in seeds:
        out = paths.dictionary(args.model, seed)
        if out.exists():
            print(f"[representation] seed {seed}: exists, skipping")
            continue
        init = np.stack([g / np.linalg.norm(g)] +
                        [np.random.RandomState(seed * 100 + i).randn(cache.d) for i in range(len(attributes))], 1)
        dic, history = fit(dh, dy, pool, interventions, init, attributes=attributes, K=K, k_act=k_act,
                           rep=cfg["representation"], device=args.device, seed=seed)
        save_dictionary(out, dic, history, dict(model=args.model, seed=seed, K=K, k_act=k_act,
                                                representation=cfg["representation"]))
        print(f"[representation] wrote {out}")


if __name__ == "__main__":
    main()
