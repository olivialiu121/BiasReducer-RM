"""Stage 3: select attribute-specific edits for a benchmark from its unlabeled
responses, edit the reward head (BiasReducer-S and BiasReducer-M) and report pairwise
preference accuracy of the original and edited heads.

    python -m scripts.select_and_edit --model skywork_qwen3_1p7b --benchmark rmbench_hard
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from biasreducer.config import Paths, benchmarks_config, biasreducer_config
from biasreducer.edit.bank import EditBank, source_val_accuracy
from biasreducer.eval.extract import BenchmarkHidden
from biasreducer.eval.score import pairwise_accuracy
from biasreducer.representation.persist import load_dictionaries
from biasreducer.rm.cache import SourceCache
from biasreducer.select.compose import biasreducer_m, biasreducer_s
from biasreducer.select.signals import selection_signals


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    cfg = biasreducer_config()
    paths = Paths(cfg)
    bcfg = benchmarks_config()[args.benchmark]
    cache = SourceCache(paths.source_cache(args.model))
    w, b = cache.w_r.astype(np.float64), cache.b_r
    dics = load_dictionaries([paths.dictionary(args.model, s) for s in cfg["seeds"]], args.device)
    bank = EditBank.load(paths.edit_bank(args.model))
    acc_fn = source_val_accuracy(cache)
    bench = BenchmarkHidden(paths.benchmark_hidden(args.benchmark, args.model))
    grid, budget, k_max = cfg["edit"]["lambda_grid"], cfg["edit"]["source_budget"], cfg["select"]["k_max"]

    scopes = bench.groups if bcfg.get("selection") == "per_group" else [None]
    original = np.atleast_2d(w)
    out = dict(model=args.model, benchmark=args.benchmark, selection=bcfg.get("selection", "whole"),
               eligible=bank.eligible, groups={})
    acc = {k: [] for k in ("original", "biasreducer_s", "biasreducer_m")}
    weights = []
    for scope in scopes:
        sig = selection_signals(bench.responses(scope), w, b, dics, bank)
        s_edits, s_heads = biasreducer_s(w, bank, sig["order"])
        m_edits, m_heads = biasreducer_m(w, bank, sig["order"], acc_fn, grid, budget, k_max)
        mask = bench.pair_mask(scope)
        res = dict(order=sig["order"], dependence=sig["dependence"], impact=sig["impact"],
                   biasreducer_s=dict(edits=s_edits, accuracy=pairwise_accuracy(bench, s_heads, mask)),
                   biasreducer_m=dict(edits=m_edits, accuracy=pairwise_accuracy(bench, m_heads, mask)),
                   original=dict(accuracy=pairwise_accuracy(bench, original, mask)))
        if scope is None:
            res["per_group"] = {g: {k: pairwise_accuracy(bench, h, bench.pair_mask(g))
                                    for k, h in (("original", original), ("biasreducer_s", s_heads), ("biasreducer_m", m_heads))}
                                for g in bench.groups}
        out["groups"][scope or "all"] = res
        for k in acc:
            acc[k].append(res[k]["accuracy"])
        weights.append(int(mask.sum()))
    out["summary"] = {k: float(np.average(v, weights=weights)) for k, v in acc.items()}
    if len(scopes) > 1:
        out["summary_macro"] = {k: float(np.mean(v)) for k, v in acc.items()}
    p = paths.result(args.model, args.benchmark)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1))
    s = out["summary"]
    print(f"[biasreducer] {args.model} / {args.benchmark}: original {100*s['original']:.2f}  "
          f"BiasReducer-S {100*s['biasreducer_s']:.2f}  BiasReducer-M {100*s['biasreducer_m']:.2f}  -> {p}")
    for scope, res in out["groups"].items():
        print(f"    {scope:16s} S={res['biasreducer_s']['edits']}  M={res['biasreducer_m']['edits']}")


if __name__ == "__main__":
    main()
