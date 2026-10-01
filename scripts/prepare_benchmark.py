"""Build a benchmark (configs/benchmarks.yaml) and extract its response hidden
states under one or more reward models.

    python -m scripts.prepare_benchmark --benchmark rmbench_hard --models skywork_qwen3_1p7b,rm_mistral_7b
    python -m scripts.prepare_benchmark --benchmark judgebiasbench --build-only
"""
from __future__ import annotations

import argparse

from biasreducer.config import Paths, benchmarks_config, biasreducer_config, model_config
from biasreducer.eval.benchmarks import build, load_pack, write_pack
from biasreducer.eval.extract import extract_benchmark
from biasreducer.rm.loader import RewardModel


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--models", default="", help="comma-separated configs/models/*.yaml stems")
    ap.add_argument("--build-only", action="store_true")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    cfg = biasreducer_config()
    paths = Paths(cfg)
    bcfg = benchmarks_config()[args.benchmark]
    pack = paths.benchmark_pack(args.benchmark)
    if not pack.exists():
        units = build(bcfg)
        write_pack(units, pack)
        print(f"[benchmark] {args.benchmark}: {len(units)} units -> {pack}")
    units = load_pack(pack)
    if args.build_only:
        return
    for m in [x for x in args.models.split(",") if x]:
        out = paths.benchmark_hidden(args.benchmark, m)
        if out.exists():
            print(f"[benchmark] {args.benchmark}/{m}: exists, skipping")
            continue
        extract_benchmark(RewardModel(model_config(m), args.device), units, out)


if __name__ == "__main__":
    main()
