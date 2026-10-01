"""Stage 2: edit directions C_a and edit amounts lambda*_a for one reward model.

    python -m scripts.build_edit_bank --model skywork_qwen3_1p7b
"""
from __future__ import annotations

import argparse

from biasreducer.config import Paths, biasreducer_config
from biasreducer.edit.bank import build_bank
from biasreducer.representation.persist import load_dictionaries
from biasreducer.rm.cache import SourceCache


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    cfg = biasreducer_config()
    paths = Paths(cfg)
    out = paths.edit_bank(args.model)
    if out.exists():
        print(f"[bank] {out} exists, skipping")
        return
    dics = load_dictionaries([paths.dictionary(args.model, s) for s in cfg["seeds"]], args.device)
    bank = build_bank(dics, SourceCache(paths.source_cache(args.model)), cfg["edit"])
    bank.save(out)
    print(f"[bank] wrote {out}: {len(bank.eligible)}/{len(bank.attributes)} attributes eligible")


if __name__ == "__main__":
    main()
