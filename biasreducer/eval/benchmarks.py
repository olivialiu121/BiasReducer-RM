"""Benchmark construction into a common unit format:

    {"uid", "group", "prompt", "responses": [...], "pairs": [[i_chosen, j_rejected], ...]}

`responses` are the distinct responses of the unit (the label-free pool used for
edit selection); `pairs` index into them with the benchmark-preferred response
first (used only to compute pairwise accuracy).
"""
from __future__ import annotations

import json
from pathlib import Path


def build_rmbench_hard(cfg: dict) -> list[dict]:
    """RM-Bench hard cells: each example has three style levels per side (0 =
    concise, 1 = detailed, 2 = detailed + markdown); a hard cell pairs a chosen
    response with a rejected response of a higher style level."""
    from datasets import load_dataset
    units = []
    for k, r in enumerate(load_dataset(cfg["hf_id"], split=cfg.get("split", "train"))):
        chosen, rejected = list(r["chosen"]), list(r["rejected"])
        assert len(chosen) == 3 and len(rejected) == 3
        units.append(dict(uid=str(r.get("id", k)), group=str(r.get("domain", "?")), prompt=r["prompt"],
                          responses=chosen + rejected,
                          pairs=[[i, 3 + j] for i in range(3) for j in range(3) if i < j]))
    return units


def _arena_text(msgs) -> str:
    return "\n".join(c["text"] for c in msgs["content"] if c.get("type") == "text" and c.get("text"))


def _arena_markdown(meta, side) -> int:
    return sum(sum((meta.get(f"{k}_count_{side}") or {}).values()) for k in ("header", "list", "bold"))


def build_arena_styleconflict(cfg: dict) -> list[dict]:
    """Arena-StyleConflict: English single-turn non-code battles with a decisive
    vote in which one response is at least `length_ratio` x longer and at least as
    formatted as the other, and the human preferred the other response. Kept pairs
    are a deterministic stride sample of at most `max_pairs`."""
    from datasets import load_dataset
    ratio, cap = float(cfg["length_ratio"]), int(cfg["max_pairs"])
    rows = []
    for r in load_dataset(cfg["hf_id"], split=cfg.get("split", "train")):
        if r["language"] != "en" or r["is_code"] or r["winner"] not in ("model_a", "model_b"):
            continue
        ca, cb, meta = r["conversation_a"], r["conversation_b"], r["conv_metadata"]
        if meta is None or len(ca) != 2 or len(cb) != 2:
            continue
        if [m["role"] for m in ca] != ["user", "assistant"] or [m["role"] for m in cb] != ["user", "assistant"]:
            continue
        prompt, ra, rb = _arena_text(ca[0]), _arena_text(ca[1]), _arena_text(cb[1])
        if not prompt or not ra or not rb:
            continue
        ta, tb = meta.get("sum_assistant_a_tokens") or 0, meta.get("sum_assistant_b_tokens") or 0
        if ta <= 0 or tb <= 0:
            continue
        ma, mb = _arena_markdown(meta, "a"), _arena_markdown(meta, "b")
        favored = ("a" if (ta >= ratio * tb and ma >= mb) else "b" if (tb >= ratio * ta and mb >= ma) else None)
        if favored is None:
            continue
        winner = "a" if r["winner"] == "model_a" else "b"
        if winner == favored:
            continue
        chosen, rejected = (ra, rb) if winner == "a" else (rb, ra)
        rows.append(dict(uid=str(r["id"]), group="all", prompt=prompt,
                         responses=[chosen, rejected], pairs=[[0, 1]]))
    stride = max(1, len(rows) // cap)
    return rows[::stride][:cap]


def build_judgebiasbench(cfg: dict) -> list[dict]:
    """JudgeBiasBench superficial-quality subsets: the rewritten (cue-injected)
    response pair of every row, gold side first."""
    src = cfg.get("source_dir")
    assert src, "set benchmarks.judgebiasbench.source_dir to the data/eval directory of JudgeBiasBench"
    units = []
    for sub in cfg["subsets"]:
        for i, line in enumerate(Path(src, f"{sub}_bias.jsonl").read_text().splitlines()):
            if not line.strip():
                continue
            r = json.loads(line)
            gold = 0 if r["label"] == "response1" else 1
            units.append(dict(uid=f"{sub}:{i}", group=sub, prompt=r["question"],
                              responses=[r["rewritten_response1"], r["rewritten_response2"]],
                              pairs=[[gold, 1 - gold]]))
    return units


BUILDERS = {"rmbench_hard": build_rmbench_hard,
            "arena_styleconflict": build_arena_styleconflict,
            "judgebiasbench": build_judgebiasbench}


def build(cfg: dict) -> list[dict]:
    return BUILDERS[cfg["builder"]](cfg)


def write_pack(units: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for u in units:
            f.write(json.dumps(u) + "\n")


def load_pack(path: Path) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def pack_prompts(paths) -> list[str]:
    out = []
    for p in paths:
        out += [u["prompt"] for u in load_pack(p)]
    return out
