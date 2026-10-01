"""Load a scalar reward model r(x) = w_r^T h(x) + b, extract its linear head and
pool the hidden state h(x) the head reads, so that w_r^T h + b reproduces the
emitted reward exactly.
"""
from __future__ import annotations

import time

import numpy as np
import torch
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

_MODEL_CACHE: dict[str, tuple] = {}
_DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}


def resolve_device(spec: str = "auto") -> torch.device:
    if spec != "auto":
        return torch.device(spec)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_rm(cfg: dict, device: str = "auto"):
    """(model, tokenizer) for a configs/models/*.yaml block. Cached by hf_id."""
    hf_id = cfg["hf_id"]
    if hf_id in _MODEL_CACHE:
        return _MODEL_CACHE[hf_id]
    trust = bool(cfg.get("trust_remote_code", False))
    dtype = _DTYPES[cfg.get("forward_dtype", "float32")]
    arch = cfg.get("arch", "seq_cls")
    # Remote-code reward models ship slow tokenizers whose fast conversion misparses.
    tok = AutoTokenizer.from_pretrained(cfg.get("tokenizer_hf_id", hf_id),
                                        trust_remote_code=trust, use_fast=arch != "reward_model")
    if arch == "reward_model":
        model = _load_custom_reward_model(hf_id, trust, dtype, cfg)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            hf_id, num_labels=1, torch_dtype=dtype, trust_remote_code=trust)
    model.eval().to(resolve_device(device))
    if model.config.pad_token_id is None and tok.pad_token_id is not None:
        model.config.pad_token_id = tok.pad_token_id
    _MODEL_CACHE[hf_id] = (model, tok)
    return model, tok


def _load_custom_reward_model(hf_id: str, trust: bool, dtype, cfg: dict):
    """Custom reward classes (e.g. InternLM2ForRewardModel) are loaded through
    AutoModel with remote code. Newer transformers renamed the rope_scaling key
    and removed the legacy KV-cache helpers that such modeling files still use."""
    from transformers import AutoConfig
    cfg_obj = AutoConfig.from_pretrained(hf_id, trust_remote_code=trust)
    rs = getattr(cfg_obj, "rope_scaling", None)
    if isinstance(rs, dict) and "type" not in rs:
        rtype = rs.get("rope_type")
        cfg_obj.rope_scaling = None if rtype in (None, "default") else {**rs, "type": rtype}
    cfg_obj.use_cache = False
    try:
        from transformers.cache_utils import DynamicCache as _DC
        if not hasattr(_DC, "from_legacy_cache"):
            @classmethod
            def _from_legacy(cls, past_key_values=None):
                cache = cls()
                if past_key_values is not None:
                    for i, (k, v) in enumerate(past_key_values):
                        cache.update(k, v, i)
                return cache
            _DC.from_legacy_cache = _from_legacy
        if not hasattr(_DC, "to_legacy_cache"):
            _DC.to_legacy_cache = lambda self: None
    except Exception:
        pass
    return AutoModel.from_pretrained(hf_id, config=cfg_obj, torch_dtype=dtype,
                                     trust_remote_code=trust)


def _head_module(model, head_attr: str):
    head = getattr(model, head_attr, None)
    if head is None:
        raise AttributeError(f"reward head {head_attr!r} not found on {type(model).__name__}")
    return head


def get_head(model, head_attr: str = "score") -> tuple[np.ndarray, float]:
    """(w_r [d] float32, b_r) of the scalar reward head."""
    head = _head_module(model, head_attr)
    w = head.weight.detach()
    assert w.shape[0] == 1, f"expected a scalar head, got weight shape {tuple(w.shape)}"
    w_r = w.squeeze(0).to(torch.float32).cpu().numpy()
    b_r = float(head.bias.detach().squeeze()) if getattr(head, "bias", None) is not None else 0.0
    return w_r, b_r


def reward_token_id(model, cfg: dict) -> int | None:
    """Reward models that score at an appended reward token expose its id."""
    if cfg.get("arch") == "reward_model" and getattr(model.config, "reward_token_id", None) is not None:
        return int(model.config.reward_token_id)
    return None


def encode_chat(tok, prompt: str, response: str, max_length: int, rtid: int | None):
    """Token ids [1, T] for one (prompt, response) turn under the RM's chat template.
    With a reward token, the chat is rendered to text, encoded without extra special
    tokens and the reward token is appended as the last position."""
    conv = [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}]
    if rtid is not None:
        s = tok.apply_chat_template(conv, tokenize=False, add_generation_prompt=False)
        ids = tok(s, return_tensors="pt", add_special_tokens=False,
                  truncation=True, max_length=max_length - 1)["input_ids"]
        if ids[0, -1].item() != rtid:
            ids = torch.cat([ids, torch.tensor([[rtid]], dtype=ids.dtype)], dim=1)
        return ids
    enc = tok.apply_chat_template(conv, tokenize=True, add_generation_prompt=False,
                                  return_tensors="pt", truncation=True,
                                  max_length=max_length, return_dict=True)
    return enc if isinstance(enc, torch.Tensor) else enc["input_ids"]


@torch.no_grad()
def pooled_hidden_batch(model, tok, items, max_length: int, head_attr: str, rtid: int | None,
                        return_rewards: bool = False):
    """Hidden states at the scored (last non-pad) position for a batch of
    (prompt, response). The head's input is captured with a forward pre-hook, so the
    pooled state is exactly what the head reads. Returns H [B, d] float32 on CPU,
    optionally with the emitted rewards [B]."""
    device = next(model.parameters()).device
    rows = [encode_chat(tok, p, r, max_length, rtid)[0] for p, r in items]
    lengths = torch.tensor([len(x) for x in rows])
    B, T = len(rows), int(lengths.max())
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    ids = torch.full((B, T), pad_id, dtype=rows[0].dtype)
    mask = torch.zeros((B, T), dtype=torch.long)
    for i, row in enumerate(rows):
        ids[i, :len(row)] = row
        mask[i, :len(row)] = 1

    captured = {}
    handle = _head_module(model, head_attr).register_forward_pre_hook(
        lambda _m, inp: captured.__setitem__("h", inp[0].detach()))
    try:
        out = model(input_ids=ids.to(device), attention_mask=mask.to(device))
    finally:
        handle.remove()
    h_all = captured["h"]
    ar = torch.arange(B, device=device)
    H = h_all[ar, (lengths - 1).to(device)] if h_all.dim() == 3 else h_all
    H = H.to(torch.float32).cpu()
    if not return_rewards:
        return H
    return H, _emitted_rewards(out)[ar].detach().to(torch.float32).cpu()


def _emitted_rewards(out) -> torch.Tensor:
    if isinstance(out, torch.Tensor):
        return out.reshape(out.shape[0], -1)[:, 0]
    for attr in ("logits", "scores"):
        v = getattr(out, attr, None)
        if v is not None:
            return v.reshape(v.shape[0], -1)[:, 0]
    if isinstance(out, (tuple, list)):
        return out[0].reshape(out[0].shape[0], -1)[:, 0]
    raise AttributeError(f"cannot find the emitted reward on {type(out)}")


class RewardModel:
    """A loaded reward model with its head and forward helpers."""

    def __init__(self, cfg: dict, device: str = "auto"):
        self.cfg = cfg
        self.model, self.tok = load_rm(cfg, device)
        self.head_attr = cfg.get("head_attr", "score")
        self.w_r, self.b_r = get_head(self.model, self.head_attr)
        self.rtid = reward_token_id(self.model, cfg)
        self.batch_size = int(cfg.get("batch_size", 8))
        self.max_length = int(cfg.get("max_length", 4096))

    @property
    def d(self) -> int:
        return int(self.w_r.shape[0])

    def hidden(self, items, verbose: bool = True, tag: str = "") -> np.ndarray:
        """Pooled hidden states [N, d] for a list of (prompt, response)."""
        H, t0 = [], time.time()
        for s in range(0, len(items), self.batch_size):
            H.append(pooled_hidden_batch(self.model, self.tok, items[s:s + self.batch_size],
                                         self.max_length, self.head_attr, self.rtid).numpy())
            if verbose and (s // self.batch_size) % 50 == 0:
                done = s + len(H[-1])
                print(f"  [{tag}] {done}/{len(items)} ({done / max(time.time() - t0, 1e-9):.1f}/s)", flush=True)
        H = np.concatenate(H).astype(np.float32) if H else np.zeros((0, self.d), np.float32)
        assert np.isfinite(H).all(), "non-finite hidden states (forward_dtype overflow?)"
        return H

    def hidden_and_rewards(self, items) -> tuple[np.ndarray, np.ndarray]:
        H, R = [], []
        for s in range(0, len(items), self.batch_size):
            h, r = pooled_hidden_batch(self.model, self.tok, items[s:s + self.batch_size],
                                       self.max_length, self.head_attr, self.rtid, return_rewards=True)
            H.append(h.numpy()); R.append(r.numpy())
        return np.concatenate(H).astype(np.float32), np.concatenate(R).astype(np.float32)
