"""Sparse encoder-decoder with designated coordinates.

    z_pre = W_enc u + b_enc,   z_post = TopK(z_pre),   u_hat = V z_post

Coordinate 0 is the rewrite-artifact coordinate, coordinates 1..|A| are the
attributes (in attributes.yaml order), the rest are free.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

ARTIFACT = "artifact"


def unit_norm_cols(V: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return V / (V.norm(dim=0, keepdim=True) + eps)


def signed_topk(z: torch.Tensor, k: int) -> torch.Tensor:
    """Keep the k largest-magnitude entries per row (signs preserved), zero the rest."""
    if k is None or k >= z.shape[-1]:
        return z
    kth = z.abs().topk(k, dim=-1).values[:, -1:]
    return z * (z.abs() >= kth)


class AttributeDictionary(nn.Module):
    def __init__(self, d: int, K: int, k_act: int, atom_names: list[str], seed: int = 0):
        super().__init__()
        assert atom_names[0] == ARTIFACT
        self.d, self.K, self.k_act = d, K, k_act
        self.atom_names = list(atom_names)
        g = torch.Generator().manual_seed(seed)
        self.V = nn.Parameter(unit_norm_cols(torch.randn(d, K, generator=g)))
        self.W_enc = nn.Linear(d, K, bias=True)

    def encode_pre(self, u: torch.Tensor) -> torch.Tensor:
        return self.W_enc(u)

    def encode(self, u: torch.Tensor) -> torch.Tensor:
        return signed_topk(self.encode_pre(u), self.k_act)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return z @ self.V.t()

    def atom(self, name: str) -> int:
        return self.atom_names.index(name)


class Dictionary:
    """Numpy-facing view of a trained AttributeDictionary."""

    def __init__(self, model: AttributeDictionary, device):
        self.model = model.eval()
        self.device = device
        self.atom_names = model.atom_names
        self.attributes = [a for a in model.atom_names if a != ARTIFACT]

    def atom(self, name: str) -> int:
        return self.model.atom(name)

    @property
    def V(self) -> np.ndarray:
        return self.model.V.detach().cpu().numpy()

    @torch.no_grad()
    def encode(self, H: np.ndarray) -> np.ndarray:
        """Post-TopK codes z_post(h) [N, K]."""
        t = torch.as_tensor(np.array(H, dtype=np.float32), device=self.device)
        return self.model.encode(t).cpu().numpy()

    @torch.no_grad()
    def encode_pre(self, H: np.ndarray) -> np.ndarray:
        """Pre-TopK codes z_pre(h) [N, K]."""
        t = torch.as_tensor(np.array(H, dtype=np.float32), device=self.device)
        return self.model.encode_pre(t).cpu().numpy()
