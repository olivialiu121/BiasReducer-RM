"""Training objective:

    L = L_rep + gamma_b scaled(L_beh) + gamma_i scaled(L_int) + gamma_a scaled(L_art)

with every auxiliary term rescaled to the magnitude of L_rep (stop-gradient).
"""
from __future__ import annotations

import torch

from .model import AttributeDictionary, signed_topk, unit_norm_cols


def batch_corr(z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Pearson correlation of every code column with y over the batch: [K]."""
    zc = z - z.mean(0, keepdim=True)
    yc = y - y.mean()
    return (zc * yc.unsqueeze(1)).sum(0) / (zc.norm(dim=0) * yc.norm() + 1e-8)


def representation_loss(model: AttributeDictionary, u: torch.Tensor, sparsity: float,
                        incoherence: float):
    """Reconstruction + L1 sparsity on post-TopK codes + dictionary incoherence."""
    z = model.encode(u)
    recon = ((u - model.decode(z)) ** 2).sum(-1).mean()
    Vn = unit_norm_cols(model.V)
    gram = Vn.t() @ Vn
    K = model.K
    incoh = (gram - torch.eye(K, device=gram.device)).pow(2).sum() / max(K * (K - 1), 1)
    return recon + sparsity * z.abs().mean() + incoherence * incoh, recon, z


def behavioral_loss(model: AttributeDictionary, u: torch.Tensor, dy: torch.Tensor,
                    m_corr: float, kappa: float):
    """Signed argmax-correlation margin per attribute + cross-attribute leakage,
    on pre-TopK codes. dy [B, |A|] aligned with u rows; attribute a is coordinate a+1."""
    z_pre = model.encode_pre(u)
    n_attr = dy.shape[1]
    l_axis = l_dec = 0.0
    corrs = []
    for a in range(n_attr):
        c = batch_corr(z_pre, dy[:, a])
        comp = c.abs().clone()
        comp[0] = -1.0                       # artifact coordinate never competes
        comp[a + 1] = -1.0
        l_axis = l_axis + torch.relu(m_corr - c[a + 1] + comp.max())
        others = [b + 1 for b in range(n_attr) if b != a]
        l_dec = l_dec + (c[others] ** 2).sum()
        corrs.append(float(c[a + 1].detach()))
    return l_axis + kappa * l_dec, corrs


def intervention_loss(model: AttributeDictionary, delta: torch.Tensor, atom: torch.Tensor,
                      sign: torch.Tensor, m_conc: float):
    """relu(m_conc - s z_pre,a(delta) + max_{k not in {a, art}} |z_post,k(delta)|)."""
    rows = torch.arange(len(delta), device=delta.device)
    z_pre = model.encode_pre(delta)
    z_post = signed_topk(z_pre, model.k_act)
    mask = torch.ones_like(z_post, dtype=torch.bool)
    mask[:, 0] = False
    mask[rows, atom] = False
    comp = z_post.abs().masked_fill(~mask, 0.0).max(-1).values
    return torch.relu(m_conc - sign * z_pre[rows, atom] + comp).mean()


def artifact_loss(model: AttributeDictionary, delta: torch.Tensor, m_conc: float):
    """relu(m_conc - (-z_pre,art(delta)) + max_{k != art} |z_post,k(delta)|)."""
    z_pre = model.encode_pre(delta)
    z_post = signed_topk(z_pre, model.k_act)
    comp = z_post.abs()[:, 1:].max(-1).values
    return torch.relu(m_conc - (-z_pre[:, 0]) + comp).mean()


def scaled(l_rep: torch.Tensor, l_x: torch.Tensor) -> torch.Tensor:
    return (l_rep.detach() / (l_x.detach() + 1e-8)) * l_x
