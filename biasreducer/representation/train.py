"""Train the attribute-aligned dictionary on source-train preference contrasts."""
from __future__ import annotations

import numpy as np
import torch

from .losses import (artifact_loss, behavioral_loss, intervention_loss,
                     representation_loss, scaled)
from .model import ARTIFACT, AttributeDictionary, Dictionary, unit_norm_cols


def fit(dh: np.ndarray, dy: np.ndarray, artifact_pool: np.ndarray,
        interventions: tuple[np.ndarray, np.ndarray, np.ndarray], init: np.ndarray, *,
        attributes: list[str], K: int, k_act: int, rep: dict,
        device: str = "cpu", seed: int = 0, verbose: bool = True) -> tuple[Dictionary, list[dict]]:
    """dh [N, d] contrasts; dy [N, |A|] standardized behavioral differences;
    artifact_pool [M, d] pooled rewrite differences; interventions = (delta [M', d],
    atom [M'], sign [M']); init [d, 1 + |A|] initial columns for the artifact and
    attribute coordinates. `rep` is the `representation:` block of biasreducer.yaml."""
    N, d = dh.shape
    torch.manual_seed(seed)
    np.random.seed(seed)
    atom_names = [ARTIFACT] + list(attributes)
    model = AttributeDictionary(d, K, k_act, atom_names, seed=seed).to(device)
    with torch.no_grad():
        A = torch.as_tensor(init, dtype=torch.float32, device=device)
        model.V.data[:, :A.shape[1]] = unit_norm_cols(A)

    dh_t = torch.as_tensor(dh, dtype=torch.float32, device=device)
    dy_t = torch.as_tensor(dy, dtype=torch.float32, device=device)
    art_t = torch.as_tensor(artifact_pool, dtype=torch.float32, device=device)
    iv = tuple(torch.as_tensor(x).to(device) for x in interventions)
    opt = torch.optim.Adam(model.parameters(), lr=rep["lr"])
    bs = min(int(rep["batch_size"]), N)
    bc = max(1, round(rep["intervention_batch_frac"] * bs))
    g = torch.Generator().manual_seed(seed + 1)

    history = []
    for epoch in range(int(rep["epochs"])):
        perm = torch.randperm(N, device=device)
        last = None
        for i in range(0, N, bs):
            idx = perm[i:i + bs]
            a_idx = torch.randint(0, len(art_t), (bc,), generator=g).to(device)
            j = torch.randint(0, len(iv[0]), (bc,), generator=g).to(device)

            l_rep, recon, z = representation_loss(model, dh_t[idx], rep["sparsity"], rep["incoherence"])
            l_beh, corrs = behavioral_loss(model, dh_t[idx], dy_t[idx], rep["m_corr"], rep["kappa_decor"])
            l_art = artifact_loss(model, art_t[a_idx], rep["m_conc"])
            l_int = intervention_loss(model, iv[0][j], iv[1][j], iv[2][j], rep["m_conc"])
            loss = (l_rep + rep["gamma_behavioral"] * scaled(l_rep, l_beh)
                    + rep["gamma_artifact"] * scaled(l_rep, l_art)
                    + rep["gamma_intervention"] * scaled(l_rep, l_int))
            opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                model.V.data = unit_norm_cols(model.V.data)
            last = dict(loss=loss.item(), recon=recon.item(), l_beh=l_beh.item(),
                        l_art=l_art.item(), l_int=l_int.item(),
                        corr_min=min(corrs), corr_mean=float(np.mean(corrs)),
                        L0=(z.detach().abs() > 1e-3).float().sum(-1).mean().item())
        history.append(last)
        if verbose and (epoch % max(int(rep["epochs"]) // 10, 1) == 0 or epoch == int(rep["epochs"]) - 1):
            print(f"  epoch {epoch:4d} loss={last['loss']:.3f} recon={last['recon']:.3f} "
                  f"beh={last['l_beh']:.3f} corr(min/mean)={last['corr_min']:.2f}/{last['corr_mean']:.2f} "
                  f"L0={last['L0']:.0f}", flush=True)
    return Dictionary(model, device), history
