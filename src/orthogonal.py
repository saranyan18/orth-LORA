"""Orthogonality tools.

Two flavours (ablation):

1. **Parameter-space** (O-LoRA style): penalise ``||A_new A_old^T||_F^2``.
2. **Activation null-space** (GPM / InfLoRA style): estimate the input-activation subspace of old tasks per
   layer, and force ``A_new`` into its orthogonal complement, so ``A_new x_old ~= 0`` and the new slice adds
   (approximately) nothing to old-task outputs.

Also provides ``leakage_terms``: a layer-local first-order quantity that upper-bounds the relative change
of an old task's layer output caused by a new slice.  It is a diagnostic (compare it with measured forgetting),
not a formal end-to-end guarantee.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import torch

from .lora_pool import PoolLoRALinear, RoutingContext


@torch.no_grad()
def update_basis(old_basis: Optional[torch.Tensor], X: torch.Tensor, energy_thr: float = 0.90,
                 max_new: int = 128, max_total: Optional[int] = None) -> Optional[torch.Tensor]:
    """GPM-style subspace growth.

    X: (N, in) activations of the task that just finished (already on the compute device).
    Returns an orthonormal (in, k') basis that contains ``old_basis`` plus the top residual directions
    needed to capture ``energy_thr`` of X's energy.
    """
    X = X.float()
    total = (X ** 2).sum()
    if total <= 0:
        return old_basis
    old_k = 0 if old_basis is None else old_basis.shape[1]
    if old_basis is None:
        captured, Xr = torch.zeros((), device=X.device), X
    else:
        proj = X @ old_basis
        captured = (proj ** 2).sum()
        Xr = X - proj @ old_basis.T
    need = energy_thr * total - captured
    if need <= 0:
        return old_basis
    _, S, Vh = torch.linalg.svd(Xr, full_matrices=False)
    cum = torch.cumsum(S ** 2, 0)
    k = int((cum < need).sum().item()) + 1
    k = min(k, max_new, S.numel())
    if max_total is not None:
        k = min(k, max(max_total - old_k, 0))
    if k <= 0:
        return old_basis
    new = Vh[:k].T
    merged = new if old_basis is None else torch.cat([old_basis, new], dim=1)
    q, _ = torch.linalg.qr(merged)  # re-orthonormalise (numerical hygiene)
    return q


@torch.no_grad()
def collect_activations(model, ctx: RoutingContext, wrapped: Dict[str, PoolLoRALinear], batches: List[dict],
                        n_tokens: int, device, autocast_fn=None) -> Dict[str, torch.Tensor]:
    """Run ``batches`` and sample ~``n_tokens`` input activations per wrapped layer (stored fp16 on CPU)."""
    per_batch = max(n_tokens // max(len(batches), 1), 1)
    for m in wrapped.values():
        m.start_capture(per_batch)
    model.eval()
    try:
        for b in batches:
            ids = b["input_ids"].to(device)
            mask = b["attention_mask"].to(device)
            ctx.token_mask = mask.bool()
            if autocast_fn is not None:
                with autocast_fn():
                    model(input_ids=ids, attention_mask=mask, use_cache=False)
            else:
                model(input_ids=ids, attention_mask=mask, use_cache=False)
    finally:
        ctx.token_mask = None
    return {n: m.stop_capture() for n, m in wrapped.items()}


@torch.no_grad()
def update_all_bases(wrapped: Dict[str, PoolLoRALinear], acts: Dict[str, torch.Tensor], energy_thr: float,
                     max_new: int, max_total_frac: float, device) -> Dict[str, int]:
    dims = {}
    for name, m in wrapped.items():
        X = acts.get(name)
        if X is None:
            continue
        max_total = int(max_total_frac * m.in_features)
        old = None if m.old_basis is None else m.old_basis.to(device)
        new = update_basis(old, X.to(device), energy_thr, max_new, max_total)
        if new is not None:
            m.old_basis = new.to(device)
            dims[name] = new.shape[1]
    return dims


def parameter_orthogonality_penalty(wrapped: Dict[str, PoolLoRALinear], tid: int, old_tids: List[int]):
    """O-LoRA-style penalty: sum_l ||A_new A_old^T||_F^2 (old slices detached)."""
    total = None
    for m in wrapped.values():
        if str(tid) not in m.adapters:
            continue
        A = m.adapters[str(tid)].A
        for o in old_tids:
            if str(o) in m.adapters:
                term = (A @ m.adapters[str(o)].A.detach().T).pow(2).sum()
                total = term if total is None else total + term
    if total is None:
        return torch.zeros(())
    return total


@torch.no_grad()
def leakage_terms(module: PoolLoRALinear, tid: int, X_old: torch.Tensor) -> Dict[str, float]:
    """How much can slice ``tid`` change this layer's output on old-task activations ``X_old`` (N, in)?

    ||dY||_F <= scale * ||B||_2 * ||X A^T||_F   =>   relative bound = that / ||X||_F.
    """
    ad = module.adapters[str(tid)]
    A = module.effective_A(tid).float()
    X = X_old.to(A.device).float()
    leak = (X @ A.T).norm()
    xn = X.norm() + 1e-12
    b_spec = torch.linalg.matrix_norm(ad.B.float(), ord=2)
    return {
        "leak": float(leak),
        "rel_leak": float(leak / (xn * (A.norm() + 1e-12))),
        "rel_output_bound": float(ad.scale * b_spec * leak / xn),
    }
