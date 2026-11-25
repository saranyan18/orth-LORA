"""Rank-pool LoRA.

Every targeted ``nn.Linear`` is wrapped in a :class:`PoolLoRALinear` that owns a *pool* of
``capacity`` LoRA ranks.  Each task claims a slice of that pool (its own ``A_t`` / ``B_t``),
whose size is decided by a budget (uniform / causal / gradient, see ``causal_probe.py``).

A shared :class:`RoutingContext` tells every wrapped layer which slices to apply:

* ``off``   - base model only (used for router features and the causal probe)
* ``sum``   - add the slices listed in ``ctx.active`` with weight 1 (training / "cumulative" inference)
* ``route`` - add slice ``t`` scaled by per-sequence weight ``ctx.weights[b, t]`` (router inference)

Activation null-space orthogonality: if a layer has an ``old_basis`` (orthonormal columns spanning
the input-activation subspace of previous tasks), the new task's ``A`` is projected on the fly so that
``A_eff @ x_old ~= 0``.  Because the projection is part of the forward pass (re-parametrisation), it holds
under any optimiser (unlike gradient projection, which Adam breaks).
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class RoutingContext:
    """Mutable switchboard shared by all wrapped layers."""

    def __init__(self) -> None:
        self.mode: str = "off"
        self.active: List[int] = []
        self.weights: Optional[torch.Tensor] = None  # (B, T) for mode == "route"
        self.route_active: List[int] = []
        self.token_mask: Optional[torch.Tensor] = None  # (B, L) bool, used for activation capture

    def off(self) -> None:
        self.mode, self.active, self.weights, self.route_active = "off", [], None, []

    def sum(self, active: List[int]) -> None:
        self.mode, self.active, self.weights, self.route_active = "sum", list(active), None, []

    def route(self, weights: torch.Tensor) -> None:
        self.mode, self.active, self.weights = "route", [], weights
        nz = (weights > 0).any(dim=0).tolist()  # single host sync
        self.route_active = [t for t, flag in enumerate(nz) if flag]

    def state(self):
        return (self.mode, list(self.active), self.weights, list(self.route_active))

    def load(self, st) -> None:
        self.mode, self.active, self.weights, self.route_active = st[0], list(st[1]), st[2], list(st[3])


class TaskAdapter(nn.Module):
    """One task's LoRA slice: delta = scale * B (A x)."""

    def __init__(self, in_f: int, out_f: int, rank: int, scale: float, device, dtype=torch.float32):
        super().__init__()
        self.r = rank
        self.scale = scale
        self.A = nn.Parameter(torch.empty(rank, in_f, device=device, dtype=dtype))
        self.B = nn.Parameter(torch.zeros(out_f, rank, device=device, dtype=dtype))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def num_params(self) -> int:
        return self.A.numel() + self.B.numel()


class PoolLoRALinear(nn.Module):
    def __init__(self, base: nn.Module, name: str, ctx: RoutingContext,
                 capacity: int = 64, alpha_ratio: float = 2.0) -> None:
        super().__init__()
        self.base = base
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.name = name
        self.ctx = ctx
        self.capacity = capacity
        self.scale = alpha_ratio  # alpha = alpha_ratio * r  ->  scale is rank-independent
        self.adapters = nn.ModuleDict()
        self.register_buffer("old_basis", None, persistent=False)  # (in, k) orthonormal
        self._project: set = set()
        self._capture: Optional[dict] = None

    # ---- convenience -------------------------------------------------------------------
    @property
    def weight(self):  # some HF code peeks at .weight.dtype/.device
        return self.base.weight

    @property
    def used_rank(self) -> int:
        return sum(ad.r for ad in self.adapters.values())

    @property
    def remaining(self) -> int:
        return self.capacity - self.used_rank

    # ---- task management -----------------------------------------------------------------
    def add_task(self, tid: int, rank: int, project: bool = False, trainable: bool = True) -> bool:
        if rank <= 0:
            return False
        if rank > self.remaining:
            raise ValueError(f"{self.name}: asked for rank {rank}, only {self.remaining} left in the pool")
        device = self.base.weight.device
        ad = TaskAdapter(self.in_features, self.out_features, rank, self.scale, device)
        if project and self.old_basis is not None:
            if self.old_basis.shape[1] + rank > self.in_features:
                raise ValueError(f"{self.name}: null space exhausted")
            ad.A.data = self._proj(ad.A.data)
            self._project.add(tid)
        for p in ad.parameters():
            p.requires_grad_(trainable)
        self.adapters[str(tid)] = ad
        return True

    def set_trainable(self, tid: int, flag: bool) -> None:
        if str(tid) in self.adapters:
            for p in self.adapters[str(tid)].parameters():
                p.requires_grad_(flag)

    def freeze_task(self, tid: int) -> None:
        """Bake the null-space projection into A and freeze the slice."""
        key = str(tid)
        if key not in self.adapters:
            return
        ad = self.adapters[key]
        if tid in self._project:
            ad.A.data = self._proj(ad.A.data)
            self._project.discard(tid)
        for p in ad.parameters():
            p.requires_grad_(False)

    def _proj(self, A: torch.Tensor) -> torch.Tensor:
        U = self.old_basis
        if U is None:
            return A
        U = U.to(device=A.device, dtype=A.dtype)
        return A - (A @ U) @ U.T

    def effective_A(self, tid: int) -> torch.Tensor:
        A = self.adapters[str(tid)].A
        return self._proj(A) if tid in self._project else A

    # ---- activation capture (for null-space estimation) --------------------------------------
    def start_capture(self, tokens_per_batch: int) -> None:
        self._capture = {"k": tokens_per_batch, "buf": []}

    def stop_capture(self) -> Optional[torch.Tensor]:
        buf = self._capture["buf"] if self._capture else []
        self._capture = None
        return torch.cat(buf) if buf else None

    def _maybe_capture(self, x: torch.Tensor) -> None:
        cap = self._capture
        if cap is None:
            return
        mask = self.ctx.token_mask
        xd = x.detach()
        xs = xd[mask] if (mask is not None and mask.shape == xd.shape[:2]) else xd.reshape(-1, xd.shape[-1])
        if xs.shape[0] > cap["k"]:
            idx = torch.randperm(xs.shape[0], device=xs.device)[: cap["k"]]
            xs = xs[idx]
        cap["buf"].append(xs.half().cpu())

    # ---- forward ----------------------------------------------------------------------------------
    def _delta(self, ad: TaskAdapter, tid: int, x: torch.Tensor) -> torch.Tensor:
        A = self._proj(ad.A) if tid in self._project else ad.A
        h = F.linear(x.to(A.dtype), A)
        return F.linear(h, ad.B) * ad.scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._maybe_capture(x)
        y = self.base(x)
        ctx = self.ctx
        if ctx.mode == "off" or not self.adapters:
            return y
        if ctx.mode == "sum":
            for t in ctx.active:
                ad = self.adapters[str(t)] if str(t) in self.adapters else None
                if ad is not None:
                    y = y + self._delta(ad, t, x).to(y.dtype)
        elif ctx.mode == "route":
            w = ctx.weights
            for t in ctx.route_active:
                key = str(t)
                if key not in self.adapters:
                    continue
                wt = w[:, t].view(-1, *([1] * (x.dim() - 1))).to(y.dtype)
                y = y + wt * self._delta(self.adapters[key], t, x).to(y.dtype)
        return y


def wrap_model(model: nn.Module, target_names: List[str], ctx: RoutingContext,
               capacity: int = 64, alpha_ratio: float = 2.0) -> Dict[str, PoolLoRALinear]:
    """Replace every child linear whose attribute name is in ``target_names``."""
    wrapped: Dict[str, PoolLoRALinear] = {}
    for parent_name, parent in list(model.named_modules()):
        for child_name, child in list(parent.named_children()):
            if (child_name in target_names and not isinstance(child, PoolLoRALinear)
                    and hasattr(child, "in_features") and hasattr(child, "out_features")):
                full = f"{parent_name}.{child_name}" if parent_name else child_name
                new = PoolLoRALinear(child, full, ctx, capacity, alpha_ratio)
                setattr(parent, child_name, new)
                wrapped[full] = new
    if not wrapped:
        raise RuntimeError(f"No modules matched {target_names}")
    return wrapped


def allocate_ranks(scores: Dict[str, float], remaining: Dict[str, int], total_budget: int,
                   r_min: int = 0, r_max: int = 64, step: int = 1) -> Dict[str, int]:
    """Split ``total_budget`` ranks across modules proportionally to ``scores``.

    Respects per-module caps (``r_max`` and the pool's remaining capacity) by iterative water-filling,
    with largest-remainder rounding so the budget is spent exactly whenever capacity allows.
    """
    names = list(scores)
    cap = {n: min(r_max, remaining[n]) for n in names}
    r = {n: min(r_min, cap[n]) for n in names}
    left = total_budget - sum(r.values())
    while left >= step:
        act = [n for n in names if cap[n] - r[n] >= step]
        if not act:
            break
        w = torch.tensor([max(float(scores[n]), 0.0) + 1e-12 for n in act], dtype=torch.float64)
        ideal = left * w / w.sum()
        units = torch.floor(ideal / step).long()
        gave = 0
        for n, u in zip(act, units.tolist()):
            g = min(u * step, ((cap[n] - r[n]) // step) * step)
            r[n] += g
            gave += g
        left -= gave
        if gave == 0:
            frac = (ideal / step - units).tolist()
            n = act[max(range(len(act)), key=lambda i: frac[i])]
            r[n] += step
            left -= step
    return r


def pool_report(wrapped: Dict[str, PoolLoRALinear]) -> Dict[str, float]:
    used = sum(m.used_rank for m in wrapped.values())
    cap = sum(m.capacity for m in wrapped.values())
    params = sum(ad.num_params() for m in wrapped.values() for ad in m.adapters.values())
    return {"rank_used": used, "rank_capacity": cap, "fill": used / max(cap, 1), "adapter_params": params}
