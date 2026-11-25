"""Causal importance of each layer/module for a task, used to budget LoRA ranks.

Headline method - **activation patching (mean ablation)**.  For a group of modules we replace their output by
its reference mean (a ``do()`` intervention that removes input-specific information while keeping the activation
distribution plausible) and measure the increase in task loss: the *indirect effect* of that group.

Cheap proxy - **attribution patching**: first-order Taylor estimate of the same effect,
``sum (mean - out) * dLoss/dout``, from one backward pass per batch instead of one forward per group.

Also: ``causal_overlap`` / ``conflict_adjust`` implement "causal conflict detection" (give extra rank where the
new task's causal profile overlaps with earlier tasks').
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional

import torch

from .lora_pool import PoolLoRALinear, RoutingContext
from .utils import autocast, model_device

_LAYER_RE = re.compile(r"(?:layers|h|blocks)\.(\d+)\.")


def group_modules(names, granularity: str = "layer") -> Dict[str, List[str]]:
    groups: Dict[str, List[str]] = {}
    for n in names:
        if granularity == "module":
            key = n
        else:
            mt = _LAYER_RE.search(n)
            key = f"layer{mt.group(1)}" if mt else n
        groups.setdefault(key, []).append(n)
    return groups


def regroup_scores(scores: Dict[str, float], granularity: str) -> Dict[str, float]:
    """Spread each group's total score equally over its members (so layer-level and module-level are comparable)."""
    out = {}
    for _, names in group_modules(scores.keys(), granularity).items():
        tot = sum(scores[n] for n in names)
        for n in names:
            out[n] = tot / len(names)
    return out


def uniform_scores(names) -> Dict[str, float]:
    return {n: 1.0 for n in names}


def _nll(model, batch, device, amp_dtype):
    ids = batch["input_ids"].to(device)
    mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)
    with autocast(device, amp_dtype):
        out = model(input_ids=ids, attention_mask=mask, labels=labels, use_cache=False)
    n = int((labels[:, 1:] != -100).sum())
    return float(out.loss.float()) * n, n


@torch.no_grad()
def _mean_loss(model, batches, device, amp_dtype) -> float:
    tot, cnt = 0.0, 0
    for b in batches:
        s, n = _nll(model, b, device, amp_dtype)
        tot, cnt = tot + s, cnt + n
    return tot / max(cnt, 1)


@torch.no_grad()
def _collect_mean_outputs(model, wrapped: Dict[str, PoolLoRALinear], batches, device, amp_dtype):
    sums: Dict[str, torch.Tensor] = {}
    count = [0]
    state = {"mask": None}

    def make(n):
        def hook(mod, inp, out):
            m = state["mask"].to(out.dtype).unsqueeze(-1)
            s = (out.float() * m.float()).sum(dim=(0, 1))
            sums[n] = s if n not in sums else sums[n] + s
        return hook

    handles = [m.base.register_forward_hook(make(n)) for n, m in wrapped.items()]
    try:
        for b in batches:
            state["mask"] = b["attention_mask"].to(device)
            count[0] += int(state["mask"].sum())
            ids, mask = b["input_ids"].to(device), b["attention_mask"].to(device)
            with autocast(device, amp_dtype):
                model(input_ids=ids, attention_mask=mask, use_cache=False)
    finally:
        for h in handles:
            h.remove()
    return {n: s / max(count[0], 1) for n, s in sums.items()}


@torch.no_grad()
def causal_effects(model, ctx: RoutingContext, wrapped: Dict[str, PoolLoRALinear], batches: List[dict],
                   reference_batches: Optional[List[dict]] = None, granularity: str = "layer",
                   amp_dtype=torch.bfloat16) -> Dict[str, float]:
    """Indirect effect (loss increase under mean-ablation) per module.  Cost ~ n_groups x len(batches) forwards."""
    device = model_device(model)
    model.eval()
    saved = ctx.state()
    ctx.off()
    try:
        means = _collect_mean_outputs(model, wrapped, reference_batches or batches, device, amp_dtype)
        clean = _mean_loss(model, batches, device, amp_dtype)
        effects: Dict[str, float] = {}
        for _, names in group_modules(wrapped.keys(), granularity).items():
            def make(n):
                def hook(mod, inp, out):
                    return means[n].to(out.dtype).view(1, 1, -1).expand_as(out).contiguous()
                return hook
            handles = [wrapped[n].base.register_forward_hook(make(n)) for n in names]
            try:
                loss = _mean_loss(model, batches, device, amp_dtype)
            finally:
                for h in handles:
                    h.remove()
            eff = max(loss - clean, 0.0)
            for n in names:
                effects[n] = eff / len(names)
    finally:
        ctx.load(saved)
    return effects


def attribution_scores(model, ctx: RoutingContext, wrapped: Dict[str, PoolLoRALinear], batches: List[dict],
                       reference_batches: Optional[List[dict]] = None, granularity: str = "layer",
                       amp_dtype=torch.bfloat16) -> Dict[str, float]:
    """First-order (gradient x activation-difference) approximation of ``causal_effects``."""
    device = model_device(model)
    model.eval()
    saved = ctx.state()
    ctx.off()
    try:
        means = _collect_mean_outputs(model, wrapped, reference_batches or batches, device, amp_dtype)
        acc = {n: 0.0 for n in wrapped}
        state = {"mask": None}

        def make(n):
            def hook(mod, inp, out):
                if not out.requires_grad:
                    return
                o = out.detach()
                mu = means[n].to(device=out.device, dtype=out.dtype)
                mask = state["mask"].to(out.dtype).unsqueeze(-1)

                def ghook(g, n=n, o=o, mu=mu, mask=mask):
                    acc[n] += float((((mu - o) * g * mask).float().sum()).abs())
                out.register_hook(ghook)
            return hook

        handles = [m.base.register_forward_hook(make(n)) for n, m in wrapped.items()]
        try:
            for b in batches:
                ids, mask = b["input_ids"].to(device), b["attention_mask"].to(device)
                labels = b["labels"].to(device)
                state["mask"] = mask
                with torch.enable_grad():
                    emb = model.get_input_embeddings()(ids).detach().requires_grad_(True)
                    with autocast(device, amp_dtype):
                        out = model(inputs_embeds=emb, attention_mask=mask, labels=labels, use_cache=False)
                    out.loss.backward()
        finally:
            for h in handles:
                h.remove()
            model.zero_grad(set_to_none=True)
    finally:
        ctx.load(saved)
    return regroup_scores(acc, granularity)


# ---- causal conflict detection --------------------------------------------------------------------------------
def causal_overlap(new: Dict[str, float], olds: List[Dict[str, float]]) -> Dict[str, float]:
    """Per-module overlap in [0, 1] between the new causal profile and the strongest earlier one."""
    if not olds:
        return {n: 0.0 for n in new}
    def norm(p):
        mx = max(p.values()) + 1e-12
        return {k: v / mx for k, v in p.items()}
    pn = norm(new)
    po = [norm(o) for o in olds]
    return {n: pn[n] * max(o.get(n, 0.0) for o in po) for n in new}


def conflict_adjust(scores: Dict[str, float], overlap: Dict[str, float], lam: float) -> Dict[str, float]:
    """Boost rank where tasks collide causally (more room to keep the slices apart)."""
    return {n: s * (1.0 + lam * overlap.get(n, 0.0)) for n, s in scores.items()}
