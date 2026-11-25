"""Task-ID-free routing.

Features: mean-pooled mid-depth hidden state of the *prompt*, computed with all adapters switched off (so features
are identical before and after any adapter is trained and never drift).

Routers (both return per-sequence weights over task slices, ``(N, n_tasks)``):

* :class:`PrototypeRouter` - class means + shared shrunk covariance -> squared Mahalanobis distance. Stores only
  ``D`` floats per task plus one ``D x D`` matrix; no raw data is kept, nothing is trained.
* :class:`LearnedGateRouter` - tiny linear gate trained on a few stored *feature vectors* per task (feature replay).

Invariance: fit routers on features of original *and* perturbed prompts (see ``data.perturb_prompt``) so they do not
latch onto surface cues such as templates or casing.
"""
from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .lora_pool import RoutingContext
from .utils import model_device


class FeatureExtractor:
    def __init__(self, model, tok, ctx: RoutingContext, layer=None, max_len: int = 512, batch_size: int = 16):
        self.model, self.tok, self.ctx = model, tok, ctx
        self.layer, self.max_len, self.bs = layer, max_len, batch_size
        self.backbone = getattr(model, "model", None) or model.base_model

    @torch.no_grad()
    def __call__(self, texts: List[str]) -> torch.Tensor:
        dev = model_device(self.model)
        saved = self.ctx.state()
        self.ctx.off()
        was_training = self.model.training
        self.model.eval()
        side = self.tok.padding_side
        self.tok.padding_side = "right"
        feats = []
        try:
            for i in range(0, len(texts), self.bs):
                enc = self.tok(texts[i:i + self.bs], return_tensors="pt", padding=True,
                               truncation=True, max_length=self.max_len).to(dev)
                out = self.backbone(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                                    output_hidden_states=True, use_cache=False)
                hs = out.hidden_states
                idx = self.layer if self.layer is not None else len(hs) // 2
                h = hs[idx].float()
                m = enc["attention_mask"].unsqueeze(-1).float()
                feats.append(((h * m).sum(1) / m.sum(1)).cpu())
        finally:
            self.tok.padding_side = side
            self.ctx.load(saved)
            self.model.train(was_training)
        return torch.cat(feats)


def topk_weights(scores: torch.Tensor, k: int) -> torch.Tensor:
    """softmax over tasks, keep top-k, renormalise."""
    p = F.softmax(scores, dim=-1)
    if k >= p.shape[1]:
        return p
    vals, idx = p.topk(k, dim=-1)
    out = torch.zeros_like(p).scatter(1, idx, vals)
    return out / out.sum(dim=1, keepdim=True)


class PrototypeRouter:
    def __init__(self, dim: int, shrinkage: float = 0.1, temperature: float = 1.0, top_k: int = 1):
        self.dim, self.shrinkage, self.temperature, self.top_k = dim, shrinkage, temperature, top_k
        self.means: Dict[int, torch.Tensor] = {}
        self.scatter = torch.zeros(dim, dim)
        self.dof = 0
        self._prec = None

    @property
    def n_tasks(self) -> int:
        return (max(self.means) + 1) if self.means else 0

    def fit_task(self, tid: int, feats: torch.Tensor) -> None:
        feats = feats.float()
        mu = feats.mean(0)
        c = feats - mu
        self.scatter += c.T @ c
        self.dof += max(feats.shape[0] - 1, 1)
        self.means[tid] = mu
        self._prec = None

    def _precision(self) -> torch.Tensor:
        if self._prec is None:
            cov = self.scatter / max(self.dof, 1)
            tr = torch.trace(cov) / self.dim
            cov = (1 - self.shrinkage) * cov + self.shrinkage * tr * torch.eye(self.dim)
            self._prec = torch.linalg.inv(cov + 1e-6 * tr * torch.eye(self.dim))
        return self._prec

    def distances(self, feats: torch.Tensor) -> torch.Tensor:
        P = self._precision()
        d = torch.full((feats.shape[0], self.n_tasks), float("inf"))
        for t, mu in self.means.items():
            diff = feats.float() - mu
            d[:, t] = ((diff @ P) * diff).sum(-1)
        return d

    def weights(self, feats: torch.Tensor) -> torch.Tensor:
        d = self.distances(feats) / self.dim
        return topk_weights(-d / self.temperature, self.top_k)

    def state_dict(self):
        return {"means": self.means, "scatter": self.scatter, "dof": self.dof}


class LearnedGateRouter:
    def __init__(self, dim: int, max_tasks: int, top_k: int = 1, n_keep: int = 256, steps: int = 200,
                 lr: float = 1e-2, wd: float = 1e-3):
        self.dim, self.max_tasks, self.top_k = dim, max_tasks, top_k
        self.n_keep, self.steps, self.lr, self.wd = n_keep, steps, lr, wd
        self.bank: Dict[int, torch.Tensor] = {}
        self.gate = nn.Linear(dim, max_tasks)
        self.mu = torch.zeros(dim)
        self.sd = torch.ones(dim)

    @property
    def n_tasks(self) -> int:
        return (max(self.bank) + 1) if self.bank else 0

    def fit_task(self, tid: int, feats: torch.Tensor) -> None:
        g = torch.Generator().manual_seed(tid)
        idx = torch.randperm(feats.shape[0], generator=g)[: self.n_keep]
        self.bank[tid] = feats[idx].float().clone()
        X = torch.cat(list(self.bank.values()))
        y = torch.cat([torch.full((v.shape[0],), t) for t, v in self.bank.items()])
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        Xn = (X - self.mu) / self.sd
        self.gate = nn.Linear(self.dim, self.max_tasks)
        opt = torch.optim.AdamW(self.gate.parameters(), lr=self.lr, weight_decay=self.wd)
        for _ in range(self.steps):
            logits = self.gate(Xn)[:, : self.n_tasks]
            loss = F.cross_entropy(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()

    @torch.no_grad()
    def weights(self, feats: torch.Tensor) -> torch.Tensor:
        logits = self.gate((feats.float() - self.mu) / self.sd)[:, : self.n_tasks]
        return topk_weights(logits, self.top_k)

    def state_dict(self):
        return {"gate": self.gate.state_dict(), "mu": self.mu, "sd": self.sd, "bank": self.bank}


def one_hot_weights(tid: int, n: int, n_tasks: int) -> torch.Tensor:
    w = torch.zeros(n, n_tasks)
    w[:, tid] = 1.0
    return w


@torch.no_grad()
def router_accuracy(router, feats: torch.Tensor, tid: int) -> float:
    pred = router.weights(feats).argmax(-1)
    return float((pred == tid).float().mean())
