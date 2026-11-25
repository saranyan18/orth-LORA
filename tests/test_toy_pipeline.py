"""End-to-end smoke test on a tiny causal LM (no downloads): probe, attribution, capture, training, null-space."""
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.causal_probe import attribution_scores, causal_effects
from src.lora_pool import RoutingContext, allocate_ranks, wrap_model
from src.orthogonal import collect_activations, update_all_bases
from src.train_stream import train_task


class Block(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.q_proj, self.down_proj = nn.Linear(d, d), nn.Linear(d, d)

    def forward(self, x):
        return x + self.down_proj(torch.relu(self.q_proj(x)))


class Inner(nn.Module):
    def __init__(self, V, d, n):
        super().__init__()
        self.embed = nn.Embedding(V, d)
        self.layers = nn.ModuleList([Block(d) for _ in range(n)])


class ToyLM(nn.Module):
    def __init__(self, V=50, d=32, n=3):
        super().__init__()
        self.model = Inner(V, d, n)
        self.head = nn.Linear(d, V)

    def get_input_embeddings(self):
        return self.model.embed

    def forward(self, input_ids=None, inputs_embeds=None, attention_mask=None, labels=None, use_cache=False):
        x = inputs_embeds if inputs_embeds is not None else self.model.embed(input_ids)
        for l in self.model.layers:
            x = l(x)
        logits = self.head(x)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), labels[:, 1:].reshape(-1), ignore_index=-100)
        return SimpleNamespace(loss=loss, logits=logits)


def make_feats(n=64, L=12, seed=0, shift=0):
    g = torch.Generator().manual_seed(seed)
    feats = []
    for _ in range(n):
        ids = ((torch.randint(0, 20, (L,), generator=g) + shift) % 50).tolist()
        feats.append({"input_ids": ids, "labels": [-100] * 4 + ids[4:]})
    return feats


def setup():
    torch.manual_seed(0)
    model = ToyLM()
    for p in model.parameters():
        p.requires_grad_(False)
    ctx = RoutingContext()
    wrapped = wrap_model(model, ["q_proj", "down_proj"], ctx, capacity=16)
    return model, ctx, wrapped


def test_probe_attribution_train_and_nullspace():
    from src.data import make_batches
    model, ctx, wrapped = setup()
    dev = torch.device("cpu")
    feats = make_feats()
    batches = make_batches(feats, 8, 0)

    eff = causal_effects(model, ctx, wrapped, batches, granularity="layer", amp_dtype=torch.float32)
    att = attribution_scores(model, ctx, wrapped, batches, granularity="layer", amp_dtype=torch.float32)
    assert set(eff) == set(att) == set(wrapped) and all(v >= 0 for v in eff.values()) and sum(att.values()) > 0
    assert ctx.mode == "off"

    ranks = allocate_ranks(eff, {n: m.remaining for n, m in wrapped.items()}, total_budget=12, r_max=8)
    assert sum(ranks.values()) == 12
    for n, m in wrapped.items():
        m.add_task(0, ranks[n], project=False)
    params = [p for m in wrapped.values() if "0" in m.adapters for p in m.adapters["0"].parameters()]
    cfg = {"seed": 0, "train": {"epochs": 3, "bs": 8, "grad_accum": 1, "lr": 1e-2, "warmup": 0.1}}
    ctx.sum([0])
    l0 = float(model(**batches[0]).loss.detach())
    train_task(model, ctx, params, feats, cfg, 0, dev, torch.float32, tag="toy")
    l1 = float(model(**batches[0]).loss.detach())
    assert l1 < l0
    for m in wrapped.values():
        m.freeze_task(0)

    acts = collect_activations(model, ctx, wrapped, batches, 256, dev)
    assert all(a is not None and a.shape[1] == wrapped[n].in_features for n, a in acts.items())
    dims = update_all_bases(wrapped, acts, 0.99, 32, 0.9, dev)
    assert all(d >= 1 for d in dims.values())

    # slice 1 is built in the null space of task-0 activations: it must disturb task-0 inputs far less
    # than the very same slice with the projection switched off.
    for n, m in wrapped.items():
        m.add_task(1, 4, project=True)
        m.adapters["1"].B.data.normal_()
        m.adapters["1"].A.data.normal_()  # raw (un-projected) A: the forward-pass projection must handle it
    with torch.no_grad():
        ctx.sum([0]); ref = model(**batches[0]).logits
        ctx.sum([0, 1]); projected = (model(**batches[0]).logits - ref).abs().mean()
        for m in wrapped.values():
            m._project.discard(1)
        unprojected = (model(**batches[0]).logits - ref).abs().mean()
    assert projected < 0.35 * unprojected, (float(projected), float(unprojected))
