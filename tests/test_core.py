import random

import torch
import torch.nn as nn

from src.causal_probe import causal_overlap, conflict_adjust, group_modules, regroup_scores
from src.eval.metrics import cl_metrics
from src.lora_pool import PoolLoRALinear, RoutingContext, allocate_ranks, wrap_model
from src.orthogonal import leakage_terms, parameter_orthogonality_penalty, update_basis
from src.router import LearnedGateRouter, PrototypeRouter, topk_weights
from src.data import perturb_prompt


def test_allocate_ranks_budget_and_caps():
    scores = {"a": 10.0, "b": 1.0, "c": 0.0, "d": 5.0}
    rem = {"a": 64, "b": 64, "c": 64, "d": 4}
    r = allocate_ranks(scores, rem, total_budget=40, r_max=20)
    assert sum(r.values()) == 40
    assert r["a"] <= 20 and r["d"] <= 4
    assert r["a"] > r["b"] >= r["c"]


def test_allocate_ranks_capacity_exhausted():
    r = allocate_ranks({"a": 1.0, "b": 1.0}, {"a": 3, "b": 2}, total_budget=100)
    assert r == {"a": 3, "b": 2}


def test_null_space_projection_kills_old_activations():
    torch.manual_seed(0)
    ctx = RoutingContext()
    m = PoolLoRALinear(nn.Linear(64, 32), "lin", ctx, capacity=16)
    X_old = torch.randn(200, 8) @ torch.randn(8, 64)  # old activations live in an 8-dim subspace
    m.old_basis = update_basis(None, X_old, energy_thr=0.999, max_new=32)
    assert m.old_basis.shape[1] >= 8
    assert torch.allclose(m.old_basis.T @ m.old_basis, torch.eye(m.old_basis.shape[1]), atol=1e-4)
    m.add_task(1, 8, project=True)
    m.adapters["1"].B.data.normal_()  # make the slice non-trivial
    ctx.sum([1])
    delta = m(X_old) - m.base(X_old)
    assert delta.abs().max() < 1e-3 * m.base(X_old).abs().max() + 1e-4
    # a new-task input outside the old subspace is still affected
    x_new = torch.randn(10, 64)
    assert (m(x_new) - m.base(x_new)).abs().max() > 1e-2
    # leakage diagnostic agrees
    assert leakage_terms(m, 1, X_old)["rel_leak"] < 1e-3
    # baking the projection (freeze) preserves behaviour
    before = m(x_new)
    m.freeze_task(1)
    assert torch.allclose(before, m(x_new), atol=1e-5)


def test_projection_survives_optimizer_steps():
    torch.manual_seed(1)
    ctx = RoutingContext()
    m = PoolLoRALinear(nn.Linear(32, 16), "lin", ctx, capacity=8)
    X_old = torch.randn(100, 4) @ torch.randn(4, 32)
    m.old_basis = update_basis(None, X_old, 0.999, 16)
    m.add_task(0, 4, project=True)
    ctx.sum([0])
    opt = torch.optim.Adam(m.adapters["0"].parameters(), lr=1e-1)
    for _ in range(20):
        x = torch.randn(16, 32)
        loss = m(x).pow(2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    assert (m(X_old) - m.base(X_old)).abs().max() < 1e-3


def test_routing_modes():
    ctx = RoutingContext()
    m = PoolLoRALinear(nn.Linear(8, 8), "lin", ctx, capacity=8)
    m.add_task(0, 2); m.add_task(1, 2)
    for t in (0, 1):
        m.adapters[str(t)].B.data.normal_()
    x = torch.randn(4, 5, 8)
    base = m.base(x)
    ctx.off(); assert torch.equal(m(x), base)
    ctx.sum([0]); y0 = m(x)
    ctx.sum([1]); y1 = m(x)
    ctx.sum([0, 1]); y01 = m(x)
    assert torch.allclose(y01 - base, (y0 - base) + (y1 - base), atol=1e-5)
    w = torch.tensor([[1., 0.], [0., 1.], [0.5, 0.5], [0., 0.]])
    ctx.route(w)
    y = m(x)
    assert torch.allclose(y[0], y0[0], atol=1e-5)
    assert torch.allclose(y[1], y1[1], atol=1e-5)
    assert torch.allclose(y[2] - base[2], 0.5 * (y0[2] - base[2]) + 0.5 * (y1[2] - base[2]), atol=1e-5)
    assert torch.allclose(y[3], base[3], atol=1e-6)


def test_wrap_model_and_pool_overflow():
    ctx = RoutingContext()
    net = nn.Sequential(nn.Linear(8, 8), nn.ReLU())
    net.add_module("q_proj", nn.Linear(8, 8))
    wrapped = wrap_model(net, ["q_proj"], ctx, capacity=4)
    assert list(wrapped) == ["q_proj"] and isinstance(net.q_proj, PoolLoRALinear)
    wrapped["q_proj"].add_task(0, 3)
    try:
        wrapped["q_proj"].add_task(1, 2)
        assert False
    except ValueError:
        pass


def test_parameter_penalty_zero_when_orthogonal():
    ctx = RoutingContext()
    m = PoolLoRALinear(nn.Linear(8, 8), "lin", ctx, capacity=8)
    m.add_task(0, 2); m.add_task(1, 2)
    m.adapters["0"].A.data = torch.eye(8)[:2]
    m.adapters["1"].A.data = torch.eye(8)[2:4]
    assert parameter_orthogonality_penalty({"lin": m}, 1, [0]).item() == 0.0
    m.adapters["1"].A.data = torch.eye(8)[:2]
    assert parameter_orthogonality_penalty({"lin": m}, 1, [0]).item() > 0


def test_prototype_and_gate_routers_separate_tasks():
    torch.manual_seed(0)
    D = 32
    centers = [torch.randn(D) * 3 for _ in range(3)]
    data = [c + torch.randn(200, D) for c in centers]
    for router in (PrototypeRouter(D, top_k=1), LearnedGateRouter(D, max_tasks=3, top_k=1)):
        for t in range(3):
            router.fit_task(t, data[t][:100])
        for t in range(3):
            acc = (router.weights(data[t][100:]).argmax(-1) == t).float().mean()
            assert acc > 0.95
    w = topk_weights(torch.tensor([[3., 2., -5.]]), 2)
    assert abs(w.sum().item() - 1) < 1e-6 and w[0, 2] == 0


def test_causal_helpers():
    names = ["model.layers.0.q_proj", "model.layers.0.k_proj", "model.layers.1.q_proj"]
    g = group_modules(names, "layer")
    assert set(g) == {"layer0", "layer1"}
    rs = regroup_scores({names[0]: 2.0, names[1]: 0.0, names[2]: 1.0}, "layer")
    assert rs[names[0]] == rs[names[1]] == 1.0 and rs[names[2]] == 1.0
    ov = causal_overlap({"a": 1.0, "b": 0.1}, [{"a": 1.0, "b": 0.0}])
    assert ov["a"] > ov["b"]
    assert conflict_adjust({"a": 1.0}, {"a": 1.0}, 2.0)["a"] == 3.0


def test_metrics():
    nan = float("nan")
    R = [[0.8, nan], [0.7, 0.9]]
    m = cl_metrics(R, base=[0.1, 0.2])
    assert abs(m["avg_acc"] - 0.8) < 1e-9
    assert abs(m["bwt"] + 0.1) < 1e-9 and abs(m["forgetting"] - 0.1) < 1e-9


def test_perturb_prompt_runs():
    rng = random.Random(0)
    assert isinstance(perturb_prompt("### Instruction:\nHi\n\n### Response:\n", rng), str)
