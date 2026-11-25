"""Continual learning over a task stream with the causal-orthogonal LoRA pool.

    python -m src.train_stream --config configs/ours.yaml --set tasks=[gsm8k,mbpp] seed=1

Configuration axes (every baseline / ablation is just a config):
  lora.adapter_mode : separate (one slice per task)  | shared (one slice reused = sequential-LoRA baseline)
  budget            : uniform | causal | gradient    (how ranks are split across layers)
  orth              : none | parameter | activation
  router.type       : prototype | gate               (used when eval.routing == router)
  eval.routing      : router | oracle | cumulative
"""
from __future__ import annotations

import argparse
import math
import os
import random
import time
from typing import Dict, List

import torch

from .causal_probe import (attribution_scores, causal_effects, causal_overlap, conflict_adjust,
                           uniform_scores)
from .data import TaskSpec, build_features, load_task, make_batches, perturb_prompt
from .eval.evaluate import evaluate_task
from .eval.metrics import cl_metrics
from .lora_pool import RoutingContext, allocate_ranks, pool_report, wrap_model
from .orthogonal import (collect_activations, leakage_terms, parameter_orthogonality_penalty,
                         update_all_bases)
from .router import (FeatureExtractor, LearnedGateRouter, PrototypeRouter, one_hot_weights,
                     router_accuracy)
from .utils import (autocast, dtype_from_str, dump_json, load_config, model_device, peak_vram_mb,
                    set_seed)


def load_model(mcfg: dict):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    dtype = dtype_from_str(mcfg.get("dtype", "bfloat16"))
    kw = dict(torch_dtype=dtype, device_map={"": 0} if torch.cuda.is_available() else None)
    if mcfg.get("load_in_4bit", True) and torch.cuda.is_available():
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(mcfg["name"], **kw)
    tok = AutoTokenizer.from_pretrained(mcfg["name"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if mcfg.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = True
    return model, tok


def train_task(model, ctx, params, feats, cfg, pad_id, device, amp_dtype, aux_loss=None, tag=""):
    tcfg = cfg["train"]
    accum = tcfg["grad_accum"]
    opt = torch.optim.AdamW(params, lr=tcfg["lr"], weight_decay=0.0)
    n_batches = math.ceil(len(feats) / tcfg["bs"])
    total = max(math.ceil(n_batches * tcfg["epochs"] / accum), 1)
    warm = max(int(tcfg.get("warmup", 0.03) * total), 1)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / warm, max(0.0, (total - s) / max(total - warm, 1))))
    scaler = torch.amp.GradScaler('cuda', enabled=(amp_dtype == torch.float16 and device.type == "cuda"))
    model.train()
    step, t0 = 0, time.time()
    for ep in range(tcfg["epochs"]):
        batches = make_batches(feats, tcfg["bs"], pad_id, shuffle=True, seed=cfg["seed"] + ep)
        for i, b in enumerate(batches):
            ids, att, lab = (b[k].to(device) for k in ("input_ids", "attention_mask", "labels"))
            with autocast(device, amp_dtype):
                loss = model(input_ids=ids, attention_mask=att, labels=lab, use_cache=False).loss
            if aux_loss is not None:
                loss = loss + aux_loss().to(loss.device)
            scaler.scale(loss / accum).backward()
            if (i + 1) % accum == 0 or i == len(batches) - 1:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
                step += 1
                if step % 20 == 0:
                    print(f"[{tag}] step {step}/{total} loss {float(loss.detach()):.4f} ({time.time() - t0:.0f}s)", flush=True)
    model.eval()
    return time.time() - t0


def compute_ranks(cfg, model, ctx, wrapped, probe_batches, profiles, amp_dtype) -> Dict[str, int]:
    lcfg, pcfg = cfg["lora"], cfg.get("probe", {})
    names = list(wrapped)
    remaining = {n: wrapped[n].remaining for n in names}
    total = lcfg["avg_rank"] * len(names)
    mode = cfg.get("budget", "causal")
    if mode == "uniform":
        scores = uniform_scores(names)
    elif mode == "gradient":
        scores = attribution_scores(model, ctx, wrapped, probe_batches, None, pcfg.get("granularity", "layer"), amp_dtype)
    elif mode == "causal":
        scores = causal_effects(model, ctx, wrapped, probe_batches, None, pcfg.get("granularity", "layer"), amp_dtype)
        lam = pcfg.get("conflict_lambda", 0.0)
        profiles_prev = list(profiles)
        profiles.append(dict(scores))
        if lam > 0 and profiles_prev:
            scores = conflict_adjust(scores, causal_overlap(scores, profiles_prev), lam)
    else:
        raise ValueError(mode)
    if sum(max(v, 0) for v in scores.values()) <= 0:  # degenerate probe -> fall back to uniform
        scores = uniform_scores(names)
    return allocate_ranks(scores, remaining, total, lcfg.get("r_min", 0), lcfg.get("r_max", 32))


def run(cfg: dict) -> dict:
    set_seed(cfg["seed"])
    out_dir = cfg["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    model, tok = load_model(cfg["model"])
    device = model_device(model)
    amp_dtype = dtype_from_str(cfg["model"].get("dtype", "bfloat16"))
    for p in model.parameters():
        p.requires_grad_(False)
    ctx = RoutingContext()
    lcfg, ocfg, rcfg, ecfg = cfg["lora"], cfg.get("orth_cfg", {}), cfg.get("router", {}), cfg["eval"]
    wrapped = wrap_model(model, lcfg["targets"], ctx, lcfg["capacity"], lcfg.get("alpha_ratio", 2.0))
    print(f"wrapped {len(wrapped)} linear layers")

    shared = lcfg.get("adapter_mode", "separate") == "shared"
    orth = cfg.get("orth", "activation")
    tasks: List[TaskSpec] = [load_task(n, cfg["data"]["n_train"], cfg["data"]["n_eval"], cfg["seed"]) for n in cfg["tasks"]]
    T = len(tasks)
    feats_train = [build_features(tok, t.train, cfg["model"]["max_len"]) for t in tasks]
    pad_id = tok.pad_token_id

    # ---- routing machinery -------------------------------------------------------------------------------------
    dim = model.config.hidden_size
    extractor = FeatureExtractor(model, tok, ctx, rcfg.get("layer"), max_len=512)
    if rcfg.get("type", "prototype") == "gate":
        router = LearnedGateRouter(dim, max_tasks=T, top_k=rcfg.get("top_k", 1))
    else:
        router = PrototypeRouter(dim, rcfg.get("shrinkage", 0.1), rcfg.get("temperature", 1.0), rcfg.get("top_k", 1))

    def make_setter(mode: str, seen: List[int]):
        def setter(prompts: List[str], true_tid: int) -> None:
            if mode == "cumulative":
                ctx.sum(seen)
            elif mode == "oracle":
                if true_tid in seen:
                    ctx.route(one_hot_weights(true_tid, len(prompts), max(seen) + 1).to(device))
                else:
                    ctx.off()
            elif mode == "router":
                ctx.route(router.weights(extractor(prompts)).to(device))
            else:
                ctx.off()
        return setter

    def eval_all(mode: str, seen: List[int], upto: int) -> List[float]:
        setter = make_setter(mode, seen)
        accs = []
        for j, task in enumerate(tasks):
            if j < upto:
                accs.append(evaluate_task(model, tok, task, setter, j, ecfg, device, amp_dtype))
            else:
                accs.append(float("nan"))
        return accs

    # ---- base model row -----------------------------------------------------------------------------------------
    base = None
    if ecfg.get("base", True):
        ctx.off()
        base = [evaluate_task(model, tok, t, make_setter("off", []), j, ecfg, device, amp_dtype) for j, t in enumerate(tasks)]
        print("base accuracy:", base, flush=True)

    # ---- the stream ---------------------------------------------------------------------------------------------
    R: List[List[float]] = []
    router_accs: List[Dict[str, float]] = []
    rank_log, params_log, time_log, leak_log, dims_log = [], [], [], [], []
    profiles: List[Dict[str, float]] = []
    act_store: Dict[int, Dict[str, torch.Tensor]] = {}
    primary = ecfg.get("routing", "router")
    if shared:
        primary = "cumulative"
    train_active_cfg = cfg["train"].get("active", "auto")
    cumulative_train = (train_active_cfg == "cumulative") or (train_active_cfg == "auto" and primary == "cumulative")
    replay_buf: List[dict] = []

    for t, task in enumerate(tasks):
        print(f"\n===== task {t}: {task.name} =====", flush=True)
        tid = 0 if shared else t
        # 1. allocate this task's slice of the pool
        if shared and t > 0:
            for m in wrapped.values():
                m.set_trainable(0, True)
        else:
            nb = make_batches(feats_train[t][: cfg["probe"].get("n_examples", 200)], cfg["probe"].get("bs", 4), pad_id)
            ranks = compute_ranks(cfg, model, ctx, wrapped, nb, profiles, amp_dtype)
            for n, m in wrapped.items():
                m.add_task(tid, ranks.get(n, 0), project=(orth == "activation" and not shared), trainable=True)
            rank_log.append(ranks)
        params = [p for m in wrapped.values() if str(tid) in m.adapters for p in m.adapters[str(tid)].parameters()]
        params_log.append(sum(p.numel() for p in params))
        print(f"trainable adapter params: {params_log[-1]:,} | pool: {pool_report(wrapped)}", flush=True)

        # 2. train
        if shared:
            active = [0]
        else:
            active = list(range(t + 1)) if cumulative_train else [tid]
        ctx.sum(active)
        aux = None
        if orth == "parameter" and t > 0 and not shared:
            lam = ocfg.get("lambda_param", 1.0)
            aux = lambda: lam * parameter_orthogonality_penalty(wrapped, tid, list(range(t)))  # noqa: E731
        feats = list(feats_train[t])
        rr = cfg["train"].get("replay_ratio", 0.0)
        if rr > 0 and replay_buf:
            feats += random.Random(cfg["seed"]).sample(replay_buf, min(int(rr * len(feats)), len(replay_buf)))
        time_log.append(train_task(model, ctx, params, feats, cfg, pad_id, device, amp_dtype, aux, tag=task.name))
        if rr > 0:
            replay_buf += random.Random(t).sample(feats_train[t], min(cfg["train"].get("replay_buffer", 200), len(feats_train[t])))

        # 3. freeze (bakes projection) and update the old-activation subspace for the next task
        for m in wrapped.values():
            m.freeze_task(tid)
        if orth == "activation" and not shared:
            ctx.sum(active)
            ab = make_batches(feats_train[t][:64], 4, pad_id)
            acts = collect_activations(model, ctx, wrapped, ab, ocfg.get("n_tokens", 512), device,
                                       lambda: autocast(device, amp_dtype))
            if ocfg.get("keep_activations", False):
                act_store[t] = {n: a for n, a in acts.items() if a is not None}
            dims = update_all_bases(wrapped, acts, ocfg.get("energy_thr", 0.90), ocfg.get("max_new", 128),
                                    ocfg.get("max_total_frac", 0.6), device)
            dims_log.append(float(sum(dims.values())) / max(len(dims), 1))
            del acts
            if ocfg.get("keep_activations", False) and t > 0:  # leakage of slice t onto earlier tasks' activations
                per_old = {}
                for o in range(t):
                    vals = [leakage_terms(m, tid, act_store[o][n]) for n, m in wrapped.items()
                            if str(tid) in m.adapters and n in act_store[o]]
                    per_old[o] = {k: sum(v[k] for v in vals) / max(len(vals), 1) for k in ("rel_leak", "rel_output_bound")}
                leak_log.append(per_old)

        # 4. fit router on prompt features (original + perturbed -> invariance)
        if not shared and primary == "router":
            rng = random.Random(cfg["seed"] + t)
            prompts = [e["prompt"] for e in task.train[: rcfg.get("n_fit", 500)]]
            f = extractor(prompts)
            if rcfg.get("augment", True):
                f = torch.cat([f, extractor([perturb_prompt(p, rng) for p in prompts])])
            router.fit_task(tid, f)

        # 5. evaluate
        seen = [0] if shared else list(range(t + 1))
        upto = T if ecfg.get("eval_future", False) else t + 1
        accs = eval_all(primary, seen, upto)
        R.append(accs)
        print(f"after task {t}: acc = {accs}", flush=True)
        if primary == "router" and not shared:
            ra = {}
            for j in seen:
                ra[tasks[j].name] = router_accuracy(router, extractor([e["prompt"] for e in tasks[j].eval[:100]]), j)
            router_accs.append(ra)
            print("router accuracy:", ra, flush=True)

    # ---- final extras: other inference policies on the final model --------------------------------------------------------
    extras = {}
    for mode in ecfg.get("extra_routings", []):
        if mode == "router" and (shared or primary != "router"):
            continue
        extras[mode] = eval_all(mode, [0] if shared else list(range(T)), T)

    result = {
        "config": cfg, "tasks": [t.name for t in tasks], "base_acc": base, "acc_matrix": R,
        "metrics": cl_metrics(R, base), "final_extra_routings": extras,
        "extra_metrics": {k: float(sum(v) / len(v)) for k, v in extras.items()},
        "router_accuracy": router_accs, "ranks_per_task": rank_log, "trainable_params_per_task": params_log,
        "train_seconds_per_task": time_log, "pool": pool_report(wrapped), "peak_vram_mb": peak_vram_mb(),
        "mean_basis_dim_per_task": dims_log, "leakage": leak_log,
    }
    dump_json(result, os.path.join(out_dir, "results.json"))
    adapters = {n: {k: ad.state_dict() for k, ad in m.adapters.items()} for n, m in wrapped.items()}
    torch.save({"adapters": adapters, "router": router.state_dict()}, os.path.join(out_dir, "adapters.pt"))
    print("\nFINAL:", result["metrics"], f"| peak VRAM {result['peak_vram_mb']:.0f} MB")
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[], help="dotted overrides, e.g. train.lr=1e-4 tasks=[gsm8k,mbpp]")
    args = ap.parse_args()
    run(load_config(args.config, args.set))


if __name__ == "__main__":
    main()
