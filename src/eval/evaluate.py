"""Evaluate one task under a given routing policy.

``set_ctx(prompts, true_tid)`` is called before every batch; it is responsible for configuring the
``RoutingContext`` (oracle / router / cumulative / off).
"""
from __future__ import annotations

import re
from typing import Callable, List

import torch

from ..data import TaskSpec
from ..utils import autocast
from .code_exec import extract_code, run_tests


def _extract_number(text: str):
    seg = text.split("####", 1)[1] if "####" in text else text
    nums = re.findall(r"-?\d[\d,]*\.?\d*", seg)
    if not nums:
        return None
    s = (nums[0] if "####" in text else nums[-1]).replace(",", "").rstrip(".")
    try:
        return float(s)
    except ValueError:
        return None


def score_gsm8k(ex: dict, text: str) -> bool:
    pred = _extract_number(text)
    return pred is not None and abs(pred - float(ex["gold"])) < 1e-4


def score_mbpp(ex: dict, text: str) -> bool:
    return run_tests(extract_code(text), ex["tests"], ex.get("setup", ""))


SCORERS = {"gsm8k": score_gsm8k, "mbpp": score_mbpp}


def _last_logits(model, enc):
    try:
        out = model(**enc, use_cache=False, logits_to_keep=1)
    except TypeError:
        out = model(**enc, use_cache=False)
    return out.logits[:, -1, :].float()


@torch.no_grad()
def evaluate_task(model, tok, task: TaskSpec, set_ctx: Callable[[List[str], int], None], tid: int,
                  eval_cfg: dict, device, amp_dtype) -> float:
    model.eval()
    bs = eval_cfg.get("bs", 8)
    max_prompt = eval_cfg.get("max_prompt_len", 768)
    side = tok.padding_side
    tok.padding_side = "left"
    cand_ids = None
    if task.kind == "choice":
        cand_ids = torch.tensor([tok(c, add_special_tokens=False)["input_ids"][0] for c in task.candidates], device=device)
        assert len(set(cand_ids.tolist())) == len(task.candidates), "candidate first-tokens collide"
    correct = 0
    try:
        for i in range(0, len(task.eval), bs):
            chunk = task.eval[i:i + bs]
            prompts = [e["prompt"] for e in chunk]
            set_ctx(prompts, tid)
            enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_prompt).to(device)
            with autocast(device, amp_dtype):
                if task.kind == "choice":
                    pred = _last_logits(model, enc)[:, cand_ids].argmax(-1).tolist()
                    correct += sum(int(p == e["label"]) for p, e in zip(pred, chunk))
                else:
                    mnt = min(task.max_new_tokens, eval_cfg.get("max_new_tokens", task.max_new_tokens))
                    gen = model.generate(**enc, max_new_tokens=mnt, do_sample=False,
                                         pad_token_id=tok.pad_token_id, use_cache=True)
                    texts = tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
                    correct += sum(int(SCORERS[task.metric](e, t)) for e, t in zip(chunk, texts))
    finally:
        tok.padding_side = side
    return correct / max(len(task.eval), 1)
