"""Task registry: math (GSM8K), code (MBPP), science QA (ARC / SciQ), sentiment (SST-2), topic (AG News).

Every example is ``{"prompt": str, "answer": str, ...}``.  Tasks are either
* ``generate`` - free-form generation scored by ``metric`` ("gsm8k" numeric match, "mbpp" unit tests), or
* ``choice``   - the model's next-token logits are compared over ``candidates`` (accuracy).
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List

import torch

LETTERS = ["A", "B", "C", "D"]


def fmt_prompt(instruction: str) -> str:
    return f"### Instruction:\n{instruction}\n\n### Response:\n"


@dataclass
class TaskSpec:
    name: str
    kind: str                      # "generate" | "choice"
    train: List[dict]
    eval: List[dict]
    metric: str = "accuracy"
    candidates: List[str] = field(default_factory=list)
    max_new_tokens: int = 256


def _sub(rows, n, rng):
    rows = list(rows)
    rng.shuffle(rows)
    return rows[:n]


def _mc_prompt(question: str, options: List[str]) -> str:
    opts = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(options))
    return fmt_prompt(f"{question}\n{opts}\nAnswer with the letter of the correct option.")


def load_gsm8k(n_train, n_eval, seed) -> TaskSpec:
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main")
    rng = random.Random(seed)

    def conv(r):
        ans = re.sub(r"<<.*?>>", "", r["answer"])
        return {"prompt": fmt_prompt(r["question"]), "answer": ans, "gold": r["answer"].split("####")[-1].strip().replace(",", "")}
    return TaskSpec("gsm8k", "generate", [conv(r) for r in _sub(ds["train"], n_train, rng)],
                    [conv(r) for r in _sub(ds["test"], n_eval, rng)], metric="gsm8k", max_new_tokens=256)


def load_mbpp(n_train, n_eval, seed) -> TaskSpec:
    from datasets import load_dataset
    ds = load_dataset("google-research-datasets/mbpp", "full")
    rng = random.Random(seed)

    def conv(r):
        tests = "\n".join(r["test_list"])
        return {"prompt": fmt_prompt(f"{r['text']}\nYour code should pass these tests:\n{tests}"),
                "answer": r["code"].replace("\r\n", "\n"), "tests": r["test_list"], "setup": r.get("test_setup_code", "")}
    return TaskSpec("mbpp", "generate", [conv(r) for r in _sub(ds["train"], n_train, rng)],
                    [conv(r) for r in _sub(ds["test"], n_eval, rng)], metric="mbpp", max_new_tokens=384)


def load_arc(n_train, n_eval, seed, config="ARC-Challenge") -> TaskSpec:
    from datasets import load_dataset
    ds = load_dataset("allenai/ai2_arc", config)
    rng = random.Random(seed)

    def ok(r):
        return len(r["choices"]["text"]) == 4 and r["answerKey"] in r["choices"]["label"]

    def conv(r):
        lab = r["choices"]["label"].index(r["answerKey"])
        return {"prompt": _mc_prompt(r["question"], r["choices"]["text"]), "answer": LETTERS[lab], "label": lab}
    tr = [r for r in ds["train"] if ok(r)]
    te = [r for r in ds["test"] if ok(r)]
    return TaskSpec(config.lower().replace("-", "_"), "choice", [conv(r) for r in _sub(tr, n_train, rng)],
                    [conv(r) for r in _sub(te, n_eval, rng)], candidates=LETTERS)


def load_sciq(n_train, n_eval, seed) -> TaskSpec:
    from datasets import load_dataset
    ds = load_dataset("allenai/sciq")
    rng = random.Random(seed)

    def conv(r):
        opts = [r["distractor1"], r["distractor2"], r["distractor3"], r["correct_answer"]]
        order = list(range(4))
        rng.shuffle(order)
        opts = [opts[i] for i in order]
        lab = order.index(3)
        return {"prompt": _mc_prompt(r["question"], opts), "answer": LETTERS[lab], "label": lab}
    return TaskSpec("sciq", "choice", [conv(r) for r in _sub(ds["train"], n_train, rng)],
                    [conv(r) for r in _sub(ds["test"], n_eval, rng)], candidates=LETTERS)


def load_sst2(n_train, n_eval, seed) -> TaskSpec:
    from datasets import load_dataset
    ds = load_dataset("nyu-mll/glue", "sst2")
    rng = random.Random(seed)
    names = ["negative", "positive"]

    def conv(r):
        return {"prompt": fmt_prompt(f"Review: {r['sentence'].strip()}\nIs the sentiment positive or negative?"),
                "answer": names[r["label"]], "label": r["label"]}
    return TaskSpec("sst2", "choice", [conv(r) for r in _sub(ds["train"], n_train, rng)],
                    [conv(r) for r in _sub(ds["validation"], n_eval, rng)], candidates=names)


def load_ag_news(n_train, n_eval, seed) -> TaskSpec:
    from datasets import load_dataset
    ds = load_dataset("fancyzhx/ag_news")
    rng = random.Random(seed)
    names = ["World", "Sports", "Business", "Technology"]

    def conv(r):
        return {"prompt": fmt_prompt(f"News: {r['text']}\nTopic (World, Sports, Business or Technology)?"),
                "answer": names[r["label"]], "label": r["label"]}
    return TaskSpec("ag_news", "choice", [conv(r) for r in _sub(ds["train"], n_train, rng)],
                    [conv(r) for r in _sub(ds["test"], n_eval, rng)], candidates=names)


REGISTRY: Dict[str, Callable[..., TaskSpec]] = {
    "gsm8k": load_gsm8k, "mbpp": load_mbpp, "arc_challenge": load_arc,
    "arc_easy": lambda a, b, c: load_arc(a, b, c, "ARC-Easy"),
    "sciq": load_sciq, "sst2": load_sst2, "ag_news": load_ag_news,
}


def load_task(name: str, n_train: int, n_eval: int, seed: int) -> TaskSpec:
    return REGISTRY[name](n_train, n_eval, seed)


# ---- tokenisation / batching -----------------------------------------------------------------------------------
def build_features(tok, examples: List[dict], max_len: int) -> List[dict]:
    feats = []
    for ex in examples:
        p = tok(ex["prompt"])["input_ids"]
        a = tok(ex["answer"], add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
        if len(p) + len(a) > max_len:
            continue
        feats.append({"input_ids": p + a, "labels": [-100] * len(p) + a})
    return feats


def make_batches(feats: List[dict], bs: int, pad_id: int, shuffle: bool = False, seed: int = 0) -> List[dict]:
    idx = list(range(len(feats)))
    if shuffle:
        random.Random(seed).shuffle(idx)
    batches = []
    for i in range(0, len(idx), bs):
        chunk = [feats[j] for j in idx[i:i + bs]]
        L = max(len(c["input_ids"]) for c in chunk)
        ids = torch.full((len(chunk), L), pad_id, dtype=torch.long)
        lab = torch.full((len(chunk), L), -100, dtype=torch.long)
        att = torch.zeros((len(chunk), L), dtype=torch.long)
        for k, c in enumerate(chunk):
            n = len(c["input_ids"])
            ids[k, :n] = torch.tensor(c["input_ids"])
            lab[k, :n] = torch.tensor(c["labels"])
            att[k, :n] = 1
        batches.append({"input_ids": ids, "attention_mask": att, "labels": lab})
    return batches


# ---- prompt perturbations for invariant routing ------------------------------------------------------------------
def perturb_prompt(text: str, rng: random.Random) -> str:
    ops = [
        lambda t: t.lower(),
        lambda t: re.sub(r"\s+", " ", t),
        lambda t: t.replace("### Instruction:", "Task:").replace("### Response:", "Answer:"),
        lambda t: t.replace("### Instruction:\n", "").replace("\n\n### Response:\n", "\n"),
        lambda t: t + "  \n",
        lambda t: "Please help with the following.\n" + t,
    ]
    return rng.choice(ops)(text)
