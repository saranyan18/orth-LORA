"""Small shared helpers: seeding, config loading, autocast, VRAM tracking."""
from __future__ import annotations

import contextlib
import json
import os
import random
from typing import Any, Dict, Iterable, Optional

import numpy as np
import torch
import yaml


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: str, overrides: Optional[Iterable[str]] = None) -> Dict[str, Any]:
    """Load YAML and apply dotted overrides, e.g. ``train.lr=1e-4`` or ``tasks=[gsm8k,mbpp]``."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for ov in overrides or []:
        key, val = ov.split("=", 1)
        val = yaml.safe_load(val)
        parts = key.split(".")
        cur = cfg
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
        cur[parts[-1]] = val
    return cfg


def dtype_from_str(name: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
            "float16": torch.float16, "fp16": torch.float16,
            "float32": torch.float32, "fp32": torch.float32}[name]


def model_device(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device


def autocast(device: torch.device, dtype: torch.dtype):
    if device.type == "cuda":
        return torch.autocast("cuda", dtype=dtype)
    return contextlib.nullcontext()


def peak_vram_mb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 2**20
    return 0.0


def dump_json(obj: Any, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=lambda o: float(o) if isinstance(o, (np.floating,)) else str(o))
