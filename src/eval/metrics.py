"""Continual-learning metrics from the accuracy matrix.

``R[i][j]`` = accuracy on task j after training on task i (NaN where not evaluated).
``base[j]`` = accuracy of the untouched base model on task j.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np


def cl_metrics(R: List[List[float]], base: Optional[List[float]] = None) -> Dict[str, float]:
    R = np.array(R, dtype=float)
    T = R.shape[0]
    last = R[T - 1]
    out: Dict[str, float] = {"avg_acc": float(np.nanmean(last[:T]))}
    if T > 1:
        out["bwt"] = float(np.nanmean([R[T - 1, j] - R[j, j] for j in range(T - 1)]))
        out["forgetting"] = float(np.nanmean([np.nanmax(R[j:T - 1, j]) - R[T - 1, j] for j in range(T - 1)]))
    if base is not None and T > 1:
        # forward transfer: accuracy on task j *before* training on it (needs eval on future tasks) vs the base model
        vals = [R[j - 1, j] - base[j] for j in range(1, T) if not np.isnan(R[j - 1, j])]
        if vals:
            out["fwt"] = float(np.mean(vals))
    return out
