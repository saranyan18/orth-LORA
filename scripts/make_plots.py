"""Plot accuracy matrices and a method comparison from runs/*/results.json.

    python scripts/make_plots.py runs/ours runs/seq_lora runs/oracle_lora --out figures
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load(path):
    with open(os.path.join(path, "results.json")) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", default="figures")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    results = {os.path.basename(r.rstrip("/")): load(r) for r in args.runs}

    # 1. accuracy matrix heatmaps
    fig, axes = plt.subplots(1, len(results), figsize=(4.2 * len(results), 3.8), squeeze=False)
    for ax, (name, res) in zip(axes[0], results.items()):
        R = np.array(res["acc_matrix"], dtype=float)
        im = ax.imshow(R, vmin=0, vmax=1, cmap="viridis")
        ax.set_title(name)
        ax.set_xticks(range(len(res["tasks"])), res["tasks"], rotation=45, ha="right")
        ax.set_yticks(range(R.shape[0]), [f"after {t}" for t in res["tasks"][: R.shape[0]]])
        for i in range(R.shape[0]):
            for j in range(R.shape[1]):
                if not np.isnan(R[i, j]):
                    ax.text(j, i, f"{R[i, j]:.2f}", ha="center", va="center", color="w", fontsize=8)
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8)
    fig.savefig(os.path.join(args.out, "acc_matrices.png"), dpi=160, bbox_inches="tight")

    # 2. forgetting curves: accuracy on the FIRST task as the stream progresses
    fig, ax = plt.subplots(figsize=(5, 3.5))
    for name, res in results.items():
        R = np.array(res["acc_matrix"], dtype=float)
        ax.plot(range(R.shape[0]), R[:, 0], marker="o", label=name)
    ax.set_xlabel("tasks learned"); ax.set_ylabel(f"accuracy on {next(iter(results.values()))['tasks'][0]}")
    ax.legend(); ax.grid(alpha=0.3)
    fig.savefig(os.path.join(args.out, "forgetting_curve.png"), dpi=160, bbox_inches="tight")

    # 3. summary table (markdown)
    lines = ["| run | avg acc | forgetting | BWT | peak VRAM (MB) | params/task |", "|---|---|---|---|---|---|"]
    for name, res in results.items():
        m = res["metrics"]
        lines.append(f"| {name} | {m.get('avg_acc', float('nan')):.3f} | {m.get('forgetting', float('nan')):.3f} | "
                     f"{m.get('bwt', float('nan')):.3f} | {res['peak_vram_mb']:.0f} | "
                     f"{int(np.mean(res['trainable_params_per_task'])):,} |")
    with open(os.path.join(args.out, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
