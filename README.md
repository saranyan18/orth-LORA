# Causal-Orthogonal LoRA pool for continual learning in small LMs

A pre-allocated LoRA rank pool per layer. Each task claims a slice whose size is set by **causal importance**
(activation patching), whose `A` matrix lives in the **null space of old-task activations**, and a
**task-ID-free router** picks slices at inference.

```bash
pip install -r requirements.txt
pytest -q                                        # CPU unit + toy end-to-end tests
python -m src.train_stream --config configs/ours.yaml
bash scripts/run_all.sh                          # baselines + ablations + plots
```

| file | role |
|---|---|
| `src/lora_pool.py` | rank pool, routing context, rank allocation |
| `src/orthogonal.py` | activation null-space (GPM-style), parameter-space penalty, leakage bound |
| `src/causal_probe.py` | mean-ablation patching, attribution patching, conflict detection |
| `src/router.py` | feature extractor, Mahalanobis prototype router, learned gate |
| `src/data.py`, `src/eval/*` | GSM8K / MBPP / ARC / SciQ / SST-2 / AG News, metrics (ACC, BWT, forgetting, FWT) |
| `src/train_stream.py` | the continual-learning loop; every baseline/ablation is a config |

Notes: on a T4 set `model.dtype: float16`. MBPP scoring executes generated code (see `src/eval/code_exec.py`),
so run inside a container/Colab. Results are written to `runs/<name>/results.json`.
