#!/usr/bin/env bash
set -e
for cfg in seq_lora oracle_lora ours; do
  python -m src.train_stream --config configs/$cfg.yaml
done
# ablations
python -m src.train_stream --config configs/ours.yaml --set budget=uniform out_dir=runs/abl_uniform
python -m src.train_stream --config configs/ours.yaml --set budget=gradient out_dir=runs/abl_gradient
python -m src.train_stream --config configs/ours.yaml --set orth=parameter out_dir=runs/abl_param_orth
python -m src.train_stream --config configs/ours.yaml --set orth=none out_dir=runs/abl_no_orth
python -m src.train_stream --config configs/ours.yaml --set router.type=gate out_dir=runs/abl_gate
python scripts/make_plots.py runs/seq_lora runs/oracle_lora runs/ours runs/abl_* --out figures
