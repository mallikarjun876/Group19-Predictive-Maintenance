# Group 19 — Explainable Hybrid AI for Predictive Maintenance

This folder contains the Python implementation, experiment launcher, scoring
utilities and MetroPT-3 pilot pipeline for the accompanying report.

## 1. Package layout

```text
code/
  model/
    cmapss.py       C-MAPSS loading, normalisation, RUL labels and windows
    nets.py         concatenation, independent-attention and relational models
    train.py        one training/evaluation run; writes a JSON result
    experiment.py   multi-dataset, multi-model, multi-seed and lambda runner
  scoring/
    attribution.py  gradient x input, occlusion and random maps
    iou.py          top-k IoU, chance distribution, z-score and permutation p
    reference.py    train-derived C-MAPSS and MetroPT reference helpers
    validation.py  random/wrong-sensor/shuffled controls and faithfulness curves
  data/
    metropt.py      MetroPT-3 loading, event labels, windows and time splits
  demo_pilot.py    synthetic metric pilot and optional real MetroPT pilot
  summarize.py     compact summary of per-run JSON results
  run_all.sh       full C-MAPSS experiment-matrix launcher
  requirements.txt Python dependencies
```

## 2. Environment setup

Python 3.10 or newer is recommended. From this directory:

```bash
python -m venv .venv
# Windows:
.venv\\Scripts\\activate
# macOS/Linux:
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Install the PyTorch build appropriate for your CPU/GPU from the official
PyTorch installation selector if the default pip command is unsuitable.

## 3. Dataset placement

The C-MAPSS loader expects the official text files with these exact names:

```text
4_dataset/CMAPSS/
  train_FD002.txt  test_FD002.txt  RUL_FD002.txt
  train_FD004.txt  test_FD004.txt  RUL_FD004.txt
```

The archive contains sample-format files only. Replace/add the official full
NASA C-MAPSS files before running full experiments. The code does not fabricate
missing rows or reconstruct full datasets from samples.

For MetroPT-3, download the official CSV and provide its path to `demo_pilot.py`.
Failure intervals in `scoring/reference.py` are explicit project inputs and
must be checked against the dataset documentation before treating them as
authoritative labels.

## 4. Run one configuration

Run from this `code/` directory:

```bash
python -m model.train --data_dir ../4_dataset/CMAPSS --fd FD002 --model concat --seed 0
python -m model.train --data_dir ../4_dataset/CMAPSS --fd FD002 --model independent --seed 0
python -m model.train --data_dir ../4_dataset/CMAPSS --fd FD002 --model relational --seed 0
python -m model.train --data_dir ../4_dataset/CMAPSS --fd FD002 --model relational --phys_lambda 0.25 --seed 0
```

Each run saves one JSON file under `code/results/` unless `--result_path` is
specified. It reports RMSE, NASA score, attribution IoU and (for the relational
model) attention IoU separately.

## 5. Run the experiment matrix

```bash
python -m model.experiment --data_dir ../4_dataset/CMAPSS \\
  --fds FD002 FD004 --seeds 0 1 2 3 4 --epochs 20
```

The matrix runs:
- channel-concatenation baseline;
- independent per-channel temporal-attention baseline;
- relational-attention model;
- relational-attention physics-loss sweep at lambda = 0, 0.1, 0.25, 0.5, 1.0;
- five independent seeds by default (0–4).

Per-run JSON records, a combined CSV and grouped mean/SD/95% interval summary
are written under `results/experiment_matrix/`. The reported interval uses
the Student-t critical value 2.776 for five seeds (df=4). The script only
summarises results that actually exist; it does not create placeholder results.

On macOS/Linux the same matrix can be launched with:

```bash
bash run_all.sh
```

On Windows, use the `python -m model.experiment ...` command directly.

## 6. Explanation metrics and controls

`scoring.iou` provides top-k IoU, a random-selection null distribution, chance
mean, z-score and permutation p-value. Attention importance and gradient x
input attribution are scored as separate maps in `model.train`.

`scoring.validation` provides reusable random, wrong-sensor and shuffled-
reference controls, plus deletion/insertion prediction trajectories and
trapezoidal AUC. These utilities are provided for evaluation scripts; they are
not automatically included in the main training matrix.

## 7. MetroPT-3 pilot

Synthetic pipeline demonstration (not real-world evidence):

```bash
python demo_pilot.py
```

Real CSV pilot:

```bash
python demo_pilot.py "path/to/MetroPT3(AirCompressor).csv"
```

The MetroPT script performs chronological splitting and a simple classifier
pilot. It is not a C-MAPSS RUL experiment and does not establish generalisation
from four logged failure events.

## 8. Important interpretation and reproducibility notes

- The reference sensors used by the C-MAPSS training script are selected from
  training trajectories using absolute Spearman correlation with training RUL.
  This is a train-derived proxy/reference, not independent expert-validated
  physical ground truth.
- The physics auxiliary loss acts on relational attention mass. Attention IoU
  is therefore partly optimised directly; attribution IoU is a separate,
  post-hoc gradient x input metric.
- The included `validation.py` controls and faithfulness functions are utilities
  and require an evaluation caller to supply the appropriate model, reference
  and windows. Their presence does not mean those evaluations have already
  been executed.
- Architecture ablations, cross-dataset transfer, full MetroPT-3 evaluation,
  and final faithfulness statistics must be run and recorded before claiming
  those results. This package does not contain completed experiment logs.
- Do not report any metrics as measured results unless they came from actual
  runs on the full datasets. No model training or experiments are performed
  merely by installing this package.

## 9. Output files

The experiment runner creates:
- one JSON per (dataset, model, lambda, seed);
- `all_seed_results.csv`;
- `summary_mean_sd_ci95.json`.

Use these outputs to regenerate result tables from actual experiment runs.
