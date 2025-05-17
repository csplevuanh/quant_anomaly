# RN‑F Quantization Residual Detector — ICML 2025

[![CI](https://github.com/<your-handle>/icml2025-rnf/actions/workflows/ci.yml/badge.svg)](https://github.com/<your-handle>/icml2025-rnf/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)


This repository reproduces **all experiments** from the paper  
*“Quantization Residuals are Free Signals for Detecting Contaminated TinyML Models”* (ICML 2025).

It covers:
* Download & preprocessing of the **M5Product** tri‑modal corpus (image + text + tabular).
* Post‑Training Quantization (PTQ) of three tiny workloads:
  * **TinyViT‑IoT‑S4** (4.7 M params, vision)
  * **TinyStories‑GPT2‑XS** (13 M params, language)
  * **TabPFNGen** (1.2 M params, tabular)
* Detection of data contamination with **RN‑F** and baselines (CDD, BAIT, ConStat).
* Full evaluation scripts to reproduce Tables 1–2 & Figures 4–6.

> **Hardware**: all runs finish on a single free‑tier NVIDIA T4 GPU (16 GB) in Google Colab  
> **Runtime**: ≤ 40 s calibration, ≤ 5 % latency overhead vs. the 4‑bit baseline

---

## Quick‑start

```bash
git clone https://github.com/<your‑handle>/icml2025-rnf.git
cd icml2025-rnf

# Create fresh environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# ☕ 1. Download dataset & checkpoints (≈ 4 GB)
python scripts/setup.py --all

# 🚀 2. Reproduce the full paper in one line
python scripts/run_all.py
```

Individual workloads:

```bash
# Quantise TinyStories‑GPT2‑XS to 4‑bit
python scripts/quantize.py --model gpt2-xs --bits 4

# Run RN‑F detector
python scripts/detect.py --model gpt2-xs --detector rnf

# Aggregate AUROC, FPR@95, ECE
python scripts/evaluate.py --workload gpt2-xs
```

---

## Repository layout
```
icml2025-rnf/
├── src/rnf_experiments/   ← library code
│   ├── data/              ← dataset wrappers
│   ├── models/            ← model loaders
│   ├── quantization/      ← PTQ / GPTQ utilities
│   ├── detectors/         ← RN‑F & baselines
│   └── evaluation/        ← metrics & helpers
├── scripts/               ← CLI entry‑points
├── configs/               ← YAML experiment recipes
└── papers/                ← LaTeX template (optional)
```

---

### Citation
```bibtex
@inproceedings{le2025rnf,
  title     = {Quantization Residuals are Free Signals for Detecting Contaminated TinyML Models},
  author    = {Vu Anh Le and ...},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2025}
}
```

MIT License. Happy hacking ♥

---

## Visualisations

After running detection, each script saves a compact `.npz` log with arrays. 
Generate the figures exactly as in the paper:

```bash
# Figure 4 — ROC curve
python scripts/plot_roc.py --npz logs/gpt2-xs_rnf.npz --out figures/figure4_roc.png

# Figure 5 — Residual histogram
python scripts/plot_residuals.py --npz logs/gpt2-xs_rnf.npz --out figures/figure5_hist.png

# Figure 6 — Calibration / reliability diagram
python scripts/plot_reliability.py --npz logs/gpt2-xs_rnf.npz --out figures/figure6_reliability.png
```

All figures are 300 dpi, ready for publication.

---

## Development workflow

```bash
# 1. Create dev environment
make dev

# 2. Run code quality checks
make lint format

# 3. Execute unit tests
make test

# 4. Build Docker image (reproduces paper environment)
make docker-build
```

## Continuous integration

All pushes and pull requests trigger the **CI** pipeline defined in
`.github/workflows/ci.yml`, which runs linting and the full test suite
on Python 3.10 & 3.11.

## Docker

```bash
docker build -t rnf-experiments .
docker run --gpus all -it rnf-experiments quantize --model gpt2-xs --bits 4
```
