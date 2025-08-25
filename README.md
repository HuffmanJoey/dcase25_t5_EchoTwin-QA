# dcase25_t5_EchoTwin-QA
Official Implementation of a submission EchoTwin-QA to DCASE 2025 Task 5 Audio Question Answering 

# AQA2025 — Dual-Tower BEATs × BERT for Audio Question Answering

Minimal, reproducible baseline for **DCASE 2025 Task 5 (Audio Question Answering)**.

This repo implements a light **dual-tower** model:
- **Audio tower:** [BEATs] encoder (frozen by default)
- **Text tower:** HuggingFace **BERT** (trainable)
- **Fusion:** `[audio_emb ; text_emb] → MLP → logits` (multi-choice)

It trains on per-QA JSON files with linked audio and produces a **submission.csv** in the required `(question, answer)` format.

---

## Table of Contents
- [Features](#features)
- [Install](#install)
- [Local Model Assets (Required)](#local-model-assets-required)
- [Data Format & Layout](#data-format--layout)
- [BEATs Checkpoint](#beats-checkpoint)
- [Quickstart: Training](#quickstart-training)
- [Dev Evaluation + Submission CSV](#dev-evaluation--submission-csv)
- [Expected Directory Tree](#expected-directory-tree)
- [Design Choices](#design-choices)
- [Troubleshooting / Gotchas](#troubleshooting--gotchas)
- [Reproducibility](#reproducibility)
- [License](#license)
- [Acknowledgements](#acknowledgements)
- [Citation](#citation)

---

## Features
- Defensive audio I/O (torchaudio with soundfile fallback), mono-mix, peak-norm, 16 kHz resample, 10 s fixed length (truncate/pad).
- Robustness guards in the forward pass (`torch.nan_to_num` on BEATs outputs).
- Two-group optimizer with distinct LRs for text vs. audio tower.
- Optional partial unfreezing of the last **N** BEATs blocks; BEATs **LayerNorms remain frozen** for stability.
- Dev metrics: overall accuracy + per-question-type accuracy.
- Eval inference → `submission.csv` for leaderboard.

---

## Install

Python ≥ 3.9, PyTorch ≥ 1.13 (CUDA recommended).

```bash
pip install torch torchaudio transformers soundfile tensorboard tqdm
