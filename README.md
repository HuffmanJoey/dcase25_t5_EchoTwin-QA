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
- [Citation](#citation)
- [License](#license)
- [Acknowledgements](#acknowledgements)


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
```

## Local Model Assets (Required)

Models are loaded from local folders (local_files_only=True). Place these inside src/hf_models/:
In specific countries and regions where direct connection to hugging face is not possible for your server, please download the hfmodel and manually introduce it here

```
src/hf_models/
 ├─ bert-base-uncased/           # a full HF checkpoint dir (config.json, model.safetensors, etc.)
 └─ all-MiniLM-L6-v2/            # sentence-transformers tokenizer assets used by dataset tokenization
```

## Data format & layout

The dataset uses **JSON files** (one per QA) with fields that your code expects:

```json
{
  "audio_url": "https://.../clip.wav",
  "question": "What is the first sound?",
  "choice": ["A. Writing", "B. Door", "C. Laugh", "D. Cough"],
  "answer": "A. Writing",
  "question_type": "both"
}
```

For training and dev, you keep **local copies** of both JSONs and audio:

```
data/
 ├─ train_json/    *.json  (train)
 ├─ dev_json/      *.json  (dev)
 └─ audio/         *.wav   (shared by train/dev; names must match URL basenames)
```
**Important details implemented in code:**
- **Waveforms** are loaded with `torchaudio` (fallback to `soundfile`), mono‑mixed, peak‑normalized, resampled to **16 kHz**, and clamped away from zero to avoid NaNs.
- Each sample is **10 s** fixed length (160,000 samples): truncated or zero‑padded.
- Questions are tokenized as: `question + " " + " ".join(choice)`.
- Labels are parsed from the answer’s leading letter (`A/B/C/D → 0/1/2/3`).
- A sample is **skipped** if its audio is missing or shorter than **25 ms** (defensive cleaning).

## BEATs checkpoint 
https://github.com/microsoft/unilm/tree/master/beats
Download a BEATs pretrained checkpoint (e.g., BEATs iter3) and pass its path to `--checkpoint` / `--beats_ckpt`.

- The loader expects a `dict` containing either `cfg`/`args` and `model`, or a raw state dict.
- Missing keys are tolerated (`strict=False`), and we set a couple of safe defaults.

Example placement:

```
checkpoints/
 └─ BEATs_iter3_plus_AS2M.pt
```

## Quickstart: Training

From `src/`:

```bash
python train.py   --json_dir_train ../data/train_json   --json_dir_val   ../data/dev_json   --audio_dir      ../data/audio   --checkpoint     ../checkpoints/BEATs_iter3_plus_AS2M.pt   --save_dir       ../runs/exp1   --batch_size     32   --epochs         15   --lr_text        1e-5   --lr_audio       1e-6   --unfreeze_audio_layers 0   --amp   --weight_decay   0.0
```

What happens:
- Audio encoder is **frozen** by default (set `--unfreeze_audio_layers N` to fine‑tune the last `N` BEATs blocks).
- If you unfreeze, **LayerNorms stay frozen** (stability) per code.
- Two‑group optimizer: separate LRs for text/audio.
- Cosine LR schedule across all steps.
- Mixed‑precision (`--amp`) with grad‑scaler + gradient clipping.
- TensorBoard logs in `save_dir`.

**View logs:**
```bash
tensorboard --logdir ../runs/exp1
```

**Output:**
- Best checkpoint: `runs/exp1/best.pt`
- Last checkpoint: `runs/exp1/last.pt`
- Console prints overall and per‑type validation accuracy each epoch.


- **Defensive audio I/O** — Fallback to `soundfile` and clamp waveforms to avoid NaNs/Infs; `forward()` also applies `torch.nan_to_num` before pooling.
- **Local HF checkpoints** — `local_files_only=True` for offline runs and fast CI. Remove that flag if you prefer auto‑download.
- **Simple fusion head** — Concatenate pooled audio and `[CLS]` text embeddings → MLP. Easy to extend with cross‑attention later.
- **LayerNorm freezing when unfreezing BEATs** — Improves stability with small LR.


## Dev evaluation + Submission CSV

Use the provided **evaluate.py** to (1) score on dev, and (2) generate a CSV for the hidden eval set.

```bash
python evaluate.py   --beats_ckpt        ../checkpoints/BEATs_iter3_plus_AS2M.pt   --model_checkpoint  ../runs/exp1/best.pt   --hf_text_dir       ./hf_models/bert-base-uncased   --dev_json_dir      ../data/dev_json   --dev_audio_dir     ../data/audio   --eval_json_dir     ../data/eval_json   --eval_audio_dir    ../data/audio   --batch_size        32   --output_csv        ../submission.csv   --freeze_audio
```

- **`--hf_text_dir`** is explicitly required here (same folder as during training).
- Dev metrics (overall + per `question_type`) are printed.
- The eval pass writes **`submission.csv`** with:

```csv
question,answer
<question_id>,<verbatim selected option text>
```

This matches the expected format: question ID and selected answer text.

> If the challenge requires a `.zip` with metadata, include your CSV and any YAML as requested by the organizers alongside your technical report.


## Expected directory tree

```
AQA2025/
 ├─ src/
 │   ├─ dataset.py
 │   ├─ model.py
 │   ├─ train.py
 │   ├─ evaluate.py
 │   └─ hf_models/
 │       ├─ bert-base-uncased/...
 │       └─ all-MiniLM-L6-v2/...
 ├─ data/
 │   ├─ train/*.json
 │   ├─ dev/*.json
 │   ├─ eval/*.json
 │   └─ audio/*.wav
 ├─ checkpoints/
 │   └─ BEATs_iter3_plus_AS2M.pt
 └─ runs/
     └─ exp1/  (tensorboard logs + best.pt/last.pt)
```



## Citation

If you build on this baseline in a DCASE2025 submission or paper, please cite the DCASE Task 5 technical report and BEATs/BERT as appropriate.

## License



## Acknowledgements

- BEATs authors and maintainers  
- HuggingFace Transformers  
- DCASE2025 Task 5 organizers and dataset providers



