# evaluate.py

import argparse
import json
import csv
from pathlib import Path
from collections import defaultdict

import torch
import torchaudio
import soundfile as sf
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

from model import DualTowerAQA

# ─── Inference Dataset ─────────────────────────────────────────────────────
class InferenceDataset(Dataset):
    def __init__(self, json_dir, audio_dir, tokenizer, max_length=128):
        self.json_paths = sorted(Path(json_dir).glob("*.json"))
        self.audio_dir  = Path(audio_dir)
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.json_paths)

    def _load_wav(self, wav_path):
        try:
            wav, sr = torchaudio.load(wav_path)
        except Exception:
            return torch.zeros(160000)
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)
        wav = wav.squeeze(0)
        mv = wav.abs().max().item()
        if mv < 1e-9:
            wav = torch.full_like(wav, 1e-6)
        else:
            wav = wav / mv
            wav = wav.sign() * wav.abs().clamp(min=1e-6)
        if sr != 16000:
            wav = torchaudio.functional.resample(wav.unsqueeze(0), sr, 16000).squeeze(0)
        wav = wav[:160000]
        if wav.numel() < 160000:
            pad = 160000 - wav.numel()
            wav = torch.nn.functional.pad(wav, (0, pad), value=1e-6)
        return wav

    def __getitem__(self, idx):
        jp   = self.json_paths[idx]
        meta = json.loads(jp.read_text(encoding="utf-8"))

        # audio
        wav_path = self.audio_dir / Path(meta["audio_url"]).name
        wav      = self._load_wav(wav_path)

        # text + choices
        text = meta["question"] + " " + " ".join(meta["choice"])
        toks = self.tokenizer(
            text, truncation=True, padding="max_length",
            max_length=self.max_length, return_tensors="pt"
        )

        # label (dev only)
        ans   = meta.get("answer")
        label = ord(ans.split(".")[0].strip().upper()) - ord("A") if ans else None

        # question type
        qtype = meta.get("question_type", "unknown")

        return (
            wav,
            toks["input_ids"].squeeze(0),
            toks["attention_mask"].squeeze(0),
            label,
            qtype,
            meta["choice"],
            jp.stem
        )

def collate_fn(batch):
    wavs, ids, masks, labels, qtypes, choices, qids = zip(*batch)
    return (
        torch.stack(wavs),
        torch.stack(ids),
        torch.stack(masks),
        labels,
        qtypes,
        choices,
        qids
    )

# ─── Main ─────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--beats_ckpt",        required=True)
    parser.add_argument("--model_checkpoint",  required=True)
    parser.add_argument("--hf_text_dir",       required=True)
    parser.add_argument("--dev_json_dir",      required=True)
    parser.add_argument("--dev_audio_dir",     required=True)
    parser.add_argument("--eval_json_dir",     required=True)
    parser.add_argument("--eval_audio_dir",    required=True)
    parser.add_argument("--batch_size",  type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--output_csv",  default="submission.csv")
    parser.add_argument("--freeze_audio", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load trained checkpoint & extract state_dict
    ckpt       = torch.load(args.model_checkpoint, map_location="cpu")
    state_dict = ckpt.get("model", ckpt)

    # detect num_labels
    out_dim = state_dict["fusion.2.weight"].shape[0]
    print(f"Detected num_labels = {out_dim}")

    # instantiate and load model
    model = DualTowerAQA(
        beats_ckpt=args.beats_ckpt,
        num_labels=out_dim,
        freeze_audio=args.freeze_audio
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.hf_text_dir, local_files_only=True)

    # ── DEV EVALUATION ───────────────────────────────────────────
    dev_ds     = InferenceDataset(args.dev_json_dir, args.dev_audio_dir, tokenizer)
    dev_loader = DataLoader(dev_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers,
                            collate_fn=collate_fn)

    total_correct = 0
    total_samples = 0
    correct_by_type = defaultdict(int)
    count_by_type   = defaultdict(int)

    with torch.no_grad():
        for wav, ids, masks, labels, qtypes, *_ in dev_loader:
            wav, ids, masks = wav.to(device), ids.to(device), masks.to(device)
            logits = model(wav, ids, masks)
            preds  = logits.argmax(dim=1).cpu().tolist()

            for p, gt, qt in zip(preds, labels, qtypes):
                count_by_type[qt] += 1
                if gt is not None and p == gt:
                    correct_by_type[qt] += 1
                    total_correct     += 1
                total_samples      += 1

    overall_acc = total_correct / total_samples if total_samples else 0.0
    print(f"\nDev overall accuracy: {overall_acc*100:.2f}% ({total_correct}/{total_samples})\n")
    for qt, count in count_by_type.items():
        acc = correct_by_type[qt] / count
        print(f"  {qt:20s} : {acc*100:5.2f}% ({correct_by_type[qt]}/{count})")
    print()

    # ── EVAL INFERENCE & CSV WRITE ───────────────────────────────
    eval_ds     = InferenceDataset(args.eval_json_dir, args.eval_audio_dir, tokenizer)
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_workers,
                             collate_fn=collate_fn)

    rows = []
    with torch.no_grad():
        for wav, ids, masks, _, _, choices, qids in eval_loader:
            wav, ids, masks = wav.to(device), ids.to(device), masks.to(device)
            logits = model(wav, ids, masks)
            preds  = logits.argmax(dim=1).cpu().tolist()
            for p, ch_list, qid in zip(preds, choices, qids):
                rows.append({"question": qid, "answer": ch_list[p]})

    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["question", "answer"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} lines to {args.output_csv}")

if __name__ == "__main__":
    main()
