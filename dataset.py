# src/dataset.py
import json
import torch
import torchaudio
import random
from pathlib import Path
from transformers import AutoTokenizer
import soundfile as sf   # pip install soundfile
import numpy as np
from torchaudio import transforms as T

hf_dir = Path(__file__).parent / "hf_models" / "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(str(hf_dir), local_files_only=True)

class AQADataset(torch.utils.data.Dataset):
    def __init__(self, json_dir, audio_root, split="train", max_len=None):
        all_paths = list(Path(json_dir).glob("*.json"))
        valid = []
        min_frames = int(16000 * 0.025)
        for jp in all_paths:
            meta = json.loads(jp.read_text(encoding="utf-8"))
            wav_p = Path(audio_root) / Path(meta["audio_url"]).name
            if not wav_p.exists(): continue
            # skip too-short
            try:
                with sf.SoundFile(str(wav_p)) as f:
                    if f.frames < min_frames: continue
            except:
                continue
            valid.append(jp)
        if max_len:
            valid = valid[:max_len]
        random.shuffle(valid)
        self.items = valid
        self.audio_root = Path(audio_root)
        self.split = split          # remember mode

        # build SpecAug pipeline once (train-only)
        if split == "train":
            self.spec      = T.Spectrogram(
                n_fft=400, hop_length=160, win_length=400,
                power=2.0, center=True, pad_mode="reflect")
            self.freq_mask = T.FrequencyMasking(freq_mask_param=20)
            self.time_mask = T.TimeMasking(time_mask_param=80)
            self.griffin   = T.GriffinLim(
                n_fft=400, hop_length=160, win_length=400,
                power=2.0, n_iter=16)
        print(f"[DEBUG] {split}: {len(self.items)} JSONs")

    def _load_wav(self, wav_path):
        # load (C, T) or fallback
        try:
            wav, sr = torchaudio.load(wav_path)  # float32 in [-1,1]
        except RuntimeError:
            wav_np, sr = sf.read(str(wav_path))
            wav = torch.from_numpy(wav_np).float()
            if wav.dim() == 1:
                wav = wav.unsqueeze(0)

        # mono → (1, T)
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)

        # flatten → (T,)
        wav = wav.squeeze(0)

        # normalize amplitude
        max_val = wav.abs().max().item()
        if max_val < 1e-9:
            wav = torch.full_like(wav, 1e-6)
        else:
            wav = wav / max_val
            wav = wav.sign() * wav.abs().clamp(min=1e-6)

        # resample if needed
        if sr != 16000:
            wav = torchaudio.functional.resample(wav.unsqueeze(0), sr, 16000).squeeze(0)

        # truncate & pad (with 1e-6, not 0)
        wav = wav[:160000]
        if wav.size(0) < 160000:
            pad = 160000 - wav.size(0)
            wav = torch.nn.functional.pad(wav, (0, pad), mode='constant', value=1e-6)

        return wav  # shape (160000,), no exact zeros anywhere

        # ---------------------------------------------------------
    def _augment(self, wav: torch.Tensor) -> torch.Tensor:
        """Cheap but effective waveform aug."""
        # 1) random time-shift ±1 s
        if random.random() < 0.5:
            shift = random.randint(-16000, 16000)
            wav = torch.roll(wav, shifts=shift, dims=0)
            if shift > 0: wav[:shift] = 0
            else:         wav[shift:] = 0

        # 2) SpecAug on magnitude spectrogram
        spec = self.spec(wav.unsqueeze(0))          # (1,F,T)
        spec = self.freq_mask(spec)
        spec = self.time_mask(spec)
        wav  = self.griffin(spec.squeeze(0))        # back to waveform
        wav  = wav / wav.abs().max().clamp(min=1e-6)

        # 3) random gain (-6 … +6 dB)
        if random.random() < 0.5:
            gain_db = random.uniform(-6, 6)
            wav *= 10 ** (gain_db / 20)

        # 4) additive white noise (SNR 10–30 dB)
        if random.random() < 0.3:
            snr_db = random.uniform(10, 30)
            sig_p  = wav.pow(2).mean()
            noise_p = sig_p / (10 ** (snr_db / 10))
            wav += torch.randn_like(wav) * noise_p.sqrt()

        return wav


    def __getitem__(self, idx):
        jp = self.items[idx]
        meta = json.loads(jp.read_text(encoding="utf-8"))
        wav = self._load_wav(self.audio_root / Path(meta["audio_url"]).name)
        if self.split == "train":          # ← add this guard
            wav = self._augment(wav)
        # text
        text = meta["question"] + " " + " ".join(meta["choice"])
        toks = tokenizer(text, truncation=True, padding="max_length", max_length=128, return_tensors="pt")
        label = ord(meta["answer"].split('.')[0].strip().upper()) - ord('A')
        return wav, toks["input_ids"].squeeze(0), toks["attention_mask"].squeeze(0), label

    def __len__(self):
        return len(self.items)

def collate(batch):
    wavs, ids, masks, labels = zip(*batch)
    return (
        torch.stack(wavs),        # (B, 160000)
        torch.stack(ids),         # (B, 64)
        torch.stack(masks),
        torch.tensor(labels)      # (B,)
    )
