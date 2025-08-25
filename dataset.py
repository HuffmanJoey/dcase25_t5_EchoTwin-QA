import json
import torch
import torchaudio
from pathlib import Path
from transformers import AutoTokenizer
import soundfile as sf

hf_dir    = Path(__file__).parent / "hf_models" / "the hf model you want"
tokenizer = AutoTokenizer.from_pretrained(str(hf_dir), local_files_only=True)

class AQADataset(torch.utils.data.Dataset):
    def __init__(self, json_dir, audio_root, split="train", max_len=None):
        all_paths  = list(Path(json_dir).glob("*.json"))
        valid      = []
        min_frames = int(16000 * 0.025)
        for jp in all_paths:
            meta  = json.loads(jp.read_text(encoding="utf-8"))
            wav_p = Path(audio_root) / Path(meta["audio_url"]).name
            if not wav_p.exists(): continue
            try:
                with sf.SoundFile(str(wav_p)) as f:
                    if f.frames < min_frames: continue
            except:
                continue
            valid.append(jp)
        if max_len:
            valid = valid[:max_len]
        random.shuffle(valid)
        self.items      = valid
        self.audio_root = Path(audio_root)
        print(f"[DEBUG] {split}: {len(self.items)} JSONs")

    def _load_wav(self, wav_path):
        try:
            wav, sr = torchaudio.load(wav_path)
        except RuntimeError:
            wav_np, sr = sf.read(str(wav_path))
            wav = torch.from_numpy(wav_np).float().unsqueeze(0)
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
        jp   = self.items[idx]
        meta = json.loads(jp.read_text(encoding="utf-8"))

        wav = self._load_wav(self.audio_root / Path(meta["audio_url"]).name)

        text = meta["question"] + " " + " ".join(meta["choice"])
        toks = tokenizer(text, truncation=True, padding="max_length",
                         max_length=128, return_tensors="pt")

        label = ord(meta["answer"].split('.')[0].strip().upper()) - ord('A')
        qtype = meta.get("question_type", "unknown")

        return (
            wav,
            toks["input_ids"].squeeze(0),
            toks["attention_mask"].squeeze(0),
            label,
            qtype
        )

    def __len__(self):
        return len(self.items)


def collate(batch):
    wavs, ids, masks, labels, qtypes = zip(*batch)
    return (
        torch.stack(wavs),       # (B, 160000)
        torch.stack(ids),        # (B, 128)
        torch.stack(masks),      # (B, 128)
        torch.tensor(labels),    # (B,)
        list(qtypes)             # length B
    )
