# src/model.py
import torch
import torch.nn as nn
from transformers import AutoModel
from types import SimpleNamespace
from BEATs import BEATs
from pathlib import Path

class DualTowerAQA(nn.Module):
    def __init__(self, beats_ckpt: str, num_labels: int, freeze_audio: bool = True):
        super().__init__()
        loaded = torch.load(beats_ckpt, map_location="cpu")
        cfg_dict = loaded.get("cfg") or loaded.get("args")
        if cfg_dict is None:
            raise ValueError("No cfg in checkpoint")
        state = loaded.get("model", loaded)

        # minimal defaults
        cfg_dict.setdefault("finetuned_model", False)
        cfg = SimpleNamespace(**cfg_dict)

        self.audio_enc = BEATs(cfg)
        self.audio_enc.load_state_dict(state, strict=False)
        if freeze_audio:
            for p in self.audio_enc.parameters():
                p.requires_grad = False

        hf_dir = Path(__file__).parent / "hf_models" / "bert-base-uncased"
        self.text_enc = AutoModel.from_pretrained(str(hf_dir), local_files_only=True)

        audio_dim = cfg.encoder_embed_dim
        text_dim  = self.text_enc.config.hidden_size
        self.fusion = nn.Sequential(
            nn.Linear(audio_dim + text_dim, 512),
            nn.ReLU(),
            nn.Linear(512, num_labels),
        )

    def forward(self, wavs: torch.Tensor, ids: torch.Tensor, masks: torch.Tensor):
        # inside DualTowerAQA.forward, replace the first few lines with:

        # 1) AUDIO TOWER
        out = self.audio_enc(wavs)           # {"x": (B, seq_len, D)}

        # —— IMMEDIATELY ZERO‐OUT ANY NaNs OR INFs ——
        x = out["x"]
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        # now safe to collapse
        a_emb = x.mean(dim=1)                # (B, D)


        t_out = self.text_enc(input_ids=ids, attention_mask=masks)
        t_emb = t_out.last_hidden_state[:, 0, :]  # (B, text_dim)

        x = torch.cat([a_emb, t_emb], dim=-1)
        return self.fusion(x)                # (B, num_labels)
