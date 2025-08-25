# src/train.py
import argparse, random, time, json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np


from dataset import AQADataset, collate
from model import DualTowerAQA

torch.autograd.set_detect_anomaly(True)

def set_seeds(seed: int = 1337):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def mixup_in_class(wav, y, alpha=0.4):
    """
    Mix each sample with another random sample of the **same label**.
    Returns mixed wav; labels unchanged.
    """
    lam = np.random.beta(alpha, alpha)
    perm = torch.randperm(y.size(0), device=wav.device)
    same = (y == y[perm])
    # only mix where labels match; others unchanged
    wav[same] = lam * wav[same] + (1 - lam) * wav[perm[same]]
    return wav


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def save_ckpt(state, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)

def get_max_choices(json_dir):
    max_c = 0
    for jp in Path(json_dir).glob("*.json"):
        data = json.loads(jp.read_text(encoding="utf-8"))
        max_c = max(max_c, len(data.get("choice", [])))
    return max_c

def train_one_epoch(model, loader, optimizer, scaler, device, scheduler=None):
    """
    Train for one epoch. Returns the average training loss.
    """
    model.train()
    total_loss = 0.0
    n_samples = 0

    loop = tqdm(loader, desc="Training", leave=False)
    for wav, ids, mask, y in loop:
        wav, ids, mask, y = [t.to(device) for t in (wav, ids, mask, y)]
        optimizer.zero_grad(set_to_none=True)

        # new added
        if random.random() < 0.5:
            wav = mixup_in_class(wav, y)
        #MixUp Try
            
        with torch.cuda.amp.autocast(enabled=(scaler is not None)):
            logits = model(wav, ids, mask)
            loss = F.cross_entropy(logits, y, label_smoothing=0.05)

        # backward + step
        if scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
        else:
            loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        if scaler:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        # step scheduler if provided
        if scheduler is not None:
            scheduler.step()

        batch_size = y.size(0)
        total_loss += loss.item() * batch_size
        n_samples  += batch_size
        loop.set_postfix(loss=f"{loss.item():.4f}")

    # Avoid division by zero
    if n_samples == 0:
        return 0.0
    return total_loss / n_samples

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total   = 0
    for wav, ids, mask, y in loader:
        wav, ids, mask, y = [t.to(device) for t in (wav, ids, mask, y)]
        logits = model(wav, ids, mask)
        correct += (logits.argmax(1) == y).sum().item()
        total   += y.size(0)
    return correct / total if total > 0 else 0.0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_dir_train", required=True)
    parser.add_argument("--json_dir_val",   required=True)
    parser.add_argument("--audio_dir",      required=True)
    parser.add_argument("--checkpoint",     required=True)
    parser.add_argument("--save_dir",       default="runs/exp1")
    parser.add_argument("--batch_size",     type=int, default=32)
    parser.add_argument("--epochs",         type=int, default=15)
    parser.add_argument("--lr_text",        type=float, default=1e-5)
    parser.add_argument("--lr_audio",       type=float, default=1e-6)
    parser.add_argument("--unfreeze_audio_layers", type=int, default=0)
    parser.add_argument("--seed",           type=int, default=1337)
    parser.add_argument("--num_workers",    type=int, default=4)
    parser.add_argument("--amp",           action="store_true")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    args = parser.parse_args()

    set_seeds(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # figure out how many answer classes
    max_train = get_max_choices(args.json_dir_train)
    max_val   = get_max_choices(args.json_dir_val)
    num_labels = max(max_train, max_val)
    print(f"▶ Using num_labels = {num_labels}")

    # data
    train_set = AQADataset(args.json_dir_train, args.audio_dir, split="train")
    val_set   = AQADataset(args.json_dir_val,   args.audio_dir, split="val")
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate, num_workers=args.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate, num_workers=args.num_workers, pin_memory=True
    )

    # model
    model = DualTowerAQA(args.checkpoint, num_labels=num_labels, freeze_audio=True).to(device)
    # optionally unfreeze some audio layers
    if args.unfreeze_audio_layers > 0:
        blocks = model.audio_enc.encoder.layers
        for layer in blocks[-args.unfreeze_audio_layers:]:
            for p in layer.parameters():
                p.requires_grad = True
        # freeze LayerNorm in audio to avoid instability
        for module in model.audio_enc.modules():
            if isinstance(module, torch.nn.LayerNorm):
                for p in module.parameters():
                    p.requires_grad = False

    print(f"Trainable parameters: {count_params(model):,}")

    # optimizer & scheduler
    text_params  = [p for n,p in model.named_parameters() if p.requires_grad and not n.startswith("audio_enc")]
    audio_params = [p for n,p in model.named_parameters() if p.requires_grad and   n.startswith("audio_enc")]
    optimizer = AdamW([
        {"params": text_params,  "lr": args.lr_text, "weight_decay": args.weight_decay},
        {"params": audio_params, "lr": args.lr_audio, "weight_decay": args.weight_decay},
    ])
    total_steps = len(train_loader) * args.epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    writer = SummaryWriter(log_dir=args.save_dir)

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, device, scheduler)
        val_acc    = evaluate(model, val_loader, device)

        # log to TensorBoard
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Acc/val",    val_acc,     epoch)

        print(f"Epoch {epoch:02d} | "
              f"Train loss {train_loss:.4f} | "
              f"Val acc {val_acc:.4f} | "
              f"time {(time.time()-start):.1f}s")

        # save best
        if val_acc > best_acc:
            best_acc = val_acc
            save_ckpt({
                "epoch": epoch,
                "model": model.state_dict(),
                "best_acc": best_acc
            }, Path(args.save_dir) / "best.pt")

    # final save
    save_ckpt({
        "epoch": args.epochs,
        "model": model.state_dict(),
        "best_acc": best_acc
    }, Path(args.save_dir) / "last.pt")
    writer.close()
    print(f"Training finished. Best val acc = {best_acc:.4f}")

if __name__ == "__main__":
    main()
