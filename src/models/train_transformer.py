"""Fine-tune DistilBERT to classify reviews as human-written (0) or generated (1).

Trains on the full training split, evaluates on validation after every epoch
and keeps the best epoch (by validation PR-AUC). The test split is never used.
Run from the repository root, ideally on a GPU (Colab T4: ~5 min per epoch):

    python -m src.models.train_transformer
    python -m src.models.train_transformer --epochs 1 --max-train 4000   # quick check
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from src.common import DISTILBERT_DIR, RESULTS_DIR, SEED, SPLIT_DIR, binary_metrics, load_split

MODEL_NAME = "distilbert/distilbert-base-uncased"


class ReviewDataset(Dataset):
    def __init__(self, df, tokenizer, max_length):
        # Tokenise once up front; dynamic padding happens in the collator.
        self.enc = tokenizer(df["text"].tolist(), truncation=True, max_length=max_length)
        self.labels = df["label"].tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.enc.items()}
        item["labels"] = self.labels[i]
        return item


def make_collate(tokenizer):
    def collate(batch):
        labels = torch.tensor([b.pop("labels") for b in batch])
        out = tokenizer.pad(batch, return_tensors="pt")
        out["labels"] = labels
        return out
    return collate


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    y, s = [], []
    for batch in loader:
        labels = batch.pop("labels")
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(**{k: v.to(device) for k, v in batch.items()}).logits
        y.extend(labels.tolist())
        s.extend(torch.softmax(logits.float(), dim=-1)[:, 1].cpu().tolist())
    return np.array(y), np.array(s)


def subsample(df, n):
    if n > 0 and len(df) > n:
        return df.groupby("label", group_keys=False).sample(frac=n / len(df), random_state=SEED)
    return df


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split-dir", default=str(SPLIT_DIR))
    p.add_argument("--out", default=str(DISTILBERT_DIR))
    p.add_argument("--max-train", type=int, default=0, help="0 = use the full training split")
    p.add_argument("--max-val", type=int, default=0)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--length", type=int, default=256, help="max tokens; 128 truncates many long reviews")
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--warmup", type=float, default=0.06)
    args = p.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device, torch.cuda.get_device_name(0) if device.type == "cuda" else "")

    train = subsample(load_split("train", args.split_dir), args.max_train)
    val = subsample(load_split("val", args.split_dir), args.max_val)
    print(f"Train {len(train)} | Val {len(val)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2, id2label={0: "GENUINE", 1: "GENERATED"},
        label2id={"GENUINE": 0, "GENERATED": 1}).to(device)

    collate = make_collate(tokenizer)
    train_loader = DataLoader(ReviewDataset(train, tokenizer, args.length), batch_size=args.batch_size,
                              shuffle=True, collate_fn=collate)
    val_loader = DataLoader(ReviewDataset(val, tokenizer, args.length), batch_size=args.batch_size * 2,
                            collate_fn=collate)

    total_steps = args.epochs * len(train_loader)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(args.warmup * total_steps), total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    out = Path(args.out)
    best, history = -1.0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, running = time.time(), 0.0
        for step, batch in enumerate(tqdm(train_loader, desc=f"epoch {epoch}"), 1):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                loss = model(**batch).loss
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running += loss.item()

        y, s = predict(model, val_loader, device)
        m = binary_metrics(y, s)
        m.update(epoch=epoch, train_loss=running / len(train_loader), minutes=(time.time() - t0) / 60)
        history.append(m)
        print(f"epoch {epoch}: train loss {m['train_loss']:.4f} | val acc {m['accuracy']:.4f} "
              f"f1 {m['f1']:.4f} pr-auc {m['pr_auc']:.4f}")

        if m["pr_auc"] > best:
            best = m["pr_auc"]
            model.save_pretrained(out)
            tokenizer.save_pretrained(out)
            (out / "validation_metrics.json").write_text(json.dumps(m, indent=2))
            (out / "training_args.json").write_text(json.dumps(vars(args), indent=2))
            print("  saved best model to", out)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "distilbert_training_history.json").write_text(json.dumps(history, indent=2))
    print(f"Done. Best val PR-AUC {best:.4f}. Next: python -m src.models.evaluate")


if __name__ == "__main__":
    main()
