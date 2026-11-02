"""Train a DistilBERT binary review classifier on ReviewShield's parquet splits.
Run from repository root: python -m src.models.train_transformer
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

MODEL_NAME = "distilbert/distilbert-base-uncased"

class ReviewDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length):
        self.texts = dataframe["text"].astype(str).tolist()
        self.labels = dataframe["label"].astype(int).tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        encoded = self.tokenizer(self.texts[index], max_length=self.max_length,
                                 truncation=True, padding="max_length", return_tensors="pt")
        result = {key: value.squeeze(0) for key, value in encoded.items()}
        result["labels"] = torch.tensor(self.labels[index], dtype=torch.long)
        return result

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    y_true, y_score = [], []
    for batch in tqdm(loader, desc="Validation", leave=False):
        labels = batch.pop("labels")
        logits = model(**{k: v.to(device) for k, v in batch.items()}).logits
        y_true.extend(labels.tolist())
        y_score.extend(torch.softmax(logits.float(), dim=-1)[:, 1].cpu().tolist())
    y_true, y_score = np.array(y_true), np.array(y_score)
    pred = (y_score >= 0.5).astype(int)
    return {"accuracy": float(accuracy_score(y_true, pred)),
            "f1": float(f1_score(y_true, pred)),
            "pr_auc": float(average_precision_score(y_true, y_score)),
            "roc_auc": float(roc_auc_score(y_true, y_score))}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", default="data/processed/train2.parquet")
    p.add_argument("--val", default="data/processed/val.parquet")
    p.add_argument("--out", default="models/distilbert")
    p.add_argument("--max-train", type=int, default=4000)
    p.add_argument("--max-val", type=int, default=1000)
    p.add_argument("--steps", type=int, default=120)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--accumulation", type=int, default=4)
    p.add_argument("--length", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-5)
    args = p.parse_args()

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
    train = pd.read_parquet(args.train).dropna(subset=["text", "label"])
    val = pd.read_parquet(args.val).dropna(subset=["text", "label"])
    if args.max_train > 0 and len(train) > args.max_train:
        train = train.groupby("label", group_keys=False).sample(frac=args.max_train / len(train), random_state=42)
    if args.max_val > 0 and len(val) > args.max_val:
        val = val.groupby("label", group_keys=False).sample(frac=args.max_val / len(val), random_state=42)
    print(f"Training examples: {len(train)} | Validation examples: {len(val)}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2, id2label={0: "GENUINE", 1: "GENERATED"},
        label2id={"GENUINE": 0, "GENERATED": 1}
    ).to(device)
    train_loader = DataLoader(ReviewDataset(train, tokenizer, args.length), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(ReviewDataset(val, tokenizer, args.length), batch_size=args.batch_size * 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=max(1, args.steps // 10), num_training_steps=args.steps)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    iterator = iter(train_loader)
    for step in tqdm(range(args.steps), desc="Fine-tuning"):
        for _ in range(args.accumulation):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                batch = next(iterator)
            batch = {key: value.to(device) for key, value in batch.items()}
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                loss = model(**batch).loss / args.accumulation
            scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
    metrics = evaluate(model, val_loader, device)
    print("Validation metrics:", json.dumps(metrics, indent=2))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    (out / "validation_metrics.json").write_text(json.dumps(metrics, indent=2))
    print("Saved model:", out)

if __name__ == "__main__":
    main()
