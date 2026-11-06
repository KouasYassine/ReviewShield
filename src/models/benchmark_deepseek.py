
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

DATA_PATH = "data/external/modern_reviews/deepseek_test.csv"
LR_PATH = "models/tfidf_lr_clean.joblib"
BERT_PATH = "models/distilbert_clean"
THRESHOLD = 0.5
MAX_LENGTH = 128
BATCH_SIZE = 16

df = pd.read_csv(DATA_PATH)
df = df[df["label"] == "CG"].dropna(subset=["text_"]).copy()
texts = df["text_"].astype(str).tolist()

print("DeepSeek reviews:", len(texts))

# Logistic Regression
lr = joblib.load(LR_PATH)
lr_scores = lr.predict_proba(texts)[:, list(lr.classes_).index(1)]

# DistilBERT
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(BERT_PATH)
model = AutoModelForSequenceClassification.from_pretrained(
    BERT_PATH
).to(device).eval()

bert_scores = []
token_lengths = []

with torch.inference_mode():
    for start in range(0, len(texts), BATCH_SIZE):
        batch_texts = texts[start:start + BATCH_SIZE]

        token_lengths.extend(
            len(ids) for ids in tokenizer(
                batch_texts,
                truncation=False,
                add_special_tokens=True
            )["input_ids"]
        )

        inputs = tokenizer(
            batch_texts,
            truncation=True,
            max_length=MAX_LENGTH,
            padding=True,
            return_tensors="pt"
        ).to(device)

        logits = model(**inputs).logits
        probabilities = torch.softmax(logits.float(), dim=-1)[:, 1]
        bert_scores.extend(probabilities.cpu().numpy())

bert_scores = np.array(bert_scores)
token_lengths = np.array(token_lengths)

for name, scores in [
    ("Logistic Regression", lr_scores),
    ("DistilBERT", bert_scores)
]:
    print(f"\n{name}")
    print("Generated recall:", round(np.mean(scores >= THRESHOLD), 4))
    print("Mean AI probability:", round(float(scores.mean()), 6))
    print("Median AI probability:", round(float(np.median(scores)), 6))

    short = token_lengths <= MAX_LENGTH
    print("Reviews <=128 tokens:", int(short.sum()))
    if short.any():
        print(
            "Recall on short reviews:",
            round(float(np.mean(scores[short] >= THRESHOLD)), 4)
        )

output = Path("results/deepseek_predictions.csv")
output.parent.mkdir(parents=True, exist_ok=True)

df["lr_probability"] = lr_scores
df["bert_probability"] = bert_scores
df["token_length"] = token_lengths
df.to_csv(output, index=False)

print("\nSaved:", output)
