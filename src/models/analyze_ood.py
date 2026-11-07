"""Error analysis on the external AiGen-FoodReview set: score distributions,
performance by review length (truncation), and the most confident mistakes.

    python -m src.models.analyze_ood
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = Path(
    "data/external/aigen-foodreview/data/test.csv"
)
from src.common import DISTILBERT_DIR as BERT_PATH, RESULTS_DIR, TFIDF_PATH as LR_PATH

OUTPUT_DIR = RESULTS_DIR / "ood_analysis"

THRESHOLD = 0.5
BATCH_SIZE = 16
MAX_LENGTH = 256  # must match the length used in training

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(DATA_PATH)
df = df.dropna(subset=["text", "label"]).copy()
df["text"] = df["text"].astype(str)
df["label"] = df["label"].astype(int)

texts = df["text"].tolist()
y_true = df["label"].to_numpy()

print("\nTotal reviews:", len(df))
print("Class distribution:", df["label"].value_counts().to_dict())

# ============================================================
# LOGISTIC REGRESSION
# ============================================================

print("\nLoading Logistic Regression...")

lr = joblib.load(LR_PATH)

lr_positive_index = list(
    lr.classes_
).index(1)

lr_scores = lr.predict_proba(texts)[:, lr_positive_index]

# ============================================================
# DISTILBERT
# ============================================================

print("\nLoading DistilBERT...")

tokenizer = AutoTokenizer.from_pretrained(BERT_PATH)

bert = AutoModelForSequenceClassification.from_pretrained(
    BERT_PATH
).to(device)

bert.eval()

bert_scores = []
token_lengths = []

with torch.inference_mode():

    for start in range(0, len(texts), BATCH_SIZE):

        batch_texts = texts[start:start + BATCH_SIZE]

        encoded = tokenizer(
            batch_texts,
            truncation=False,
            add_special_tokens=True,
        )

        token_lengths.extend(
            len(ids) for ids in encoded["input_ids"]
        )

        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        ).to(device)

        logits = bert(**inputs).logits

        probabilities = torch.softmax(
            logits.float(), dim=-1
        )[:, 1]

        bert_scores.extend(
            probabilities.cpu().numpy()
        )

        if (start // BATCH_SIZE) % 25 == 0:
            print(
                f"Processed {min(start + BATCH_SIZE, len(texts))}"
                f"/{len(texts)} reviews"
            )

bert_scores = np.asarray(bert_scores)
token_lengths = np.asarray(token_lengths)

df["lr_probability"] = lr_scores
df["bert_probability"] = bert_scores
df["token_length"] = token_lengths
df["truncated"] = token_lengths > MAX_LENGTH

# ============================================================
# SCORE DISTRIBUTIONS
# ============================================================

def analyze_scores(name, scores):

    print(f"\n{'=' * 55}")
    print(name)
    print("=" * 55)

    predictions = (scores >= THRESHOLD).astype(int)

    print("Accuracy:", round(
        accuracy_score(y_true, predictions), 4
    ))
    print("ROC-AUC:", round(
        roc_auc_score(y_true, scores), 4
    ))
    print("PR-AUC:", round(
        average_precision_score(y_true, scores), 4
    ))

    print("\nConfusion matrix:")
    print(confusion_matrix(y_true, predictions))

    for label, class_name in [
        (0, "HUMAN"),
        (1, "AI-GENERATED"),
    ]:

        class_scores = scores[y_true == label]

        print(f"\n{class_name} — {len(class_scores)} reviews")
        print("Mean:", round(class_scores.mean(), 6))
        print("Median:", round(np.median(class_scores), 6))

        print(
            "Percentiles [5, 25, 50, 75, 95]:",
            np.round(
                np.percentile(
                    class_scores, [5, 25, 50, 75, 95]
                ),
                6
            )
        )

        print(
            "Predicted AI:",
            int((class_scores >= THRESHOLD).sum())
        )


analyze_scores("LOGISTIC REGRESSION", lr_scores)
analyze_scores("DISTILBERT", bert_scores)

# ============================================================
# REVIEW LENGTH AND TRUNCATION
# ============================================================

print("\n" + "=" * 55)
print("TOKEN LENGTH ANALYSIS")
print("=" * 55)

for label, name in [(0, "HUMAN"), (1, "AI-GENERATED")]:

    subset = df[df["label"] == label]

    print(f"\n{name}")

    print("Mean token length:", round(
        subset["token_length"].mean(), 2
    ))

    print("Median token length:", round(
        subset["token_length"].median(), 2
    ))

    print(f"Truncated at {MAX_LENGTH}:", int(
        subset["truncated"].sum()
    ), "/", len(subset))

    print("Truncation rate:", round(
        100 * subset["truncated"].mean(), 2
    ), "%")

# ============================================================
# PERFORMANCE BY REVIEW LENGTH
# ============================================================

print("\n" + "=" * 55)
print("PERFORMANCE BY TOKEN LENGTH")
print("=" * 55)

df["length_group"] = pd.cut(
    df["token_length"],
    bins=[0, 64, 128, 256, 512, float("inf")],
    labels=[
        "1-64",
        "65-128",
        "129-256",
        "257-512",
        "513+",
    ],
    include_lowest=True,
)

for group, subset in df.groupby(
    "length_group", observed=True
):

    if len(subset) < 2 or subset["label"].nunique() < 2:
        continue

    print(f"\nLength group: {group}")
    print("Samples:", len(subset))

    for model_name, column in [
        ("LR", "lr_probability"),
        ("DistilBERT", "bert_probability"),
    ]:

        auc = roc_auc_score(
            subset["label"], subset[column]
        )

        accuracy = accuracy_score(
            subset["label"],
            (subset[column] >= THRESHOLD).astype(int),
        )

        print(
            f"{model_name}: "
            f"Accuracy={accuracy:.4f}, "
            f"ROC-AUC={auc:.4f}"
        )

# ============================================================
# MOST CONFIDENT WRONG PREDICTIONS
# ============================================================

for name, column in [
    ("logistic_regression", "lr_probability"),
    ("distilbert", "bert_probability"),
]:

    wrong = df[
        ((df["label"] == 1) & (df[column] < THRESHOLD))
        |
        ((df["label"] == 0) & (df[column] >= THRESHOLD))
    ].copy()

    wrong["confidence_in_wrong_class"] = np.where(
        wrong["label"] == 1,
        1 - wrong[column],
        wrong[column],
    )

    wrong = wrong.sort_values(
        "confidence_in_wrong_class",
        ascending=False,
    )

    columns = [
        "text",
        "label",
        column,
        "token_length",
        "truncated",
        "confidence_in_wrong_class",
    ]

    wrong[columns].head(50).to_csv(
        OUTPUT_DIR / f"{name}_top_errors.csv",
        index=False,
        encoding="utf-8-sig",
    )

print("\nCLASS DISTRIBUTION BY LENGTH")

print(
    pd.crosstab(
        df["length_group"],
        df["label"],
        normalize="index"
    ).round(3)
)
# ============================================================
# SAVE RESULTS
# ============================================================

df.to_csv(
    OUTPUT_DIR / "all_predictions.csv",
    index=False,
    encoding="utf-8-sig",
)

print("\nAnalysis completed.")
print("Results saved in:", OUTPUT_DIR)
