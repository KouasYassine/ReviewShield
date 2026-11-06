
import numpy as np
import pandas as pd
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    average_precision_score,
    roc_auc_score
)

# ============================================================
# 1. CONFIGURATION
# ============================================================

MODEL_PATH = "models/distilbert_clean"
TEST_PATH = "data/processed/clean/test.parquet"

THRESHOLD = 0.5
BATCH_SIZE = 16
MAX_LENGTH = 128

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

# ============================================================
# 2. LOAD MODEL AND TOKENIZER
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH
).to(device)

model.eval()

# ============================================================
# 3. LOAD TEST DATA
# ============================================================

df = pd.read_parquet(TEST_PATH)
df = df.dropna(subset=["text", "label"])

texts = df["text"].astype(str).tolist()
y_true = df["label"].astype(int).to_numpy()

print("Test samples:", len(df))
print("Class distribution:", df["label"].value_counts().to_dict())

# ============================================================
# 4. MODEL INFERENCE
# ============================================================

scores = []

with torch.inference_mode():

    for start in range(0, len(texts), BATCH_SIZE):

        batch_texts = texts[start:start + BATCH_SIZE]

        batch = tokenizer(
            batch_texts,
            truncation=True,
            padding=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        ).to(device)

        logits = model(**batch).logits

        probabilities = torch.softmax(
            logits, dim=-1
        )[:, 1]

        scores.extend(
            probabilities.cpu().numpy()
        )

scores = np.asarray(scores)

# ============================================================
# 5. FIXED DECISION THRESHOLD
# ============================================================

y_pred = (scores >= THRESHOLD).astype(int)

# ============================================================
# 6. EVALUATION METRICS
# ============================================================

print("\n========== TEST RESULTS ==========")

print(f"\nThreshold: {THRESHOLD}")

print(
    "Accuracy:",
    round(accuracy_score(y_true, y_pred), 4)
)

print(
    "Precision:",
    round(precision_score(y_true, y_pred, zero_division=0), 4)
)

print(
    "Recall:",
    round(recall_score(y_true, y_pred, zero_division=0), 4)
)

print(
    "F1:",
    round(f1_score(y_true, y_pred, zero_division=0), 4)
)

print(
    "ROC-AUC:",
    round(roc_auc_score(y_true, scores), 4)
)

print(
    "PR-AUC:",
    round(average_precision_score(y_true, scores), 4)
)

print("\nCONFUSION MATRIX")
print(confusion_matrix(y_true, y_pred))

print("\nCLASSIFICATION REPORT")
print(
    classification_report(
        y_true,
        y_pred,
        target_names=["Genuine", "Generated"],
        digits=4,
        zero_division=0
    )
)

# ============================================================
# 7. PREDICTION STATISTICS
# ============================================================

print("\nPREDICTION STATISTICS")

print("Predicted generated:", int(y_pred.sum()))
print("Actual generated:", int(y_true.sum()))

print("\nMean predicted P(generated):")

for label, name in [(0, "Genuine"), (1, "Generated")]:
    print(
        f"{name}: {scores[y_true == label].mean():.6f}"
    )
