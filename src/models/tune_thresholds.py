from pathlib import Path
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_curve

TRAIN_PATH = Path("data/processed/val.parquet")
MODEL_PATH = Path("models/baseline_tfidf_lr.joblib")
THRESHOLD_PATH = Path("models/thresholds.json")

TARGET_RECALL_CONSUMER = 0.95
TARGET_PRECISION_STRICT = 0.95
TARGET_PRECISION_BALANCED = 0.90  # <- corporate “85–90% confiant”: change to 0.85 if you want

BETA = 0.5  # F0.5 favors precision

def f_beta(p, r, beta: float):
    if p == 0 and r == 0:
        return 0.0
    b2 = beta * beta
    denom = (b2 * p + r)
    return 0.0 if denom == 0 else (1 + b2) * (p * r) / denom

def pick_max_threshold(precision, recall, thresholds, *, min_recall=None, min_precision=None):
    best = None
    for p, r, t in zip(precision, recall, thresholds):
        if min_recall is not None and r < min_recall:
            continue
        if min_precision is not None and p < min_precision:
            continue
        if best is None or t > best[0]:
            best = (t, p, r)
    return best  # (t, p, r) or None

def pick_max_fbeta(precision, recall, thresholds, beta: float):
    best = None
    for p, r, t in zip(precision, recall, thresholds):
        score = f_beta(float(p), float(r), beta)
        if best is None or score > best[0]:
            best = (score, t, p, r)
    return best  # (fbeta, t, p, r)

def main():
    train_df = pd.read_parquet(TRAIN_PATH)
    X_train = train_df["text"]
    y_train = train_df["label"]

    model = joblib.load(MODEL_PATH)
    y_scores = model.predict_proba(X_train)[:, 1]

    precision, recall, thresholds = precision_recall_curve(y_train, y_scores)
    thresholds = np.append(thresholds, 1.0)  # align

    # Consumer: strict recall
    cons = pick_max_threshold(precision, recall, thresholds, min_recall=TARGET_RECALL_CONSUMER)
    # Corporate strict: strict precision
    corp_strict = pick_max_threshold(precision, recall, thresholds, min_precision=TARGET_PRECISION_STRICT)
    # Corporate balanced: either F0.5 optimum OR precision >= 0.90
    corp_bal_f = pick_max_fbeta(precision, recall, thresholds, beta=BETA)
    corp_bal_p = pick_max_threshold(precision, recall, thresholds, min_precision=TARGET_PRECISION_BALANCED)

    results = {
        "consumer": {
            "threshold": float(cons[0]), "precision": float(cons[1]), "recall": float(cons[2]),
            "rule": f"max threshold with recall>={TARGET_RECALL_CONSUMER}"
        },
        "corporate_strict": {
            "threshold": None if corp_strict is None else float(corp_strict[0]),
            "precision": None if corp_strict is None else float(corp_strict[1]),
            "recall": None if corp_strict is None else float(corp_strict[2]),
            "rule": f"max threshold with precision>={TARGET_PRECISION_STRICT}"
        },
        "corporate_balanced_f0.5": {
            "threshold": float(corp_bal_f[1]), "precision": float(corp_bal_f[2]), "recall": float(corp_bal_f[3]),
            "f0.5": float(corp_bal_f[0]),
            "rule": f"maximize F{BETA}"
        },
        "corporate_balanced_precision90": {
            "threshold": None if corp_bal_p is None else float(corp_bal_p[0]),
            "precision": None if corp_bal_p is None else float(corp_bal_p[1]),
            "recall": None if corp_bal_p is None else float(corp_bal_p[2]),
            "rule": f"max threshold with precision>={TARGET_PRECISION_BALANCED}"
        },
    }

    THRESHOLD_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(THRESHOLD_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    print(json.dumps(results, indent=4))

if __name__ == "__main__":
    main()
