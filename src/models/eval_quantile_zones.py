from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

TEST_PATH = Path("data/processed/test.parquet")
MODEL_PATH = Path("models/tfidf_lr_calibrated_train_full.joblib")
ZONES_PATH = Path("models/quantile_zones.json")

def main():
    test_df = pd.read_parquet(TEST_PATH)
    X = test_df["text"]
    y = test_df["label"]

    model = joblib.load(MODEL_PATH)
    zones = json.loads(ZONES_PATH.read_text(encoding="utf-8"))
    t_low = zones["thresholds"]["t_low"]
    t_high = zones["thresholds"]["t_high"]

    scores = model.predict_proba(X)[:, 1]

    # Decisions: 0=real, 1=fake, -1=needs_review
    decisions = []
    for s in scores:
        if s < t_low:
            decisions.append(0)
        elif s > t_high:
            decisions.append(1)
        else:
            decisions.append(-1)

    test_df = test_df.copy()
    test_df["score"] = scores
    test_df["decision"] = decisions

    n = len(test_df)
    n_review = int((test_df["decision"] == -1).sum())
    n_auto = n - n_review

    print(f"Total: {n}")
    print(f"Needs review: {n_review} ({n_review/n:.2%})")
    print(f"Auto-decided: {n_auto} ({n_auto/n:.2%})\n")

    # Evaluate only auto-decided cases
    auto = test_df[test_df["decision"] != -1]
    y_auto = auto["label"]
    y_pred_auto = auto["decision"]

    print("=== Auto-decided performance (excluding needs_review) ===")
    print("Confusion matrix:\n", confusion_matrix(y_auto, y_pred_auto), "\n")
    print(classification_report(y_auto, y_pred_auto, digits=4))

    # Precision inside the high-confidence fake zone
    flagged = test_df[test_df["decision"] == 1]
    if len(flagged) > 0:
        precision_flag = (flagged["label"] == 1).mean()
        print(f"\nFlag zone size: {len(flagged)} ({len(flagged)/n:.2%})")
        print(f"Flag zone precision: {precision_flag:.4f}")
    else:
        print("\nNo samples in flag zone (t_high too high).")

if __name__ == "__main__":
    main()
