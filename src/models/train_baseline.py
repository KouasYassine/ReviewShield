"""TF-IDF + logistic regression baseline, with calibrated probabilities.

Calibration (Platt scaling with 5-fold CV on the training split) makes the
scores usable as probabilities, which the human-review zone relies on.

    python -m src.models.train_baseline
"""
import argparse
import json

import joblib
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.common import RESULTS_DIR, SEED, SPLIT_DIR, TFIDF_PATH, binary_metrics, load_split


def build_model(calibrate: bool = True, min_df: int = 2):
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=min_df, max_df=0.9,
                                  sublinear_tf=True)),
        ("lr", LogisticRegression(max_iter=2000, C=4.0, random_state=SEED)),
    ])
    return CalibratedClassifierCV(pipe, method="sigmoid", cv=5) if calibrate else pipe


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split-dir", default=str(SPLIT_DIR))
    p.add_argument("--out", default=str(TFIDF_PATH))
    args = p.parse_args()

    train = load_split("train", args.split_dir)
    val = load_split("val", args.split_dir)
    print(f"Train {len(train)} | Val {len(val)}")

    model = build_model()
    model.fit(train["text"], train["label"])
    metrics = binary_metrics(val["label"], model.predict_proba(val["text"])[:, 1])
    print("Validation:", json.dumps({k: v for k, v in metrics.items() if k != "confusion_matrix"}, indent=2))

    joblib.dump(model, args.out)
    out = RESULTS_DIR / "tfidf_val_metrics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2))
    print("Saved model to", args.out)


if __name__ == "__main__":
    main()
