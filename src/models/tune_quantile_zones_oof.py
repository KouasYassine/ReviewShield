from pathlib import Path
import json
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

TRAIN_FULL_PATH = Path("data/processed/train.parquet")
OUT_PATH = Path("models/quantile_zones_oof.json")

UNCERTAIN_RATE = 0.15
MIN_BAND = 0.05
FOLDS = 5
SEED = 42

def build_model():
    base_pipeline = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.9,
                stop_words="english",
            )),
            ("lr", LogisticRegression(max_iter=2000)),
        ]
    )

    calibrated = CalibratedClassifierCV(
        estimator=base_pipeline,
        method="sigmoid",
        cv=3
    )

    return calibrated

def main():
    df = pd.read_parquet(TRAIN_FULL_PATH)
    X = df["text"]
    y = df["label"].values

    skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    oof_scores = np.zeros(len(df))

    for train_idx, val_idx in skf.split(X, y):
        model = build_model()
        model.fit(X.iloc[train_idx], y[train_idx])
        oof_scores[val_idx] = model.predict_proba(X.iloc[val_idx])[:, 1]

    low_q = (1 - UNCERTAIN_RATE) / 2
    high_q = 1 - low_q

    t_low = float(np.quantile(oof_scores, low_q))
    t_high = float(np.quantile(oof_scores, high_q))

    if (t_high - t_low) < MIN_BAND:
        mid = 0.5 * (t_low + t_high)
        t_low = float(mid - MIN_BAND / 2)
        t_high = float(mid + MIN_BAND / 2)

    results = {
        "mode": "quantile_oof",
        "uncertain_rate": UNCERTAIN_RATE,
        "quantiles": {"low_q": low_q, "high_q": high_q},
        "thresholds": {"t_low": t_low, "t_high": t_high},
        "min_band": MIN_BAND,
        "folds": FOLDS
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(results, indent=4), encoding="utf-8")

    print(json.dumps(results, indent=4))

if __name__ == "__main__":
    main()
