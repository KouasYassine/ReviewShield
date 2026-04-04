from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

TRAIN_PATH = Path("data/processed/train.parquet")
MODEL_PATH = Path("models/tfidf_lr_calibrated_train_full.joblib")

OUT_PATH = Path("models/quantile_zones.json")

UNCERTAIN_RATE = 0.15  # 15% needs_review budget
MIN_BAND = 0.05        # avoid "always unsure" degenerate band

def main():
    df = pd.read_parquet(TRAIN_PATH)
    X = df["text"]

    model = joblib.load(MODEL_PATH)
    scores = model.predict_proba(X)[:, 1]

    low_q = (1 - UNCERTAIN_RATE) / 2
    high_q = 1 - low_q

    t_low = float(np.quantile(scores, low_q))
    t_high = float(np.quantile(scores, high_q))

    # guardrail: ensure band isn't too tiny
    if (t_high - t_low) < MIN_BAND:
        mid = 0.5 * (t_low + t_high)
        t_low = float(mid - MIN_BAND / 2)
        t_high = float(mid + MIN_BAND / 2)

    results = {
        "mode": "corporate_quantile",
        "uncertain_rate": UNCERTAIN_RATE,
        "quantiles": {"low_q": low_q, "high_q": high_q},
        "thresholds": {"t_low": t_low, "t_high": t_high},
        "min_band": MIN_BAND,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(results, indent=4), encoding="utf-8")
    print(json.dumps(results, indent=4))

if __name__ == "__main__":
    main()
