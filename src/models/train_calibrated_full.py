from pathlib import Path
import joblib
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    average_precision_score,
)

TRAIN2_PATH = Path("data/processed/train2.parquet")
VAL_PATH = Path("data/processed/val.parquet")
TEST_PATH = Path("data/processed/test.parquet")

OUT_MODEL_PATH = Path("models/tfidf_lr_calibrated_train_full.joblib")

def build_pipeline():
    return Pipeline(
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

def main():

    train2_df = pd.read_parquet(TRAIN2_PATH)
    val_df = pd.read_parquet(VAL_PATH)
    test_df = pd.read_parquet(TEST_PATH)


    train_full = pd.concat([train2_df, val_df], ignore_index=True)

    X_train_full = train_full["text"]
    y_train_full = train_full["label"]

    X_test = test_df["text"]
    y_test = test_df["label"]

    print(f"train_full shape: {train_full.shape} | label dist: {train_full['label'].value_counts().to_dict()}")
    print(f"test shape:       {test_df.shape} | label dist: {test_df['label'].value_counts().to_dict()}")

    base_pipeline = build_pipeline()


    calibrated = CalibratedClassifierCV(
        estimator=base_pipeline,
        method="sigmoid",
        cv=5
    )

    calibrated.fit(X_train_full, y_train_full)


    y_pred = calibrated.predict(X_test)
    y_proba = calibrated.predict_proba(X_test)[:, 1]

    pr_auc = average_precision_score(y_test, y_proba)
    print("\n=== Test (calibrated, trained on train_full with CV) ===")
    print(f"PR-AUC: {pr_auc:.4f}")
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred, digits=4))


    OUT_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(calibrated, OUT_MODEL_PATH)
    print(f"\nSaved calibrated model to {OUT_MODEL_PATH}")

if __name__ == "__main__":
    main()
