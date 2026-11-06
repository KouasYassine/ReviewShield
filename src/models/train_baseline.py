from pathlib import Path
import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    average_precision_score,
)

TRAIN_PATH = Path("data/processed/train2.parquet")
TEST_PATH = Path("data/processed/test.parquet")
MODEL_PATH = Path("models/baseline_tfidf_lr.joblib")

def main():
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)

    X_train, y_train = train_df["text"], train_df["label"]
    X_test, y_test = test_df["text"], test_df["label"]


    model = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(
                analyzer='char_wb',
                ngram_range=(3, 5),
                min_df=10,
                max_features=60000,
                lowercase=True
            )),
            ("clf", LogisticRegression(
                max_iter=5000,
                class_weight='balanced',
                solver='saga',
                n_jobs=-1,
                random_state=42
            )),
        ]
    )
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Metrics
    pr_auc = average_precision_score(y_test, y_proba)
    print(f"PR-AUC: {pr_auc:.4f}\n")
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred), "\n")
    print(classification_report(y_test, y_pred, digits=4))

    # Save model
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"\nSaved model to {MODEL_PATH}")

if __name__ == "__main__":
    main()
