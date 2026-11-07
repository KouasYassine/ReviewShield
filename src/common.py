"""Shared paths, I/O and metrics. Run every script from the repository root."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)

ROOT = Path(__file__).resolve().parent.parent
RAW_CSV = ROOT / "data" / "raw" / "fake reviews dataset.csv"
LABELED = ROOT / "data" / "processed" / "reviews_labeled.parquet"
SPLIT_DIR = ROOT / "data" / "processed" / "clean"
MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

TFIDF_PATH = MODEL_DIR / "tfidf_lr_clean.joblib"
DISTILBERT_DIR = MODEL_DIR / "distilbert_clean"

# 0 = written by a human (OR, original), 1 = computer-generated (CG)
LABEL_MAP = {"OR": 0, "CG": 1}
SEED = 42


def read_table(path) -> pd.DataFrame:
    path = Path(path)
    return pd.read_csv(path) if path.suffix == ".csv" else pd.read_parquet(path)


def write_table(df: pd.DataFrame, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False) if path.suffix == ".csv" else df.to_parquet(path, index=False)


def load_split(name: str, split_dir=SPLIT_DIR) -> pd.DataFrame:
    """Load train / val / test, whichever of .parquet or .csv exists."""
    for ext in (".parquet", ".csv"):
        path = Path(split_dir) / f"{name}{ext}"
        if path.exists():
            df = read_table(path).dropna(subset=["text", "label"])
            df["text"] = df["text"].astype(str)
            df["label"] = df["label"].astype(int)
            return df.reset_index(drop=True)
    raise FileNotFoundError(f"No {name}.parquet or {name}.csv in {split_dir}. Run src.data.make_splits first.")


def binary_metrics(y_true, scores, threshold: float = 0.5) -> dict:
    y_true, scores = np.asarray(y_true), np.asarray(scores)
    pred = (scores >= threshold).astype(int)
    out = {"n": int(len(y_true)), "threshold": threshold,
           "accuracy": float(accuracy_score(y_true, pred)),
           "precision": float(precision_score(y_true, pred, zero_division=0)),
           "recall": float(recall_score(y_true, pred, zero_division=0)),
           "f1": float(f1_score(y_true, pred, zero_division=0)),
           "confusion_matrix": confusion_matrix(y_true, pred, labels=[0, 1]).tolist()}
    if len(np.unique(y_true)) == 2:  # ranking metrics need both classes
        out["roc_auc"] = float(roc_auc_score(y_true, scores))
        out["pr_auc"] = float(average_precision_score(y_true, scores))
    return out
