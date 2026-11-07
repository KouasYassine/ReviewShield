"""Zero-shot robustness test on reviews from generators NOT seen in training.

The training data was generated with GPT-2. Here we score reviews written by
newer LLMs, without any retraining, to see whether the detectors generalise.

    # AiGen-FoodReview (labels 0 = human, 1 = generated)
    python -m src.models.evaluate_external --data data/external/aigen-foodreview/data/test.csv --name aigen_foodreview

    # Reviews you generated with DeepSeek (only AI texts, label column "CG")
    python -m src.models.evaluate_external --data data/external/modern_reviews/deepseek_test.csv \
        --name deepseek --text-col text_ --label-col label

Labels may be 0/1 or OR/CG. If the file only contains generated reviews,
only the detection rate (recall) is meaningful and is reported as such.
Exact duplicates with the training split are reported, so "unseen" is checked.
"""
import argparse
import json

import numpy as np
import pandas as pd

from src.common import LABEL_MAP, RESULTS_DIR, SPLIT_DIR, binary_metrics, load_split
from src.models.scorers import available_scorers, load_scorer


def normalise(s: str) -> str:
    return " ".join(str(s).lower().split())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--name", required=True, help="short name used for the output file")
    p.add_argument("--text-col", default="text")
    p.add_argument("--label-col", default="label")
    p.add_argument("--models", nargs="*", default=None)
    args = p.parse_args()

    df = pd.read_csv(args.data)[[args.text_col, args.label_col]].dropna()
    df.columns = ["text", "label"]
    if not pd.api.types.is_numeric_dtype(df["label"]):
        df["label"] = df["label"].map(LABEL_MAP)
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() > 0].reset_index(drop=True)

    train_texts = set(load_split("train", SPLIT_DIR)["text"].map(normalise))
    overlap = int(df["text"].map(normalise).isin(train_texts).sum())
    print(f"{args.name}: {len(df)} reviews {df['label'].value_counts().to_dict()} | overlap with train: {overlap}")

    result = {"dataset": args.data, "n": len(df), "class_counts": {int(k): int(v) for k, v in df["label"].value_counts().items()},
              "overlap_with_train": overlap, "models": {}}
    for name in args.models or available_scorers():
        scores = load_scorer(name)(df["text"])
        m = binary_metrics(df["label"], scores)
        m["mean_score_generated"] = float(np.mean(scores[df["label"] == 1])) if (df["label"] == 1).any() else None
        m["mean_score_human"] = float(np.mean(scores[df["label"] == 0])) if (df["label"] == 0).any() else None
        result["models"][name] = m
        print(f"  {name:10s} acc {m['accuracy']:.4f} | recall (AI caught) {m['recall']:.4f} "
              + (f"| roc-auc {m['roc_auc']:.4f}" if "roc_auc" in m else "| single-class file"))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"external_{args.name}.json"
    out.write_text(json.dumps(result, indent=2))
    print("Saved", out)


if __name__ == "__main__":
    main()
