"""In-domain test-set evaluation of every trained detector.

    python -m src.models.evaluate

Writes results/test_metrics.json, results/test_metrics.md (README table)
and results/test_predictions.csv.
"""
import argparse
import json

import pandas as pd

from src.common import RESULTS_DIR, SPLIT_DIR, binary_metrics, load_split
from src.models.scorers import available_scorers, load_scorer

PRETTY = {"tfidf_lr": "TF-IDF + logistic regression", "distilbert": "DistilBERT (fine-tuned)"}


def markdown_table(results: dict) -> str:
    lines = ["| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |", "|---|---|---|---|---|---|---|"]
    for name, m in results.items():
        lines.append(f"| {PRETTY.get(name, name)} | {m['accuracy']:.4f} | {m['precision']:.4f} | "
                     f"{m['recall']:.4f} | {m['f1']:.4f} | {m.get('roc_auc', float('nan')):.4f} | "
                     f"{m.get('pr_auc', float('nan')):.4f} |")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="*", default=None)
    p.add_argument("--split-dir", default=str(SPLIT_DIR))
    args = p.parse_args()

    names = args.models or available_scorers()
    if not names:
        raise SystemExit("No trained model found in models/. Train one first.")
    test = load_split("test", args.split_dir)
    print(f"Test reviews: {len(test)} | {test['label'].value_counts().to_dict()}")

    results, preds = {}, test[["text", "label"]].copy()
    for name in names:
        scores = load_scorer(name)(test["text"])
        preds[f"{name}_score"] = scores
        results[name] = binary_metrics(test["label"], scores)
        print(f"{name:10s} acc {results[name]['accuracy']:.4f} f1 {results[name]['f1']:.4f} "
              f"pr-auc {results[name]['pr_auc']:.4f} | confusion {results[name]['confusion_matrix']}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "test_metrics.json").write_text(json.dumps(results, indent=2))
    (RESULTS_DIR / "test_metrics.md").write_text(markdown_table(results) + "\n")
    preds.to_csv(RESULTS_DIR / "test_predictions.csv", index=False)
    print("\n" + markdown_table(results))


if __name__ == "__main__":
    main()
