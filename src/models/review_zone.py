"""Selective classification: auto-decide confident cases, send the rest to a human.

Instead of one 0.5 threshold, we pick two thresholds on the VALIDATION split:
    score <= t_low            -> "Human"            (auto)
    score >= t_high           -> "AI-generated"     (auto)
    t_low < score < t_high    -> "Needs review"     (sent to a moderator)

t_high is the lowest threshold whose flagged reviews are at least `precision`
correct; t_low is the highest threshold whose cleared reviews are at least
`precision` correct. The review band is therefore as small as possible for the
requested error rate on automatic decisions. We then report, on the TEST split,
how many reviews still need a human and how accurate the automatic decisions are.

    python -m src.models.review_zone --model tfidf_lr --precision 0.99
    python -m src.models.review_zone --model distilbert --precision 0.99
"""
import argparse
import json

import numpy as np

from src.common import MODEL_DIR, RESULTS_DIR, SPLIT_DIR, load_split


def pick_thresholds(y, s, precision: float):
    """Return (t_low, t_high) on scores s for labels y (1 = generated)."""
    y, s = np.asarray(y), np.asarray(s)
    cand = np.unique(s)

    # t_high: smallest t such that P(y=1 | s >= t) >= precision (needs >= 1 flagged item)
    t_high = 1.0 + 1e-9  # default: flag nothing automatically
    for t in cand:
        flagged = y[s >= t]
        if len(flagged) and flagged.mean() >= precision:
            t_high = float(t)
            break

    # t_low: largest t such that P(y=0 | s <= t) >= precision
    t_low = -1e-9  # default: clear nothing automatically
    for t in cand[::-1]:
        cleared = y[s <= t]
        if len(cleared) and (1 - cleared.mean()) >= precision:
            t_low = float(t)
            break

    if t_low >= t_high:  # very separable data: both rules overlap, meet in the middle
        mid = 0.5 * (t_low + t_high)
        t_low, t_high = mid - 1e-12, mid
    return t_low, t_high


def apply_zone(s, t_low, t_high):
    """1 = AI-generated, 0 = human, -1 = needs review."""
    s = np.asarray(s)
    return np.where(s >= t_high, 1, np.where(s <= t_low, 0, -1))


def zone_report(y, s, t_low, t_high) -> dict:
    y = np.asarray(y)
    d = apply_zone(s, t_low, t_high)
    auto = d != -1
    flagged, cleared = d == 1, d == 0
    return {
        "n": int(len(y)),
        "review_rate": float((~auto).mean()),
        "auto_rate": float(auto.mean()),
        "auto_accuracy": float((d[auto] == y[auto]).mean()) if auto.any() else None,
        "flag_precision": float(y[flagged].mean()) if flagged.any() else None,
        "clear_precision": float(1 - y[cleared].mean()) if cleared.any() else None,
        "generated_caught_automatically": float((flagged & (y == 1)).sum() / max(1, (y == 1).sum())),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="tfidf_lr", choices=["tfidf_lr", "distilbert"])
    p.add_argument("--precision", type=float, default=0.99, help="target precision of automatic decisions")
    p.add_argument("--split-dir", default=str(SPLIT_DIR))
    args = p.parse_args()

    from src.models.scorers import load_scorer
    scorer = load_scorer(args.model)
    val, test = load_split("val", args.split_dir), load_split("test", args.split_dir)

    s_val, s_test = scorer(val["text"]), scorer(test["text"])
    t_low, t_high = pick_thresholds(val["label"], s_val, args.precision)
    report = {"model": args.model, "target_precision": args.precision,
              "thresholds": {"t_low": t_low, "t_high": t_high},
              "val": zone_report(val["label"], s_val, t_low, t_high),
              "test": zone_report(test["label"], s_test, t_low, t_high)}
    print(json.dumps(report, indent=2))

    (MODEL_DIR / f"review_zone_{args.model}.json").write_text(json.dumps(report, indent=2))
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"review_zone_{args.model}.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
