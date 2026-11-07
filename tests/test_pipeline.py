"""End-to-end tests on a small synthetic dataset (no real data, no GPU needed).

    python -m pytest -q
"""
import numpy as np
import pandas as pd
import pytest

from src.common import binary_metrics, load_split, write_table
from src.models.review_zone import apply_zone, pick_thresholds, zone_report
from src.models.train_baseline import build_model

HUMAN = ["the zipper broke after a week but support replaced it",
         "my kid loves it, though the paint chipped fast",
         "arrived late and the box was dented, works fine though",
         "honestly smaller than the photos, returned it",
         "battery lasts two days which is ok for the price"]
GENERATED = ["this product is great and i love it very much",
             "i would highly recommend this product to anyone",
             "great quality product and works perfectly as described",
             "this is the best product i have ever purchased",
             "excellent product great value would buy again"]


def synthetic(n=400, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        label = i % 2
        base = rng.choice(GENERATED if label else HUMAN)
        rows.append({"text": f"{base} {rng.choice(['', 'really', 'item', 'today'])} #{i}", "label": label})
    return pd.DataFrame(rows)


@pytest.fixture
def split_dir(tmp_path):
    for i, name in enumerate(["train", "val", "test"]):
        write_table(synthetic(400 if name == "train" else 200, seed=i), tmp_path / f"{name}.csv")
    return tmp_path


def test_load_split_reads_csv(split_dir):
    df = load_split("val", split_dir)
    assert set(df.columns) >= {"text", "label"} and set(df["label"]) == {0, 1}


def test_baseline_learns_and_is_calibrated(split_dir):
    train, test = load_split("train", split_dir), load_split("test", split_dir)
    model = build_model(min_df=1).fit(train["text"], train["label"])
    p = model.predict_proba(test["text"])[:, 1]
    assert ((p >= 0) & (p <= 1)).all()
    assert binary_metrics(test["label"], p)["accuracy"] > 0.95


def test_explain_returns_ngrams(split_dir, tmp_path):
    import joblib
    from src.models.scorers import TfidfScorer

    train = load_split("train", split_dir)
    path = tmp_path / "m.joblib"
    joblib.dump(build_model(min_df=1).fit(train["text"], train["label"]), path)
    exp = TfidfScorer(path).explain("this product is great, i would recommend it")
    assert exp["towards_generated"], "generic praise should push towards 'generated'"


def test_thresholds_reach_target_precision():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 5000)
    s = np.clip(rng.normal(0.3 + 0.4 * y, 0.15), 0, 1)  # overlapping classes
    t_low, t_high = pick_thresholds(y, s, precision=0.95)
    assert t_low < t_high
    r = zone_report(y, s, t_low, t_high)
    assert r["flag_precision"] >= 0.95 and r["clear_precision"] >= 0.95
    assert 0 < r["review_rate"] < 1


def test_unreachable_precision_flags_nothing_instead_of_fake_result():
    # Regression test for the old bug: "precision >= 0.95" returned threshold 1.0 with recall 0.
    y = np.array([0, 1, 0, 1, 0, 1])
    s = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    t_low, t_high = pick_thresholds(y, s, precision=0.999)
    d = apply_zone(s, t_low, t_high)
    assert not (d == 1).any() or y[d == 1].mean() >= 0.999


def test_zone_decisions():
    d = apply_zone([0.05, 0.5, 0.95], 0.2, 0.8)
    assert d.tolist() == [0, -1, 1]
