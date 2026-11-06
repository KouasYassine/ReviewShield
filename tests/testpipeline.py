import joblib
import json
from pathlib import Path

def test_model_loading_and_prediction():
    model_path = Path("models/tfidf_lr_calibrated_train_full.joblib")
    model = joblib.load(model_path)

    sample_text = "This product is amazing and perfect, I love it!"
    proba = model.predict_proba([sample_text])[0][1]

    assert 0.0 <= proba <= 1.0

def test_thresholds_format():
    thresholds_path = Path("models/thresholds.json")
    thresholds = json.loads(thresholds_path.read_text(encoding="utf-8"))

    assert "corporate_strict" in thresholds
    assert "threshold" in thresholds["corporate_strict"]