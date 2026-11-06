from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import json
from pathlib import Path

app = FastAPI()

MODEL_PATH = Path("models/tfidf_lr_calibrated_train_full.joblib")
ZONES_PATH = Path("models/quantile_zones.json")

model = joblib.load(MODEL_PATH)
zones = json.loads(ZONES_PATH.read_text(encoding="utf-8"))

class ReviewRequest(BaseModel):
    text: str

@app.post("/predict")
def predict_fraud(request: ReviewRequest):
    proba = float(model.predict_proba([request.text])[0][1])

    t_low = zones["thresholds"]["t_low"]
    t_high = zones["thresholds"]["t_high"]

    if proba < t_low:
        decision = "Real"
    elif proba > t_high:
        decision = "Fake"
    else:
        decision = "Needs Review"

    return {
        "probability": proba,
        "t_low": t_low,
        "t_high": t_high,
        "decision": decision
    }