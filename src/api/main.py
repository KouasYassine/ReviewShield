"""REST API.   uvicorn src.api.main:app --reload   then POST /predict {"text": "..."}"""
import json
from functools import lru_cache

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.common import MODEL_DIR
from src.models.review_zone import apply_zone
from src.models.scorers import available_scorers, load_scorer

app = FastAPI(title="ReviewShield")
LABELS = {1: "AI-generated", 0: "Human", -1: "Needs review"}


class ReviewRequest(BaseModel):
    text: str
    model: str = "tfidf_lr"


@lru_cache
def scorer(name):
    return load_scorer(name)


@lru_cache
def zone(name):
    path = MODEL_DIR / f"review_zone_{name}.json"
    return json.loads(path.read_text())["thresholds"] if path.exists() else {"t_low": 0.5, "t_high": 0.5}


@app.get("/models")
def models():
    return available_scorers()


@app.post("/predict")
def predict(req: ReviewRequest):
    if req.model not in available_scorers():
        raise HTTPException(404, f"Model {req.model!r} is not trained. Available: {available_scorers()}")
    p = float(scorer(req.model)([req.text])[0])
    t = zone(req.model)
    return {"model": req.model, "probability_generated": p, **t,
            "decision": LABELS[int(apply_zone([p], t["t_low"], t["t_high"])[0])]}
