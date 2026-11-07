"""One interface for both detectors: score(texts) -> P(generated) in [0, 1].

PyTorch / transformers are imported only when the DistilBERT scorer is used,
so the TF-IDF model, the app and the tests run without them.
"""
import json
from pathlib import Path

import joblib
import numpy as np

from src.common import DISTILBERT_DIR, TFIDF_PATH


class TfidfScorer:
    name = "tfidf_lr"

    def __init__(self, path=TFIDF_PATH):
        self.model = joblib.load(path)
        self.pos = list(self.model.classes_).index(1)

    def __call__(self, texts):
        return self.model.predict_proba(list(texts))[:, self.pos]

    def _pipelines(self):
        # Calibrated model = one TF-IDF + LR pipeline per CV fold.
        if hasattr(self.model, "calibrated_classifiers_"):
            return [c.estimator for c in self.model.calibrated_classifiers_]
        return [self.model]

    def explain(self, text: str, k: int = 8):
        """Words / bigrams that pushed this review towards 'generated' (+) or 'human' (-).
        Contribution = tf-idf weight x logistic-regression coefficient, averaged over folds."""
        totals = {}
        pipes = self._pipelines()
        for pipe in pipes:
            vec, lr = pipe.named_steps["tfidf"], pipe.named_steps["lr"]
            x = vec.transform([text]).tocoo()
            vocab = vec.get_feature_names_out()
            sign = 1.0 if list(lr.classes_).index(1) == 1 else -1.0
            for j, w in zip(x.col, x.data):
                totals[vocab[j]] = totals.get(vocab[j], 0.0) + float(sign * w * lr.coef_[0, j]) / len(pipes)
        ranked = sorted(totals.items(), key=lambda kv: kv[1])
        return {"towards_generated": [t for t in ranked[::-1] if t[1] > 0][:k],
                "towards_human": [t for t in ranked if t[1] < 0][:k]}


class DistilBertScorer:
    name = "distilbert"

    def __init__(self, path=DISTILBERT_DIR, max_length=None, batch_size=32, device=None):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path).to(self.device).eval()
        # Guard against a silently swapped label order.
        self.pos = self.model.config.label2id.get("GENERATED", 1)
        if max_length is None:
            # Use the length the model was trained with (models trained before
            # training_args.json existed used the old default of 128 tokens).
            args_file = Path(path) / "training_args.json"
            max_length = json.loads(args_file.read_text()).get("length", 128) if args_file.exists() else 128
        self.max_length, self.batch_size = max_length, batch_size

    def __call__(self, texts):
        texts, scores = list(texts), []
        with self.torch.inference_mode():
            for i in range(0, len(texts), self.batch_size):
                batch = self.tokenizer(texts[i:i + self.batch_size], truncation=True, padding=True,
                                       max_length=self.max_length, return_tensors="pt").to(self.device)
                logits = self.model(**batch).logits.float()
                scores.extend(self.torch.softmax(logits, dim=-1)[:, self.pos].cpu().tolist())
        return np.asarray(scores)


def available_scorers():
    """Names of models that exist on disk."""
    names = []
    if Path(TFIDF_PATH).exists():
        names.append("tfidf_lr")
    if (Path(DISTILBERT_DIR) / "config.json").exists():
        names.append("distilbert")
    return names


def load_scorer(name: str, **kwargs):
    return {"tfidf_lr": TfidfScorer, "distilbert": DistilBertScorer}[name](**kwargs)
