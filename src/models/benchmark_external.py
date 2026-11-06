"""Zero-shot external benchmark: TF-IDF/LR vs DistilBERT on AiGen-FoodReview.
Run from the root of ReviewShield. External test labels must be 0=human, 1=generated.
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def metrics(y, score):
    pred = (score >= 0.5).astype(int)
    return {
        'accuracy': float(accuracy_score(y, pred)),
        'precision': float(precision_score(y, pred, zero_division=0)),
        'recall': float(recall_score(y, pred, zero_division=0)),
        'f1': float(f1_score(y, pred, zero_division=0)),
        'roc_auc': float(roc_auc_score(y, score)),
        'pr_auc': float(average_precision_score(y, score)),
        'confusion_matrix': confusion_matrix(y, pred, labels=[0, 1]).tolist()
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--train', default='data/processed/clean/train.parquet')
    p.add_argument('--external', default='data/external/aigen-foodreview/data/test.csv')
    p.add_argument('--transformer', default='models/distilbert_clean')
    p.add_argument('--lr-model', default='models/tfidf_lr_clean.joblib')
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--max-length', type=int, default=128)
    p.add_argument('--output', default='results/aigen_foodreview_benchmark.json')
    args = p.parse_args()

    train = pd.read_parquet(args.train)[['text', 'label']].dropna()
    external = pd.read_csv(args.external)[['text', 'label']].dropna()
    for name, frame in [('train', train), ('external', external)]:
        frame['label'] = frame['label'].astype(int)
        assert set(frame['label'].unique()) == {0, 1}, f'{name}: unexpected labels'
        frame['text'] = frame['text'].astype(str).str.strip()
    external = external[external['text'].str.len() > 0].copy()
    train = train[train['text'].str.len() > 0].copy()

    norm = lambda s: ' '.join(s.lower().split())
    overlap = len(set(train['text'].map(norm)) & set(external['text'].map(norm)))
    print(f'Train: {len(train)} | External: {len(external)} | normalized text overlap: {overlap}')
    print('External class counts:', external['label'].value_counts().to_dict())
    if overlap:
        print('WARNING: Exact normalized overlaps detected; examine before reporting this as fully unseen data.')

    lr_path = Path(args.lr_model)
    if lr_path.exists():
        print('Loading existing clean LR:', lr_path)
        lr = joblib.load(lr_path)
    else:
        print('Training TF-IDF + Logistic Regression on CLEAN train split...')
        lr = Pipeline([
            ('tfidf', TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2, max_df=0.9, stop_words='english')),
            ('clf', LogisticRegression(max_iter=2000)),
        ])
        lr.fit(train['text'], train['label'])
        lr_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(lr, lr_path)
        print('Saved:', lr_path)
    lr_score = lr.predict_proba(external['text'])[:, list(lr.classes_).index(1)]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Transformer inference device:', device)
    tokenizer = AutoTokenizer.from_pretrained(args.transformer)
    model = AutoModelForSequenceClassification.from_pretrained(args.transformer).to(device).eval()
    assert model.config.label2id.get('GENERATED') == 1, 'Check positive label mapping in saved model'
    bert_scores = []
    texts = external['text'].tolist()
    with torch.inference_mode():
        for i in range(0, len(texts), args.batch_size):
            batch = tokenizer(texts[i:i+args.batch_size], truncation=True, padding=True,
                              max_length=args.max_length, return_tensors='pt').to(device)
            logits = model(**batch).logits
            bert_scores.extend(torch.softmax(logits.float(), dim=-1)[:, 1].cpu().tolist())
            if (i // args.batch_size + 1) % 50 == 0:
                print(f'Processed {min(i + args.batch_size, len(texts))}/{len(texts)} reviews')

    y = external['label'].to_numpy()
    result = {'dataset': str(args.external), 'n': len(y), 'threshold': 0.5,
              'overlap_normalized_texts': overlap,
              'logistic_regression': metrics(y, np.asarray(lr_score)),
              'distilbert': metrics(y, np.asarray(bert_scores))}
    for k in ('logistic_regression', 'distilbert'):
        print('\n', k.upper(), '\n', json.dumps(result[k], indent=2))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print('\nSaved results:', out)


if __name__ == '__main__':
    main()
