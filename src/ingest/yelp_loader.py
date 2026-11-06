import pandas as pd
from pathlib import Path
import json

def process_yelp_data(input_path: str, output_path: str):
    input_file = Path(input_path)
    out_file = Path(output_path)

    if not input_file.exists():
        return

    texts = []
    ratings = []

    with open(input_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 5000:
                break
            data = json.loads(line)
            texts.append(data.get('text', ''))
            ratings.append(data.get('stars', 5))

    df_new = pd.DataFrame({
        "text": texts,
        "label": 0,
        "rating": ratings,
        "category": "Yelp_Human"
    })

    df_new["text"] = df_new["text"].astype(str).str.strip()
    df_new = df_new[df_new["text"].str.len() > 10]

    if out_file.exists():
        df_existing = pd.read_parquet(out_file)
        df_final = pd.concat([df_existing, df_new], ignore_index=True)
        df_final = df_final.drop_duplicates(subset=["text"])
    else:
        df_final = df_new

    out_file.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_parquet(out_file, index=False)

if __name__ == "__main__":
    process_yelp_data(
        'data/raw/yelp_academic_dataset_review.json',
        'data/processed/train2.parquet'
    )