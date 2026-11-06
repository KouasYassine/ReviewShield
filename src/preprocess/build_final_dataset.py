import pandas as pd
import json
from pathlib import Path

YELP_RAW = Path("data/raw/yelp_academic_dataset_review.json")
HC3_RAW = Path("data/raw/all.jsonl")
DAIGT_RAW = Path("data/raw/daigt_v2.csv")
FINAL_TRAIN_PATH = Path("data/processed/multisource_train.parquet")
def load_yelp_sample(n=25000):
    texts = []
    try:
        with open(YELP_RAW, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= n: break
                data = json.loads(line)
                texts.append(data['text'])
        return pd.DataFrame({"text": texts, "label": 0, "category": "Yelp_Human"})
    except Exception:
        return pd.DataFrame()

def load_hc3_sample():
    h_txt, a_txt = [], []
    try:
        with open(HC3_RAW, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                if data.get('human_answers'): h_txt.extend(data['human_answers'])
                if data.get('chatgpt_answers'): a_txt.extend(data['chatgpt_answers'])

        df_h = pd.DataFrame({"text": h_txt, "label": 0, "category": "HC3_Human"})
        df_a = pd.DataFrame({"text": a_txt, "label": 1, "category": "HC3_ChatGPT"})
        return pd.concat([df_h, df_a])
    except Exception:
        return pd.DataFrame()

def load_daigt_sample():
    if not DAIGT_RAW.exists():
        return pd.DataFrame()

    df = pd.read_csv(DAIGT_RAW)
    df = df[['text', 'label']].copy()
    df['category'] = 'DAIGT_Mixed'
    return df

def main():
    df_yelp = load_yelp_sample(25000)
    df_hc3 = load_hc3_sample()
    df_daigt = load_daigt_sample()

    df_final = pd.concat([df_yelp, df_hc3, df_daigt], ignore_index=True)
    df_final = df_final.drop_duplicates(subset=["text"])
    df_final = df_final[df_final["text"].str.len() > 50]

    df_final.to_parquet(FINAL_TRAIN_PATH, index=False)

    print(df_final["label"].value_counts())
    print(df_final["category"].value_counts())

if __name__ == "__main__":
    main()