import pandas as pd
from pathlib import Path
import json

RAW_HC3_PATH = Path("data/raw/all.jsonl")
TRAIN_PATH = Path("data/processed/train2.parquet")

def main():
    if not RAW_HC3_PATH.exists():
        print(f"Fichier introuvable : {RAW_HC3_PATH}")
        return

    human_texts = []
    ai_texts = []

    with open(RAW_HC3_PATH, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 2000:
                break

            try:
                data = json.loads(line)
                if data.get('human_answers') and len(data['human_answers']) > 0:
                    human_texts.append(data['human_answers'][0])
                if data.get('chatgpt_answers') and len(data['chatgpt_answers']) > 0:
                    ai_texts.append(data['chatgpt_answers'][0])
            except json.JSONDecodeError:
                continue

    df_human = pd.DataFrame({
        "text": human_texts,
        "label": 0,
        "rating": 5,
        "category": "HC3_Human"
    })

    df_ai = pd.DataFrame({
        "text": ai_texts,
        "label": 1,
        "rating": 5,
        "category": "HC3_ChatGPT"
    })

    new_data = pd.concat([df_human, df_ai], ignore_index=True)
    new_data["text"] = new_data["text"].astype(str).str.strip()
    new_data = new_data[new_data["text"].str.len() > 10]

    if TRAIN_PATH.exists():
        df_existing = pd.read_parquet(TRAIN_PATH)
        df_final = pd.concat([df_existing, new_data], ignore_index=True)
        df_final = df_final.drop_duplicates(subset=["text"])
        df_final.to_parquet(TRAIN_PATH, index=False)
        print(f"Succès : {len(new_data)} lignes injectées dans {TRAIN_PATH.name}")
    else:
        new_data.to_parquet(TRAIN_PATH, index=False)
        print(f"Fichier créé : {TRAIN_PATH.name} avec {len(new_data)} lignes.")

if __name__ == "__main__":
    main()