"""Raw Kaggle CSV -> data/processed/reviews_labeled.parquet

Dataset: "Fake Reviews Dataset" (Salminen et al., 2022): 40K Amazon reviews,
half original (OR, human) and half computer-generated (CG, GPT-2).
Put the file at: data/raw/fake reviews dataset.csv

    python -m src.data.make_dataset
"""
import pandas as pd

from src.common import LABELED, RAW_CSV, write_table


def main():
    df = pd.read_csv(RAW_CSV).rename(columns={"text_": "text"})
    df = df[["text", "label", "rating", "category"]].copy()
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() > 0].drop_duplicates(subset=["text", "label"])

    print("Shape:", df.shape)
    print("Labels:", df["label"].value_counts().to_dict())
    write_table(df, LABELED)
    print("Saved", LABELED)


if __name__ == "__main__":
    main()
