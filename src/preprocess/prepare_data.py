from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

IN_PATH = Path("data/processed/reviews_labeled.parquet")
OUT_TRAIN = Path("data/processed/train.parquet")
OUT_TEST = Path("data/processed/test.parquet")

LABEL_MAP = {"OR": 0, "CG": 1}

def main():
    df = pd.read_parquet(IN_PATH)

    # rename text column and map labels
    df = df.rename(columns={"text_": "text"})
    df["label"] = df["label"].map(LABEL_MAP)

    # basic cleanup
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() > 0].drop_duplicates(subset=["text"])

    # stratified split so classes stay balanced
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["label"]
    )

    train_df.to_parquet(OUT_TRAIN, index=False)
    test_df.to_parquet(OUT_TEST, index=False)

    print("Saved:")
    print(" -", OUT_TRAIN, train_df.shape, "label dist:", train_df["label"].value_counts().to_dict())
    print(" -", OUT_TEST, test_df.shape, "label dist:", test_df["label"].value_counts().to_dict())

if __name__ == "__main__":
    main()
