from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

IN_TRAIN = Path("data/processed/train.parquet")   # current 80%
OUT_TRAIN = Path("data/processed/train2.parquet") # new train
OUT_VAL = Path("data/processed/val.parquet")      # new validation

VAL_SIZE = 0.15  # 15% of the current train (=> overall 12% of total)

def main():
    df = pd.read_parquet(IN_TRAIN)

    train_df, val_df = train_test_split(
        df,
        test_size=VAL_SIZE,
        random_state=42,
        stratify=df["label"]
    )

    train_df.to_parquet(OUT_TRAIN, index=False)
    val_df.to_parquet(OUT_VAL, index=False)

    print("Saved:")
    print(" -", OUT_TRAIN, train_df.shape, train_df["label"].value_counts().to_dict())
    print(" -", OUT_VAL, val_df.shape, val_df["label"].value_counts().to_dict())

if __name__ == "__main__":
    main()
