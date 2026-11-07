"""Stratified 70 / 15 / 15 train / val / test split, with exact duplicates removed
first so the same review can never appear in two splits.

    python -m src.data.make_splits
"""
from sklearn.model_selection import train_test_split

from src.common import LABEL_MAP, LABELED, SEED, SPLIT_DIR, read_table, write_table


def main():
    df = read_table(LABELED)
    df["label"] = df["label"].map(LABEL_MAP)
    df["text"] = df["text"].astype(str).str.strip()
    df = df.dropna(subset=["label"])
    df = df[df["text"].str.len() > 0].drop_duplicates(subset=["text"])
    df["label"] = df["label"].astype(int)

    train, temp = train_test_split(df, test_size=0.30, stratify=df["label"], random_state=SEED)
    val, test = train_test_split(temp, test_size=0.50, stratify=temp["label"], random_state=SEED)

    for name, part in [("train", train), ("val", val), ("test", test)]:
        write_table(part, SPLIT_DIR / f"{name}.parquet")
        print(f"{name:5s} {len(part):6d}  {part['label'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
