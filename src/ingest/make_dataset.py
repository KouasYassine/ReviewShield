from pathlib import Path
import pandas as pd

RAW_PATH = Path("data/raw/fake reviews dataset.csv")
OUT_PATH = Path("data/processed/reviews_labeled.parquet")

def main():
    df = pd.read_csv(RAW_PATH)
    df = df.rename(columns={"text_": "text"})

    df = df[["text", "label", "rating", "category"]].copy()

    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() > 0].drop_duplicates(subset=["text", "label"])
    print("Shape:", df.shape)
    print("Labels:\n", df["label"].value_counts(dropna=False))
    print("Rating stats:\n", df["rating"].describe())

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)
    print(f"Saved to {OUT_PATH}")

if __name__ == "__main__":
    main()
