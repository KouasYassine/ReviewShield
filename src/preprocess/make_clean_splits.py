
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

source = Path("data/processed/reviews_labeled.parquet")
output = Path("data/processed/clean")
output.mkdir(parents=True, exist_ok=True)

df = pd.read_parquet(source).copy()

df["label"] = df["label"].map({"OR": 0, "CG": 1})
df["text"] = df["text"].astype(str).str.strip()

df = df.dropna(subset=["label"])
df = df[df["text"].str.len() > 0]
df = df.drop_duplicates(subset=["text"])

train, temp = train_test_split(
    df,
    test_size=0.30,
    stratify=df["label"],
    random_state=42
)

val, test = train_test_split(
    temp,
    test_size=0.50,
    stratify=temp["label"],
    random_state=42
)

for name, data in [
    ("train", train),
    ("val", val),
    ("test", test)
]:
    path = output / f"{name}.parquet"
    data.to_parquet(path, index=False)

    print(name, len(data))
    print(data["label"].value_counts().to_dict())
