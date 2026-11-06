import pandas as pd

def load_and_clean_yelp(json_input, parquet_output):
    reader = pd.read_json(json_input, lines=True, chunksize=50000)
    for chunk in reader:
        filtered = chunk[['user_id', 'business_id', 'stars', 'text', 'date']]
        filtered.to_parquet(parquet_output, engine='pyarrow', index=False)
        break

if __name__ == "__main__":
    load_and_clean_yelp('data/raw/review.json', 'data/processed/reviews.parquet')