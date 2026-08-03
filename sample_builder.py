import pandas as pd
import random

categories = [
    "Amazon_Fashion",
    "Electronics",
    "Home_and_Kitchen",
    "Beauty_and_Personal_Care",
    "Sports_and_Outdoors"
]

def build_demo_sample(reviews_per_category=60, seed=None):
    if seed is not None:
        random.seed(seed)

    all_samples = []
    for category in categories:
        df = pd.read_parquet(f"data/{category}.parquet")
        low = df[df["rating"] <= 2]
        mid = df[df["rating"] == 3]
        high = df[df["rating"] >= 4]

        n_low = min(len(low), int(reviews_per_category * 0.35))
        n_mid = min(len(mid), int(reviews_per_category * 0.15))
        n_high = reviews_per_category - n_low - n_mid

        sampled = pd.concat([
            low.sample(n_low, random_state=seed),
            mid.sample(n_mid, random_state=seed),
            high.sample(n_high, random_state=seed)
        ])
        all_samples.append(sampled)

    demo_df = pd.concat(all_samples, ignore_index=True)
    demo_df = demo_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return demo_df


if __name__ == "__main__":
    sample = build_demo_sample()  
    print(f"Built demo sample: {len(sample)} reviews")
    print(sample["product_category"].value_counts())
    print(sample["rating"].describe())
    sample.to_parquet("data/demo_sample.parquet", index=False)
    print("Saved to data/demo_sample.parquet")