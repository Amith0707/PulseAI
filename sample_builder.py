import pandas as pd
import random

categories = [
    "Amazon_Fashion",
    "Electronics",
    "Home_and_Kitchen",
    "Beauty_and_Personal_Care",
    "Sports_and_Outdoors"
]

WINDOW_START = pd.Timestamp("2022-01-01")
WINDOW_END = pd.Timestamp("2022-06-30")
Q1_END = pd.Timestamp("2022-03-31")  # boundary between Q1 (Jan-Mar) and Q2 (Apr-Jun)


def load_windowed_category(category):
    """Load one category's Parquet file, filtered to the Jan-Jun 2022
    window, with a quarter label attached.

    Parameters
    ----------
    category : str
        One of the 5 product category names.

    Returns
    -------
    pandas.DataFrame
        Rows within [WINDOW_START, WINDOW_END], with a new ``quarter``
        column ("Q1 2022" or "Q2 2022").
    """
    df = pd.read_parquet(f"data/{category}.parquet")
    df["review_date"] = pd.to_datetime(df["timestamp"], unit="ms")
    windowed = df[(df["review_date"] >= WINDOW_START) & (df["review_date"] <= WINDOW_END)].copy()
    windowed["quarter"] = windowed["review_date"].apply(
        lambda d: "Q1 2022" if d <= Q1_END else "Q2 2022"
    )
    return windowed


def build_demo_sample(reviews_per_category=200, seed=None):
    """Build a deliberate, mixed, quarter-labeled demo sample from the
    Jan-Jun 2022 window.

    Parameters
    ----------
    reviews_per_category : int, optional
        Number of reviews to select per product category (default 200,
        giving 1000 total across 5 categories).
    seed : int, optional
        Random seed for reproducibility (default None = fresh random
        sample each run).

    Returns
    -------
    pandas.DataFrame
        The combined, quarter-labeled, deliberately-sampled demo set.
    """
    if seed is not None:
        random.seed(seed)

    all_samples = []
    for category in categories:
        df = load_windowed_category(category)

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
    sample["review_id"] = sample["asin"] + "_" + sample["user_id"].astype(str)
    print(f"Built demo sample: {len(sample)} reviews")
    print(f"Unique review_ids: {sample['review_id'].nunique()}")
    print(sample["product_category"].value_counts())
    print(sample["quarter"].value_counts())
    print(sample["rating"].describe())
    sample.to_parquet("data/demo_sample.parquet", index=False)
    print("Saved to data/demo_sample.parquet")