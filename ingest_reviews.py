import pandas as pd
from db import save_review


def ingest_demo_sample(filepath="data/demo_sample.parquet"):
    """Load the demo sample and insert every row into the reviews table.

    This populates raw source data only — no classification happens
    here. Classification is a separate, later step, matching the
    real-world pattern of data ingestion being decoupled from
    processing.

    Parameters
    ----------
    filepath : str, optional
        Path to the Parquet file containing the demo sample.

    Returns
    -------
    int
        Number of reviews successfully ingested.
    """
    df = pd.read_parquet(filepath)
    count = 0

    for _, row in df.iterrows():
        save_review(
            review_id=str(row["review_id"]),
            product_category=row["product_category"],
            product_title=row.get("product_title", "Unknown Product"),
            review_text=row["text"],
            rating=row["rating"],
            review_date=row["review_date"],
            quarter=row["quarter"],
            verified_purchase=bool(row.get("verified_purchase", False))
        )
        count += 1

    return count


if __name__ == "__main__":
    n = ingest_demo_sample()
    print(f"Ingested {n} reviews into the reviews table")