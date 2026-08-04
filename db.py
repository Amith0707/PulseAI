import os
import psycopg2
from dotenv import load_dotenv
from psycopg2.extras import RealDictCursor

load_dotenv()

def get_connection():
    """Create and return a PostgreSQL database connection.

    Returns
    -------
    psycopg2.extensions.connection
        An active PostgreSQL connection object.
    """
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=os.getenv("DB_PORT", "5432"),
        dbname=os.getenv("DB_NAME", "pulseai"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD", "")
    )


def init_db():
    """Initialize the reviews and classifications tables if they don't
    already exist.

    ``reviews`` holds raw source data — no AI involvement, just the
    original customer review as it exists in the source dataset.
    ``classifications`` holds AI-generated outputs, referencing
    ``reviews`` by ``review_id`` via a foreign key, so a classification
    can never exist for a review that hasn't actually been ingested.
    """
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS reviews (
            review_id TEXT PRIMARY KEY,
            product_category TEXT NOT NULL,
            product_title TEXT,
            review_text TEXT NOT NULL,
            rating FLOAT,
            review_date TIMESTAMP,
            quarter TEXT,
            verified_purchase BOOLEAN,
            ingested_at TIMESTAMP DEFAULT NOW()
        );
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS classifications (
            id SERIAL PRIMARY KEY,
            review_id TEXT REFERENCES reviews(review_id),
            feedback_category TEXT NOT NULL,
            sentiment TEXT NOT NULL,
            urgency TEXT NOT NULL,
            reasoning TEXT NOT NULL,
            classified_at TIMESTAMP DEFAULT NOW()
        );
    """)
    conn.commit()
    cur.close()
    conn.close()


def save_review(review_id, product_category, product_title, review_text,
                 rating, review_date, quarter, verified_purchase):
    """Persist one raw review to the ``reviews`` table.

    This stores the source data only — no AI classification involved.
    Must be called before :func:`save_result` for the same review_id,
    since ``classifications`` references ``reviews`` via a foreign key.

    Parameters
    ----------
    review_id : str
        Unique identifier for the review (the product's ``asin``).
    product_category : str
        Which product category this review belongs to.
    product_title : str
        The product's name, joined in from metadata.
    review_text : str
        The raw review text as written by the customer.
    rating : float
        The star rating given (1.0 to 5.0).
    review_date : str or datetime
        When the review was originally posted.
    quarter : str
        Which quarter this review falls into (e.g., "Q1 2022").
    verified_purchase : bool
        Whether Amazon verified this reviewer purchased the product.

    Returns
    -------
    None

    Notes
    -----
    Uses ``ON CONFLICT DO NOTHING`` so re-running ingestion on the same
    review_id is safe and does not raise a duplicate-key error.
    """
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO reviews (review_id, product_category, product_title,
                              review_text, rating, review_date, quarter,
                              verified_purchase)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (review_id) DO NOTHING;
    """, (
        review_id, product_category, product_title, review_text,
        rating, review_date, quarter, verified_purchase
    ))
    conn.commit()
    cur.close()
    conn.close()


def save_result(review_id, result):
    """Persist a classification result to the ``classifications`` table.

    Parameters
    ----------
    review_id : str
        The review_id this classification belongs to. Must already
        exist in ``reviews``, since this is a foreign key reference.
    result : dict
        The classification result, with keys feedback_category,
        sentiment, urgency, reasoning.

    Returns
    -------
    int
        The database-generated primary key ID of the newly inserted row.
    """
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO classifications (review_id, feedback_category, sentiment,
                                      urgency, reasoning)
        VALUES (%s, %s, %s, %s, %s)
        RETURNING id;
    """, (
        review_id, result["feedback_category"], result["sentiment"],
        result["urgency"], result["reasoning"]
    ))
    new_id = cur.fetchone()[0]
    conn.commit()
    cur.close()
    conn.close()
    return new_id


def get_all_results(limit=1500):
    """Retrieve reviews joined with their classifications, if any.

    Uses a LEFT JOIN so reviews without a classification yet still
    appear (with NULL classification fields), rather than being
    silently excluded — reflecting the real state of a pipeline where
    not all ingested data has necessarily been processed yet.

    Parameters
    ----------
    limit : int, optional
        Maximum number of records to return (default 1500).

    Returns
    -------
    list of dict
        Joined review + classification records, newest reviews first.
    """
    conn = get_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute("""
        SELECT r.review_id, r.product_category, r.product_title, r.review_text,
               r.rating, r.review_date, r.quarter, r.verified_purchase,
               c.feedback_category, c.sentiment, c.urgency, c.reasoning,
               c.classified_at
        FROM reviews r
        LEFT JOIN classifications c ON r.review_id = c.review_id
        ORDER BY r.review_date DESC
        LIMIT %s;
    """, (limit,))
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows