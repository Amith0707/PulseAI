import os
import psycopg2
from psycopg2.extras import RealDictCursor

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
    """Initialize the feedback table if it doesn't already exist."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id SERIAL PRIMARY KEY,
            review_id TEXT NOT NULL,
            product_category TEXT NOT NULL,
            review_text TEXT NOT NULL,
            rating FLOAT,
            feedback_category TEXT NOT NULL,
            sentiment TEXT NOT NULL,
            urgency TEXT NOT NULL,
            reasoning TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT NOW()
        );
    """)
    conn.commit()
    cur.close()
    conn.close()


def save_result(review_id, product_category, review_text, rating, result):
    """Persist a classified review to the database.

    Parameters
    ----------
    review_id : str
        Unique identifier for the review.
    product_category : str
        Which product category this review belongs to (Fashion, Electronics, etc.).
    review_text : str
        The raw review text.
    rating : float
        The original star rating from the source data.
    result : dict
        The classification result, with keys feedback_category, sentiment,
        urgency, reasoning.

    Returns
    -------
    int
        The database-generated primary key ID of the newly inserted row.
    """
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO feedback (review_id, product_category, review_text, rating,
                               feedback_category, sentiment, urgency, reasoning)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING id;
    """, (
        review_id, product_category, review_text, rating,
        result["feedback_category"], result["sentiment"],
        result["urgency"], result["reasoning"]
    ))
    new_id = cur.fetchone()[0]
    conn.commit()
    cur.close()
    conn.close()
    return new_id


def get_all_results(limit=500):
    """Retrieve classified reviews from the database.

    Parameters
    ----------
    limit : int, optional
        Maximum number of records to return (default 500).

    Returns
    -------
    list of dict
        Classified review records, newest first.
    """
    conn = get_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute("SELECT * FROM feedback ORDER BY created_at DESC LIMIT %s;", (limit,))
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows