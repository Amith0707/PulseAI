import json
import pandas as pd
from db import get_all_results,get_connection
from classifier import generate_summary
from schema_info import validate_date_range

def get_aggregated_stats(product_category=None, feedback_category=None,
                          sentiment=None, urgency=None):
    """Compute aggregate statistics across classified feedback.

    Supports optional filtering on any combination of dimensions, so the
    same function powers both the full, unfiltered dashboard view and
    any sliced view a user selects (e.g., only Electronics, only
    Negative sentiment).

    Parameters
    ----------
    product_category : str, optional
        If given, restrict to this product category only (e.g.,
        "Electronics"). Default is None (no filter).
    feedback_category : str, optional
        If given, restrict to this feedback category only (e.g.,
        "Product Quality/Defect"). Default is None (no filter).
    sentiment : str, optional
        If given, restrict to this sentiment only. Default is None.
    urgency : str, optional
        If given, restrict to this urgency level only. Default is None.

    Returns
    -------
    dict
        A dictionary with keys:

        - ``total_count`` : int, number of reviews matching the filters
        - ``category_counts`` : dict, feedback_category -> count
        - ``sentiment_counts`` : dict, sentiment -> count
        - ``urgency_counts`` : dict, urgency -> count
        - ``product_category_counts`` : dict, product_category -> count
        - ``filtered_rows`` : pandas.DataFrame, the underlying filtered
          rows, for drill-down display

    Examples
    --------
    >>> stats = get_aggregated_stats(product_category="Electronics")
    >>> stats["total_count"]
    60
    """
    df = pd.DataFrame(get_all_results(limit=1000))

    if product_category:
        df = df[df["product_category"] == product_category]
    if feedback_category:
        df = df[df["feedback_category"] == feedback_category]
    if sentiment:
        df = df[df["sentiment"] == sentiment]
    if urgency:
        df = df[df["urgency"] == urgency]

    return {
        "total_count": len(df),
        "category_counts": df["feedback_category"].value_counts().to_dict(),
        "sentiment_counts": df["sentiment"].value_counts().to_dict(),
        "urgency_counts": df["urgency"].value_counts().to_dict(),
        "product_category_counts": df["product_category"].value_counts().to_dict(),
        "filtered_rows": df
    }


def get_top_themes(n=5, **filters):
    """Return the top N most frequent feedback categories, with counts.

    Parameters
    ----------
    n : int, optional
        Number of top themes to return (default 5).
    **filters
        Any of product_category, feedback_category, sentiment, urgency,
        passed through to :func:`get_aggregated_stats`.

    Returns
    -------
    list of tuple
        (feedback_category, count) pairs, sorted descending by count.
    """
    stats = get_aggregated_stats(**filters)
    sorted_themes = sorted(stats["category_counts"].items(), key=lambda x: -x[1])
    return sorted_themes[:n]

def generate_and_store_summary(window_label, date_start, date_end):
    """Compute stats and generate a narrative summary for an arbitrary
    date range, storing (or replacing) the result under window_label.

    Always validates the requested range against the database's live
    current coverage first, so this stays correct even as new data is
    appended in the future — no hardcoded date assumptions.

    Parameters
    ----------
    window_label : str
        Human-readable label for this window (e.g., "Q1 2022", "Feb 2022").
    date_start : str
        Start date, YYYY-MM-DD format.
    date_end : str
        End date, YYYY-MM-DD format.

    Returns
    -------
    dict
        Keys: "success" (bool), and either "narrative"/"stats" or
        "error" depending on outcome.
    """
    is_valid, error_message = validate_date_range(date_start, date_end)
    if not is_valid:
        return {"success": False, "error": error_message}

    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT c.feedback_category, c.sentiment, c.urgency, r.product_category
        FROM reviews r
        LEFT JOIN classifications c ON r.review_id = c.review_id
        WHERE r.review_date BETWEEN %s AND %s;
    """, (date_start, date_end))
    rows = cur.fetchall()

    if len(rows) == 0:
        cur.close()
        conn.close()
        return {"success": False, "error": "No reviews found in this specific range."}

    import pandas as pd
    df = pd.DataFrame(rows, columns=["feedback_category", "sentiment", "urgency", "product_category"])

    window_stats = {
        "total_count": len(df),
        "category_counts": df["feedback_category"].value_counts().to_dict(),
        "sentiment_counts": df["sentiment"].value_counts().to_dict(),
        "urgency_counts": df["urgency"].value_counts().to_dict(),
        "product_category_counts": df["product_category"].value_counts().to_dict(),
    }

    narrative = generate_summary(window_stats)

    cur.execute("""
        INSERT INTO summaries (window_label, date_start, date_end, total_reviews, stats, narrative)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (window_label) DO UPDATE
        SET date_start = EXCLUDED.date_start,
            date_end = EXCLUDED.date_end,
            total_reviews = EXCLUDED.total_reviews,
            stats = EXCLUDED.stats,
            narrative = EXCLUDED.narrative,
            generated_at = NOW();
    """, (window_label, date_start, date_end, window_stats["total_count"], json.dumps(window_stats), narrative))
    conn.commit()
    cur.close()
    conn.close()

    return {"success": True, "narrative": narrative, "stats": window_stats}


def get_stored_summary(window_label):
    """Retrieve a previously generated summary by its window label."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT narrative, stats, date_start, date_end, generated_at FROM summaries WHERE window_label = %s;", (window_label,))
    row = cur.fetchone()
    cur.close()
    conn.close()
    if row is None:
        return None
    return {"narrative": row[0], "stats": row[1], "date_start": row[2], "date_end": row[3], "generated_at": row[4]}