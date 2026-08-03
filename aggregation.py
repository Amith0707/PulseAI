import pandas as pd
from db import get_all_results


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