import pandas as pd
from db import get_connection
ALLOWED_TABLES = ["reviews", "classifications"]


def get_table_preview(table_name, limit=50):
    """Fetch a preview of rows from a specific table.

    Parameters
    ----------
    table_name : str
        Must be one of ALLOWED_TABLES — never accept arbitrary table
        names from user input directly, to avoid SQL injection via
        table name interpolation.
    limit : int, optional
        Number of rows to preview (default 50).

    Returns
    -------
    dict
        Keys "columns" (list of column names) and "rows" (list of dicts).

    Raises
    ------
    ValueError
        If table_name is not in ALLOWED_TABLES.
    """
    if table_name not in ALLOWED_TABLES:
        raise ValueError(f"Table '{table_name}' is not accessible. Allowed: {ALLOWED_TABLES}")

    conn = get_connection()
    cur = conn.cursor()
    # table_name is validated against an allowlist above, safe to use in f-string here —
    # never do this with unvalidated user input, since normal parameterization
    # (%s) does not work for table/column names in SQL, only for values
    cur.execute(f"SELECT * FROM {table_name} LIMIT %s;", (limit,))
    columns = [desc[0] for desc in cur.description]
    rows = cur.fetchall()
    cur.close()
    conn.close()

    return {
        "columns": columns,
        "rows": [dict(zip(columns, row)) for row in rows]
    }

def get_schema_info():
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) FROM reviews;")
    row_count = cur.fetchone()[0]

    cur.execute("SELECT MIN(review_date), MAX(review_date) FROM reviews;")
    min_date, max_date = cur.fetchone()

    cur.execute("SELECT DISTINCT feedback_category FROM classifications;")
    feedback_categories = [r[0] for r in cur.fetchall()]

    cur.execute("SELECT DISTINCT product_category FROM reviews;")
    product_categories = [r[0] for r in cur.fetchall()]

    cur.execute("SELECT DISTINCT sentiment FROM classifications;")
    sentiments = [r[0] for r in cur.fetchall()]

    cur.execute("SELECT DISTINCT urgency FROM classifications;")
    urgencies = [r[0] for r in cur.fetchall()]

    cur.execute("SELECT DISTINCT quarter FROM reviews;")
    quarters = [r[0] for r in cur.fetchall()]

    cur.close()
    conn.close()

    return {
        "table_name": "reviews joined with classifications",
        "row_count": row_count,
        "source_data_date_range": {"min": str(min_date), "max": str(max_date)},
        "columns": [
            {"name": "review_id", "type": "identifier",
             "note": "Unique ID for a specific review. Use for direct lookup of one review, not for filtering/counting."},
            {"name": "product_category", "type": "categorical", "possible_values": product_categories},
            {"name": "feedback_category", "type": "categorical", "possible_values": feedback_categories},
            {"name": "sentiment", "type": "categorical", "possible_values": sentiments},
            {"name": "urgency", "type": "categorical", "possible_values": urgencies},
            {"name": "quarter", "type": "categorical", "possible_values": quarters},
            {"name": "rating", "type": "numeric", "range": "1.0 to 5.0"},
            {"name": "review_date", "type": "date",
             "range": f"{min_date} to {max_date}",
             "note": "Use for specific date-range questions, not just quarter labels."},
            {"name": "verified_purchase", "type": "boolean", "possible_values": [True, False],
             "note": "Whether Amazon verified the reviewer actually purchased the product."},
        ]
    }

def validate_date_range(date_start, date_end):
    """Check a requested date range against the database's actual
    current coverage — always queried live, never hardcoded, so this
    stays correct even as new data is appended in the future.

    Returns
    -------
    tuple
        (is_valid: bool, message: str) — message explains the actual
        available range if invalid.
    """
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT MIN(review_date), MAX(review_date) FROM reviews;")
    data_min, data_max = cur.fetchone()
    cur.close()
    conn.close()

    req_start = pd.Timestamp(date_start)
    req_end = pd.Timestamp(date_end)

    if req_end < data_min or req_start > data_max:
        return False, f"No data available in that range. Currently available: {data_min.date()} to {data_max.date()}."
    return True, None