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