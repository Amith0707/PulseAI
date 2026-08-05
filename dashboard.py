import dash
from dash import dcc, html, Input, Output, State
import plotly.express as px
import pandas as pd

from schema_info import get_schema_info, get_table_preview, ALLOWED_TABLES
from db import get_connection
from aggregation import get_aggregated_stats, generate_and_store_summary
from classifier import parse_natural_language_query, suggest_followup_questions

app = dash.Dash(__name__)
app.title = "PulseAI — Feedback Dashboard"

PRODUCT_CATEGORIES = [
    "Amazon_Fashion", "Electronics", "Home_and_Kitchen",
    "Beauty_and_Personal_Care", "Sports_and_Outdoors"
]
FEEDBACK_CATEGORIES = [
    "Product Quality/Defect", "Not As Described", "Shipping/Delivery Issue",
    "Pricing/Value Complaint", "Usability/Performance Issue",
    "Customer Service Experience", "Praise/Positive", "Insufficient Context"
]
SENTIMENTS = ["Positive", "Neutral", "Negative"]
URGENCIES = ["High", "Medium", "Low"]

app.layout = html.Div([

    # ============ Header ============
    html.Div([
        html.Span("PA", className="mark"),
        "PulseAI"
    ], className="pulseai-header"),

    # ============ Genie-style query panel ============
    html.Div([
        # LEFT SIDEBAR
        html.Div([
            html.H3("PulseAI"),
            html.Div([
                html.Div("Browse a table", className="sidebar-label"),
                dcc.Dropdown(
                    id="table-selector",
                    options=[{"label": t, "value": t} for t in ALLOWED_TABLES],
                    placeholder="Select a table...",
                    className="table-dropdown"
                ),
            ], className="sidebar-section"),
            html.Div([
                html.Div("Recent questions", className="sidebar-label"),
                html.Div(
                    id="chat-history-list",
                    children=html.Span("No questions asked yet.", className="no-questions-text")
                ),
            ], className="sidebar-section"),
        ], className="genie-sidebar"),

        # MAIN QUERY/PREVIEW AREA
        html.Div([
            html.Div([
                dcc.Input(
                    id="query-input",
                    type="text",
                    placeholder="Ask a question, e.g. 'how many negative reviews'",
                    className="genie-input"
                ),
                html.Button("Ask", id="ask-btn", className="genie-ask-btn"),
            ], className="query-input-row"),
            html.Div(
                id="query-result-area",
                children=html.Span("Select a table, or ask a question above.", className="no-questions-text")
            ),
        ], className="genie-main"),
    ], className="genie-panel"),

    # ============ Analytics dashboard ============
    html.Div([
        html.H1("Weekly Feedback Dashboard"),

        html.Div([
            html.Div([
                html.Label("Product category"),
                dcc.Dropdown(
                    id="filter-product-category",
                    options=[{"label": c, "value": c} for c in PRODUCT_CATEGORIES],
                    multi=True,
                    placeholder="All product categories"
                ),
            ], className="filter-col"),

            html.Div([
                html.Label("Feedback category"),
                dcc.Dropdown(
                    id="filter-feedback-category",
                    options=[{"label": c, "value": c} for c in FEEDBACK_CATEGORIES],
                    multi=True,
                    placeholder="All feedback categories"
                ),
            ], className="filter-col"),

            html.Div([
                html.Label("Sentiment"),
                dcc.Dropdown(
                    id="filter-sentiment",
                    options=[{"label": s, "value": s} for s in SENTIMENTS],
                    multi=True,
                    placeholder="All sentiments"
                ),
            ], className="filter-col"),

            html.Div([
                html.Label("Urgency"),
                dcc.Dropdown(
                    id="filter-urgency",
                    options=[{"label": u, "value": u} for u in URGENCIES],
                    multi=True,
                    placeholder="All urgency levels"
                ),
            ], className="filter-col"),
        ], className="filter-row"),

        html.Div(id="total-count-display", className="total-count-banner"),

        html.Div([
            html.Div(dcc.Graph(id="chart-feedback-category"), className="chart-card"),
            html.Div(dcc.Graph(id="chart-sentiment"), className="chart-card"),
        ], className="chart-row"),

        html.Div([
            html.Div(dcc.Graph(id="chart-urgency"), className="chart-card"),
            html.Div(dcc.Graph(id="chart-product-category"), className="chart-card"),
        ], className="chart-row"),
    ], className="analytics-section"),

    # ============ Rolling summary section ============
    html.Div([
        html.H2("Rolling Summary"),
        dcc.Input(
            id="summary-query-input",
            type="text",
            placeholder="e.g. 'Summarize Q1 2022' or 'How was sentiment from Jan to March 2022'",
            className="genie-input"
        ),
        html.Button("Get Summary", id="summary-ask-btn", className="genie-ask-btn"),
        html.Div(id="summary-display-area"),
    ], className="analytics-section", id="summary-section"),
])

@app.callback(
    Output("query-result-area", "children", allow_duplicate=True),
    Input("ask-btn", "n_clicks"),
    State("query-input", "value"),
    prevent_initial_call=True
)
def handle_query(n_clicks, question):
    """Parse a natural language question and return a count, lookup,
    or unsupported response."""
    if not question:
        return html.Span("Type a question first.", className="no-questions-text")

    schema = get_schema_info()
    parsed = parse_natural_language_query(question, schema)

    if parsed["query_type"] == "unsupported":
        return html.Div([
            html.Div(parsed["explanation"], className="answer-explanation")
        ], className="answer-card")

    conn = get_connection()
    cur = conn.cursor()

    if parsed["query_type"] == "lookup":
        review_id = parsed.get("review_id")
        cur.execute("""
            SELECT r.review_id, r.product_category, r.product_title, r.review_text,
                   r.rating, r.review_date, r.verified_purchase,
                   c.feedback_category, c.sentiment, c.urgency, c.reasoning
            FROM reviews r
            LEFT JOIN classifications c ON r.review_id = c.review_id
            WHERE r.review_id = %s;
        """, (review_id,))
        row = cur.fetchone()
        cur.close()
        conn.close()

        if row is None:
            return html.Div([
                html.Div(f"No review found with ID '{review_id}'.", className="answer-explanation")
            ], className="answer-card")

        columns = ["review_id", "product_category", "product_title", "review_text",
                   "rating", "review_date", "verified_purchase",
                   "feedback_category", "sentiment", "urgency", "reasoning"]
        result_dict = dict(zip(columns, row))

        return html.Div([
            html.Div(parsed["explanation"], className="answer-explanation"),
            html.Table([
                html.Tr([html.Td(html.B(k)), html.Td(str(v))]) for k, v in result_dict.items()
            ], className="preview-table", style={"marginTop": "12px"})
        ], className="answer-card")

    if parsed["query_type"] == "breakdown":
        column = parsed.get("breakdown_column")
        valid_columns = {
            "product_category": "r.product_category",
            "feedback_category": "c.feedback_category",
            "sentiment": "c.sentiment",
            "urgency": "c.urgency",
            "quarter": "r.quarter",
            "verified_purchase": "r.verified_purchase",
        }

        if column not in valid_columns:
            cur.close()
            conn.close()
            return html.Div([
                html.Div(f"'{column}' isn't a valid column to break down by.", className="answer-explanation")
            ], className="answer-card")

        sql_column = valid_columns[column]

        conditions = []
        params = []
        for field in ["product_category", "feedback_category", "sentiment", "urgency"]:
            value = parsed.get(field)
            if value:
                col = "r.product_category" if field == "product_category" else f"c.{field}"
                conditions.append(f"{col} = %s")
                params.append(value)

        where_clause = " AND ".join(conditions) if conditions else "TRUE"
        sql = f"""
            SELECT {sql_column}, COUNT(*) as count
            FROM reviews r
            LEFT JOIN classifications c ON r.review_id = c.review_id
            WHERE {where_clause}
            GROUP BY {sql_column}
            ORDER BY count DESC;
        """
        cur.execute(sql, params)
        rows = cur.fetchall()
        cur.close()
        conn.close()

        table_rows = [html.Tr([html.Td(str(r[0]) if r[0] is not None else "null"), html.Td(str(r[1]))]) for r in rows]

        return html.Div([
            html.Div(parsed["explanation"], className="answer-explanation"),
            html.Table(
                [html.Tr([html.Th(column), html.Th("count")])] + table_rows,
                className="preview-table", style={"marginTop": "12px"}
            ),
            html.Details([
                html.Summary("View generated query"),
                html.Pre(sql, style={"fontSize": "12px", "background": "var(--slate-100)", "padding": "10px"})
            ])
        ], className="answer-card")

    # query_type == "count"
    conditions = []
    params = []
    for field in ["product_category", "feedback_category", "sentiment", "urgency"]:
        value = parsed.get(field)
        if value:
            column = "r.product_category" if field == "product_category" else f"c.{field}"
            conditions.append(f"{column} = %s")
            params.append(value)

    if parsed.get("verified_purchase") is not None:
        conditions.append("r.verified_purchase = %s")
        params.append(parsed["verified_purchase"])

    date_start = parsed.get("date_start")
    date_end = parsed.get("date_end")
    if date_start and date_end:
        conditions.append("r.review_date BETWEEN %s AND %s")
        params.append(date_start)
        params.append(date_end)

    where_clause = " AND ".join(conditions) if conditions else "TRUE"
    sql = f"""
        SELECT COUNT(*) FROM reviews r
        LEFT JOIN classifications c ON r.review_id = c.review_id
        WHERE {where_clause};
    """
    cur.execute(sql, params)
    count = cur.fetchone()[0]
    cur.close()
    conn.close()

    return html.Div([
        html.Div(str(count), className="answer-value"),
        html.Div(parsed["explanation"], className="answer-explanation"),
        html.Details([
            html.Summary("View generated query"),
            html.Pre(sql, style={"fontSize": "12px", "background": "var(--slate-100)", "padding": "10px"})
        ])
    ], className="answer-card")

# ============ Table browsing callback ============
@app.callback(
    Output("query-result-area", "children", allow_duplicate=True),
    Input("table-selector", "value"),
    prevent_initial_call=True
)
def show_table_preview(table_name):
    """Render a styled preview of the selected table's rows."""
    if not table_name:
        return html.Span("Select a table to preview its records.", className="no-questions-text")

    preview = get_table_preview(table_name, limit=20)

    header = html.Tr([html.Th(col) for col in preview["columns"]])
    body = [
        html.Tr([html.Td(str(row.get(col, ""))[:80]) for col in preview["columns"]])
        for row in preview["rows"]
    ]

    return html.Div([
        html.H4(f"Previewing '{table_name}' — first {len(preview['rows'])} rows", className="result-heading"),
        html.Table([header] + body, className="preview-table")
    ])


# ============ Analytics dashboard callback ============
@app.callback(
    [
        Output("total-count-display", "children"),
        Output("chart-feedback-category", "figure"),
        Output("chart-sentiment", "figure"),
        Output("chart-urgency", "figure"),
        Output("chart-product-category", "figure"),
    ],
    [
        Input("filter-product-category", "value"),
        Input("filter-feedback-category", "value"),
        Input("filter-sentiment", "value"),
        Input("filter-urgency", "value"),
    ]
)
def update_dashboard(product_categories, feedback_categories, sentiments, urgencies):
    """Recompute all charts whenever any filter dropdown changes."""
    stats = get_aggregated_stats()
    df = stats["filtered_rows"]

    if product_categories:
        df = df[df["product_category"].isin(product_categories)]
    if feedback_categories:
        df = df[df["feedback_category"].isin(feedback_categories)]
    if sentiments:
        df = df[df["sentiment"].isin(sentiments)]
    if urgencies:
        df = df[df["urgency"].isin(urgencies)]

    total_text = f"Showing {len(df)} reviews matching current filters"

    fig_category = px.bar(
        df["feedback_category"].value_counts().reset_index(),
        x="feedback_category", y="count", title="Feedback category breakdown"
    )
    fig_sentiment = px.pie(
        df["sentiment"].value_counts().reset_index(),
        names="sentiment", values="count", title="Sentiment distribution"
    )
    fig_urgency = px.bar(
        df["urgency"].value_counts().reset_index(),
        x="urgency", y="count", title="Urgency distribution",
        category_orders={"urgency": ["Low", "Medium", "High"]}
    )
    fig_product = px.bar(
        df["product_category"].value_counts().reset_index(),
        x="product_category", y="count", title="Reviews by product category"
    )

    return total_text, fig_category, fig_sentiment, fig_urgency, fig_product


# ============ Rolling summary callback ============
@app.callback(
    Output("summary-display-area", "children"),
    Input("summary-ask-btn", "n_clicks"),
    State("summary-query-input", "value"),
    prevent_initial_call=True
)
def handle_summary_request(n_clicks, question):
    """Generate one summary for a user's typed request, showing
    follow-up suggestions as static placeholders (not yet clickable)."""
    if not question:
        return html.Span("Type a request first.", className="no-questions-text")

    schema = get_schema_info()
    parsed = parse_natural_language_query(question, schema)

    if parsed["query_type"] != "summary" or not parsed.get("date_start"):
        return html.Div(parsed["explanation"], className="answer-explanation")

    window_label = f"Custom: {parsed['date_start']} to {parsed['date_end']}"
    result = generate_and_store_summary(window_label, parsed["date_start"], parsed["date_end"])

    if not result["success"]:
        return html.Div(result["error"], className="answer-explanation")

    followups = suggest_followup_questions(result["narrative"], result["stats"])

    return html.Div([
        html.Div(result["narrative"], className="answer-explanation"),
        html.Div("Suggested follow-up questions:", style={"marginTop": "16px", "fontWeight": "600"}),
        html.Div([html.Button(q, className="suggestion-chip", disabled=True) for q in followups])
    ], className="answer-card")


if __name__ == "__main__":
    app.run(debug=True, port=8050)