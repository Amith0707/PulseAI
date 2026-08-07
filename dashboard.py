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

VALID_GROUP_COLUMNS = {
    "product_category": "r.product_category",
    "feedback_category": "c.feedback_category",
    "sentiment": "c.sentiment",
    "urgency": "c.urgency",
    "quarter": "r.quarter",
    "verified_purchase": "r.verified_purchase",
}


def build_filter_conditions(parsed, include_date=True, include_verified=False):
    """Build a parameterized WHERE clause from a parsed query's filter
    fields. Shared across count/breakdown/chart so the SQL-building logic
    exists in exactly one place."""
    conditions = []
    params = []
    for field in ["product_category", "feedback_category", "sentiment", "urgency"]:
        value = parsed.get(field)
        if value:
            col = "r.product_category" if field == "product_category" else f"c.{field}"
            conditions.append(f"{col} = %s")
            params.append(value)

    if include_verified and parsed.get("verified_purchase") is not None:
        conditions.append("r.verified_purchase = %s")
        params.append(parsed["verified_purchase"])

    if include_date and parsed.get("date_start") and parsed.get("date_end"):
        conditions.append("r.review_date BETWEEN %s AND %s")
        params.append(parsed["date_start"])
        params.append(parsed["date_end"])

    where_clause = " AND ".join(conditions) if conditions else "TRUE"
    return where_clause, params


def run_lookup_query(parsed):
    """Look up a single review by ID. Returns a plain-data dict — never a
    Dash component — so it can be rendered by any UI (Genie panel or
    chat) and safely stored in a dcc.Store."""
    review_id = parsed.get("review_id")
    conn = get_connection()
    cur = conn.cursor()
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
        return {"kind": "message", "explanation": f"No review found with ID '{review_id}'."}

    columns = ["review_id", "product_category", "product_title", "review_text",
               "rating", "review_date", "verified_purchase",
               "feedback_category", "sentiment", "urgency", "reasoning"]
    return {
        "kind": "keyvalue",
        "explanation": parsed["explanation"],
        "pairs": [[col, str(val)] for col, val in zip(columns, row)],
    }


def run_breakdown_query(parsed):
    """Count reviews grouped by one column. Returns a plain-data dict."""
    column = parsed.get("breakdown_column")
    if column not in VALID_GROUP_COLUMNS:
        return {"kind": "message", "explanation": f"'{column}' isn't a valid column to break down by."}

    sql_column = VALID_GROUP_COLUMNS[column]
    where_clause, params = build_filter_conditions(parsed, include_date=False, include_verified=False)
    sql = f"""
        SELECT {sql_column}, COUNT(*) as count
        FROM reviews r
        LEFT JOIN classifications c ON r.review_id = c.review_id
        WHERE {where_clause}
        GROUP BY {sql_column}
        ORDER BY count DESC;
    """
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(sql, params)
    rows = cur.fetchall()
    cur.close()
    conn.close()

    return {
        "kind": "table",
        "explanation": parsed["explanation"],
        "columns": [column, "count"],
        "rows": [[r[0] if r[0] is not None else "null", r[1]] for r in rows],
        "sql": sql,
    }


def run_count_query(parsed):
    """Count reviews matching the parsed filters. Returns a plain-data dict."""
    where_clause, params = build_filter_conditions(parsed, include_date=True, include_verified=True)
    sql = f"""
        SELECT COUNT(*) FROM reviews r
        LEFT JOIN classifications c ON r.review_id = c.review_id
        WHERE {where_clause};
    """
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(sql, params)
    count = cur.fetchone()[0]
    cur.close()
    conn.close()

    return {"kind": "count", "explanation": parsed["explanation"], "count": count, "sql": sql}


def run_chart_query(parsed):
    """Fetch grouped counts for a chart request. Returns a plain-data dict
    describing the raw rows plus which chart type(s) to render — the
    actual Plotly figures get built at render time, from these rows, in
    render_answer_card (never stored as figure/Component objects)."""
    column = parsed.get("group_by_column")
    if column not in VALID_GROUP_COLUMNS:
        return {"kind": "message", "explanation": f"'{column}' isn't a valid column to chart by."}

    sql_column = VALID_GROUP_COLUMNS[column]
    where_clause, params = build_filter_conditions(parsed, include_date=True, include_verified=False)
    sql = f"""
        SELECT {sql_column}, COUNT(*) as count
        FROM reviews r
        LEFT JOIN classifications c ON r.review_id = c.review_id
        WHERE {where_clause}
        GROUP BY {sql_column}
        ORDER BY count DESC;
    """
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(sql, params)
    rows = cur.fetchall()
    cur.close()
    conn.close()

    if not rows:
        return {"kind": "message", "explanation": "No data found matching those filters."}

    requested_types = [t for t in (parsed.get("chart_types") or ["bar"]) if t in ("bar", "line", "pie")] or ["bar"]

    notes = []
    if "line" in requested_types:
        notes.append("A line chart isn't meaningful for a categorical breakdown like this, so it's rendered as a bar chart instead.")
        requested_types = list(dict.fromkeys("bar" if t == "line" else t for t in requested_types))

    return {
        "kind": "chart",
        "explanation": parsed["explanation"],
        "notes": notes,
        "column": column,
        "rows": [[r[0] if r[0] is not None else "null", r[1]] for r in rows],
        "chart_types": requested_types,
    }


def run_summary_query(parsed):
    """Generate (or fetch a stored) narrative summary via the existing
    aggregation.generate_and_store_summary — never reimplemented here."""
    date_start, date_end = parsed.get("date_start"), parsed.get("date_end")
    if not date_start or not date_end:
        return {"kind": "message", "explanation": parsed["explanation"]}

    window_label = f"Custom: {date_start} to {date_end}"
    result = generate_and_store_summary(window_label, date_start, date_end)
    if not result["success"]:
        return {"kind": "message", "explanation": result["error"]}

    followups = suggest_followup_questions(result["narrative"], result["stats"])
    return {"kind": "summary", "narrative": result["narrative"], "followups": followups}


def run_parsed_query(parsed):
    """Route a parsed query to the right data-fetching function. Returns a
    plain-data dict (never a Dash component), so any caller can render it
    however fits its own UI, and it can be safely persisted in a
    dcc.Store (e.g. the chat history) without a JSON-serialization issue."""
    query_type = parsed["query_type"]
    if query_type == "lookup":
        return run_lookup_query(parsed)
    if query_type == "breakdown":
        return run_breakdown_query(parsed)
    if query_type == "chart":
        return run_chart_query(parsed)
    if query_type == "summary":
        return run_summary_query(parsed)
    if query_type == "count":
        return run_count_query(parsed)
    return {"kind": "message", "explanation": parsed["explanation"]}


def render_answer_card(result):
    """Render a plain-data result (from run_parsed_query) into the same
    'answer-card' markup used throughout the app. This is the single
    place that turns query results into Dash components, reused by the
    Genie panel, the Rolling Summary box, and the chat interface."""
    kind = result["kind"]

    if kind == "message":
        return html.Div([
            html.Div(result["explanation"], className="answer-explanation")
        ], className="answer-card")

    if kind == "keyvalue":
        return html.Div([
            html.Div(result["explanation"], className="answer-explanation"),
            html.Table([
                html.Tr([html.Td(html.B(k)), html.Td(v)]) for k, v in result["pairs"]
            ], className="preview-table", style={"marginTop": "12px"})
        ], className="answer-card")

    if kind == "table":
        header = html.Tr([html.Th(c) for c in result["columns"]])
        body = [html.Tr([html.Td(str(v)) for v in row]) for row in result["rows"]]
        return html.Div([
            html.Div(result["explanation"], className="answer-explanation"),
            html.Table([header] + body, className="preview-table", style={"marginTop": "12px"}),
            html.Details([
                html.Summary("View generated query"),
                html.Pre(result["sql"], style={"fontSize": "12px", "background": "var(--slate-100)", "padding": "10px"})
            ])
        ], className="answer-card")

    if kind == "count":
        return html.Div([
            html.Div(str(result["count"]), className="answer-value"),
            html.Div(result["explanation"], className="answer-explanation"),
            html.Details([
                html.Summary("View generated query"),
                html.Pre(result["sql"], style={"fontSize": "12px", "background": "var(--slate-100)", "padding": "10px"})
            ])
        ], className="answer-card")

    if kind == "chart":
        children = [html.Div(result["explanation"], className="answer-explanation")]
        for note in result["notes"]:
            children.append(html.Div(note, className="answer-explanation", style={"marginTop": "6px", "fontStyle": "italic"}))

        df = pd.DataFrame(result["rows"], columns=[result["column"], "count"])
        graphs = []
        for chart_type in result["chart_types"]:
            if chart_type == "pie":
                fig = px.pie(df, names=result["column"], values="count", title=f"{result['column']} breakdown")
            else:
                fig = px.bar(df, x=result["column"], y="count", title=f"{result['column']} breakdown")
            graphs.append(html.Div(dcc.Graph(figure=fig, config={"displayModeBar": True, "responsive": True}), className="chart-card"))

        children.append(html.Div(graphs, className="chart-row", style={"marginTop": "14px", "flexWrap": "wrap"}))
        return html.Div(children, className="answer-card")

    if kind == "summary":
        return html.Div([
            html.Div(result["narrative"], className="answer-explanation"),
            html.Div("Suggested follow-up questions:", style={"marginTop": "16px", "fontWeight": "600"}),
            html.Div([html.Button(q, className="suggestion-chip", disabled=True) for q in result["followups"]])
        ], className="answer-card")

    return html.Div("Unable to render this result.", className="no-questions-text")


CHAT_PLACEHOLDER_TEXT = (
    "Ask about counts, breakdowns, summaries, or request a chart — "
    "e.g. \"show me a bar and pie chart of sentiment\"."
)


def render_chat_thread(history):
    """Render the full chat history. Entries are stored oldest-to-newest;
    rendered newest-first in the DOM so that, combined with the
    column-reverse container in the CSS, the latest message lands at the
    bottom automatically — no JS needed — with older turns reachable by
    scrolling up."""
    if not history:
        return html.Span(CHAT_PLACEHOLDER_TEXT, className="no-questions-text")

    bubbles = []
    for entry in reversed(history):
        if entry["role"] == "user":
            bubbles.append(html.Div(entry["text"], className="chat-bubble-user"))
        else:
            bubbles.append(html.Div(render_answer_card(entry), className="chat-bubble-assistant"))
    return bubbles


app.layout = html.Div([

    html.Div([
        html.Span("PA", className="mark"),
        "PulseAI"
    ], className="pulseai-header"),

    html.Div([
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

        html.Div([
            html.Div([
                dcc.Input(
                    id="query-input",
                    type="text",
                    placeholder="Ask a question, e.g. 'how many negative reviews'",
                    className="genie-input",
                    style={"width": "100%", "boxSizing": "border-box", "flex": "1"}
                ),
                html.Button("Ask", id="ask-btn", className="genie-ask-btn"),
            ], className="query-input-row", style={"display": "flex", "gap": "12px", "width": "100%"}),
            html.Div(
                id="query-result-area",
                children=html.Span("Select a table, or ask a question above.", className="no-questions-text")
            ),
        ], className="genie-main"),
    ], className="genie-panel"),

    html.Div([
        html.Div([
            html.H3("Chats"),
            html.Div([
                html.Button("+ New chat", id="chat-new-btn", className="genie-ask-btn chat-new-btn"),
                html.Div("Sessions", className="sidebar-label"),
                html.Div(
                    html.Div("Current session", className="recent-question-item chat-session-active"),
                ),
            ], className="sidebar-section"),
        ], className="genie-sidebar"),

        html.Div([
            dcc.Store(id="chat-history-store", data=[]),
            html.Div(id="chat-thread", className="chat-thread", children=render_chat_thread([])),
            html.Div([
                dcc.Input(
                    id="chat-input",
                    type="text",
                    placeholder="Ask a question, request a summary, or ask for a chart...",
                    className="genie-input",
                    style={"width": "100%", "boxSizing": "border-box", "flex": "1"}
                ),
                html.Button("Send", id="chat-send-btn", className="genie-ask-btn"),
            ], className="query-input-row chat-input-row", style={"display": "flex", "gap": "12px", "width": "100%"}),
        ], className="genie-main"),
    ], className="genie-panel", id="chat-section"),

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
            html.Div(dcc.Graph(id="chart-feedback-category", config={"responsive": True}), className="chart-card"),
            html.Div(dcc.Graph(id="chart-sentiment", config={"responsive": True}), className="chart-card"),
        ], className="chart-row"),

        html.Div([
            html.Div(dcc.Graph(id="chart-urgency", config={"responsive": True}), className="chart-card"),
            html.Div(dcc.Graph(id="chart-product-category", config={"responsive": True}), className="chart-card"),
        ], className="chart-row"),
    ], className="analytics-section"),

    html.Div([
        html.H2("Rolling Summary"),
        html.Div([
            dcc.Input(
                id="summary-query-input",
                type="text",
                placeholder="e.g. 'Summarize Q1 2022' or 'How was sentiment from Jan to March 2022'",
                className="genie-input",
                style={"width": "100%", "boxSizing": "border-box", "flex": "1"}
            ),
            html.Button("Get Summary", id="summary-ask-btn", className="genie-ask-btn"),
        ], className="query-input-row", style={"display": "flex", "gap": "12px", "width": "100%"}),
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
    breakdown, chart, summary, or unsupported response."""
    if not question:
        return html.Span("Type a question first.", className="no-questions-text")

    schema = get_schema_info()
    parsed = parse_natural_language_query(question, schema)
    result = run_parsed_query(parsed)
    return render_answer_card(result)

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


@app.callback(
    [Output("chat-thread", "children", allow_duplicate=True),
     Output("chat-history-store", "data", allow_duplicate=True),
     Output("chat-input", "value")],
    Input("chat-send-btn", "n_clicks"),
    [State("chat-input", "value"),
     State("chat-history-store", "data")],
    prevent_initial_call=True
)
def handle_chat_message(n_clicks, message, history):
    """Route one chat message through the same parsing + query-execution
    pipeline used by the Genie panel and Rolling Summary box, append the
    turn to the running session history, and re-render the whole thread."""
    if not message:
        return dash.no_update, dash.no_update, dash.no_update

    history = history or []
    schema = get_schema_info()
    parsed = parse_natural_language_query(message, schema, conversation_history=history)
    result = run_parsed_query(parsed)

    history = history + [
        {"role": "user", "text": message},
        {"role": "assistant", **result},
    ]

    return render_chat_thread(history), history, ""


@app.callback(
    [Output("chat-thread", "children", allow_duplicate=True),
     Output("chat-history-store", "data", allow_duplicate=True)],
    Input("chat-new-btn", "n_clicks"),
    prevent_initial_call=True
)
def handle_new_chat(n_clicks):
    """Clear the current session's conversation."""
    return render_chat_thread([]), []


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

    if parsed["query_type"] != "summary":
        return html.Div(parsed["explanation"], className="answer-explanation")

    result = run_summary_query(parsed)
    return render_answer_card(result)


if __name__ == "__main__":
    app.run(debug=True, port=8050)
