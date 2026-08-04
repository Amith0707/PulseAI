import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import pandas as pd
from aggregation import get_aggregated_stats
from schema_info import get_table_preview, ALLOWED_TABLES

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
])


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


if __name__ == "__main__":
    app.run(debug=True, port=8050)