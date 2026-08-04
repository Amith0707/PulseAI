import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import pandas as pd
from aggregation import get_aggregated_stats

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
    html.H1("PulseAI — Weekly Feedback Dashboard"),

    html.Div([
        html.Div([
            html.Label("Product category"),
            dcc.Dropdown(
                id="filter-product-category",
                options=[{"label": c, "value": c} for c in PRODUCT_CATEGORIES],
                multi=True,
                placeholder="All product categories"
            ),
        ], style={"width": "23%", "display": "inline-block", "marginRight": "2%"}),

        html.Div([
            html.Label("Feedback category"),
            dcc.Dropdown(
                id="filter-feedback-category",
                options=[{"label": c, "value": c} for c in FEEDBACK_CATEGORIES],
                multi=True,
                placeholder="All feedback categories"
            ),
        ], style={"width": "23%", "display": "inline-block", "marginRight": "2%"}),

        html.Div([
            html.Label("Sentiment"),
            dcc.Dropdown(
                id="filter-sentiment",
                options=[{"label": s, "value": s} for s in SENTIMENTS],
                multi=True,
                placeholder="All sentiments"
            ),
        ], style={"width": "23%", "display": "inline-block", "marginRight": "2%"}),

        html.Div([
            html.Label("Urgency"),
            dcc.Dropdown(
                id="filter-urgency",
                options=[{"label": u, "value": u} for u in URGENCIES],
                multi=True,
                placeholder="All urgency levels"
            ),
        ], style={"width": "23%", "display": "inline-block"}),
    ], style={"marginBottom": "30px"}),

    html.Div(id="total-count-display", style={"fontSize": "18px", "marginBottom": "20px"}),

    html.Div([
        dcc.Graph(id="chart-feedback-category", style={"width": "48%", "display": "inline-block"}),
        dcc.Graph(id="chart-sentiment", style={"width": "48%", "display": "inline-block"}),
    ]),

    html.Div([
        dcc.Graph(id="chart-urgency", style={"width": "48%", "display": "inline-block"}),
        dcc.Graph(id="chart-product-category", style={"width": "48%", "display": "inline-block"}),
    ]),
])


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
    """Recompute all charts whenever any filter dropdown changes.

    Since Dash's multi-select dropdowns return a list of selected values
    (or None if empty), this fetches the full dataset once and applies
    all selected filters together via pandas boolean indexing, rather
    than calling get_aggregated_stats with single-value filters.

    Parameters
    ----------
    product_categories : list of str or None
        Selected product categories to filter to, or None for all.
    feedback_categories : list of str or None
        Selected feedback categories to filter to, or None for all.
    sentiments : list of str or None
        Selected sentiments to filter to, or None for all.
    urgencies : list of str or None
        Selected urgency levels to filter to, or None for all.

    Returns
    -------
    tuple
        (total count text, feedback category figure, sentiment figure,
        urgency figure, product category figure)
    """
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