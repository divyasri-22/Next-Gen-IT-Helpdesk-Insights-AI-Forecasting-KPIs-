<<<<<<< HEAD
from pathlib import Path
from datetime import datetime, timedelta
=======
import pathlib
from datetime import datetime, timedelta
import math
>>>>>>> 72ba820ed17af07eec6034180189e486d24c3c5e

import numpy as np
import pandas as pd
import plotly.express as px
<<<<<<< HEAD
from dash import Dash, dcc, html, dash_table, Input, Output

# -----------------------------------------------------------------------------
# Paths  (app.py and data folder are in the same "data-orchestrator core" folder)
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "tech_support_tickets.csv"   # <= your CSV here


# -----------------------------------------------------------------------------
# Load & prepare data
# -----------------------------------------------------------------------------
=======
import plotly.graph_objects as go

from dash import Dash, html, dcc, Input, Output

# -----------------------------------------------------------------------------
# Paths & Data
# -----------------------------------------------------------------------------
BASE_DIR = pathlib.Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "tech_support_tickets.csv"


>>>>>>> 72ba820ed17af07eec6034180189e486d24c3c5e
def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"CSV not found at: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

<<<<<<< HEAD
    # Normalise column names just in case
    df.columns = [c.strip().lower() for c in df.columns]

    # Parse dates if present
=======
    # normalize column names a bit
    df.columns = [c.strip() for c in df.columns]

    # parse dates if present
>>>>>>> 72ba820ed17af07eec6034180189e486d24c3c5e
    for col in ["created_at", "resolved_at"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

<<<<<<< HEAD
    # Ensure helpful fallback columns
    if "ticket_id" not in df.columns:
        df["ticket_id"] = np.arange(1, len(df) + 1)
=======
    # ------------------------------------------------------------------
    # Auto-create a simple sentiment column if it doesn't exist
    # ------------------------------------------------------------------
    if "sentiment" not in df.columns:
        text_col = None
        for cand in ["description", "issue_sum", "subject", "title", "body"]:
            if cand in df.columns:
                text_col = cand
                break

        if text_col is not None:
            pos_words = {
                "resolved",
                "fixed",
                "working",
                "thank",
                "great",
                "good",
                "fast",
                "quick",
                "helpful",
            }
            neg_words = {
                "error",
                "fail",
                "failure",
                "crash",
                "slow",
                "issue",
                "problem",
                "cannot",
                "unable",
                "down",
                "not working",
            }

            def simple_sentiment(text: str) -> str:
                if not isinstance(text, str):
                    return "Neutral"
                low = text.lower()
                score = 0
                for w in pos_words:
                    if w in low:
                        score += 1
                for w in neg_words:
                    if w in low:
                        score -= 1
                if score > 0:
                    return "Positive"
                if score < 0:
                    return "Negative"
                return "Neutral"

            df["sentiment"] = df[text_col].apply(simple_sentiment)
        else:
            df["sentiment"] = "Neutral"
>>>>>>> 72ba820ed17af07eec6034180189e486d24c3c5e

    return df


df_raw = load_data()

<<<<<<< HEAD

# -----------------------------------------------------------------------------
# Helper to filter data
# -----------------------------------------------------------------------------
def apply_filters(
    df: pd.DataFrame,
    status_values,
    priority_values,
=======
# -----------------------------------------------------------------------------
# Theme definitions (simple dark / light)
# -----------------------------------------------------------------------------
THEMES = {
    "dark": {
        "name": "dark",
        "background": "#020617",
        "card_bg": "rgba(15, 23, 42, 0.98)",
        "highlight": "#22c55e",
        "text": "#e5e7eb",
        "muted": "#9ca3af",
        "plot_template": "plotly_dark",
    },
    "light": {
        "name": "light",
        "background": "#f3f4f6",
        "card_bg": "#ffffff",
        "highlight": "#16a34a",
        "text": "#111827",
        "muted": "#6b7280",
        "plot_template": "plotly_white",
    },
}

# -----------------------------------------------------------------------------
# Dash app setup
# -----------------------------------------------------------------------------
app = Dash(__name__)
app.title = "Tech Support Analytics Dashboard"

# Custom CSS / HTML shell
app.index_string = """
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
        body {
            margin: 0;
            font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            background: radial-gradient(circle at top left, #0f172a 0, #020617 40%, #000 100%);
            color: #e5e7eb;
        }
        .app-root {
            padding: 24px 32px 40px 32px;
            max-width: 1400px;
            margin: 0 auto;
        }
        .main-title {
            font-size: 2.6rem;
            font-weight: 800;
            letter-spacing: 0.15em;
            text-transform: uppercase;
            background: linear-gradient(90deg, #22c55e, #a855f7, #f97316);
            -webkit-background-clip: text;
            color: transparent;
        }
        .subtitle {
            margin-top: 4px;
            color: #9ca3af;
            font-size: 0.9rem;
        }
        .filters-row {
            margin-top: 20px;
            padding: 12px 16px 18px 16px;
            border-radius: 18px;
            background: rgba(15, 23, 42, 0.98);
            border: 1px solid rgba(55, 65, 81, 0.9);
            box-shadow: 0 22px 50px rgba(15, 23, 42, 0.95);
        }
        .metric-card {
            background: radial-gradient(circle at top left, #22c55e, #0f172a 60%, #020617);
            padding: 14px 18px;
            border-radius: 16px;
            border: 1px solid rgba(148, 163, 184, 0.6);
            box-shadow: 0 22px 55px rgba(15, 23, 42, 0.95);
        }
        .metric-label {
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #cbd5f5;
        }
        .metric-value {
            font-size: 2rem;
            font-weight: 700;
        }
        .metric-sub {
            font-size: 0.75rem;
            color: #e5e7eb;
            opacity: 0.8;
        }
        .section-title {
            font-size: 1.05rem;
            font-weight: 700;
            margin-bottom: 8px;
        }
        .section-block {
            margin-top: 26px;
        }
        .chart-card {
            background: rgba(15, 23, 42, 0.98);
            border-radius: 18px;
            border: 1px solid rgba(55, 65, 81, 0.9);
            box-shadow: 0 20px 55px rgba(15, 23, 42, 0.95);
            padding: 8px 12px 4px 12px;
        }
        .sentiment-samples {
            font-size: 0.85rem;
        }
        a, a:visited {
            color: #93c5fd;
        }

        /* -----------------------------
           DROPDOWN styling – always readable
           ----------------------------- */
        .Select-control {
            background-color: #020617 !important;
            border-color: #4b5563 !important;
            color: #e5e7eb !important;
        }
        .Select-menu-outer {
            background-color: #020617 !important;
            border-color: #4b5563 !important;
            color: #e5e7eb !important;
            z-index: 9999;
        }
        .Select-value {
            background-color: #111827 !important;
            border-color: #4b5563 !important;
        }
        .Select-value-label {
            color: #e5e7eb !important;
        }
        .Select-placeholder {
            color: #9ca3af !important;
        }
        .Select-arrow-zone, .Select-arrow {
            border-color: #e5e7eb transparent transparent !important;
        }

        /* Date picker text color */
        .DateInput_input, .DateRangePickerInput {
            background-color: #020617;
            color: #e5e7eb;
        }

        /* Simple KPI number fade-in */
        .kpi-animated {
            transition: all 0.4s ease-out;
        }
        </style>
    </head>
    <body>
        <div class="app-root">
            {%app_entry%}
        </div>
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
"""

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def get_filter_options(col_name: str):
    if col_name not in df_raw.columns:
        return []
    vals = sorted([v for v in df_raw[col_name].dropna().unique().tolist()])
    return [{"label": str(v), "value": v} for v in vals]


def apply_filters(
    df: pd.DataFrame,
    status_values,
>>>>>>> 72ba820ed17af07eec6034180189e486d24c3c5e
    agent_values,
    start_date,
    end_date,
) -> pd.DataFrame:
<<<<<<< HEAD
    out = df.copy()

    if status_values:
        out = out[out["status"].isin(status_values)]

    if priority_values:
        out = out[out["priority"].isin(priority_values)]

    if agent_values:
        out = out[out["agent"].isin(agent_values)]

    if start_date and end_date and "created_at" in out.columns:
        start = datetime.fromisoformat(start_date).replace(hour=0, minute=0, second=0)
        end = datetime.fromisoformat(end_date).replace(hour=23, minute=59, second=59)
        out = out[(out["created_at"] >= start) & (out["created_at"] <= end)]

    return out


# -----------------------------------------------------------------------------
# Dash app + layout
# -----------------------------------------------------------------------------
app = Dash(__name__)

# Compute dropdown options from raw data
status_options = (
    sorted(df_raw["status"].dropna().unique().tolist()) if "status" in df_raw.columns else []
)
priority_options = (
    sorted(df_raw["priority"].dropna().unique().tolist()) if "priority" in df_raw.columns else []
)
agent_options = (
    sorted(df_raw["agent"].dropna().unique().tolist()) if "agent" in df_raw.columns else []
)

# Date range defaults
if "created_at" in df_raw.columns:
    min_date = df_raw["created_at"].min().date()
    max_date = df_raw["created_at"].max().date()
else:
    min_date = max_date = None

app.layout = html.Div(
    style={
        "backgroundColor": "#020617",
        "color": "#e5e7eb",
        "minHeight": "100vh",
        "padding": "20px 40px",
        "fontFamily": "system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
    },
    children=[
        # TITLE
        html.Div(
            [
                html.H1(
                    "Tech Support Analytics Dashboard",
                    style={
                        "fontSize": "30px",
                        "fontWeight": "800",
                        "letterSpacing": "0.15em",
                        "textTransform": "uppercase",
                        "background": "linear-gradient(90deg,#60a5fa,#a855f7,#f97316)",
                        "WebkitBackgroundClip": "text",
                        "color": "transparent",
                        "marginBottom": "4px",
                    },
                ),
                html.Div(
                    "Use the filters to explore tickets, SLA, agents, sentiment, keywords & forecast.",
                    style={"fontSize": "14px", "color": "#9ca3af"},
                ),
            ],
            style={"marginBottom": "20px"},
        ),

        # FILTERS
        html.Div(
            [
                # Status
                html.Div(
                    [
                        html.Label("Status", style={"fontSize": "13px"}),
                        dcc.Dropdown(
                            id="status-filter",
                            options=[{"label": s, "value": s} for s in status_options],
                            value=status_options,
                            multi=True,
                            placeholder="Select status",
                            style={"backgroundColor": "#020617", "color": "#000"},
                        ),
                    ],
                    style={"flex": 1, "marginRight": "8px"},
                ),

                # Priority
                html.Div(
                    [
                        html.Label("Priority", style={"fontSize": "13px"}),
                        dcc.Dropdown(
                            id="priority-filter",
                            options=[{"label": p, "value": p} for p in priority_options],
                            value=priority_options,
                            multi=True,
                            placeholder="Select priority",
                            style={"backgroundColor": "#020617", "color": "#000"},
                        ),
                    ],
                    style={"flex": 1, "marginRight": "8px"},
                ),

                # Agent
                html.Div(
                    [
                        html.Label("Agent", style={"fontSize": "13px"}),
                        dcc.Dropdown(
                            id="agent-filter",
                            options=[{"label": a, "value": a} for a in agent_options],
                            value=agent_options,
                            multi=True,
                            placeholder="Select agent",
                            style={"backgroundColor": "#020617", "color": "#000"},
                        ),
                    ],
                    style={"flex": 1, "marginRight": "8px"},
                ),

                # Date range
                html.Div(
                    [
                        html.Label("Created date range", style={"fontSize": "13px"}),
                        dcc.DatePickerRange(
                            id="date-filter",
                            start_date=min_date,
                            end_date=max_date,
                            display_format="YYYY-MM-DD",
                            style={"fontSize": "12px"},
                        ),
                    ],
                    style={"flex": 1},
                ),
            ],
            style={
                "display": "flex",
                "flexDirection": "row",
                "marginBottom": "18px",
                "gap": "8px",
            },
        ),

        # KEY METRICS
        html.Div(
            [
                html.Div(
                    [
                        html.Div("Total Tickets", style={"fontSize": "13px", "color": "#9ca3af"}),
                        html.Div(
                            id="metric-total",
                            style={"fontSize": "26px", "fontWeight": "700"},
                        ),
                    ],
                    style={
                        "flex": 1,
                        "padding": "12px 16px",
                        "borderRadius": "14px",
                        "border": "1px solid #1f2937",
                        "background": "radial-gradient(circle at top left,#3b82f6,#020617)",
                        "boxShadow": "0 18px 45px rgba(15,23,42,0.9)",
                    },
                ),
                html.Div(
                    [
                        html.Div(
                            "Avg Resolution Time (min)",
                            style={"fontSize": "13px", "color": "#9ca3af"},
                        ),
                        html.Div(
                            id="metric-avg-res",
                            style={"fontSize": "26px", "fontWeight": "700"},
                        ),
                    ],
                    style={
                        "flex": 1,
                        "padding": "12px 16px",
                        "borderRadius": "14px",
                        "border": "1px solid #1f2937",
                        "background": "radial-gradient(circle at top left,#22c55e,#020617)",
                        "boxShadow": "0 18px 45px rgba(15,23,42,0.9)",
                    },
                ),
                html.Div(
                    [
                        html.Div(
                            "Resolved Ticket %",
                            style={"fontSize": "13px", "color": "#9ca3af"},
                        ),
                        html.Div(
                            id="metric-res-pct",
                            style={"fontSize": "26px", "fontWeight": "700"},
                        ),
                    ],
                    style={
                        "flex": 1,
                        "padding": "12px 16px",
                        "borderRadius": "14px",
                        "border": "1px solid #1f2937",
                        "background": "radial-gradient(circle at top left,#eab308,#020617)",
                        "boxShadow": "0 18px 45px rgba(15,23,42,0.9)",
                    },
                ),
            ],
            style={"display": "flex", "gap": "12px", "marginBottom": "24px"},
        ),

        # ROW 1: Status + SLA
        html.Div(
            [
                html.Div(
                    [
                        html.H3("Ticket Status Distribution", style={"fontSize": "16px"}),
                        dcc.Graph(id="status-chart"),
                    ],
                    style={"flex": 1, "marginRight": "10px"},
                ),
                html.Div(
                    [
                        html.H3("SLA Heatmap (by Day & Priority)", style={"fontSize": "16px"}),
                        dcc.Graph(id="sla-chart"),
                        html.Div(id="sla-text", style={"fontSize": "12px", "color": "#9ca3af"}),
                    ],
                    style={"flex": 1},
                ),
            ],
            style={"display": "flex", "marginBottom": "24px"},
        ),

        # ROW 2: Agent + Sentiment
        html.Div(
            [
                html.Div(
                    [
                        html.H3("Agent Performance", style={"fontSize": "16px"}),
                        dcc.Graph(id="agent-chart"),
                    ],
                    style={"flex": 1, "marginRight": "10px"},
                ),
                html.Div(
                    [
                        html.H3("Sentiment Analytics", style={"fontSize": "16px"}),
                        dcc.Graph(id="sentiment-chart"),
                        html.Div(
                            [
                                html.Label(
                                    "Preview tickets for sentiment:",
                                    style={"fontSize": "13px"},
                                ),
                                dcc.Dropdown(
                                    id="sentiment-select",
                                    placeholder="Select sentiment",
                                    style={"backgroundColor": "#020617", "color": "#000"},
                                ),
                                dash_table.DataTable(
                                    id="sentiment-table",
                                    page_size=5,
                                    style_header={
                                        "backgroundColor": "#111827",
                                        "color": "#e5e7eb",
                                        "fontWeight": "600",
                                    },
                                    style_data={
                                        "backgroundColor": "#020617",
                                        "color": "#e5e7eb",
                                        "border": "1px solid #111827",
                                        "fontSize": "12px",
                                    },
                                ),
                            ]
                        ),
                    ],
                    style={"flex": 1},
                ),
            ],
            style={"display": "flex", "marginBottom": "24px"},
        ),

        # ROW 3: Keywords + Forecast
        html.Div(
            [
                html.Div(
                    [
                        html.H3("Top Issue Keywords", style={"fontSize": "16px"}),
                        dcc.Graph(id="keywords-chart"),
                    ],
                    style={"flex": 1, "marginRight": "10px"},
                ),
                html.Div(
                    [
                        html.H3("Ticket Volume & 7-day Forecast", style={"fontSize": "16px"}),
                        dcc.Graph(id="volume-chart"),
                    ],
                    style={"flex": 1},
                ),
            ],
            style={"display": "flex", "marginBottom": "24px"},
        ),
    ],
)


# -----------------------------------------------------------------------------
# Callback: update everything based on filters
=======
    filtered = df.copy()

    if status_values and "status" in filtered.columns:
        filtered = filtered[filtered["status"].isin(status_values)]

    if agent_values and "agent" in filtered.columns:
        filtered = filtered[filtered["agent"].isin(agent_values)]

    if start_date and end_date and "created_at" in filtered.columns:
        start = datetime.combine(pd.to_datetime(start_date).date(), datetime.min.time())
        end = datetime.combine(pd.to_datetime(end_date).date(), datetime.max.time())
        filtered = filtered[
            (filtered["created_at"] >= start) & (filtered["created_at"] <= end)
        ]

    return filtered


# -----------------------------------------------------------------------------
# Layout
# -----------------------------------------------------------------------------
app.layout = html.Div(
    children=[
        # Theme + title row
        html.Div(
            style={
                "display": "flex",
                "justifyContent": "space-between",
                "alignItems": "center",
                "gap": "16px",
            },
            children=[
                html.Div(
                    children=[
                        html.Div(
                            "TECH SUPPORT ANALYTICS DASHBOARD", className="main-title"
                        ),
                        html.Div(
                            "Interactive filters, animated KPIs, live sentiment & forecasting.",
                            className="subtitle",
                        ),
                    ]
                ),
                # Theme toggle + tiny "animation" (GIF instead of Lottie to avoid extra deps)
                html.Div(
                    style={
                        "display": "flex",
                        "flexDirection": "column",
                        "alignItems": "flex-end",
                        "gap": "4px",
                    },
                    children=[
                        dcc.RadioItems(
                            id="theme-toggle",
                            options=[
                                {"label": "🌙 Dark", "value": "dark"},
                                {"label": "☀️ Light", "value": "light"},
                            ],
                            value="dark",
                            labelStyle={
                                "display": "inline-block",
                                "marginRight": "8px",
                                "cursor": "pointer",
                            },
                            inputStyle={"marginRight": "4px"},
                            style={"fontSize": "0.8rem"},
                        ),
                        html.Img(
                            src="https://assets9.lottiefiles.com/packages/lf20_u8o7BL.json.gif",
                            style={
                                "height": "40px",
                                "borderRadius": "999px",
                                "boxShadow": "0 0 15px rgba(56,189,248,0.6)",
                            },
                        ),
                    ],
                ),
            ],
        ),

        # Filters row
        html.Div(
            className="filters-row",
            children=[
                html.Div(
                    style={"display": "flex", "gap": "12px", "flexWrap": "wrap"},
                    children=[
                        html.Div(
                            style={"flex": "1 1 220px"},
                            children=[
                                html.Label("Status"),
                                dcc.Dropdown(
                                    id="status-filter",
                                    options=get_filter_options("status"),
                                    value=[
                                        opt["value"]
                                        for opt in get_filter_options("status")
                                    ],
                                    multi=True,
                                    placeholder="Select status",
                                ),
                            ],
                        ),
                        html.Div(
                            style={"flex": "1 1 260px"},
                            children=[
                                html.Label("Agent"),
                                dcc.Dropdown(
                                    id="agent-filter",
                                    options=get_filter_options("agent"),
                                    value=[
                                        opt["value"]
                                        for opt in get_filter_options("agent")
                                    ],
                                    multi=True,
                                    placeholder="Select agents",
                                ),
                            ],
                        ),
                        html.Div(
                            style={"flex": "1 1 260px"},
                            children=[
                                html.Label("Created date range"),
                                dcc.DatePickerRange(
                                    id="date-filter",
                                    start_date=df_raw["created_at"].min().date()
                                    if "created_at" in df_raw.columns
                                    else None,
                                    end_date=df_raw["created_at"].max().date()
                                    if "created_at" in df_raw.columns
                                    else None,
                                ),
                            ],
                        ),
                    ],
                )
            ],
        ),

        # KPI row
        html.Div(
            style={
                "display": "grid",
                "gridTemplateColumns": "repeat(3, minmax(0, 1fr))",
                "gap": "16px",
                "marginTop": "22px",
            },
            children=[
                html.Div(
                    className="metric-card",
                    children=[
                        html.Div("Total Tickets", className="metric-label"),
                        html.Div(id="metric-total", className="metric-value kpi-animated"),
                        html.Div(
                            "After applying current filters",
                            className="metric-sub",
                        ),
                    ],
                ),
                html.Div(
                    className="metric-card",
                    children=[
                        html.Div("Avg Resolution Time (min)", className="metric-label"),
                        html.Div(
                            id="metric-avg-resolution",
                            className="metric-value kpi-animated",
                        ),
                        html.Div(
                            "Based on resolved tickets",
                            className="metric-sub",
                        ),
                    ],
                ),
                html.Div(
                    className="metric-card",
                    children=[
                        html.Div("Resolved Ticket %", className="metric-label"),
                        html.Div(
                            id="metric-resolved-pct",
                            className="metric-value kpi-animated",
                        ),
                        html.Div(
                            "Resolved / total in current view",
                            className="metric-sub",
                        ),
                    ],
                ),
            ],
        ),

        # What-if slider row
        html.Div(
            className="section-block",
            children=[
                html.Div(
                    className="chart-card",
                    children=[
                        html.Div(
                            "What-if: ticket growth vs agents needed",
                            className="section-title",
                        ),
                        html.Div(
                            style={
                                "display": "grid",
                                "gridTemplateColumns": "2.2fr 1fr",
                                "gap": "18px",
                                "alignItems": "center",
                            },
                            children=[
                                dcc.Slider(
                                    id="growth-slider",
                                    min=0,
                                    max=200,
                                    step=10,
                                    value=0,
                                    marks={
                                        0: "0%",
                                        50: "50%",
                                        100: "100%",
                                        150: "150%",
                                        200: "200%",
                                    },
                                    tooltip={
                                        "placement": "bottom",
                                        "always_visible": True,
                                    },
                                ),
                                html.Div(
                                    className="metric-card",
                                    style={"marginTop": "0"},
                                    children=[
                                        html.Div(
                                            "Projected Agents Needed",
                                            className="metric-label",
                                        ),
                                        html.Div(
                                            id="whatif-agents",
                                            className="metric-value kpi-animated",
                                        ),
                                        html.Div(
                                            "Based on current staffing & growth %",
                                            className="metric-sub",
                                        ),
                                    ],
                                ),
                            ],
                        ),
                    ],
                )
            ],
        ),

        # Row: Status pie + SLA heatmap
        html.Div(
            className="section-block",
            children=[
                html.Div(
                    style={
                        "display": "grid",
                        "gridTemplateColumns": "minmax(0, 1.1fr) minmax(0, 1.3fr)",
                        "gap": "16px",
                    },
                    children=[
                        html.Div(
                            className="chart-card",
                            children=[
                                html.Div(
                                    "Ticket Status Distribution",
                                    className="section-title",
                                ),
                                dcc.Graph(
                                    id="status-pie",
                                    style={"height": "360px"},
                                ),
                            ],
                        ),
                        html.Div(
                            className="chart-card",
                            children=[
                                html.Div(
                                    "Tickets per Day & Status (heatmap)",
                                    className="section-title",
                                ),
                                dcc.Graph(
                                    id="sla-heatmap",
                                    style={"height": "360px"},
                                ),
                            ],
                        ),
                    ],
                )
            ],
        ),

        # Row: Agent + Sentiment
        html.Div(
            className="section-block",
            children=[
                html.Div(
                    style={
                        "display": "grid",
                        "gridTemplateColumns": "minmax(0, 1.1fr) minmax(0, 1.3fr)",
                        "gap": "16px",
                    },
                    children=[
                        html.Div(
                            className="chart-card",
                            children=[
                                html.Div("Agent Performance", className="section-title"),
                                dcc.Graph(
                                    id="agent-performance",
                                    style={"height": "360px"},
                                ),
                            ],
                        ),
                        html.Div(
                            className="chart-card",
                            children=[
                                html.Div("Sentiment Overview", className="section-title"),
                                dcc.Graph(
                                    id="sentiment-bar",
                                    style={"height": "360px"},
                                ),
                                html.Div(
                                    style={"marginTop": "8px"},
                                    children=[
                                        html.Label(
                                            "View sample tickets by sentiment (from full dataset):"
                                        ),
                                        dcc.Dropdown(
                                            id="sentiment-choice",
                                            options=get_filter_options("sentiment"),
                                            value=(
                                                get_filter_options("sentiment")[0][
                                                    "value"
                                                ]
                                                if get_filter_options("sentiment")
                                                else None
                                            ),
                                            placeholder="Select sentiment",
                                            style={"marginTop": "4px"},
                                        ),
                                        html.Div(
                                            id="sentiment-samples",
                                            className="sentiment-samples",
                                            style={"marginTop": "8px"},
                                        ),
                                    ],
                                ),
                            ],
                        ),
                    ],
                )
            ],
        ),

        # Row: Keywords + Forecast
        html.Div(
            className="section-block",
            children=[
                html.Div(
                    style={
                        "display": "grid",
                        "gridTemplateColumns": "minmax(0, 1.1fr) minmax(0, 1.3fr)",
                        "gap": "16px",
                    },
                    children=[
                        html.Div(
                            className="chart-card",
                            children=[
                                html.Div(
                                    "Top Issue Keywords (word-cloud style)",
                                    className="section-title",
                                ),
                                dcc.Graph(
                                    id="keywords-bar",
                                    style={"height": "360px"},
                                ),
                            ],
                        ),
                        html.Div(
                            className="chart-card",
                            children=[
                                html.Div(
                                    "Ticket Volume & 7-Day Forecast",
                                    className="section-title",
                                ),
                                dcc.Graph(
                                    id="volume-forecast",
                                    style={"height": "360px"},
                                ),
                            ],
                        ),
                    ],
                )
            ],
        ),
    ]
)

# -----------------------------------------------------------------------------
# Main dashboard callback  (filters -> metrics + charts + what-if)
>>>>>>> 72ba820ed17af07eec6034180189e486d24c3c5e
# -----------------------------------------------------------------------------
@app.callback(
    [
        Output("metric-total", "children"),
<<<<<<< HEAD
        Output("metric-avg-res", "children"),
        Output("metric-res-pct", "children"),
        Output("status-chart", "figure"),
        Output("sla-chart", "figure"),
        Output("sla-text", "children"),
        Output("agent-chart", "figure"),
        Output("sentiment-chart", "figure"),
        Output("sentiment-select", "options"),
        Output("sentiment-select", "value"),
        Output("sentiment-table", "data"),
        Output("sentiment-table", "columns"),
        Output("keywords-chart", "figure"),
        Output("volume-chart", "figure"),
    ],
    [
        Input("status-filter", "value"),
        Input("priority-filter", "value"),
        Input("agent-filter", "value"),
        Input("date-filter", "start_date"),
        Input("date-filter", "end_date"),
        Input("sentiment-select", "value"),
    ],
)
def update_dashboard(
    status_values, priority_values, agent_values, start_date, end_date, sentiment_value
):
    # Ensure lists
    if isinstance(status_values, str) or status_values is None:
        status_values = status_options
    if isinstance(priority_values, str) or priority_values is None:
        priority_values = priority_options
    if isinstance(agent_values, str) or agent_values is None:
        agent_values = agent_options

    df = apply_filters(df_raw, status_values, priority_values, agent_values, start_date, end_date)

    # ===== METRICS =====
    total_tickets = len(df)

    # Avg resolution time
    if {"created_at", "resolved_at"}.issubset(df.columns):
        resolved = df.dropna(subset=["created_at", "resolved_at"]).copy()
        if not resolved.empty:
            resolved["resolution_minutes"] = (
                resolved["resolved_at"] - resolved["created_at"]
            ).dt.total_seconds() / 60.0
            avg_res_val = resolved["resolution_minutes"].mean()
            avg_res_str = f"{avg_res_val:,.1f}"
        else:
            avg_res_str = "N/A"
    else:
        avg_res_str = "N/A"

    # Resolution percentage
    if "status" in df.columns and total_tickets > 0:
        status_lower = df["status"].astype(str).str.lower()
        resolved_mask = status_lower.isin(["resolved", "closed", "done"])
        pct_val = resolved_mask.sum() / total_tickets * 100
        pct_str = f"{pct_val:,.1f}%"
    else:
        pct_str = "N/A"

    metric_total = f"{total_tickets:,}"
    metric_avg_res = avg_res_str
    metric_res_pct = pct_str

    # ===== STATUS PIE =====
    if "status" in df.columns and not df.empty:
        status_counts = df["status"].value_counts().reset_index(name="count")
        status_counts.rename(columns={"index": "status"}, inplace=True)
        status_fig = px.pie(
            status_counts,
=======
        Output("metric-avg-resolution", "children"),
        Output("metric-resolved-pct", "children"),
        Output("whatif-agents", "children"),
        Output("status-pie", "figure"),
        Output("sla-heatmap", "figure"),
        Output("agent-performance", "figure"),
        Output("sentiment-bar", "figure"),
        Output("keywords-bar", "figure"),
        Output("volume-forecast", "figure"),
    ],
    [
        Input("status-filter", "value"),
        Input("agent-filter", "value"),
        Input("date-filter", "start_date"),
        Input("date-filter", "end_date"),
        Input("growth-slider", "value"),
        Input("theme-toggle", "value"),
    ],
)
def update_dashboard(
    status_values,
    agent_values,
    start_date,
    end_date,
    growth_percent,
    theme_name,
):
    theme = THEMES.get(theme_name, THEMES["dark"])
    template = theme["plot_template"]
    text_color = theme["text"]

    df = apply_filters(df_raw, status_values, agent_values, start_date, end_date)

    # ---------- Metrics ----------
    total = len(df)

    # avg resolution
    if {"created_at", "resolved_at"}.issubset(df.columns) and not df.empty:
        resolved = df.dropna(subset=["created_at", "resolved_at"]).copy()
        if not resolved.empty:
            mins = (
                (resolved["resolved_at"] - resolved["created_at"])
                .dt.total_seconds()
                .div(60.0)
            )
            avg_res = f"{mins.mean():.1f}"
        else:
            avg_res = "N/A"
    else:
        avg_res = "N/A"

    # resolved percentage
    if "status" in df.columns and total > 0:
        status_lower = df["status"].astype(str).str.lower()
        resolved_mask = status_lower.isin(["resolved", "closed", "done"])
        pct = resolved_mask.mean() * 100.0
        resolved_pct = f"{pct:.1f}%"
    else:
        resolved_pct = "N/A"

    # ---------- What-if agents ----------
    if "agent" in df.columns and df["agent"].nunique() > 0:
        current_agents = df["agent"].nunique()
        factor = 1 + (growth_percent or 0) / 100.0
        needed = int(math.ceil(current_agents * factor))
        whatif_str = f"{needed} (now: {current_agents})"
    else:
        whatif_str = "N/A"

    # ---------- Status pie ----------
    if "status" in df.columns and not df.empty:
        counts = df["status"].value_counts().reset_index(name="count")
        counts.rename(columns={"index": "status"}, inplace=True)
        fig_status = px.pie(
            counts,
>>>>>>> 72ba820ed17af07eec6034180189e486d24c3c5e
            names="status",
            values="count",
            hole=0.4,
        )
<<<<<<< HEAD
        status_fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#020617",
            plot_bgcolor="#020617",
            margin=dict(l=0, r=0, t=30, b=0),
        )
    else:
        status_fig = px.pie(names=["No data"], values=[1])
        status_fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#020617",
            plot_bgcolor="#020617",
        )

    # ===== SLA HEATMAP =====
    sla_text = ""
    if {"created_at", "priority"}.issubset(df.columns) and not df.empty:
        df_heat = df.dropna(subset=["created_at", "priority"]).copy()
        df_heat["day"] = df_heat["created_at"].dt.date

        if "sla_breached" in df_heat.columns:
            pivot = (
                df_heat.groupby(["day", "priority"])["sla_breached"]
                .mean()
                .reset_index()
            )
            pivot["sla_breached_pct"] = pivot["sla_breached"] * 100
            sla_fig = px.density_heatmap(
                pivot,
                x="day",
                y="priority",
                z="sla_breached_pct",
                color_continuous_scale="Reds",
                labels={"sla_breached_pct": "SLA Breach %"},
            )
            overall_breach = pivot["sla_breached_pct"].mean()
            sla_text = f"Average SLA breach in current view: {overall_breach:,.1f}%"
        else:
            pivot = (
                df_heat.groupby(["day", "priority"])
                .size()
                .reset_index(name="count")
            )
            sla_fig = px.density_heatmap(
                pivot,
                x="day",
                y="priority",
                z="count",
                color_continuous_scale="Blues",
                labels={"count": "Ticket count"},
            )
            sla_text = "No 'sla_breached' column found; showing ticket counts only."
        sla_fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#020617",
            plot_bgcolor="#020617",
            margin=dict(l=0, r=0, t=30, b=0),
        )
    else:
        sla_fig = px.density_heatmap(x=[0], y=[0], z=[0])
        sla_fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#020617",
            plot_bgcolor="#020617",
            xaxis={"visible": False},
            yaxis={"visible": False},
            annotations=[
                dict(
                    text="No SLA data",
                    x=0.5,
                    y=0.5,
                    showarrow=False,
                    font=dict(color="white", size=14),
                    xref="paper",
                    yref="paper",
                )
            ],
        )
        sla_text = "No SLA data available."

    # ===== AGENT PERFORMANCE =====
    if "agent" in df.columns and not df.empty:
        perf = df.copy()
        if {"created_at", "resolved_at"}.issubset(perf.columns):
            perf = perf.dropna(subset=["created_at", "resolved_at"]).copy()
            if not perf.empty:
                perf["resolution_minutes"] = (
                    perf["resolved_at"] - perf["created_at"]
                ).dt.total_seconds() / 60.0
                agg = (
                    perf.groupby("agent")
                    .agg(
                        tickets=("agent", "size"),
                        avg_resolution=("resolution_minutes", "mean"),
                    )
                    .reset_index()
                )
            else:
                agg = pd.DataFrame(columns=["agent", "tickets", "avg_resolution"])
        else:
            agg = (
                perf.groupby("agent")
                .size()
                .reset_index(name="tickets")
            )
            agg["avg_resolution"] = np.nan

        if not agg.empty:
            agent_fig = px.bar(
                agg.sort_values("tickets", ascending=False),
                x="agent",
                y="tickets",
                hover_data=["avg_resolution"],
            )
            agent_fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="#020617",
                plot_bgcolor="#020617",
                margin=dict(l=0, r=0, t=30, b=40),
            )
        else:
            agent_fig = px.bar(x=[], y=[])
            agent_fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="#020617",
                plot_bgcolor="#020617",
                annotations=[
                    dict(
                        text="No agent data",
                        x=0.5,
                        y=0.5,
                        showarrow=False,
                        font=dict(color="white", size=14),
                        xref="paper",
                        yref="paper",
                    )
                ],
            )
    else:
        agent_fig = px.bar(x=[], y=[])
        agent_fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#020617",
            plot_bgcolor="#020617",
            annotations=[
                dict(
                    text="No agent column",
                    x=0.5,
                    y=0.5,
                    showarrow=False,
                    font=dict(color="white", size=14),
                    xref="paper",
                    yref="paper",
                )
            ],
        )

    # ===== SENTIMENT =====
    sentiment_options = []
    sentiment_fig = px.bar()
    table_data = []
    table_columns = []

    if "sentiment" in df.columns and not df.empty:
        counts = df["sentiment"].value_counts().reset_index(name="count")
        counts.rename(columns={"index": "sentiment"}, inplace=True)

        sentiment_fig = px.bar(
            counts,
            x="sentiment",
            y="count",
        )
        sentiment_fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#020617",
            plot_bgcolor="#020617",
            margin=dict(l=0, r=0, t=30, b=40),
        )

        sentiment_options = [
            {"label": s, "value": s} for s in sorted(df["sentiment"].dropna().unique())
        ]

        if sentiment_value is None and sentiment_options:
            sentiment_value = sentiment_options[0]["value"]

        if sentiment_value is not None:
            sample = df[df["sentiment"] == sentiment_value].copy()
            if not sample.empty:
                cols_to_show = [
                    c
                    for c in ["ticket_id", "agent", "status", "description"]
                    if c in sample.columns
                ]
                sample = sample[cols_to_show].head(10)
                table_data = sample.to_dict("records")
                table_columns = [{"name": c, "id": c} for c in cols_to_show]
    else:
        sentiment_fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#020617",
            plot_bgcolor="#020617",
            annotations=[
                dict(
                    text="No sentiment data",
                    x=0.5,
                    y=0.5,
                    showarrow=False,
                    font=dict(color="white", size=14),
                    xref="paper",
                    yref="paper",
                )
            ],
        )
        sentiment_value = None

    # ===== KEYWORDS =====
    if "description" in df_raw.columns:
        text = " ".join(df_raw["description"].astype(str).tolist()).lower()
        tokens = [t.strip(".,!?:;()[]") for t in text.split() if len(t) > 3]
        if tokens:
            freq = pd.Series(tokens).value_counts().reset_index(name="count")
            freq.rename(columns={"index": "word"}, inplace=True)
            top_n = freq.head(30)
            kw_fig = px.bar(
                top_n.sort_values("count"),
                x="count",
                y="word",
                orientation="h",
            )
            kw_fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="#020617",
                plot_bgcolor="#020617",
                margin=dict(l=0, r=10, t=30, b=40),
            )
        else:
            kw_fig = px.bar(x=[], y=[])
            kw_fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="#020617",
                plot_bgcolor="#020617",
                annotations=[
                    dict(
                        text="Not enough text for keywords",
                        x=0.5,
                        y=0.5,
                        showarrow=False,
                        font=dict(color="white", size=14),
                        xref="paper",
                        yref="paper",
                    )
                ],
            )
    else:
        kw_fig = px.bar(x=[], y=[])
        kw_fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#020617",
            plot_bgcolor="#020617",
            annotations=[
                dict(
                    text="No 'description' column",
                    x=0.5,
                    y=0.5,
                    showarrow=False,
                    font=dict(color="white", size=14),
                    xref="paper",
                    yref="paper",
                )
            ],
        )

    # ===== VOLUME & FORECAST =====
    if "created_at" in df_raw.columns:
        vol = df_raw.dropna(subset=["created_at"]).copy()
        vol["day"] = vol["created_at"].dt.date
        daily = (
            vol.groupby("day")
            .size()
            .reset_index(name="tickets")
            .sort_values("day")
        )

        if not daily.empty:
            daily["forecast"] = daily["tickets"].rolling(window=3, min_periods=1).mean()
            last_day = daily["day"].max()
            future_days = [last_day + timedelta(days=i) for i in range(1, 8)]
            last_forecast = daily["forecast"].iloc[-1]
            future = pd.DataFrame(
                {
                    "day": future_days,
                    "tickets": [np.nan] * len(future_days),
                    "forecast": [last_forecast] * len(future_days),
                }
            )
            combined = pd.concat([daily, future], ignore_index=True)
            vol_fig = px.line(
                combined,
                x="day",
                y=["tickets", "forecast"],
                labels={"value": "Tickets", "variable": "Series"},
            )
            vol_fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="#020617",
                plot_bgcolor="#020617",
                margin=dict(l=0, r=0, t=30, b=40),
            )
        else:
            vol_fig = px.line(x=[], y=[])
            vol_fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="#020617",
                plot_bgcolor="#020617",
                annotations=[
                    dict(
                        text="No created_at values",
                        x=0.5,
                        y=0.5,
                        showarrow=False,
                        font=dict(color="white", size=14),
                        xref="paper",
                        yref="paper",
                    )
                ],
            )
    else:
        vol_fig = px.line(x=[], y=[])
        vol_fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#020617",
            plot_bgcolor="#020617",
            annotations=[
                dict(
                    text="No 'created_at' column",
                    x=0.5,
                    y=0.5,
                    showarrow=False,
                    font=dict(color="white", size=14),
                    xref="paper",
                    yref="paper",
                )
            ],
        )

    return (
        metric_total,
        metric_avg_res,
        metric_res_pct,
        status_fig,
        sla_fig,
        sla_text,
        agent_fig,
        sentiment_fig,
        sentiment_options,
        sentiment_value,
        table_data,
        table_columns,
        kw_fig,
        vol_fig,
=======
        fig_status.update_layout(
            template=template,
            height=360,
            showlegend=True,
            font_color=text_color,
            margin=dict(l=10, r=10, t=10, b=10),
        )
    else:
        fig_status = go.Figure()
        fig_status.add_annotation(
            text="No status data for current filters.",
            showarrow=False,
        )
        fig_status.update_layout(
            template=template,
            height=360,
            font_color=text_color,
            margin=dict(l=10, r=10, t=10, b=10),
        )

    # ---------- Heatmap ----------
    if {"created_at", "status"}.issubset(df.columns) and not df.empty:
        df_heat = df.dropna(subset=["created_at", "status"]).copy()
        df_heat["day"] = df_heat["created_at"].dt.date

        agg = (
            df_heat.groupby(["day", "status"])
            .size()
            .reset_index(name="ticket_count")
        )
        pivot = agg.pivot(index="status", columns="day", values="ticket_count").fillna(0)

        fig_sla = go.Figure(
            data=go.Heatmap(
                z=pivot.values,
                x=[d.strftime("%Y-%m-%d") for d in pivot.columns],
                y=pivot.index.astype(str),
                colorbar=dict(title="Ticket count"),
            )
        )
        fig_sla.update_layout(
            template=template,
            height=360,
            margin=dict(l=40, r=10, t=10, b=40),
            xaxis_title="Day",
            yaxis_title="Status",
            font_color=text_color,
        )
    else:
        fig_sla = go.Figure()
        fig_sla.add_annotation(
            text="No data for current filters.",
            showarrow=False,
        )
        fig_sla.update_layout(
            template=template,
            height=360,
            font_color=text_color,
            margin=dict(l=10, r=10, t=10, b=10),
        )

    # ---------- Agent performance ----------
    if "agent" in df.columns and not df.empty:
        df_agents = df.copy()
        has_res_times = {"created_at", "resolved_at"}.issubset(df_agents.columns)
        if has_res_times:
            df_agents = df_agents.dropna(subset=["created_at", "resolved_at"]).copy()
            df_agents["resolution_mins"] = (
                df_agents["resolved_at"] - df_agents["created_at"]
            ).dt.total_seconds() / 60.0

        grouped = df.groupby("agent").size().reset_index(name="ticket_count")

        if has_res_times and not df_agents.empty:
            avg_res_by_agent = (
                df_agents.groupby("agent")["resolution_mins"]
                .mean()
                .reset_index(name="avg_resolution_mins")
            )
            grouped = grouped.merge(avg_res_by_agent, on="agent", how="left")
        else:
            grouped["avg_resolution_mins"] = np.nan

        grouped = grouped.sort_values("ticket_count", ascending=False)

        fig_agent = go.Figure()
        fig_agent.add_trace(
            go.Bar(
                x=grouped["agent"],
                y=grouped["ticket_count"],
                name="Tickets handled",
                yaxis="y1",
            )
        )

        if grouped["avg_resolution_mins"].notna().any():
            fig_agent.add_trace(
                go.Scatter(
                    x=grouped["agent"],
                    y=grouped["avg_resolution_mins"],
                    name="Avg resolution (min)",
                    mode="lines+markers",
                    yaxis="y2",
                )
            )

        fig_agent.update_layout(
            template=template,
            height=360,
            margin=dict(l=40, r=40, t=20, b=60),
            xaxis_title="Agent",
            yaxis=dict(title="Tickets handled"),
            yaxis2=dict(
                title="Avg resolution (min)",
                overlaying="y",
                side="right",
            ),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            font_color=text_color,
        )
    else:
        fig_agent = go.Figure()
        fig_agent.add_annotation(
            text="No agent data for current filters.",
            showarrow=False,
        )
        fig_agent.update_layout(
            template=template,
            height=360,
            font_color=text_color,
            margin=dict(l=10, r=10, t=10, b=10),
        )

    # ---------- Sentiment bar ----------
    if "sentiment" in df.columns and not df.empty:
        sent_counts = df["sentiment"].value_counts().reset_index(name="count")
        sent_counts.rename(columns={"index": "sentiment"}, inplace=True)
        fig_sent = px.bar(
            sent_counts,
            x="sentiment",
            y="count",
        )
        fig_sent.update_layout(
            template=template,
            height=360,
            margin=dict(l=40, r=10, t=20, b=40),
            xaxis_title="Sentiment",
            yaxis_title="Ticket count",
            font_color=text_color,
        )
    else:
        fig_sent = go.Figure()
        fig_sent.add_annotation(
            text="No sentiment data for current filters.",
            showarrow=False,
        )
        fig_sent.update_layout(
            template=template,
            height=360,
            font_color=text_color,
            margin=dict(l=10, r=10, t=10, b=10),
        )

    # ---------- Keywords bar ----------
    text_col = None
    for cand in ["issue_sum", "description", "subject", "title", "body"]:
        if cand in df.columns:
            text_col = cand
            break

    if text_col and not df.empty:
        from collections import Counter

        all_text = (
            df[text_col]
            .dropna()
            .astype(str)
            .str.lower()
            .str.replace(r"[^a-z0-9\\s]+", " ", regex=True)
        )

        words = []
        for line in all_text:
            words.extend(line.split())

        stop_words = {
            "the",
            "and",
            "for",
            "this",
            "that",
            "with",
            "you",
            "your",
            "are",
            "from",
            "have",
            "has",
            "when",
            "what",
            "why",
            "how",
            "can",
            "not",
            "cannot",
            "our",
            "please",
            "issue",
            "ticket",
            "help",
            "support",
        }

        words = [w for w in words if len(w) > 2 and w not in stop_words]

        if words:
            counts = Counter(words)
            top_words = counts.most_common(20)
            kw_df = pd.DataFrame(top_words, columns=["keyword", "count"])

            fig_kw = px.bar(
                kw_df.sort_values("count", ascending=True),
                x="count",
                y="keyword",
                orientation="h",
            )
            fig_kw.update_layout(
                template=template,
                height=360,
                margin=dict(l=80, r=20, t=20, b=40),
                xaxis_title="Frequency",
                yaxis_title="",
                font_color=text_color,
            )
        else:
            fig_kw = go.Figure()
            fig_kw.add_annotation(
                text="No keyword data available.",
                showarrow=False,
            )
            fig_kw.update_layout(
                template=template,
                height=360,
                font_color=text_color,
                margin=dict(l=10, r=10, t=10, b=10),
            )
    else:
        fig_kw = go.Figure()
        fig_kw.add_annotation(
            text="No keyword text column found.",
            showarrow=False,
        )
        fig_kw.update_layout(
            template=template,
            height=360,
            font_color=text_color,
            margin=dict(l=10, r=10, t=10, b=10),
        )

    # ---------- Volume & forecast (with solid black axes lines) ----------
    if "created_at" in df.columns and not df.empty:
        df_ts = df.dropna(subset=["created_at"]).copy()
        df_ts["day"] = df_ts["created_at"].dt.date

        daily_counts = (
            df_ts.groupby("day")
            .size()
            .reset_index(name="ticket_count")
            .sort_values("day")
        )

        full_days = pd.date_range(
            start=daily_counts["day"].min(),
            end=daily_counts["day"].max(),
            freq="D",
        )
        ts_full = (
            daily_counts.set_index("day")
            .reindex(full_days)
            .fillna(0.0)
            .rename_axis("day")
            .reset_index()
        )
        ts_full.rename(columns={"ticket_count": "count"}, inplace=True)

        x = np.arange(len(ts_full))
        y = ts_full["count"].values

        if len(ts_full) > 1 and np.any(y):
            coef = np.polyfit(x, y, 1)
            trend = np.poly1d(coef)

            future_days = 7
            x_future = np.arange(len(ts_full), len(ts_full) + future_days)
            y_future = np.clip(trend(x_future), a_min=0, a_max=None)

            future_dates = [
                ts_full["day"].max() + timedelta(days=i)
                for i in range(1, future_days + 1)
            ]

            fig_vol = go.Figure()
            fig_vol.add_trace(
                go.Scatter(
                    x=ts_full["day"],
                    y=ts_full["count"],
                    mode="lines+markers",
                    name="Actual tickets",
                )
            )
            fig_vol.add_trace(
                go.Scatter(
                    x=future_dates,
                    y=y_future,
                    mode="lines+markers",
                    name="Forecast (next 7 days)",
                    line=dict(dash="dash"),
                )
            )
        else:
            fig_vol = go.Figure()
            fig_vol.add_trace(
                go.Scatter(
                    x=ts_full["day"],
                    y=ts_full["count"],
                    mode="lines+markers",
                    name="Tickets",
                )
            )

        fig_vol.update_layout(
            template=template,
            height=360,
            margin=dict(l=40, r=20, t=20, b=40),
            xaxis_title="Day",
            yaxis_title="Ticket count",
            font_color=text_color,
            xaxis=dict(showline=True, linecolor="#000000", mirror=True),
            yaxis=dict(showline=True, linecolor="#000000", mirror=True),
        )
    else:
        fig_vol = go.Figure()
        fig_vol.add_annotation(
            text="No created_at data; cannot build time series.",
            showarrow=False,
        )
        fig_vol.update_layout(
            template=template,
            height=360,
            font_color=text_color,
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(showline=True, linecolor="#000000", mirror=True),
            yaxis=dict(showline=True, linecolor="#000000", mirror=True),
        )

    return (
        f"{total}",
        avg_res,
        resolved_pct,
        whatif_str,
        fig_status,
        fig_sla,
        fig_agent,
        fig_sent,
        fig_kw,
        fig_vol,
>>>>>>> 72ba820ed17af07eec6034180189e486d24c3c5e
    )


# -----------------------------------------------------------------------------
<<<<<<< HEAD
# Run app
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Use app.run (NOT app.run_server) for your Dash version
    app.run(debug=True)
=======
# Sentiment samples callback
# -----------------------------------------------------------------------------
@app.callback(
    Output("sentiment-samples", "children"),
    Input("sentiment-choice", "value"),
)
def update_sentiment_samples(sentiment_value):
    if "sentiment" not in df_raw.columns or sentiment_value is None:
        return "No sentiment data available."

    df_sent = df_raw[df_raw["sentiment"] == sentiment_value].copy()

    if df_sent.empty:
        return f"No tickets found with sentiment '{sentiment_value}'."

    text_col = None
    for cand in ["issue_sum", "description", "subject", "title", "body"]:
        if cand in df_sent.columns:
            text_col = cand
            break

    if text_col is None:
        return "No text fields available to show examples."

    n_samples = min(5, len(df_sent))
    samples = df_sent.sample(n=n_samples, random_state=42)

    items = []
    for _, row in samples.iterrows():
        created_str = ""
        if "created_at" in df_sent.columns and pd.notna(row.get("created_at")):
            created_str = f"({row['created_at'].strftime('%Y-%m-%d %H:%M')}) "

        ticket_id_str = ""
        for cand in ["ticket_id", "id", "case_id"]:
            if cand in df_sent.columns and pd.notna(row.get(cand)):
                ticket_id_str = f"[#{row[cand]}] "
                break

        text = str(row[text_col])
        items.append(
            html.Li(
                f"{created_str}{ticket_id_str}{text}",
                style={"marginBottom": "4px"},
            )
        )

    return html.Ul(items, style={"paddingLeft": "18px", "margin": 0})


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=False)
>>>>>>> 72ba820ed17af07eec6034180189e486d24c3c5e
