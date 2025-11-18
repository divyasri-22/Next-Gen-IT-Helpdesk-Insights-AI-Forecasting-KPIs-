import pathlib
from datetime import datetime, timedelta
import math

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from dash import Dash, html, dcc, Input, Output

# -----------------------------------------------------------------------------
# Paths & Data
# -----------------------------------------------------------------------------
BASE_DIR = pathlib.Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "tech_support_tickets.csv"


def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"CSV not found at: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    # normalize column names a bit
    df.columns = [c.strip() for c in df.columns]

    # parse dates if present
    for col in ["created_at", "resolved_at"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

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

    return df


df_raw = load_data()

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
    agent_values,
    start_date,
    end_date,
) -> pd.DataFrame:
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
# -----------------------------------------------------------------------------
@app.callback(
    [
        Output("metric-total", "children"),
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
            names="status",
            values="count",
            hole=0.4,
        )
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
    )


# -----------------------------------------------------------------------------
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
