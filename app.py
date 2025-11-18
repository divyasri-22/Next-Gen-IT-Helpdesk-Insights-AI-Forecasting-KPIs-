from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
from dash import Dash, dcc, html, dash_table, Input, Output

# -----------------------------------------------------------------------------
# Paths  (app.py and data folder are in the same "data-orchestrator core" folder)
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "tech_support_tickets.csv"   # <= your CSV here


# -----------------------------------------------------------------------------
# Load & prepare data
# -----------------------------------------------------------------------------
def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"CSV not found at: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    # Normalise column names just in case
    df.columns = [c.strip().lower() for c in df.columns]

    # Parse dates if present
    for col in ["created_at", "resolved_at"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Ensure helpful fallback columns
    if "ticket_id" not in df.columns:
        df["ticket_id"] = np.arange(1, len(df) + 1)

    return df


df_raw = load_data()


# -----------------------------------------------------------------------------
# Helper to filter data
# -----------------------------------------------------------------------------
def apply_filters(
    df: pd.DataFrame,
    status_values,
    priority_values,
    agent_values,
    start_date,
    end_date,
) -> pd.DataFrame:
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
# -----------------------------------------------------------------------------
@app.callback(
    [
        Output("metric-total", "children"),
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
            names="status",
            values="count",
            hole=0.4,
        )
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
    )


# -----------------------------------------------------------------------------
# Run app
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Use app.run (NOT app.run_server) for your Dash version
    app.run(debug=True)
