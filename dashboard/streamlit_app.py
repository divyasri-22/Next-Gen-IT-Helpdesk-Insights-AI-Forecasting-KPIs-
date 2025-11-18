import pathlib
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW_PATH = PROJECT_ROOT / "data" / "input" / "tech_support_tickets.csv"
SUMMARY_PATH = PROJECT_ROOT / "data" / "output" / "tech_support_summary.csv"


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
@st.cache_data
def load_data() -> pd.DataFrame:
    if not RAW_PATH.exists():
        st.error(f"❌ Could not find raw data at: {RAW_PATH}")
        return pd.DataFrame()

    df = pd.read_csv(RAW_PATH)

    # Try to parse dates if they exist
    for col in ["created_at", "resolved_at"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    return df


@st.cache_data
def load_summary() -> pd.DataFrame:
    if not SUMMARY_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(SUMMARY_PATH)


def safe_col(df: pd.DataFrame, col: str, default=None):
    return df[col] if col in df.columns else default


# -----------------------------------------------------------------------------
# Layout setup
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Tech Support Analytics Dashboard",
    layout="wide",
    page_icon="💻",
)

# ---------- ADVANCED STYLING & ANIMATIONS ----------
st.markdown(
    """
    <style>
    /* Global background */
    .stApp {
        background: radial-gradient(circle at top left, #0f172a 0, #020617 40%, #000 100%);
        color: #e5e7eb;
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1400px;
        animation: fadeSlideDown 0.8s ease-out;
    }

    @keyframes fadeSlideDown {
        from { opacity: 0; transform: translateY(-24px); }
        to   { opacity: 1; transform: translateY(0); }
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #020617 0%, #020617 45%, #000 100%);
        border-right: 1px solid rgba(37, 99, 235, 0.5);
        box-shadow: 10px 0 35px rgba(15, 23, 42, 0.9);
    }

    /* Animated title */
    .main-title {
        font-size: 2.6rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        background: linear-gradient(90deg, #60a5fa, #a855f7, #f97316);
        -webkit-background-clip: text;
        color: transparent;
        animation: titleSlide 0.9s ease-out;
    }

    @keyframes titleSlide {
        from { opacity: 0; transform: translateX(-60px); }
        to   { opacity: 1; transform: translateX(0); }
    }

    /* Tab bar */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.25rem;
        background: rgba(15, 23, 42, 0.95);
        padding: 0.35rem;
        border-radius: 999px;
        border: 1px solid rgba(148, 163, 184, 0.5);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 999px;
        padding-top: 0.4rem;
        padding-bottom: 0.4rem;
    }

    /* Dashboard sections slide from left/right */
    .dash-section-left {
        animation: slideInLeft 0.6s ease-out;
    }
    .dash-section-right {
        animation: slideInRight 0.6s ease-out;
    }

    @keyframes slideInLeft {
        from { opacity: 0; transform: translateX(-40px); }
        to   { opacity: 1; transform: translateX(0); }
    }
    @keyframes slideInRight {
        from { opacity: 0; transform: translateX(40px); }
        to   { opacity: 1; transform: translateX(0); }
    }

    /* Metric cards: neon glow + hover jump */
    .stMetric {
        background: radial-gradient(circle at top left, #3b82f6, #0f172a 55%, #020617);
        padding: 1rem 1.4rem;
        border-radius: 1.2rem;
        border: 1px solid rgba(148, 163, 184, 0.4);
        box-shadow: 0 22px 55px rgba(15, 23, 42, 0.95);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .stMetric:hover {
        transform: translateY(-4px) scale(1.01);
        box-shadow: 0 30px 70px rgba(37, 99, 235, 0.7);
    }

    /* Plotly containers */
    .stPlotlyChart {
        background: rgba(15, 23, 42, 0.98);
        padding: 1rem 1.25rem;
        border-radius: 1.3rem;
        border: 1px solid rgba(55, 65, 81, 0.9);
        box-shadow: 0 20px 55px rgba(15, 23, 42, 0.95);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .stPlotlyChart:hover {
        transform: translateY(-5px) scale(1.01);
        box-shadow: 0 30px 75px rgba(15, 23, 42, 1);
    }

    /* Dataframes */
    .stDataFrame {
        background: rgba(15, 23, 42, 0.96);
        border-radius: 1.1rem;
        border: 1px solid rgba(55, 65, 81, 0.9);
        box-shadow: 0 18px 45px rgba(15, 23, 42, 0.9);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown("## 📋 Filters")

raw = load_data()
summary = load_summary()

if raw.empty:
    st.stop()

st.sidebar.success("✅ Data Loaded Successfully")

# -----------------------------------------------------------------------------
# SIDEBAR FILTERS
# -----------------------------------------------------------------------------

# Status filter -------------------------------------------------------
if "status" in raw.columns:
    status_options = sorted(raw["status"].dropna().unique().tolist())
    selected_status = st.sidebar.multiselect(
        "Status",
        options=status_options,
        default=status_options,
    )
else:
    selected_status = None
    st.sidebar.warning("⚠ No 'status' column found in dataset.")

# Priority filter (optional) -----------------------------------------
if "priority" in raw.columns:
    priority_options = sorted(raw["priority"].dropna().unique().tolist())
    selected_priority = st.sidebar.multiselect(
        "Priority",
        options=priority_options,
        default=priority_options,
    )
else:
    selected_priority = None
    st.sidebar.warning("⚠ Priority column not found in dataset.")

# Agent filter (fixed & dynamic) -------------------------------------
if "agent" in raw.columns:
    agent_options = sorted(raw["agent"].dropna().unique().tolist())
    selected_agents = st.sidebar.multiselect(
        "Agent (optional)",
        options=agent_options,
        default=agent_options,
        help="Select one or more agents. All selected means everything is shown.",
    )
else:
    selected_agents = None
    st.sidebar.info("ℹ No 'agent' column found; agent filter disabled.")

# Date range filter ---------------------------------------------------
if "created_at" in raw.columns:
    min_date = raw["created_at"].min()
    max_date = raw["created_at"].max()
    if pd.isna(min_date) or pd.isna(max_date):
        selected_dates = None
    else:
        selected_dates = st.sidebar.date_input(
            "Created date range",
            value=(min_date.date(), max_date.date()),
        )
else:
    selected_dates = None

# -----------------------------------------------------------------------------
# APPLY FILTERS
# -----------------------------------------------------------------------------
df = raw.copy()

if selected_status is not None:
    df = df[df["status"].isin(selected_status)]

if selected_priority is not None and "priority" in df.columns:
    df = df[df["priority"].isin(selected_priority)]

if selected_agents is not None and "agent" in df.columns:
    df = df[df["agent"].isin(selected_agents)]

if selected_dates and "created_at" in df.columns:
    # date_input can return a single date or a range
    if isinstance(selected_dates, (tuple, list)) and len(selected_dates) == 2:
        start, end = selected_dates
    else:
        start = end = selected_dates

    end = datetime.combine(end, datetime.max.time())
    start = datetime.combine(start, datetime.min.time())
    df = df[(df["created_at"] >= start) & (df["created_at"] <= end)]

# -----------------------------------------------------------------------------
# HEADER
# -----------------------------------------------------------------------------
st.markdown('<div class="main-title">TECH SUPPORT ANALYTICS DASHBOARD</div>', unsafe_allow_html=True)
st.caption("Use the filters on the left to slice by status, agent, priority and date.")

# TABS to make it feel like pages sliding in
overview_tab, sla_tab, sentiment_tab, forecast_tab = st.tabs(
    ["📊 Overview", "⏰ SLA & Agents", "😊 Sentiment & Keywords", "📈 Forecast"]
)

# -----------------------------------------------------------------------------
# OVERVIEW TAB  (metrics + status pie)
# -----------------------------------------------------------------------------
with overview_tab:
    st.markdown('<div class="dash-section-left">', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Total Tickets")
        st.metric(label="After applying current filters", value=len(df))

    with col2:
        st.subheader("Avg Resolution Time (min)")
        if {"resolved_at", "created_at"}.issubset(df.columns):
            resolved = df.dropna(subset=["resolved_at", "created_at"]).copy()
            if not resolved.empty:
                resolved["resolution_minutes"] = (
                    resolved["resolved_at"] - resolved["created_at"]
                ).dt.total_seconds() / 60.0
                avg_res = resolved["resolution_minutes"].mean()
                st.metric(label="Based on resolved tickets", value=f"{avg_res:,.1f}")
            else:
                st.metric(label="Based on resolved tickets", value="N/A")
        else:
            st.metric(label="Based on resolved tickets", value="N/A")

    with col3:
        st.subheader("Resolved Ticket %")
        if "status" in df.columns and len(df) > 0:
            status_lower = df["status"].astype(str).str.lower()
            resolved_mask = status_lower.isin(["resolved", "closed", "done"])
            resolved_count = resolved_mask.sum()
            pct = resolved_count / len(df) * 100
            st.metric(
                label="Resolved / total in current view", value=f"{pct:,.1f}%"
            )
        else:
            st.metric(label="Resolved / total in current view", value="N/A")

    st.markdown("---")

    st.subheader("📊 Ticket Status Distribution")

    if "status" in df.columns and not df.empty:
        status_counts = df["status"].value_counts().reset_index(name="count")
        status_counts.rename(columns={"index": "status"}, inplace=True)

        fig_status = px.pie(
            status_counts,
            names="status",
            values="count",
            hole=0.4,
            title="Status Breakdown",
        )
        st.plotly_chart(fig_status, use_container_width=True)
    else:
        st.info("No status data available for current filters.")

    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# SLA & AGENT TAB
# -----------------------------------------------------------------------------
with sla_tab:
    st.markdown('<div class="dash-section-right">', unsafe_allow_html=True)

    # SLA HEATMAP
    st.subheader("⏰ SLA Heatmap (by Day & Priority)")

    if {"created_at", "priority"}.issubset(df.columns):
        df_heat = df.dropna(subset=["created_at", "priority"]).copy()
        if not df_heat.empty:
            df_heat["day"] = df_heat["created_at"].dt.date
            if "sla_breached" in df_heat.columns:
                pivot = (
                    df_heat.groupby(["day", "priority"])["sla_breached"]
                    .mean()
                    .reset_index()
                )
                pivot["sla_breached_pct"] = pivot["sla_breached"] * 100
                fig_sla = px.density_heatmap(
                    pivot,
                    x="day",
                    y="priority",
                    z="sla_breached_pct",
                    color_continuous_scale="Reds",
                    labels={"sla_breached_pct": "SLA Breach %"},
                )
                st.plotly_chart(fig_sla, use_container_width=True)
            else:
                st.info("No 'sla_breached' column; SLA heatmap shows only ticket counts.")
                # SAFE: no dependency on ticket_id
                pivot = (
                    df_heat.groupby(["day", "priority"])
                    .size()
                    .reset_index(name="count")
                )
                fig_sla = px.density_heatmap(
                    pivot,
                    x="day",
                    y="priority",
                    z="count",
                    color_continuous_scale="Blues",
                    labels={"count": "Ticket count"},
                )
                st.plotly_chart(fig_sla, use_container_width=True)
        else:
            st.info("No data available to plot SLA heatmap.")
    else:
        st.info("Need 'created_at' and 'priority' columns for SLA view.")

    st.markdown("---")

    # AGENT PERFORMANCE
    st.subheader("👨‍💻 Agent Performance")

    if "agent" in df.columns and not df.empty:
        perf = df.copy()

        if {"resolved_at", "created_at"}.issubset(perf.columns):
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
                fig_agent = px.bar(
                    agg.sort_values("tickets", ascending=False),
                    x="agent",
                    y="tickets",
                    hover_data=["avg_resolution"],
                    title="Tickets handled per agent",
                )
                st.plotly_chart(fig_agent, use_container_width=True)
            else:
                st.info("No resolved tickets in current filters to compute performance.")
        else:
            agg = (
                perf.groupby("agent")
                .size()
                .reset_index(name="tickets")
            )
            fig_agent = px.bar(
                agg.sort_values("tickets", ascending=False),
                x="agent",
                y="tickets",
                title="Tickets handled per agent",
            )
            st.plotly_chart(fig_agent, use_container_width=True)
    else:
        st.info("No 'agent' column; cannot compute agent performance.")

    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# SENTIMENT + KEYWORDS TAB
# -----------------------------------------------------------------------------
with sentiment_tab:
    st.markdown('<div class="dash-section-left">', unsafe_allow_html=True)

    # SENTIMENT
    st.subheader("😊 Sentiment Overview")

    if "sentiment" in df.columns and not df.empty:
        sent_counts = df["sentiment"].value_counts().reset_index(name="count")
        sent_counts.rename(columns={"index": "sentiment"}, inplace=True)

        fig_sent = px.bar(
            sent_counts,
            x="sentiment",
            y="count",
            title="Ticket count by sentiment",
        )
        st.plotly_chart(fig_sent, use_container_width=True)

        st.markdown("### Sample tickets by sentiment (from full dataset)")

        options = sorted(raw["sentiment"].dropna().unique().tolist())
        choice = st.selectbox("Sentiment", options=options)

        sample_view = raw[raw["sentiment"] == choice]

        if not sample_view.empty:
            cols_to_show = [
                c
                for c in ["ticket_id", "agent", "status", "description"]
                if c in sample_view.columns
            ]
            st.dataframe(sample_view[cols_to_show].head(10), use_container_width=True)
        else:
            st.info("No tickets with this sentiment found in the full dataset.")
    else:
        st.info("No 'sentiment' column found; sentiment analysis view disabled.")

    st.markdown("---")

    # KEYWORDS
    st.subheader("☁ Top Issue Keywords (word-cloud style)")

    if "description" in raw.columns:
        text = " ".join(raw["description"].astype(str).tolist()).lower()
        tokens = [t.strip(".,!?:;()[]") for t in text.split() if len(t) > 3]

        if tokens:
            freq = pd.Series(tokens).value_counts().reset_index(name="count")
            freq.rename(columns={"index": "word"}, inplace=True)
            top_n = freq.head(30)

            fig_kw = px.bar(
                top_n.sort_values("count"),
                x="count",
                y="word",
                orientation="h",
                title="Top issue keywords",
            )
            st.plotly_chart(fig_kw, use_container_width=True)
        else:
            st.info("Not enough text to extract keywords.")
    else:
        st.info("No 'description' column found; cannot compute keywords.")

    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# FORECAST TAB
# -----------------------------------------------------------------------------
with forecast_tab:
    st.markdown('<div class="dash-section-right">', unsafe_allow_html=True)

    st.subheader("📈 Ticket Volume & Simple 7-day Forecast")

    if "created_at" in raw.columns:
        vol = raw.dropna(subset=["created_at"]).copy()
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

            fig_fc = px.line(
                combined,
                x="day",
                y=["tickets", "forecast"],
                labels={"value": "Tickets", "variable": "Series"},
                title="Historical volume and naive 7-day forecast",
            )
            st.plotly_chart(fig_fc, use_container_width=True)
        else:
            st.info("No created_at values to plot volume.")
    else:
        st.info("Need 'created_at' column for volume & forecast.")

    st.markdown("</div>", unsafe_allow_html=True)

# End of file
