from pathlib import Path
import pandas as pd


def load_tickets(context: dict) -> pd.DataFrame:
    """
    Load raw tech support tickets from CSV.
    """
    input_path = Path(context["input_path"])
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found at: {input_path}")

    df = pd.read_csv(input_path)
    return df


def enrich_tickets(context: dict, df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and enrich tickets:
    - Parse datetime columns
    - Compute resolution_time_hours
    - Fill missing priority/status
    """
    # Parse created_at / resolved_at if present
    for col in ["created_at", "resolved_at"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Compute resolution time in hours if both datetimes exist
    if {"created_at", "resolved_at"}.issubset(df.columns):
        df["resolution_time_hours"] = (
            df["resolved_at"] - df["created_at"]
        ).dt.total_seconds() / 3600.0

    # Normalize priority / status
    if "priority" in df.columns:
        df["priority"] = df["priority"].fillna("Unknown")

    if "status" in df.columns:
        df["status"] = df["status"].fillna("Unknown")

    return df


def summarize_tickets(context: dict, df: pd.DataFrame) -> pd.DataFrame:
    """
    Create an aggregated view by priority and status:
    - ticket_count
    - avg_resolution_time_hours
    - avg_satisfaction_score
    """
    group_cols = [c for c in ["priority", "status"] if c in df.columns]
    if not group_cols:
        # Nothing to group by, just return original
        return df

    agg_dict = {"ticket_id": "count"}

    if "resolution_time_hours" in df.columns:
        agg_dict["resolution_time_hours"] = "mean"

    if "satisfaction_score" in df.columns:
        agg_dict["satisfaction_score"] = "mean"

    summary = (
        df.groupby(group_cols)
        .agg(agg_dict)
        .rename(columns={"ticket_id": "ticket_count"})
        .reset_index()
    )

    return summary


def save_summary(context: dict, df: pd.DataFrame) -> None:
    """
    Save the summary DataFrame to CSV.
    """
    output_path = Path(context["output_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
