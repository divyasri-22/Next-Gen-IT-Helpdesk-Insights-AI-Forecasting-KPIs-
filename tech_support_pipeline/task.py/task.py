from pathlib import Path
from typing import Dict, Tuple

import pandas as pd


def load_tickets(context: Dict[str, str]) -> pd.DataFrame:
    """
    Load raw tech support tickets from CSV.

    Expects context["input_path"] pointing to something like:
      data/input/tech_support_tickets.csv
    """
    input_path = Path(context["input_path"])
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found at: {input_path}")

    df = pd.read_csv(input_path)

    # Try to parse typical date columns if present
    for col in ["created_at", "updated_at", "resolved_at"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    return df


def clean_tickets(context: Dict[str, str], df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic cleaning for tech support tickets:
      - drop rows with no description
      - drop duplicate ticket_id if present
      - trim text columns
    """
    df = df.copy()

    # Drop tickets with no description at all
    text_cols = [c for c in df.columns if c.lower() in ("description", "issue", "problem")]
    if text_cols:
        main_text_col = text_cols[0]
        df = df[df[main_text_col].notna() & (df[main_text_col].str.strip() != "")]
    else:
        main_text_col = None

    # Drop duplicate ticket_ids if column exists
    ticket_id_cols = [c for c in df.columns if c.lower() in ("ticket_id", "id", "case_id")]
    if ticket_id_cols:
        df = df.drop_duplicates(subset=[ticket_id_cols[0]])

    # Trim whitespace on all object (string) columns
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str).str.strip()

    return df


def _build_text_column(df: pd.DataFrame) -> pd.Series:
    """
    Combine subject + description (if they exist) into a single text field.
    Used for simple keyword-based classification.
    """
    subject_cols = [c for c in df.columns if c.lower() in ("subject", "title", "summary")]
    desc_cols = [c for c in df.columns if c.lower() in ("description", "issue", "problem")]

    text_parts = []
    if subject_cols:
        text_parts.append(df[subject_cols[0]].fillna(""))
    if desc_cols:
        text_parts.append(df[desc_cols[0]].fillna(""))

    if not text_parts:
        # Fallback: join all object columns
        obj_cols = df.select_dtypes(include=["object"]).columns
        text = df[obj_cols].astype(str).agg(" ".join, axis=1)
    else:
        text = text_parts[0]
        for part in text_parts[1:]:
            text = text + " " + part

    return text.str.lower()


def classify_tickets(context: Dict[str, str], df: pd.DataFrame) -> pd.DataFrame:
    """
    Very simple rule-based classification:
      - priority: High / Medium / Low
      - topic: Network / Application / Hardware / Access / Other
    """
    df = df.copy()
    text = _build_text_column(df)

    # Priority heuristics
    high_keywords = ["down", "outage", "cannot login", "can’t login", "data loss"]
    medium_keywords = ["slow", "delay", "intermittent", "sometimes"]
    low_keywords = ["how to", "help", "question", "info", "inquiry"]

    def detect_priority(row_text: str) -> str:
        if any(k in row_text for k in high_keywords):
            return "High"
        if any(k in row_text for k in medium_keywords):
            return "Medium"
        if any(k in row_text for k in low_keywords):
            return "Low"
        return "Medium"

    df["priority"] = [detect_priority(t) for t in text]

    # Topic heuristics
    def detect_topic(row_text: str) -> str:
        if any(k in row_text for k in ["vpn", "network", "wifi", "lan", "wan"]):
            return "Network"
        if any(k in row_text for k in ["server", "db", "database", "api", "backend"]):
            return "Backend"
        if any(k in row_text for k in ["app", "application", "ui", "frontend", "button", "screen"]):
            return "Application"
        if any(k in row_text for k in ["laptop", "desktop", "mouse", "keyboard", "printer"]):
            return "Hardware"
        if any(k in row_text for k in ["access", "permission", "role", "authorization", "auth"]):
            return "Access"
        return "Other"

    df["topic"] = [detect_topic(t) for t in text]

    return df


def aggregate_metrics(context: Dict[str, str], df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate metrics per day x priority x topic:
      - ticket_count
    Uses 'created_at' if available; otherwise no date aggregation.
    """
    if "created_at" in df.columns:
        date_series = pd.to_datetime(df["created_at"], errors="coerce").dt.date
        df = df.copy()
        df["date"] = date_series
        group_cols = ["date", "priority", "topic"]
    else:
        group_cols = ["priority", "topic"]

    metrics = (
        df.groupby(group_cols)
        .size()
        .reset_index(name="ticket_count")
    )

    return metrics


def save_outputs(
    context: Dict[str, str],
    df: pd.DataFrame,
    metrics_df: pd.DataFrame,
) -> None:
    """
    Save:
      - cleaned & classified tickets → context["output_path"]
      - aggregated metrics          → same folder, *_metrics.csv
    """
    output_path = Path(context["output_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save cleaned + classified tickets
    df.to_csv(output_path, index=False)

    # Save metrics next to it
    metrics_path = output_path.with_name(output_path.stem + "_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)

    print(f"✅ Cleaned tickets saved to: {output_path}")
    print(f"✅ Metrics saved to:        {metrics_path}")
