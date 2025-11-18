from pathlib import Path
import pandas as pd


def load_raw_data(context: dict) -> pd.DataFrame:
    """
    Example task: load a CSV file.
    `context` will contain runtime info like input paths.
    """
    input_path = Path(context["input_path"])
    df = pd.read_csv(input_path)
    return df


def clean_data(context: dict, df: pd.DataFrame) -> pd.DataFrame:
    """
    Example task: very simple cleaning/transform.
    """
    # Drop completely empty rows
    df = df.dropna(how="all")

    # Example: fill numeric NaNs with 0
    numeric_cols = df.select_dtypes(include=["number"]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)

    return df


def save_output(context: dict, df: pd.DataFrame) -> None:
    """
    Example task: save the cleaned data to a new CSV file.
    """
    output_path = Path(context["output_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
