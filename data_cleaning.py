"""Load, clean, and prepare the historical sales dataset."""
from pathlib import Path
import pandas as pd

def load_and_clean_data(file_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    required = {"Date", "Sales"}
    missing_columns = required - set(df.columns)
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Sales"] = pd.to_numeric(df["Sales"], errors="coerce")
    df = df.dropna(subset=["Date"])
    df = df.drop_duplicates(subset=["Date"], keep="last")
    df = df.sort_values("Date").set_index("Date")

    # Fill missing sales with time interpolation, then use nearest valid values at edges.
    df["Sales"] = df["Sales"].interpolate(method="time").ffill().bfill()
    if df.empty or df["Sales"].isna().any():
        raise ValueError("The dataset has no usable sales observations.")
    return df
