"""
Data loading and preprocessing utilities.
"""
import pandas as pd
from pathlib import Path
from src.config import PATHS


def load_raw_data(path: Path = None) -> pd.DataFrame:
    """Load raw customer churn data."""
    path = path or PATHS["raw_data"]
    df = pd.read_csv(path)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and preprocess customer data."""
    df_clean = df.copy()

    if "CustomerID" in df_clean.columns:
        df_clean = df_clean.drop(columns=["CustomerID"])

    df_clean = df_clean.dropna()

    numeric_cols = ["Age", "Tenure", "Usage Frequency", "Support Calls",
                     "Payment Delay", "Total Spend", "Last Interaction", "Churn"]
    for col in numeric_cols:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")

    df_clean = df_clean.dropna()

    return df_clean


def save_processed_data(df: pd.DataFrame, path: Path = None) -> None:
    """Save cleaned data to processed folder."""
    path = path or PATHS["clean_data"]
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def load_or_create_clean_data() -> pd.DataFrame:
    """Load cleaned data or create it if it doesn't exist."""
    clean_path = PATHS["clean_data"]

    if clean_path.exists():
        df = pd.read_csv(clean_path)
        if df.isnull().sum().sum() > 0:
            print("Existing clean data has NaN, re-creating...")
    
    raw_path = PATHS["raw_data"]
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data not found at {raw_path}")

    df = load_raw_data(raw_path)
    df_clean = clean_data(df)
    save_processed_data(df_clean, clean_path)
    return df_clean


def get_data_summary(df: pd.DataFrame) -> dict:
    """Get summary statistics of the data."""
    return {
        "total_rows": len(df),
        "churn_count": int(df["Churn"].sum()),
        "churn_rate": round(df["Churn"].mean() * 100, 2),
        "columns": list(df.columns),
        "dtypes": df.dtypes.to_dict(),
    }
