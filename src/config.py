"""
Configuration and constants for the Customer Churn Pipeline.
"""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SIMULATED_DATA_DIR = DATA_DIR / "simulated"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = PROJECT_ROOT / "models"

PATHS = {
    "raw_data": RAW_DATA_DIR / "customer_churn.csv",
    "clean_data": PROCESSED_DATA_DIR / "churn_clean.csv",
    "simulated_data": SIMULATED_DATA_DIR,
    "outputs": OUTPUT_DIR,
    "models": MODELS_DIR,
}

THRESHOLDS = {
    "support_calls": {"high": 4, "critical": 7},
    "payment_delay": {"high": 17, "critical": 25},
    "last_interaction": {"high": 15, "critical": 25},
    "total_spend": {"low": 600, "very_low": 300},
    "usage_frequency": {"low": 14, "very_low": 7},
    "tenure": {"low": 6, "very_low": 3},
}

RISK_LEVELS = {
    0: "none",
    1: "low",
    2: "medium",
    3: "high",
    4: "critical",
}

API_CONFIG = {
    "base_url": "https://openrouter.ai/api/v1",
    "default_model": "anthropic/claude-3-haiku",
    "max_tokens": 500,
    "temperature": 0.7,
}


def ensure_directories():
    """Create necessary directories if they don't exist."""
    for dir_path in [DATA_DIR, PROCESSED_DATA_DIR, SIMULATED_DATA_DIR, OUTPUT_DIR, MODELS_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)


def get_project_root():
    return PROJECT_ROOT
