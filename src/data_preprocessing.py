"""
data_preprocessing.py
---------------------
Data loading, validation, and preprocessing.
All file I/O and DataFrame cleaning lives here.
"""

import logging
from pathlib import Path

import pandas as pd

from nlp_pipeline import clean_text  # single source of truth — no duplicate clean_text

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = {"job_description"}


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load and validate the jobs CSV.

    Args:
        filepath: Path to monster_jobs.csv (or equivalent).

    Returns:
        Raw DataFrame.

    Raises:
        FileNotFoundError: If the CSV does not exist.
        ValueError: If required columns are missing or dataset is empty.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(
            f"Data file not found: {filepath}\n"
            "Place monster_jobs.csv inside the data/ directory."
        )

    df = pd.read_csv(path)

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    df = df.dropna(subset=["job_description"])

    if df.empty:
        raise ValueError("Dataset is empty after dropping rows with null job_description.")

    logger.info("Loaded %d job records from %s", len(df), filepath)
    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply text cleaning to job descriptions.

    Args:
        df: Raw DataFrame with a 'job_description' column.

    Returns:
        DataFrame with an added 'cleaned' column.
    """
    df = df.copy()
    df["cleaned"] = df["job_description"].apply(clean_text)
    return df