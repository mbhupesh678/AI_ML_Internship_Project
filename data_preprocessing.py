"""
Data preprocessing for the mining-process flotation dataset.

Handles:
- Loading the raw CSV
- Parsing datetimes
- Converting comma-decimal numeric strings (European format) to floats
- Checking / reporting missing values and duplicates
- Basic dtype validation

Run directly to produce a cleaned parquet file in data/processed/.
"""

import logging
from pathlib import Path

import pandas as pd

from config import RAW_DATA_PATH, PROCESSED_DIR, DATETIME_COL

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_raw_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw CSV. Raises a clear error if the file is missing."""
    if not path.exists():
        raise FileNotFoundError(
            f"Raw data file not found at {path}.\n"
            "Download it from Kaggle (see README) and place it there."
        )
    logger.info("Loading raw data from %s", path)
    df = pd.read_csv(path)
    logger.info("Loaded %d rows, %d columns", *df.shape)
    return df


def fix_decimal_format(df: pd.DataFrame, exclude: list[str] | None = None) -> pd.DataFrame:
    """
    The source CSV stores numeric process values with a comma as the
    decimal separator (e.g. '55,2' instead of '55.2'). Convert every
    object-typed numeric-looking column to float.
    """
    exclude = exclude or [DATETIME_COL]
    df = df.copy()
    for col in df.columns:
        if col in exclude:
            continue
        if df[col].dtype == object:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(",", ".", regex=False)
                .astype(float)
            )
    return df


def parse_datetime(df: pd.DataFrame, col: str = DATETIME_COL) -> pd.DataFrame:
    """Parse the date column (day-first format as used in the source data)."""
    df = df.copy()
    df[col] = pd.to_datetime(df[col], dayfirst=True, errors="coerce")
    n_bad = df[col].isna().sum()
    if n_bad:
        logger.warning("%d rows had unparseable dates and were dropped", n_bad)
        df = df.dropna(subset=[col])
    return df.sort_values(col).reset_index(drop=True)


def report_data_quality(df: pd.DataFrame) -> pd.DataFrame:
    """Return a summary of missing values and dtypes per column."""
    summary = pd.DataFrame(
        {
            "dtype": df.dtypes,
            "n_missing": df.isna().sum(),
            "pct_missing": (df.isna().mean() * 100).round(3),
        }
    )
    n_dupes = df.duplicated().sum()
    logger.info("Duplicate rows: %d", n_dupes)
    logger.info("\n%s", summary)
    return summary


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates()
    logger.info("Dropped %d exact duplicate rows", before - len(df))
    return df


def clean_pipeline(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Full cleaning pipeline: load -> fix decimals -> parse dates -> dedupe."""
    df = load_raw_data(path)
    df = fix_decimal_format(df)
    df = parse_datetime(df)
    df = drop_duplicates(df)
    report_data_quality(df)
    return df


if __name__ == "__main__":
    cleaned = clean_pipeline()
    out_path = PROCESSED_DIR / "cleaned_data.parquet"
    cleaned.to_parquet(out_path, index=False)
    logger.info("Saved cleaned data to %s", out_path)
