"""
Time alignment between high-frequency process sensors (~20 second cadence)
and the hourly laboratory quality measurements.

Key idea: process variables are aggregated (mean) into the same time buckets
as the lab measurements so every row represents one lab-measurement instant,
described by summary statistics of the process data leading up to it.
"""

import logging

import pandas as pd

from config import DATETIME_COL, PROCESSED_DIR, RESAMPLE_FREQ, TARGET_COL, TRAIN_FRAC, VAL_FRAC

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def resample_to_frequency(df: pd.DataFrame, freq: str = RESAMPLE_FREQ) -> pd.DataFrame:
    """
    Resample all numeric columns to `freq` using the mean of the process
    values within each bucket. The lab-measured target is assumed constant
    across the hour it was recorded in, matching the source dataset's
    hourly-repeated lab values.
    """
    df = df.set_index(DATETIME_COL)
    resampled = df.resample(freq).mean()
    resampled = resampled.dropna(subset=[TARGET_COL])
    resampled = resampled.reset_index()
    logger.info("Resampled to %s frequency: %d rows remain", freq, len(resampled))
    return resampled


def chronological_split(
    df: pd.DataFrame,
    train_frac: float = TRAIN_FRAC,
    val_frac: float = VAL_FRAC,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split strictly by time order (no shuffling) to avoid leakage:
    earliest `train_frac` -> train, next `val_frac` -> validation,
    remainder -> test.
    """
    df = df.sort_values(DATETIME_COL).reset_index(drop=True)
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    logger.info(
        "Chronological split -> train: %d (%s to %s), val: %d, test: %d",
        len(train_df),
        train_df[DATETIME_COL].min(),
        train_df[DATETIME_COL].max(),
        len(val_df),
        len(test_df),
    )
    return train_df, val_df, test_df


if __name__ == "__main__":
    cleaned = pd.read_parquet(PROCESSED_DIR / "cleaned_data.parquet")
    resampled = resample_to_frequency(cleaned)
    resampled.to_parquet(PROCESSED_DIR / "resampled_data.parquet", index=False)
    logger.info("Saved resampled data to %s", PROCESSED_DIR / "resampled_data.parquet")
