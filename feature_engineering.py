"""
Feature engineering for the soft-sensor model.

Builds lag features and rolling statistics from process variables using
only information available strictly before the prediction timestamp,
to keep the pipeline leakage-safe.
"""

import logging

import pandas as pd

from config import (
    DATETIME_COL,
    LAG_STEPS,
    PROCESS_COLUMNS,
    PROCESSED_DIR,
    ROLLING_WINDOWS,
    TARGET_COL,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def add_lag_features(
    df: pd.DataFrame, columns: list[str], lags: list[int] = LAG_STEPS
) -> pd.DataFrame:
    new_cols = {
        f"{col}_lag{lag}": df[col].shift(lag) for col in columns for lag in lags
    }
    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)


def add_rolling_features(
    df: pd.DataFrame, columns: list[str], windows: list[int] = ROLLING_WINDOWS
) -> pd.DataFrame:
    """
    Rolling mean/std computed on already-shifted-by-1 data, so the window
    ending at t-1 is used to predict the value at t (no leakage from t itself).
    """
    new_cols = {}
    for col in columns:
        shifted = df[col].shift(1)
        for window in windows:
            new_cols[f"{col}_rollmean{window}"] = shifted.rolling(window).mean()
            new_cols[f"{col}_rollstd{window}"] = shifted.rolling(window).std()
    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hour"] = df[DATETIME_COL].dt.hour
    df["day_of_week"] = df[DATETIME_COL].dt.dayofweek
    df["month"] = df[DATETIME_COL].dt.month
    return df


def build_features(
    df: pd.DataFrame,
    process_columns: list[str] = PROCESS_COLUMNS,
    include_iron_concentrate: bool = False,
) -> pd.DataFrame:
    """
    Build the full modelling feature set.

    include_iron_concentrate: if True, adds lagged % Iron Concentrate as a
    feature — used only for the explicit "with lab co-measurement" comparison
    scenario described in the report. % Iron Concentrate is measured at the
    same instant as the target, so it must be lagged, never used at t=0.
    """
    df = df.copy()
    cols = list(process_columns)
    if include_iron_concentrate and "% Iron Concentrate" in df.columns:
        cols = cols + ["% Iron Concentrate"]

    df = add_lag_features(df, cols)
    df = add_rolling_features(df, cols)
    df = add_time_features(df)

    feature_cols = [c for c in df.columns if c not in (DATETIME_COL, TARGET_COL) and c not in cols]
    df_features = df[[DATETIME_COL, TARGET_COL] + feature_cols].dropna().reset_index(drop=True)

    logger.info(
        "Built %d features, %d usable rows after dropping NA from lags/rolling windows",
        len(feature_cols),
        len(df_features),
    )
    return df_features


if __name__ == "__main__":
    resampled = pd.read_parquet(PROCESSED_DIR / "resampled_data.parquet")
    features = build_features(resampled)
    features.to_parquet(PROCESSED_DIR / "features.parquet", index=False)
    logger.info("Saved feature set to %s", PROCESSED_DIR / "features.parquet")
