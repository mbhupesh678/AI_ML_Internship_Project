"""
Central configuration: file paths, column names, and constants
used across the whole pipeline.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "MiningProcess_Flotation_Plant_Database.csv"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT_DIR / "models"
OUTPUTS_DIR = ROOT_DIR / "outputs"

for d in (PROCESSED_DIR, MODELS_DIR, OUTPUTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Columns
# ---------------------------------------------------------------------------
DATETIME_COL = "date"

TARGET_COL = "% Silica Concentrate"

# % Iron Concentrate is measured at the same time as the target in the lab,
# so it is only a valid *input* feature when explicitly testing the
# "with lab co-measurement" scenario. Excluded from the default feature set.
LAB_COLUMNS = ["% Iron Concentrate", "% Silica Concentrate"]

PROCESS_COLUMNS = [
    "% Iron Feed",
    "% Silica Feed",
    "Starch Flow",
    "Amina Flow",
    "Ore Pulp Flow",
    "Ore Pulp pH",
    "Ore Pulp Density",
    "Flotation Column 01 Air Flow",
    "Flotation Column 02 Air Flow",
    "Flotation Column 03 Air Flow",
    "Flotation Column 04 Air Flow",
    "Flotation Column 05 Air Flow",
    "Flotation Column 06 Air Flow",
    "Flotation Column 07 Air Flow",
    "Flotation Column 01 Level",
    "Flotation Column 02 Level",
    "Flotation Column 03 Level",
    "Flotation Column 04 Level",
    "Flotation Column 05 Level",
    "Flotation Column 06 Level",
    "Flotation Column 07 Level",
]

# ---------------------------------------------------------------------------
# Feature engineering constants
# ---------------------------------------------------------------------------
RESAMPLE_FREQ = "1h"          # align to hourly lab measurement cadence
LAG_STEPS = [1, 2, 3, 6, 12]  # in units of RESAMPLE_FREQ
ROLLING_WINDOWS = [3, 6, 12]  # in units of RESAMPLE_FREQ

# ---------------------------------------------------------------------------
# Train / val / test split (chronological, by fraction of the time range)
# ---------------------------------------------------------------------------
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
# remaining 0.15 goes to test

RANDOM_STATE = 42
