"""
Exploratory Data Analysis for the flotation-plant dataset.

Generates and saves:
- Distribution plots for target and key process variables
- Time-series trend plot of the target
- Correlation heatmap
- Top correlated features report (printed + saved as CSV)
"""

import logging

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from config import DATETIME_COL, OUTPUTS_DIR, PROCESSED_DIR, TARGET_COL

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

sns.set_theme(style="whitegrid")


def plot_target_distribution(df: pd.DataFrame, save: bool = True) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df[TARGET_COL], kde=True, ax=ax)
    ax.set_title(f"Distribution of {TARGET_COL}")
    if save:
        fig.savefig(OUTPUTS_DIR / "target_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_target_trend(df: pd.DataFrame, save: bool = True) -> None:
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df[DATETIME_COL], df[TARGET_COL], linewidth=0.6)
    ax.set_title(f"{TARGET_COL} over time")
    ax.set_xlabel("Date")
    ax.set_ylabel(TARGET_COL)
    if save:
        fig.savefig(OUTPUTS_DIR / "target_trend.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_correlation_heatmap(df: pd.DataFrame, save: bool = True) -> pd.Series:
    numeric_df = df.select_dtypes("number")
    corr = numeric_df.corr()

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax, cbar_kws={"shrink": 0.7})
    ax.set_title("Correlation heatmap — process & quality variables")
    if save:
        fig.savefig(OUTPUTS_DIR / "correlation_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    target_corr = corr[TARGET_COL].drop(TARGET_COL).sort_values(key=abs, ascending=False)
    return target_corr


def plot_feature_distributions(df: pd.DataFrame, columns: list[str], save: bool = True) -> None:
    n = len(columns)
    n_cols = 3
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3.5 * n_rows))
    axes = axes.flatten()
    for i, col in enumerate(columns):
        sns.histplot(df[col], kde=True, ax=axes[i])
        axes[i].set_title(col, fontsize=9)
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    fig.tight_layout()
    if save:
        fig.savefig(OUTPUTS_DIR / "feature_distributions.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_eda(df: pd.DataFrame | None = None) -> pd.Series:
    if df is None:
        df = pd.read_parquet(PROCESSED_DIR / "cleaned_data.parquet")

    plot_target_distribution(df)
    plot_target_trend(df)
    target_corr = plot_correlation_heatmap(df)

    process_cols = [c for c in df.columns if c not in (DATETIME_COL, TARGET_COL)][:9]
    plot_feature_distributions(df, process_cols)

    target_corr.to_csv(OUTPUTS_DIR / "target_correlations.csv")
    logger.info("Top correlated variables with %s:\n%s", TARGET_COL, target_corr.head(10))
    logger.info("EDA plots saved to %s", OUTPUTS_DIR)
    return target_corr


if __name__ == "__main__":
    run_eda()
