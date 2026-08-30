"""
Evaluation utilities: MAE / RMSE / R2, residual analysis, and feature
importance visualisation.
"""

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

try:
    from sklearn.metrics import root_mean_squared_error
except ImportError:  # scikit-learn < 1.4 fallback
    from sklearn.metrics import mean_squared_error

    def root_mean_squared_error(y_true, y_pred):
        return mean_squared_error(y_true, y_pred, squared=False)

from config import OUTPUTS_DIR

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def compute_metrics(y_true, y_pred) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


def evaluate_models(models: dict, X_val, y_val) -> pd.DataFrame:
    """Evaluate a dict of {name: fitted_model} on the same validation set."""
    rows = []
    for name, model in models.items():
        preds = model.predict(X_val)
        metrics = compute_metrics(y_val, preds)
        metrics["model"] = name
        rows.append(metrics)
    results = pd.DataFrame(rows).set_index("model").sort_values("RMSE")
    logger.info("\n%s", results)
    return results


def plot_residuals(y_true, y_pred, model_name: str, save: bool = True) -> None:
    residuals = np.asarray(y_true) - np.asarray(y_pred)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].scatter(y_pred, residuals, s=8, alpha=0.4)
    axes[0].axhline(0, color="red", linestyle="--")
    axes[0].set_xlabel("Predicted value")
    axes[0].set_ylabel("Residual")
    axes[0].set_title(f"{model_name}: residuals vs predicted")

    axes[1].hist(residuals, bins=50)
    axes[1].set_title(f"{model_name}: residual distribution")
    fig.tight_layout()
    if save:
        fig.savefig(OUTPUTS_DIR / f"residuals_{model_name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_feature_importance(importance: pd.Series, model_name: str, top_n: int = 20, save: bool = True) -> None:
    top = importance.head(top_n).sort_values()
    fig, ax = plt.subplots(figsize=(8, max(4, 0.3 * top_n)))
    ax.barh(top.index, top.values)
    ax.set_title(f"{model_name}: top {top_n} feature importances")
    fig.tight_layout()
    if save:
        fig.savefig(OUTPUTS_DIR / f"feature_importance_{model_name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def forecast_horizon_report(
    model, X_test_by_horizon: dict[str, pd.DataFrame], y_test_by_horizon: dict[str, pd.Series]
) -> pd.DataFrame:
    """
    Compare model performance across multiple forecast horizons (e.g. predicting
    1, 3, 6 hours ahead) to study how prediction quality degrades with horizon.
    """
    rows = []
    for horizon, X_h in X_test_by_horizon.items():
        y_h = y_test_by_horizon[horizon]
        preds = model.predict(X_h)
        metrics = compute_metrics(y_h, preds)
        metrics["horizon"] = horizon
        rows.append(metrics)
    return pd.DataFrame(rows).set_index("horizon")
