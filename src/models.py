"""
Model definitions and training utilities.

Includes a mean-prediction baseline plus five regression models spanning
linear, tree-based, and boosted approaches, as used in the internship report.
"""

import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

try:
    from xgboost import XGBRegressor
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

from config import RANDOM_STATE

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class MeanBaseline:
    """Predicts the training-set mean target for every row. Sanity-check floor."""

    def __init__(self):
        self.mean_ = None

    def fit(self, X, y):
        self.mean_ = np.mean(y)
        return self

    def predict(self, X):
        return np.full(shape=(len(X),), fill_value=self.mean_)


def get_model_registry(random_state: int = RANDOM_STATE) -> dict:
    registry = {
        "mean_baseline": MeanBaseline(),
        "linear_regression": LinearRegression(),
        "decision_tree": DecisionTreeRegressor(max_depth=8, random_state=random_state),
        "random_forest": RandomForestRegressor(
            n_estimators=300, max_depth=12, n_jobs=-1, random_state=random_state
        ),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=300, max_depth=4, learning_rate=0.05, random_state=random_state
        ),
    }
    if _HAS_XGB:
        registry["xgboost"] = XGBRegressor(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            n_jobs=-1,
        )
    else:
        logger.warning("xgboost is not installed; 'xgboost' model unavailable.")
    return registry


def train_model(model_name: str, X_train: pd.DataFrame, y_train: pd.Series):
    registry = get_model_registry()
    if model_name not in registry:
        raise ValueError(f"Unknown model '{model_name}'. Options: {list(registry)}")
    model = registry[model_name]
    logger.info("Training %s on %d rows / %d features", model_name, *X_train.shape)
    model.fit(X_train, y_train)
    return model


def get_feature_importance(model, feature_names: list[str]) -> pd.Series:
    """Return a sorted feature-importance series, works for tree-based models."""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_)
    else:
        return pd.Series(dtype=float)
    return pd.Series(importances, index=feature_names).sort_values(ascending=False)
