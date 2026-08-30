"""
End-to-end orchestration script.

Usage:
    python src/train_pipeline.py --model xgboost --include-iron-concentrate
    python src/train_pipeline.py --model random_forest
"""

import argparse
import logging

import joblib
import pandas as pd

from config import DATETIME_COL, MODELS_DIR, PROCESSED_DIR, TARGET_COL
from data_preprocessing import clean_pipeline
from time_alignment import resample_to_frequency, chronological_split
from feature_engineering import build_features
from models import get_model_registry, train_model, get_feature_importance
from evaluate import evaluate_models, plot_residuals, plot_feature_importance

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train the silica-concentrate soft sensor.")
    parser.add_argument(
        "--model",
        default="xgboost",
        choices=list(get_model_registry().keys()),
        help="Which single model to train and evaluate on the test set.",
    )
    parser.add_argument(
        "--include-iron-concentrate",
        action="store_true",
        help="Include lagged %% Iron Concentrate as a feature (comparison scenario).",
    )
    parser.add_argument(
        "--compare-all",
        action="store_true",
        help="Train and compare every model on the validation set before final test evaluation.",
    )
    return parser.parse_args()


def run_pipeline(model_name: str, include_iron_concentrate: bool, compare_all: bool):
    logger.info("Step 1/6: Cleaning raw data")
    cleaned = clean_pipeline()

    logger.info("Step 2/6: Time alignment / resampling")
    resampled = resample_to_frequency(cleaned)

    logger.info("Step 3/6: Feature engineering")
    features = build_features(resampled, include_iron_concentrate=include_iron_concentrate)

    logger.info("Step 4/6: Chronological train/val/test split")
    train_df, val_df, test_df = chronological_split(features)

    feature_cols = [c for c in features.columns if c not in (DATETIME_COL, TARGET_COL)]
    X_train, y_train = train_df[feature_cols], train_df[TARGET_COL]
    X_val, y_val = val_df[feature_cols], val_df[TARGET_COL]
    X_test, y_test = test_df[feature_cols], test_df[TARGET_COL]

    logger.info("Step 5/6: Model training")
    if compare_all:
        registry = get_model_registry()
        fitted = {name: train_model(name, X_train, y_train) for name in registry}
        val_results = evaluate_models(fitted, X_val, y_val)
        val_results.to_csv(PROCESSED_DIR / "model_comparison_validation.csv")
        best_model_name = val_results["RMSE"].idxmin()
        logger.info("Best model on validation set: %s", best_model_name)
        model = fitted[best_model_name]
        model_name = best_model_name
    else:
        model = train_model(model_name, X_train, y_train)

    logger.info("Step 6/6: Final evaluation on held-out test set")
    test_preds = model.predict(X_test)
    from evaluate import compute_metrics

    test_metrics = compute_metrics(y_test, test_preds)
    logger.info("Test metrics for %s: %s", model_name, test_metrics)

    plot_residuals(y_test, test_preds, model_name)
    importance = get_feature_importance(model, feature_cols)
    if not importance.empty:
        importance.to_csv(PROCESSED_DIR / f"feature_importance_{model_name}.csv")
        plot_feature_importance(importance, model_name)

    joblib.dump(model, MODELS_DIR / f"{model_name}.joblib")
    logger.info("Saved trained model to %s", MODELS_DIR / f"{model_name}.joblib")

    return model, test_metrics


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args.model, args.include_iron_concentrate, args.compare_all)
