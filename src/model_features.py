from __future__ import annotations

from src.ml_dataset import FEATURE_COLUMNS


LOGISTIC_REGRESSION_FEATURES = [
    "volatility5",
    "volatility20_rank",
    "rebound_from_60d_low",
    "volatility20",
    "macd_histogram",
    "bb_width",
    "trend_strength60",
    "macd_signal",
    "ma60",
    "rebound_from_20d_low",
    "bb_lower",
    "ma60_slope",
]

RANDOM_FOREST_FEATURES = [
    "ma5",
    "ma10",
    "ma20",
    "ma60",
    "momentum20",
    "volatility5",
    "volatility20",
    "trend_strength20",
    "trend_strength60",
    "rebound_from_20d_low",
    "rebound_from_60d_low",
    "macd_signal",
    "macd_histogram",
    "ma20_slope",
    "ma60_slope",
    "close_vs_ma60",
]

GRADIENT_BOOSTING_FEATURES = FEATURE_COLUMNS
HIST_GRADIENT_BOOSTING_FEATURES = FEATURE_COLUMNS
LIGHTGBM_FEATURES = FEATURE_COLUMNS

MODEL_FEATURE_COLUMNS = {
    "logistic_regression": LOGISTIC_REGRESSION_FEATURES,
    "random_forest": RANDOM_FOREST_FEATURES,
    "gradient_boosting": GRADIENT_BOOSTING_FEATURES,
    "hist_gradient_boosting": HIST_GRADIENT_BOOSTING_FEATURES,
    "lightgbm": LIGHTGBM_FEATURES,
}
