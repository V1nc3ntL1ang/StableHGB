from __future__ import annotations


STABLE_HGB_MODEL_NAME = "stable_hgb"
STABLE_HGB_STRATEGY_NAME = "StableHGB"

STABLE_HGB_MODEL_PARAMS = {
    "max_iter": 150,
    "learning_rate": 0.05,
    "max_leaf_nodes": 15,
    "min_samples_leaf": 70,
    "random_state": 42,
}

STABLE_HGB_POLICY_PARAMS = {
    "mapping_type": "relative_signal_stabilizer",
    "entry_rank": 0.525,
    "exit_rank": 0.475,
    "exit_gap": 0.05,
    "confirm_days": 2,
    "min_position": 0.0,
    "max_position": 1.0,
    "smoothing_window": 1,
    "smoothing_method": "sma",
    "trend_guard_feature": "ma_alignment",
    "trend_guard_threshold": 0.5,
    "trend_guard_min_position": 1.0,
}


def get_stable_hgb_policy_params() -> dict[str, float | int | str]:
    return STABLE_HGB_POLICY_PARAMS.copy()
