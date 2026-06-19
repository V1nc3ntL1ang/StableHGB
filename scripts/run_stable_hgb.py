from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest import compute_metrics, run_backtest, write_equity_csv, write_metrics_csv
from src.experiment_config import RETURN_WEIGHT, SHARPE_WEIGHT, TEST_START
from src.ml_dataset import FEATURE_COLUMNS
from src.ml_models import get_ml_models
from src.experiment_protocol import (
    FINAL_MODEL_TRAIN_CUTOFF,
    FORMAL_SELECTION_RULE,
    FORMAL_SELECTION_RULE_DESCRIPTION,
    INVESTMENT_END,
    VALIDATION_FOLDS,
    add_formal_validation_score,
    get_worker_count,
    load_formal_frames,
    policy_to_json,
    safe_auc,
    score_metrics,
)
from src.stable_hgb import (
    STABLE_HGB_MODEL_NAME,
    STABLE_HGB_MODEL_PARAMS,
    STABLE_HGB_STRATEGY_NAME,
    get_stable_hgb_policy_params,
)
from src.paths import (
    STABLE_HGB_EQUITY_CSV,
    STABLE_HGB_METRICS_CSV,
    STABLE_HGB_VALIDATION_CSV,
    ensure_output_dirs,
)
from src.position_policy import build_policy_position


MODEL_NAME = STABLE_HGB_MODEL_NAME
STRATEGY_NAME = STABLE_HGB_STRATEGY_NAME
MODEL_PARAMS = STABLE_HGB_MODEL_PARAMS


def evaluate_validation_fold(
    labeled: pd.DataFrame,
    policy: dict[str, float | int | str],
    fold_name: str,
    fold_start: pd.Timestamp,
    fold_end: pd.Timestamp,
    worker_count: int,
) -> tuple[dict[str, float | int | str], dict[str, float]]:
    train = labeled[labeled["target_end_date"] < fold_start]
    valid_labels = labeled[
        (labeled["date"] >= fold_start)
        & (labeled["date"] < fold_end)
        & (labeled["target_end_date"] < fold_end)
    ]
    if train.empty or valid_labels.empty:
        raise ValueError(f"{fold_name} has empty train or validation labels")

    print(
        f"[{MODEL_NAME}] {fold_name}: train_labels={len(train)} valid_labels={len(valid_labels)}",
        flush=True,
    )

    model = get_ml_models(n_jobs=worker_count)[MODEL_NAME]
    model.fit(train[FEATURE_COLUMNS], train["future_up_5d"])
    probability = pd.Series(model.predict_proba(labeled[FEATURE_COLUMNS])[:, 1], index=labeled.index)

    position = build_policy_position(labeled, probability, policy)
    buy_hold_position = pd.Series(1.0, index=labeled.index)
    buy_hold_metrics = compute_metrics(
        run_backtest(labeled, buy_hold_position, test_start=fold_start, test_end=fold_end)
    )
    metrics = compute_metrics(run_backtest(labeled, position, test_start=fold_start, test_end=fold_end))
    scores = score_metrics(metrics, buy_hold_metrics)

    valid_probability = probability.loc[valid_labels.index]
    label_metrics = {
        f"{fold_name}_accuracy": float(
            accuracy_score(valid_labels["future_up_5d"], (valid_probability >= 0.5).astype(int))
        ),
        f"{fold_name}_auc": safe_auc(valid_labels["future_up_5d"], valid_probability),
    }
    row = {
        "model": MODEL_NAME,
        "strategy": STRATEGY_NAME,
        "fold": fold_name,
        "policy_params": policy_to_json(policy),
        "mapping_type": str(policy["mapping_type"]),
        **scores,
        "valid_cumulative_return": metrics["cumulative_return"],
        "valid_annualized_return": metrics["annualized_return"],
        "valid_max_drawdown": metrics["max_drawdown"],
        "valid_sharpe": metrics["sharpe"],
        "buy_hold_cumulative_return": buy_hold_metrics["cumulative_return"],
        "buy_hold_sharpe": buy_hold_metrics["sharpe"],
    }
    return row, label_metrics


def summarize_validation(rows: list[dict[str, float | int | str]]) -> dict[str, float]:
    validation = pd.DataFrame(rows)
    summary = {
        "valid_min_score": float(validation["valid_selection_score"].min()),
        "valid_mean_score": float(validation["valid_selection_score"].mean()),
        "valid_mean_return": float(validation["valid_cumulative_return"].mean()),
        "valid_return_std": float(validation["valid_cumulative_return"].std()),
        "valid_mean_sharpe": float(validation["valid_sharpe"].mean()),
        "valid_worst_drawdown": float(validation["valid_max_drawdown"].min()),
    }
    scored = add_formal_validation_score(pd.DataFrame([summary])).iloc[0]
    summary["valid_score"] = float(scored["valid_score"])
    return summary


def evaluate_final_strategy(
    labeled: pd.DataFrame,
    trading: pd.DataFrame,
    policy: dict[str, float | int | str],
    validation_summary: dict[str, float],
    validation_label_metrics: dict[str, float],
    worker_count: int,
) -> tuple[dict[str, float | int | str | bool], pd.DataFrame]:
    train = labeled[labeled["target_end_date"] < FINAL_MODEL_TRAIN_CUTOFF]
    test_labels = labeled[(labeled["date"] >= TEST_START) & (labeled["target_end_date"] <= INVESTMENT_END)]
    if train.empty or test_labels.empty:
        raise ValueError("Panel C has empty final train or test labels")

    model = get_ml_models(n_jobs=worker_count)[MODEL_NAME]
    model.fit(train[FEATURE_COLUMNS], train["future_up_5d"])

    trading_probability = pd.Series(model.predict_proba(trading[FEATURE_COLUMNS])[:, 1], index=trading.index)
    position = build_policy_position(trading, trading_probability, policy)
    equity = run_backtest(trading, position, test_start=TEST_START)
    metrics = compute_metrics(equity)

    buy_hold_position = pd.Series(1.0, index=trading.index)
    buy_hold_metrics = compute_metrics(run_backtest(trading, buy_hold_position, test_start=TEST_START))

    test_probability = pd.Series(model.predict_proba(test_labels[FEATURE_COLUMNS])[:, 1], index=test_labels.index)
    test_pred = (test_probability >= 0.5).astype(int)

    metric_row: dict[str, float | int | str | bool] = {
        "strategy": STRATEGY_NAME,
        "model": MODEL_NAME,
        "panel": "Panel C",
        "feature_count": len(FEATURE_COLUMNS),
        "features": ", ".join(FEATURE_COLUMNS),
        "model_params": policy_to_json(MODEL_PARAMS),
        "validation_method": "rolling_folds_2022_2024",
        "policy_family": "relative_signal_stabilizer_with_trend_position_guard",
        "policy_selection": FORMAL_SELECTION_RULE,
        "selection_rule": FORMAL_SELECTION_RULE_DESCRIPTION,
        "return_weight": RETURN_WEIGHT,
        "sharpe_weight": SHARPE_WEIGHT,
        "uses_moving_average_alignment_as_model_feature": False,
        "uses_trend_position_guard": True,
        "train_label_start": train["date"].min().strftime("%Y/%m/%d"),
        "train_label_end": train["date"].max().strftime("%Y/%m/%d"),
        "train_target_end": train["target_end_date"].max().strftime("%Y/%m/%d"),
        "test_label_start": test_labels["date"].min().strftime("%Y/%m/%d"),
        "test_label_end": test_labels["date"].max().strftime("%Y/%m/%d"),
        "test_target_end": test_labels["target_end_date"].max().strftime("%Y/%m/%d"),
        "investment_start": TEST_START.strftime("%Y/%m/%d"),
        "investment_end": INVESTMENT_END.strftime("%Y/%m/%d"),
        "trading_days": len(equity),
        "selected_mapping_type": str(policy["mapping_type"]),
        "selected_policy_params": policy_to_json(policy),
        **validation_summary,
        **validation_label_metrics,
        "test_accuracy": float(accuracy_score(test_labels["future_up_5d"], test_pred)),
        "test_auc": safe_auc(test_labels["future_up_5d"], test_probability),
        "buy_hold_cumulative_return": buy_hold_metrics["cumulative_return"],
        "excess_return_vs_buy_hold": metrics["cumulative_return"] - buy_hold_metrics["cumulative_return"],
        **metrics,
    }

    equity.insert(0, "strategy", STRATEGY_NAME)
    equity.insert(1, "model", MODEL_NAME)
    return metric_row, equity


def main() -> None:
    ensure_output_dirs()
    worker_count = get_worker_count()
    labeled, trading = load_formal_frames()
    policy = get_stable_hgb_policy_params()

    print(
        f"StableHGB protocol: workers={worker_count} model={MODEL_NAME} "
        f"folds={len(VALIDATION_FOLDS)} final_train_cutoff={FINAL_MODEL_TRAIN_CUTOFF.date()}",
        flush=True,
    )
    print(
        f"labeled: {labeled['date'].min().date()}..{labeled['date'].max().date()} "
        f"rows={len(labeled)} target_end_max={labeled['target_end_date'].max().date()}",
        flush=True,
    )
    print(
        f"trading: {trading['date'].min().date()}..{trading['date'].max().date()} rows={len(trading)}",
        flush=True,
    )
    print(f"StableHGB policy: {policy}", flush=True)

    model_names = set(get_ml_models(n_jobs=worker_count))
    if MODEL_NAME not in model_names:
        raise ValueError(f"Unknown model: {MODEL_NAME}")

    validation_rows: list[dict[str, float | int | str]] = []
    validation_label_metrics: dict[str, float] = {}
    for fold_name, fold_start, fold_end in VALIDATION_FOLDS:
        row, fold_label_metrics = evaluate_validation_fold(
            labeled,
            policy,
            fold_name,
            fold_start,
            fold_end,
            worker_count,
        )
        validation_rows.append(row)
        validation_label_metrics.update(fold_label_metrics)
        print(
            f"[{MODEL_NAME}] {fold_name}: return={row['valid_cumulative_return']:.6f} "
            f"score={row['valid_selection_score']:.6f}",
            flush=True,
        )

    validation_summary = summarize_validation(validation_rows)
    metric_row, equity = evaluate_final_strategy(
        labeled,
        trading,
        policy,
        validation_summary,
        validation_label_metrics,
        worker_count,
    )

    metrics = pd.DataFrame([metric_row])
    validation = pd.DataFrame(validation_rows)

    write_metrics_csv(metrics, STABLE_HGB_METRICS_CSV)
    write_metrics_csv(validation, STABLE_HGB_VALIDATION_CSV)
    write_equity_csv(equity, STABLE_HGB_EQUITY_CSV)

    print(
        f"[{MODEL_NAME}] final_return={metric_row['cumulative_return']:.6f} "
        f"excess={metric_row['excess_return_vs_buy_hold']:.6f} "
        f"max_dd={metric_row['max_drawdown']:.6f}",
        flush=True,
    )
    print(f"wrote {STABLE_HGB_METRICS_CSV} rows={len(metrics)}", flush=True)
    print(f"wrote {STABLE_HGB_VALIDATION_CSV} rows={len(validation)}", flush=True)
    print(f"wrote {STABLE_HGB_EQUITY_CSV} rows={len(equity)}", flush=True)


if __name__ == "__main__":
    main()
