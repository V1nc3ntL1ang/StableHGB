# Final Experiment Summary

The complete experiment record is in `experiments.md`. This document keeps only the final protocol and results used for the course report.

## Unified Experiment Protocol

| Item | Setting |
|---|---|
| Initial capital | 100,000 RMB |
| Investment period | 2025-01-01 to 2026-05-06 |
| Actual backtest trading days | 2025-01-02 to 2026-05-06, 321 trading days |
| Prediction label | Five-day forward return `future_ret_5d` |
| Final model training labels | Five-day label windows ending before 2024-01-01 |
| Validation evidence | Three rolling validation folds: 2022, 2023, and 2024 |
| Selection rule | `valid_score = mean(valid_cumulative_return) - 0.25 * std(valid_cumulative_return)` |
| Full experiment record | `experiments.md` |

The `future_ret_5d` label can only be generated through 2026-04-24, while daily trading features are available through 2026-05-06. Therefore, the machine-learning models are trained and validated on labeled samples, and the final trading backtest uses daily features to infer positions through 2026-05-06.

Panel A contains tradable baselines without parameter tuning. Panel B and Panel C select candidates using the rule above over the 2022, 2023, and 2024 validation folds. Ties in `valid_score` are resolved by mean validation return, mean validation selection score, mean validation Sharpe ratio, and worst validation drawdown.

## Panel A: Tradable Baseline Strategies

| Strategy | Cumulative Return | Annualized Return | Max Drawdown | Sharpe | Excess vs Buy-and-Hold |
|---|---:|---:|---:|---:|---:|
| Buy and hold | 109.21% | 78.52% | -17.26% | 2.30 | 0.00% |
| MA20 timing | 77.64% | 57.00% | -10.90% | 2.76 | -31.58% |
| 20-day momentum timing | 55.73% | 41.59% | -17.91% | 1.77 | -53.48% |
| MA20 and momentum combo | 66.94% | 49.53% | -11.69% | 2.38 | -42.27% |

## Panel B: Competitive Machine-Learning Baselines

Panel B uses a 10-worker discrete validation search. Each model evaluates 10,656 position-policy candidates per validation fold, producing 159,840 validation rows across five models and three validation folds.

| Model | Policy Mapping | Valid Score | Cumulative Return | Annualized Return | Max Drawdown | Sharpe | Excess vs Buy-and-Hold | Test AUC |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Logistic regression | power | 0.2463 | 98.80% | 71.50% | -14.84% | 2.28 | -10.41% | 0.62 |
| Random forest | rank_linear | 0.2275 | 110.90% | 79.64% | -10.79% | 2.93 | 1.68% | 0.64 |
| Gradient boosting | threshold | 0.2771 | 99.78% | 72.17% | -12.28% | 2.55 | -9.43% | 0.63 |
| Histogram gradient boosting | sigmoid | 0.1693 | 97.18% | 70.40% | -14.76% | 2.38 | -12.03% | 0.64 |
| LightGBM | rank_linear | 0.1916 | 115.22% | 82.53% | -13.49% | 2.66 | 6.01% | 0.65 |

## Panel C: StableHGB

| Strategy | Policy | Valid Score | Cumulative Return | Annualized Return | Max Drawdown | Sharpe | Excess vs Buy-and-Hold | Test AUC |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| StableHGB | Relative Signal Stabilizer + Trend Position Guard | 0.2998 | 152.03% | 106.61% | -9.84% | 3.38 | 42.82% | 0.63 |

## Conclusion

Under the final protocol, `StableHGB` is the best tradable strategy:

```text
buy_hold cumulative return = 109.21%
best ML baseline cumulative return = 115.22%
StableHGB cumulative return = 152.03%
StableHGB excess vs buy_hold = +42.82%
StableHGB max_drawdown = -9.84%
StableHGB sharpe = 3.38
```

## Final Output Files

| Content | File |
|---|---|
| Panel A metrics | `outputs/metrics/baselines.csv` |
| Panel B metrics | `outputs/metrics/ml_baselines.csv` |
| Panel C metrics | `outputs/metrics/stable_hgb_metrics.csv` |
| Panel C validation folds | `outputs/metrics/stable_hgb_validation_folds.csv` |
| Figures | `outputs/plots/` |

## Reproduction Commands

```bash
conda env create -f environment.yml
conda activate finance-stablehgb
python -u scripts/run_all_experiments.py --workers $(sysctl -n hw.ncpu)
python -u scripts/reproduce_stable_hgb.py
```
