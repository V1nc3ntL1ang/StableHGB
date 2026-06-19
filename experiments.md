# Experiment Record

This file records the reproducible experiments used for the course report.

## Evaluation Protocol

| Item | Setting |
|---|---|
| Initial capital | 100,000 RMB |
| Investment period | 2025-01-01 to 2026-05-06 |
| Trading calendar used in backtest | 2025-01-02 to 2026-05-06, 321 trading days |
| Data source for formal baselines | `data/processed/daily_features.csv` |
| Data source for formal machine-learning panels | `data/processed/daily.csv` |
| Baseline output metrics | `outputs/metrics/baselines.csv` |
| Baseline output equity curves | `outputs/equity/baselines.csv` |
| Machine-learning output metrics | `outputs/metrics/ml_baselines.csv` |
| Machine-learning output equity curves | `outputs/equity/ml_baselines.csv` |
| StableHGB output metrics | `outputs/metrics/stable_hgb_metrics.csv` |
| StableHGB validation folds | `outputs/metrics/stable_hgb_validation_folds.csv` |
| StableHGB output equity curve | `outputs/equity/stable_hgb_equity.csv` |
| Environment file | `environment.yml` |
| Label requirement | Baseline strategies do not use labels; machine-learning panels use 5-day forward return labels only for training and validation |
| Formal validation selection rule | `valid_score = mean(valid_cumulative_return) - 0.25 * std(valid_cumulative_return)` |

The formal baseline evaluation uses the full available trading feature data through 2026-05-06. It does not use `data/processed/ml_dataset.csv`, because that labeled dataset drops the final days that do not have a future 5-day label.

For the formal machine-learning panels, the final model-fitting set uses labels whose 5-day target window ends before 2024-01-01. The 2024 labels are kept as the final validation holdout, and the investment-period backtest starts on 2025-01-01.

Panel A is a direct tradable baseline panel and has no validation-based model or policy selection. Panel B and Panel C use the formal validation rule above. Ties are broken by mean validation return, mean validation selection score, mean validation Sharpe, and worst validation drawdown.

Environment setup:

```bash
conda env create -f environment.yml
conda activate finance-stablehgb
```

Full formal workflow:

```bash
python -u scripts/run_all_experiments.py --workers $(sysctl -n hw.ncpu)
```

StableHGB reproduction check without rewriting output files:

```bash
python -u scripts/reproduce_stable_hgb.py
```

## Panel A: Tradable Baseline Strategies

| Strategy | Cumulative Return | Annualized Return | Max Drawdown | Sharpe | Excess vs Buy-and-Hold |
|---|---:|---:|---:|---:|---:|
| Buy and hold | 109.21% | 78.52% | -17.26% | 2.30 | 0.00% |
| MA20 timing | 77.64% | 57.00% | -10.90% | 2.76 | -31.58% |
| 20-day momentum timing | 55.73% | 41.59% | -17.91% | 1.77 | -53.48% |
| MA20 and momentum combo | 66.94% | 49.53% | -11.69% | 2.38 | -42.27% |

Command:

```bash
python -u scripts/run_baselines.py
```

## Panel B: Competitive Machine-Learning Methods

Panel B uses the same 20 technical features for every model. The models are selected using rolling validation folds for 2022, 2023, and 2024. For each fold, training labels must end before the validation year begins. The final model is trained with labels whose target window ends before 2024-01-01 and is backtested from 2025-01-01 to 2026-05-06 using daily features through 2026-05-06.

Within each model, the selected policy maximizes `valid_score = mean(valid_cumulative_return) - 0.25 * std(valid_cumulative_return)` across the three validation folds.

Panel B only uses standard probability-to-position mappings: `linear_clipped`, `rank_linear`, `sigmoid`, `power`, and `threshold`. It does not use the StableHGB-specific `relative_signal_stabilizer` mapping or the Trend Position Guard.

The recorded Panel B run used 10 workers and a fixed discrete validation grid. The common position parameters are `min_position={0.00,0.10,0.20,0.30}`, `max_position={0.70,0.80,0.90,1.00}`, `smoothing_window={1,3,5}`, and `smoothing_method={sma}`. Combined with the mapping-specific grids, this produces 10,656 candidates per model per validation fold and 159,840 validation rows across all five models and three folds.

| Mapping | Candidates per Fold |
|---|---:|
| `linear_clipped` | 1,296 |
| `rank_linear` | 3,072 |
| `sigmoid` | 1,440 |
| `power` | 4,608 |
| `threshold` | 240 |
| Total | 10,656 |

| Model | Policy Mapping | Valid Score | Cumulative Return | Annualized Return | Max Drawdown | Sharpe | Excess vs Buy-and-Hold | Test AUC |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Logistic regression | power | 0.2463 | 98.80% | 71.50% | -14.84% | 2.28 | -10.41% | 0.62 |
| Random forest | rank_linear | 0.2275 | 110.90% | 79.64% | -10.79% | 2.93 | 1.68% | 0.64 |
| Gradient boosting | threshold | 0.2771 | 99.78% | 72.17% | -12.28% | 2.55 | -9.43% | 0.63 |
| Histogram gradient boosting | sigmoid | 0.1693 | 97.18% | 70.40% | -14.76% | 2.38 | -12.03% | 0.64 |
| LightGBM | rank_linear | 0.1916 | 115.22% | 82.53% | -13.49% | 2.66 | 6.01% | 0.65 |

Command:

```bash
PYTHONUNBUFFERED=1 FINANCE_WORKERS=10 python -u scripts/run_ml_baselines.py
```

## Panel C: StableHGB

Panel C evaluates StableHGB. It uses a histogram gradient boosting classifier with the same 20 model features as Panel B. The moving-average alignment signal `ma_alignment` is not used as a model feature; it is used only by the Trend Position Guard inside the trading policy.

The trading policy has two components. The Relative Signal Stabilizer converts predicted probabilities into expanding percentile ranks and changes position only after the rank signal is confirmed for consecutive days. The Trend Position Guard keeps the strategy fully invested when `ma_alignment > 0.5`. The reported candidate is selected by the same formal validation rule used in Panel B. Rolling folds for 2022, 2023, and 2024 are reported as validation evidence, while the final model is trained with labels whose target window ends before 2024-01-01 and is tested on the 2025-01-01 to 2026-05-06 investment period.

| Strategy | Policy | Valid Score | Cumulative Return | Annualized Return | Max Drawdown | Sharpe | Excess vs Buy-and-Hold | Test AUC |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| StableHGB | Relative Signal Stabilizer + Trend Position Guard | 0.2998 | 152.03% | 106.61% | -9.84% | 3.38 | 42.82% | 0.63 |

Validation fold details:

| Fold | Cumulative Return | Selection Score | Max Drawdown | Sharpe | Buy-and-Hold Return |
|---|---:|---:|---:|---:|---:|
| valid_2022 | 32.82% | 1.44 | -22.29% | 1.24 | 22.20% |
| valid_2023 | 20.00% | 0.54 | -7.99% | 1.58 | 47.60% |
| valid_2024 | 47.41% | 3.82 | -40.84% | 1.12 | 9.33% |

Command:

```bash
PYTHONUNBUFFERED=1 FINANCE_WORKERS=$(sysctl -n hw.ncpu) python -u scripts/run_stable_hgb.py
```

## Component Ablation

This ablation fixes the StableHGB histogram gradient boosting model and the same formal train/validation/test protocol. It changes only the position-construction components, so the comparison isolates the contribution of the Relative Signal Stabilizer and the Trend Position Guard.

| Experiment | Position rule | Valid score | Cumulative return | Excess vs buy-hold | Max drawdown | Sharpe | Test AUC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Buy-and-hold reference | constant_full_position |  | 109.21% | 0.00% | -17.26% | 2.30 |  |
| Best ML baseline reference | Panel B rank_linear | 0.1916 | 115.22% | 6.01% | -13.49% | 2.66 | 0.653 |
| HGB + standard policy | rank_linear | 0.2182 | 96.48% | -12.73% | -12.35% | 2.61 | 0.635 |
| HGB + standard policy + Trend Position Guard | rank_linear + Trend Position Guard | 0.2250 | 113.48% | 4.27% | -13.81% | 2.61 | 0.635 |
| HGB + Relative Signal Stabilizer | Relative Signal Stabilizer | 0.2081 | 117.48% | 8.27% | -7.52% | 3.55 | 0.635 |
| StableHGB | Relative Signal Stabilizer + Trend Position Guard | 0.2998 | 152.03% | 42.82% | -9.84% | 3.38 | 0.635 |

Command:

```bash
PYTHONUNBUFFERED=1 FINANCE_WORKERS=10 python -u scripts/run_stable_hgb_ablation.py
```

Outputs:

| Output | Path |
|---|---|
| Component ablation table | `outputs/metrics/stable_hgb_ablation.csv` |
| Markdown table | `outputs/metrics/stable_hgb_ablation.md` |

## Figures

Experiment figures are generated from the formal output CSV files only.

| Figure | Output |
|---|---|
| All tradable baselines | `outputs/plots/all_baselines.png` |
| All competitive machine-learning baselines with buy-and-hold reference | `outputs/plots/all_ml_baselines.png` |
| StableHGB vs best baseline and best machine-learning baseline | `outputs/plots/stable_hgb_vs_references.png` |
| Drawdown comparison against best references | `outputs/plots/drawdown_stablehgb_references.png` |
| Risk-return scatter | `outputs/plots/risk_return_scatter.png` |
| StableHGB position and equity | `outputs/plots/stable_hgb_position.png` |

Command:

```bash
PYTHONUNBUFFERED=1 python -u scripts/plot_results.py
```

## Notes

- `theoretical_optimal` is excluded from the formal baseline panel because it uses future returns and is not tradable.
- Baselines are reference strategies. StableHGB is evaluated separately against the same investment period.
- The environment is pinned in `environment.yml`.
