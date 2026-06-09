# Finance Timing Project

This project studies a simple index timing task. The workflow is organized as a few layers:

1. Data layer

   Raw daily and intraday files are stored in `data/raw/`. The main experiments use the processed daily data in `data/processed/`.

2. Feature layer

   The feature groups are defined in `src/feature_groups.py`:

   - `market`: basic price and volume features
   - `regime`: market features plus trend and risk state
   - `momentum`: market features plus normalized momentum state
   - `composite`: all available features

3. Model layer

   The implemented models are:

   - logistic regression
   - random forest
   - gradient boosting
   - histogram gradient boosting
   - LightGBM
   - LSTM (GRU) — a single-layer GRU classifier via TensorFlow/Keras (`src/lstm_model.py`)

4. Position policy layer

   Model probabilities are converted into positions with continuous mappings:

   - `linear_clipped`
   - `rank_linear`
   - `sigmoid`

   The search also includes `smoothing_window = 1, 3, 5`, where `1` means no smoothing.

5. Evaluation layer

   Parameters are selected on the validation period using:

   ```text
   valid_selection_score = 0.5 * valid_return_score + 0.5 * valid_sharpe_score
   ```

   The final comparison is reported on the test period.

## Current Results

Best test return among the searched ML combinations:

```text
logistic_regression + composite
mapping = linear_clipped
smoothing_window = 3

test_return = 107.35%
max_drawdown = -14.62%
sharpe = 2.69
excess_vs_buy_hold = +8.07%
```

Best test Sharpe:

```text
random_forest + momentum
mapping = linear_clipped
smoothing_window = 5

test_return = 87.07%
max_drawdown = -7.94%
sharpe = 2.97
```

Best LightGBM combination:

```text
lightgbm + momentum
mapping = linear_clipped
smoothing_window = 3

test_return = 105.13%
max_drawdown = -11.95%
sharpe = 2.77
```

Best LSTM combination:

```text
lstm + composite
mapping = sigmoid
smoothing_window = 5

test_return = 84.73%
max_drawdown = -15.30%
sharpe = 2.18
excess_vs_buy_hold = -14.55%
```

> **Note on LSTM performance:** The GRU model achieves AUC ≈ 0.50 across all feature
> groups — essentially random. This is expected: with only ~1,600 training samples,
> deep learning cannot extract meaningful signal from low-SNR financial data.
> Tree-based models and logistic regression are far more effective at this data
> scale. The LSTM model is kept in the codebase as a reference for comparison and
> as a starting point for larger datasets.

## Reproduce

Install dependencies (the LSTM model requires `tensorflow` in addition to the
standard packages):

```bash
pip install pandas numpy scikit-learn lightgbm matplotlib joblib tensorflow
```

Use the `finance` conda environment (if available), then run:

```bash
python scripts/run_all.py
```

To rerun only the model and feature-group search:

```bash
python scripts/run_feature_group_ablation.py
```

The number of workers can be controlled with:

```bash
FINANCE_WORKERS=4 python scripts/run_feature_group_ablation.py
```

Main outputs are written to:

- `outputs/metrics/`
- `outputs/equity/`
- `outputs/plots/`
