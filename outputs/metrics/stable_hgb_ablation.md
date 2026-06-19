| Experiment | Position rule | Valid score | Cumulative return | Excess vs buy-hold | Max drawdown | Sharpe | Test AUC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Buy-and-hold reference | constant_full_position |  | 109.21% | 0.00% | -17.26% | 2.30 |  |
| Best ML baseline reference | Panel B rank_linear | 0.1916 | 115.22% | 6.01% | -13.49% | 2.66 | 0.653 |
| HGB + standard policy | rank_linear | 0.2182 | 96.48% | -12.73% | -12.35% | 2.61 | 0.635 |
| HGB + standard policy + Trend Position Guard | rank_linear + Trend Position Guard | 0.2250 | 113.48% | 4.27% | -13.81% | 2.61 | 0.635 |
| HGB + Relative Signal Stabilizer | Relative Signal Stabilizer | 0.2081 | 117.48% | 8.27% | -7.52% | 3.55 | 0.635 |
| StableHGB | Relative Signal Stabilizer + Trend Position Guard | 0.2998 | 152.03% | 42.82% | -9.84% | 3.38 | 0.635 |
