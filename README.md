# Multi-task uncertainty weighting on UCI Bike Sharing

Compares a shared 64-64 MLP baseline with Kendall-style uncertainty weights against a per-task LightGBM model whose hyperparameters are chosen by 3-fold time-series cross-validation on the training set. Both models predict the `casual` and `registered` rider counts on the UCI Bike Sharing hourly data.

## Setup

```
pip install -r requirements.txt
python analysis.py
```

The script downloads the hourly data from the UCI archive (`hour.csv`), makes a chronological 80/20 train/test split, and writes `results.json` and `figures/main_score.png`.

## Metric

Mean over tasks of test RMSE divided by test standard deviation (NRMSE). Lower is better. Per-task MAE is also reported in `results.json`.

## Notes

- Features exclude `dteday`, `instant` and `cnt`, since `cnt` is the sum of the two targets.
- Seeds 42, 43 and 44 are used. The test set is used only for final scoring.
- The data is read directly from the UCI archive zip because the `ucimlrepo` fetch did not expose the `casual` and `registered` columns.

## Results (latest run)

On the UCI Bike Sharing hourly data, a Kendall-style uncertainty-weighted shared MLP reached a mean test NRMSE of 0.415 across casual and registered rider counts, while per-task LightGBM tuned by time-series CV reached 0.318 on the same chronological test split (lower is better).

| Metric | Value |
|---|---|
| baseline_score | 0.4146 |
| improved_score | 0.3176 |
| improvement | 0.09697 |
| baseline_score_std | 0.009439 |
| improved_score_std | 0 |
| baseline_mae_casual | 14.74 |
| improved_mae_casual | 10.94 |
| baseline_nrmse_casual | 0.4515 |
| improved_nrmse_casual | 0.3317 |
| baseline_mae_registered | 48.9 |
| improved_mae_registered | 37.13 |
| baseline_nrmse_registered | 0.3777 |
| improved_nrmse_registered | 0.3036 |

Run on 2026-10-21. Full output: [results.json](results.json) · [run.log](run.log)

Notes: Single chronological split; test set used only for final scoring. LightGBM is effectively deterministic, so seed variation is small. Baseline MLP uses early stopping on an internal split. Features exclude dteday, instant and cnt (cnt is the sum of the two targets, so it would leak). Data is read from the UCI archive hour.csv because the ucimlrepo fetch did not expose casual/registered. yr carries the long-term trend and trees cannot extrapolate beyond the training range. The paper's own model and results are not reproduced here.

## Reference

This project is motivated by the following paper:

> L Kirchdorfer, T Sesterhenn, C Bartelt (2026). *Investigating Uncertainty Weighting for Multi-Task Learning: Insights and Analytical Alternative: L. Kirchdorfer et al.*. International Journal of. https://link.springer.com/article/10.1007/s11263-025-02625-x

Paper author: Prof. Dr. Heiner Stuckenschmidt, Universität Mannheim (heiner.stuckenschmidt@uni-mannheim.de)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
