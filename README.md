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

## Reference

This project is motivated by the following paper:

> L Kirchdorfer, T Sesterhenn, C Bartelt (2026). *Investigating Uncertainty Weighting for Multi-Task Learning: Insights and Analytical Alternative: L. Kirchdorfer et al.*. International Journal of. https://link.springer.com/article/10.1007/s11263-025-02625-x

Paper author: Prof. Dr. Heiner Stuckenschmidt, Universität Mannheim (heiner.stuckenschmidt@uni-mannheim.de)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
