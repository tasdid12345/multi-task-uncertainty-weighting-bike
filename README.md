# Multi-task uncertainty weighting vs. per-task LightGBM on UCI Bike Sharing

Question: can a per-task, cross-validated LightGBM model beat an uncertainty-weighted shared MLP on the two rider-count tasks (casual, registered) of the UCI Bike Sharing (hourly) data?

Setup
- Data: UCI Bike Sharing (dataset id 275), fetched with `ucimlrepo`.
- Chronological 80/20 split: first 80% train, last 20% test. The test set is used only for final scoring.
- Baseline: shared 64-64 MLP on standardized targets, Kendall-style weights (1 / validation residual variance) refreshed over 3 rounds.
- Improved: one LightGBM regressor per task, hyperparameters chosen by 3-fold time-series CV on the training set only.
- Metric: mean over tasks of test RMSE / test std (lower is better). Secondary: test MAE per task.
- Seeds 42, 43, 44.

Run
```
pip install -r requirements.txt
python analysis.py
```
Outputs: `results.json` and `figures/main_score.png`.

Notes
- The target columns are located in either the features or the targets frame returned by `ucimlrepo`.
- `cnt` (casual + registered) is excluded from the features to avoid target leakage.

## Reference

This project is motivated by the following paper:

> L Kirchdorfer, T Sesterhenn, C Bartelt (2026). *Investigating Uncertainty Weighting for Multi-Task Learning: Insights and Analytical Alternative: L. Kirchdorfer et al.*. International Journal of. https://link.springer.com/article/10.1007/s11263-025-02625-x

Paper author: Prof. Dr. Heiner Stuckenschmidt, Universität Mannheim (heiner.stuckenschmidt@uni-mannheim.de)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
