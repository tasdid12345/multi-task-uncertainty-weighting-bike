# Per-task LightGBM vs uncertainty-weighted shared MLP on bike-sharing counts

Motivation: Multi-task networks often combine task losses with hand-set or learned weights, and "Investigating Uncertainty Weighting for Multi-Task Learning: Insights and Analytical Alternative" studies how uncertainty-based loss weighting behaves and proposes an analytical alternative. This experiment asks a simpler, practical question: when the tasks differ greatly in scale, does a shared neural network with uncertainty-style weights lose to separate, cross-validated gradient-boosted models?

Question: Can a per-task, cross-validated LightGBM model beat an uncertainty-weighted shared MLP on two rider-count tasks of different scale in the UCI Bike Sharing data?

Dataset: UCI Bike Sharing (hourly), fetched with the `ucimlrepo` package. Tasks are casual and registered rider counts. The split is chronological: first 80% train, last 20% test. This is not the paper's own dataset.

Baseline: a shared-trunk scikit-learn MLPRegressor (64-64) on standardized targets, with Kendall-style uncertainty weights approximated by inverse validation residual variance, refreshed over 3 rounds.

Improved: one LightGBM regressor per task, with hyperparameters chosen by 3-fold time-series cross-validation on the training data only.

How to run:

```
pip install -r requirements.txt
python analysis.py
```

Results: results are written to `results.json` (and a chart to `figures/main_score.png`).

## Reference

This project is motivated by the following paper:

> L Kirchdorfer, T Sesterhenn, C Bartelt (2026). *Investigating Uncertainty Weighting for Multi-Task Learning: Insights and Analytical Alternative: L. Kirchdorfer et al.*. International Journal of. https://link.springer.com/article/10.1007/s11263-025-02625-x

Paper author: Prof. Dr. Heiner Stuckenschmidt, Universität Mannheim (heiner.stuckenschmidt@uni-mannheim.de)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
