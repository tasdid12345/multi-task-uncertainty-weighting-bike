"""Baseline vs improved multi-task rider-count regression on UCI Bike Sharing (hourly).

Baseline : shared 64-64 MLP on standardized targets, Kendall-style uncertainty weights
           (1 / validation residual variance), refreshed over 3 rounds, applied by scaling outputs by sqrt(w).
Improved : one LightGBM regressor per task, hyperparameters chosen by 3-fold time-series CV on the training set.
Metric   : mean over tasks of test RMSE / test std (lower is better). Secondary: test MAE per task.
"""
import json, sys, time
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless plotting
import matplotlib.pyplot as plt
import lightgbm as lgb
import socket
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, ParameterGrid
from sklearn.metrics import mean_absolute_error
from ucimlrepo import fetch_ucirepo

SEED, SEEDS = 42, [42, 43, 44]
TASKS = ["casual", "registered"]
SOURCE_URL = "https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset"
PARAM_GRID = {"n_estimators": [300, 600], "learning_rate": [0.03, 0.1],
              "num_leaves": [15, 31, 63], "min_child_samples": [20, 50]}
socket.setdefaulttimeout(60)  # 60 s timeout for network downloads (used by ucimlrepo's HTTP calls)


def fetch_data(max_tries=5):
    """Fetch the dataset via ucimlrepo; retry with exponential backoff on 429/5xx or connection errors."""
    for i in range(max_tries):
        try:
            ds = fetch_ucirepo(id=275)
            return ds.data.features.copy(), ds.data.targets.copy()
        except Exception as e:
            code = getattr(getattr(e, "response", None), "status_code", None)
            fatal = code is not None and (code < 500 and code != 429)
            if fatal or i == max_tries - 1:
                raise
            print(f"fetch failed ({e}); retrying in {2 ** i}s")
            time.sleep(2 ** i)


def load_data():
    """Return chronologically ordered frame (features + casual/registered) and feature names."""
    X, y = fetch_data()
    X = X.drop(columns=[c for c in ["dteday", "instant"] if c in X.columns])  # ids / raw date strings
    df = pd.concat([X, y[TASKS]], axis=1).dropna().reset_index(drop=True)
    return df, [c for c in df.columns if c not in TASKS]


def rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


def fit_mlp(X, Y, w, seed):
    """Shared MLP. Outputs are scaled by sqrt(w): loss = sum_k w_k (y_k - yhat_k)^2 in standardized units."""
    m = MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=300, early_stopping=True, random_state=seed)
    return m.fit(X, Y * np.sqrt(w))


def run_baseline(train, test, feats, seed):
    """Kendall-style uncertainty-weighted shared MLP; weights = 1/val residual variance, 3 refresh rounds."""
    sc = StandardScaler().fit(train[feats])
    mu, sd = train[TASKS].mean().values, train[TASKS].std().values
    X, Xt = sc.transform(train[feats]), sc.transform(test[feats])
    Y = (train[TASKS].values - mu) / sd
    n_sub = int(0.85 * len(train))  # last 15% of training = validation set used for weight estimation
    Xa, Ya, Xv, Yv = X[:n_sub], Y[:n_sub], X[n_sub:], Y[n_sub:]
    w = np.ones(len(TASKS))
    for r in range(3):
        m = fit_mlp(Xa, Ya, w, seed)
        resid = Yv - m.predict(Xv) / np.sqrt(w)
        w = 1.0 / resid.var(axis=0)  # refreshed uncertainty weights
        print(f"  baseline round {r + 1}: weights={np.round(w, 3)}")
    m = fit_mlp(X, Y, w, seed)  # final fit on full training data with the refreshed weights
    return m.predict(Xt) / np.sqrt(w) * sd + mu


def tune_lgb(train, feats, task):
    """Pick LightGBM hyperparameters for one task by 3-fold time-series CV on the training set only."""
    best, best_s = None, np.inf
    for p in ParameterGrid(PARAM_GRID):
        errs = []
        for tr_i, va_i in TimeSeriesSplit(n_splits=3).split(train):
            m = lgb.LGBMRegressor(**p, random_state=SEED, verbose=-1)
            m.fit(train.iloc[tr_i][feats], train.iloc[tr_i][task])
            errs.append(rmse(train.iloc[va_i][task].values, m.predict(train.iloc[va_i][feats])))
        if np.mean(errs) < best_s:
            best, best_s = p, float(np.mean(errs))
    print(f"  tuned {task}: {best} (CV RMSE {best_s:.2f})")
    return best


def run_lgb(train, test, feats, params, seed):
    """Improved model: one LightGBM regressor per task with CV-chosen hyperparameters."""
    return np.column_stack([
        lgb.LGBMRegressor(**params[t], random_state=seed, verbose=-1)
        .fit(train[feats], train[t]).predict(test[feats]) for t in TASKS])


def evaluate(pred, test):
    """Return (mean NRMSE over tasks, dict of per-task MAE and NRMSE) on the test set."""
    nrmse, mae = {}, {}
    for k, t in enumerate(TASKS):
        y = test[t].values
        nrmse[t] = rmse(y, pred[:, k]) / float(np.std(y))
        mae[t] = float(mean_absolute_error(y, pred[:, k]))
    return float(np.mean(list(nrmse.values()))), mae, nrmse


def main():
    t0 = time.time()
    df, feats = load_data()
    n = len(df)
    n_tr = int(0.8 * n)  # chronological split: first 80% train, last 20% test
    train, test = df.iloc[:n_tr].reset_index(drop=True), df.iloc[n_tr:].reset_index(drop=True)
    print(f"rows={n} train={len(train)} test={len(test)} features={len(feats)}")

    print("Tuning LightGBM by 3-fold time-series CV (training data only)...")
    params = {t: tune_lgb(train, feats, t) for t in TASKS}

    b_scores, i_scores, b_mae, i_mae, b_nr, i_nr = [], [], {t: [] for t in TASKS}, {t: [] for t in TASKS}, {t: [] for t in TASKS}, {t: [] for t in TASKS}
    for seed in SEEDS:
        print(f"Seed {seed}: baseline MLP...")
        s, mae, nr = evaluate(run_baseline(train, test, feats, seed), test)
        b_scores.append(s)
        for t in TASKS: b_mae[t].append(mae[t]); b_nr[t].append(nr[t])
        print(f"Seed {seed}: improved LightGBM...")
        s, mae, nr = evaluate(run_lgb(train, test, feats, params, seed), test)
        i_scores.append(s)
        for t in TASKS: i_mae[t].append(mae[t]); i_nr[t].append(nr[t])
        print(f"  seed {seed}: baseline={b_scores[-1]:.4f} improved={i_scores[-1]:.4f}")

    bs, is_ = float(np.mean(b_scores)), float(np.mean(i_scores))
    if not np.all(np.isfinite([bs, is_])):
        print("Non-finite metrics; failing."); sys.exit(1)

    metrics = {
        "baseline_score": bs, "improved_score": is_,
        "improvement": bs - is_,  # lower is better -> baseline - improved
        "baseline_score_std": float(np.std(b_scores)), "improved_score_std": float(np.std(i_scores)),
    }
    for t in TASKS:
        metrics[f"baseline_mae_{t}"] = float(np.mean(b_mae[t]))
        metrics[f"improved_mae_{t}"] = float(np.mean(i_mae[t]))
        metrics[f"baseline_nrmse_{t}"] = float(np.mean(b_nr[t]))
        metrics[f"improved_nrmse_{t}"] = float(np.mean(i_nr[t]))

    results = {
        "question": "Can a per-task, cross-validated LightGBM model beat an uncertainty-weighted shared MLP "
                    "on two rider-count tasks of different scale in the UCI Bike Sharing data?",
        "method": "Chronological 80/20 split; baseline shared MLP with Kendall-style weights; improved per-task "
                  "LightGBM tuned by 3-fold time-series CV on training data; seeds 42-44; test set used once.",
        "dataset": {"name": "UCI Bike Sharing (hourly)", "source_url": SOURCE_URL, "n_rows": n,
                    "n_train": len(train), "n_test": len(test), "same_as_paper": False,
                    "baseline_is_papers_method": False},
        "metric_name": "mean over tasks of test RMSE / test std (NRMSE)",
        "higher_is_better": False,
        "baseline": "Shared-trunk MLPRegressor (64-64) on standardized targets with Kendall-style uncertainty "
                    "weights (1/validation residual variance) refreshed over 3 rounds, applied by scaling outputs by sqrt(w).",
        "improved": "One LightGBM regressor per task (casual, registered) with hyperparameters chosen by 3-fold "
                    "time-series CV on the training set.",
        "metrics": metrics,
        "notes": "Single chronological split; test set used only for final scoring. LightGBM is effectively "
                 "deterministic, so seed variation is small. Baseline MLP uses early stopping on an internal split. "
                 "Features exclude dteday and instant; yr carries the long-term trend and trees cannot extrapolate "
                 "beyond the training range. The paper's own model and results are not reproduced here.",
    }
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Optional chart: main metric with seed-std error bars
    import os
    os.makedirs("figures", exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["baseline MLP", "improved LGBM"], [bs, is_],
           yerr=[metrics["baseline_score_std"], metrics["improved_score_std"]], capsize=6,
           color=["grey", "steelblue"])
    ax.set_ylabel("test NRMSE (lower is better)")
    ax.set_title("Bike sharing: mean over casual/registered")
    fig.tight_layout()
    fig.savefig("figures/main_score.png", dpi=120)
    print(f"Done in {time.time() - t0:.0f}s. Wrote results.json and figures/main_score.png")


if __name__ == "__main__":
    main()
