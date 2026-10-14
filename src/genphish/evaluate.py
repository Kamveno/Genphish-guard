"""S4/S7 - evaluation metrics with uncertainty.

Every reported estimate carries a bootstrap confidence interval, so comparisons
are read with uncertainty rather than as point estimates. Expected calibration
error is included because the zero-trust layer consumes confidence, not just a
label.
"""
from __future__ import annotations
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score
from .config import N_BOOTSTRAP, SEED, TARGET_RECALL


def precision_at_recall(scores, labels, recall=TARGET_RECALL):
    pos = np.sort(scores[labels == 1])
    if len(pos) == 0:
        return 0.0
    k = int(np.floor((1 - recall) * len(pos)))
    thr = pos[min(max(k, 0), len(pos) - 1)]
    pred = (scores >= thr).astype(int)
    return float(precision_score(labels, pred, zero_division=0))


def fpr_at_threshold(scores, labels, thr):
    legit = labels == 0
    if legit.sum() == 0:
        return 0.0
    return float(np.mean(scores[legit] >= thr))


def expected_calibration_error(scores, labels, n_bins=10):
    """ECE: |confidence - accuracy| averaged over confidence bins."""
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (scores >= lo) & (scores < hi)
        if m.sum() == 0:
            continue
        conf = scores[m].mean()
        acc = labels[m].mean()          # fraction actually phishing in this bin
        ece += (m.sum() / len(scores)) * abs(conf - acc)
    return float(ece)


def bootstrap_ci(metric_fn, scores, labels, n=N_BOOTSTRAP, alpha=0.05, seed=SEED):
    """Percentile bootstrap CI for any metric_fn(scores, labels)."""
    rng = np.random.default_rng(seed)
    n_obs = len(labels)
    vals = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, n_obs, n_obs)
        vals[i] = metric_fn(scores[idx], labels[idx])
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return float(metric_fn(scores, labels)), float(lo), float(hi)


def f1_at(scores, labels, thr):
    return f1_score(labels, (scores >= thr).astype(int), zero_division=0)


def mean_offdiag_auc_ci(cache, generators, n=1000, alpha=0.05, seed=SEED):
    """Bootstrap CI for the MEAN of the off-diagonal per-pair AUCs.

    The reported point estimate is the mean of the N(N-1) cross-generator cells,
    so the interval must be built around *that* statistic. Pooling every pair's
    scores into one vector and bootstrapping the pooled AUC estimates something
    else — a different, usually smaller, number — and produces an interval that
    need not contain the figure in the abstract. Each replicate therefore
    resamples within each pair and re-averages the per-pair AUCs.
    """
    rng = np.random.default_rng(seed)
    pairs = [(a, b) for a in generators for b in generators if a != b]
    data = [cache[p] for p in pairs]
    point = float(np.mean([roc_auc_score(y, s) for s, y in data]))
    reps = np.empty(n)
    for i in range(n):
        vals = []
        for s, y in data:
            idx = rng.integers(0, len(y), len(y))
            ys = y[idx]
            # a resample with one class only has no defined AUC; redraw once
            if ys.min() == ys.max():
                idx = rng.integers(0, len(y), len(y))
                ys = y[idx]
                if ys.min() == ys.max():
                    continue
            vals.append(roc_auc_score(ys, s[idx]))
        reps[i] = np.mean(vals) if vals else np.nan
    reps = reps[~np.isnan(reps)]
    lo, hi = np.quantile(reps, [alpha / 2, 1 - alpha / 2])
    return point, float(lo), float(hi)
