"""S4 - transfer matrix.

For every (train generator, test generator) pair we score the target generator's
held-out phishing together with held-out legitimate mail, using the detector
trained on the source generator. From the cached scores we build:

  * F1 matrix at the default decision threshold (0.5) - the operational metric
  * AUC matrix (threshold-free) - the diagnostic

The diagonal is in-distribution; the off-diagonal is generator shift. The story
of the project is the contrast: AUC stays high off-diagonal (the signal survives)
while F1 collapses (the threshold is in the wrong place).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
from .config import LABEL

DEFAULT_THR = 0.5


def score_cache(detectors, ham_te, ph_te, generators):
    """cache[(src,tgt)] = (scores, labels) for target phishing + shared ham."""
    cache = {}
    for src in generators:
        det = detectors[src]
        for tgt in generators:
            eval_df = pd.concat([ph_te[tgt], ham_te], ignore_index=True)
            s = det.score(eval_df)
            y = eval_df[LABEL].values
            cache[(src, tgt)] = (s, y)
    return cache


def f1_matrix(cache, generators, thr=DEFAULT_THR):
    M = pd.DataFrame(index=generators, columns=generators, dtype=float)
    for (src, tgt), (s, y) in cache.items():
        M.loc[src, tgt] = f1_score(y, (s >= thr).astype(int))
    M.index.name = "train_model"
    return M


def auc_matrix(cache, generators):
    M = pd.DataFrame(index=generators, columns=generators, dtype=float)
    for (src, tgt), (s, y) in cache.items():
        M.loc[src, tgt] = roc_auc_score(y, s)
    M.index.name = "train_model"
    return M


def summarise(M: pd.DataFrame) -> dict:
    g = list(M.index)
    diag = np.mean([M.loc[x, x] for x in g])
    off = np.mean([M.loc[a, b] for a in g for b in g if a != b])
    return {"diagonal_mean": round(float(diag), 4),
            "off_diagonal_mean": round(float(off), 4),
            "transfer_gap_pp": round(float((diag - off) * 100), 2)}
