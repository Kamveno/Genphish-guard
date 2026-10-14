"""Reproducibility and invariant tests. `pytest` from the repo root.

These are the checks a reviewer re-runs to confirm the pipeline behaves as the
report claims: the phenomenon exists, label-free calibration helps, the drift
monitor fires on shift and not on no-shift, and runs are deterministic.
"""
import numpy as np
import pandas as pd
import pytest

from genphish.config import GENERATORS, FEATURES, LABEL, SOURCE
from genphish import corpus as C, detectors as D, transfer as T, calibration as CAL
from genphish import drift_monitor as DM, features as FE


@pytest.fixture(scope="module")
def fitted():
    df, _ = C.load_corpus()
    dets, sp = D.train_all(df, GENERATORS, "xgboost")
    ham_tr, ham_te, ph_tr, ph_te = sp
    cache = T.score_cache(dets, ham_te, ph_te, GENERATORS)
    return df, dets, sp, cache


def test_feature_extractor_shape():
    f = FE.extract("URGENT! Verify your account now: http://x.co !!")
    assert set(f) == set(FEATURES) and len(f) == 17
    assert f["url_count"] >= 1 and f["exclamation_count"] >= 2


def test_corpus_schema():
    df, _ = C.load_corpus()
    assert set(FEATURES).issubset(df.columns)
    assert {LABEL, SOURCE}.issubset(df.columns)
    assert df[LABEL].isin([0, 1]).all()


def test_phenomenon_exists(fitted):
    """AUC survives generator shift; F1 at the default threshold collapses."""
    _, _, _, cache = fitted
    F = T.f1_matrix(cache, GENERATORS)
    A = T.auc_matrix(cache, GENERATORS)
    sF, sA = T.summarise(F), T.summarise(A)
    assert sF["diagonal_mean"] > sF["off_diagonal_mean"] + 0.10   # real collapse
    assert sF["transfer_gap_pp"] > 15
    assert sA["off_diagonal_mean"] > 0.90                          # signal survives


def test_label_free_calibration_helps(fitted):
    _, _, _, cache = fitted
    recs = []
    for s in GENERATORS:
        ss, sy = cache[(s, s)]
        for t in GENERATORS:
            if s == t:
                continue
            ts, ty = cache[(s, t)]
            recs.append(CAL.calibrate_pair(s, t, ss, sy, ts, ty))
    dm = np.mean([r.f1_dm for r in recs])
    stale = np.mean([r.f1_stale for r in recs])
    oracle = np.mean([r.f1_oracle for r in recs])
    assert dm > stale + 0.05                      # distribution matching recovers
    assert stale <= dm <= oracle + 1e-6           # bounded by stale and oracle
    for r in recs:                                 # recovery is a sane fraction
        assert -0.5 <= r.recovery("dm") <= 1.5


def test_drift_monitor_fires_on_shift_not_on_control(fitted):
    """Stochastic monitor: over several seeds the control must never fire
    (Ville bounds false alarms at delta) and real shift must reliably fire."""
    df, dets, sp, cache = fitted
    ham_tr, ham_te, ph_tr, ph_te = sp
    src = "deepseek3.2"; det = dets[src]
    fired_shift = fired_ctrl = 0
    SEEDS = range(5)
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        h = ham_te.sample(frac=1, random_state=seed)
        ref_h, live_h = h.iloc[:len(h)//2], h.iloc[len(h)//2:]

        def stream(pdf, n, prev):
            npos = int(prev * n); parts = []
            if npos:
                parts.append(pdf.sample(npos, replace=len(pdf) < npos, random_state=int(rng.integers(1e9))))
            parts.append(live_h.sample(n - npos, replace=len(live_h) < n - npos, random_state=int(rng.integers(1e9))))
            return rng.permutation(det.score(pd.concat(parts)))

        band = (float(np.quantile(det.score(ref_h), 0.99)),
                float(np.quantile(det.score(ph_te[src]), 0.01)))
        cal = stream(ph_te[src], 3000, 0.25)
        pre = stream(ph_te[src], 1500, 0.25)
        fired_shift += DM.monitor_stream(cal, pre, stream(ph_te["llama-3.3-70b"], 1500, 0.25), band).fired
        fired_ctrl += DM.monitor_stream(cal, pre, stream(ph_te[src], 1500, 0.25), band).fired
    assert fired_ctrl == 0                      # no false alarms under no shift
    assert fired_shift >= len(list(SEEDS)) - 1  # detects shift in (almost) every run


def test_determinism():
    """Two independent builds give the same transfer summary."""
    def summ():
        df, _ = C.load_corpus()
        dets, sp = D.train_all(df, GENERATORS, "xgboost")
        cache = T.score_cache(dets, sp[1], sp[3], GENERATORS)
        return T.summarise(T.f1_matrix(cache, GENERATORS))
    assert summ() == summ()
