"""Error analysis - who slips through, and why.

Review 2 asks for failure cases, not just success metrics. Under a stale
(source) threshold on a target generator, which phishing emails are missed, and
what do they look like in feature space? And after label-free calibration, how
many are recovered? This module answers both and writes a per-feature profile of
the missed-vs-caught phishing so the failure mode is characterised, not just
counted.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from .config import FEATURES, LABEL


def missed_profile(target_df, scores, stale_thr, calib_thr):
    """Characterise phishing missed at the stale threshold and recovered at the
    calibrated threshold. target_df carries FEATURES + LABEL; scores aligned."""
    phish = target_df[LABEL].values == 1
    missed_stale = phish & (scores < stale_thr)
    caught_stale = phish & (scores >= stale_thr)
    recovered = missed_stale & (scores >= calib_thr)

    out = {
        "n_phish": int(phish.sum()),
        "n_missed_stale": int(missed_stale.sum()),
        "miss_rate_stale": float(missed_stale.sum() / max(phish.sum(), 1)),
        "n_recovered_by_calib": int(recovered.sum()),
        "recovery_rate": float(recovered.sum() / max(missed_stale.sum(), 1)),
    }

    # feature profile: mean of each feature for missed vs caught phishing
    X = target_df[FEATURES]
    prof = pd.DataFrame({
        "missed_mean": X[missed_stale].mean(),
        "caught_mean": X[caught_stale].mean(),
    })
    prof["abs_gap"] = (prof["missed_mean"] - prof["caught_mean"]).abs()
    prof = prof.sort_values("abs_gap", ascending=False)
    return out, prof


def summarise_missed(target_df, scores, stale_thr, calib_thr):
    out, prof = missed_profile(target_df, scores, stale_thr, calib_thr)
    top = prof.head(5).index.tolist()
    out["top_discriminating_features"] = top
    return out, prof
