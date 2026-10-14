"""Reusable experiment driver — one function, every downstream study.

S6 (ablation), S7 (robustness), the multi-seed stability study and the neural
arm all ask the same question with one knob changed: *train detectors, build the
transfer matrix, then recalibrate the off-diagonal pairs without target labels —
how much of the labelled-recalibration gap comes back?*

Rather than copy that loop four times, it lives here once. Each study passes:

    kind       which detector family           (S6 / neural arm)
    features   which feature columns to use    (S6 feature ablation)
    seed       which train/test partition      (stability study)
    transform  a function applied to the TARGET phishing at scoring time,
               after training — i.e. the attacker/paraphraser acts on mail the
               detector has never seen           (S7 robustness)
    shuffle_labels  destroy the label signal     (S6 negative control)

and gets back one ExperimentResult. Because every study goes through the same
code path, differences between them are differences in the knob, not in the
plumbing — which is the whole point of an ablation.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from .config import FEATURES, LABEL, SOURCE, HUMAN, SEED
from . import detectors as D, transfer as T, calibration as CAL

# A transform maps a frame of target phishing rows -> a modified frame.
Transform = Callable[[pd.DataFrame], pd.DataFrame]

METHODS = ["stale", "dm", "em", "self", "oracle"]
LABEL_FREE = ["dm", "em", "self"]


@dataclass
class ExperimentResult:
    """Everything one configuration produces."""
    label: str
    kind: str
    seed: int
    n_features: int
    f1: pd.DataFrame                  # N x N, default threshold
    auc: pd.DataFrame                 # N x N, threshold-free
    summary_f1: dict
    summary_auc: dict
    pairs: pd.DataFrame               # one row per off-diagonal (source, target)
    mean_f1: dict                     # method -> mean F1 over pairs
    recovery: dict                    # label-free method -> mean gap recovery
    cache: dict = field(repr=False, default_factory=dict)

    def row(self) -> dict:
        """Flat record for an ablation / robustness table."""
        return {
            "config": self.label, "model": self.kind, "seed": self.seed,
            "n_features": self.n_features,
            "diag_f1": self.summary_f1["diagonal_mean"],
            "offdiag_f1": self.summary_f1["off_diagonal_mean"],
            "transfer_gap_pp": self.summary_f1["transfer_gap_pp"],
            "offdiag_auc": self.summary_auc["off_diagonal_mean"],
            **{f"f1_{m}": self.mean_f1[m] for m in METHODS},
            **{f"rec_{m}": self.recovery[m] for m in LABEL_FREE},
        }


def run_experiment(df: pd.DataFrame, generators: list[str], *,
                   label: str = "baseline",
                   kind: str = "xgboost",
                   features: list | None = None,
                   seed: int = SEED,
                   transform: Transform | None = None,
                   shuffle_labels: bool = False) -> ExperimentResult:
    """Train, transfer, recalibrate. Returns one ExperimentResult.

    shuffle_labels is the S6 negative control; it permutes TRAINING labels only
    (see detectors.train), leaving the split and every test label intact.
    """
    features = list(features) if features else list(FEATURES)

    dets, (ham_tr, ham_te, ph_tr, ph_te) = D.train_all(
        df, generators, kind, features, seed, shuffle_y=shuffle_labels)

    # The transform models mail the detector meets in deployment, so it is
    # applied to the held-out TARGET phishing only — never to training data.
    if transform is not None:
        ph_te = {g: transform(frame) for g, frame in ph_te.items()}

    cache = T.score_cache(dets, ham_te, ph_te, generators)
    F = T.f1_matrix(cache, generators)
    A = T.auc_matrix(cache, generators)

    results = []
    for s in generators:
        ss, sy = cache[(s, s)]
        for t in generators:
            if s == t:
                continue
            ts, ty = cache[(s, t)]
            results.append(CAL.calibrate_pair(s, t, ss, sy, ts, ty))

    pairs = pd.DataFrame([vars(r) | {f"rec_{m}": r.recovery(m) for m in LABEL_FREE}
                          for r in results])
    mean_f1 = {m: float(pairs[f"f1_{m}"].mean()) for m in METHODS}
    recovery = {m: float(pairs[f"rec_{m}"].mean()) for m in LABEL_FREE}

    return ExperimentResult(
        label=label, kind=kind, seed=seed, n_features=len(features),
        f1=F, auc=A, summary_f1=T.summarise(F), summary_auc=T.summarise(A),
        pairs=pairs, mean_f1=mean_f1, recovery=recovery, cache=cache,
    )


def pair_long(res: ExperimentResult) -> pd.DataFrame:
    """Per-pair records in long form — the unit of analysis for the paired
    significance tests in stability.py."""
    rows = []
    for _, r in res.pairs.iterrows():
        for m in METHODS:
            rows.append({"config": res.label, "seed": res.seed,
                         "source": r["source"], "target": r["target"],
                         "method": m, "f1": r[f"f1_{m}"]})
    return pd.DataFrame(rows)
