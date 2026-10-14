"""S5 - label-free calibration under generator shift  (THE CONTRIBUTION).

A detector trained on a SOURCE generator deploys a threshold chosen on source
data. On a TARGET generator the discriminative signal survives (AUC stays high)
but that threshold is misplaced, so F1 collapses. These methods move the
threshold back using ONLY unlabelled target scores - no target labels - which is
the operational reality: phishing from a new model arrives before anyone has
labelled it.

Three families, the three nominated as open directions by the closest prior work
(Gutierrez et al., Frontiers 2026):

  1. Distribution matching  - re-place the threshold at the same quantile of the
     target score distribution that it occupied in the source distribution.
  2. Prior-shift / EM       - Saerens-Latinne-Decaestecker: estimate the target
     phishing prior from unlabelled scores, then correct the decision boundary.
  3. Self-training          - pseudo-label the confident tails, re-estimate the
     threshold, iterate, guarded by a prior-plausibility bound.

Each method returns a target threshold. The LABELLED oracle (best-F1 threshold
using target labels) is the upper bound; gap_recovery measures the share of the
labelled-recalibration improvement a label-free method recovers with no labels.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from sklearn.metrics import f1_score
from .config import TARGET_RECALL

EPS = 1e-9


# ----------------------------------------------------------------- threshold helpers
def threshold_at_recall(scores, labels, recall=TARGET_RECALL):
    """Lowest threshold whose recall on the positive class is >= recall."""
    pos = np.sort(scores[labels == 1])
    if len(pos) == 0:
        return 0.5
    k = int(np.floor((1 - recall) * len(pos)))
    k = min(max(k, 0), len(pos) - 1)
    return float(pos[k])


def best_f1_threshold(scores, labels):
    """Threshold maximising F1 (uses labels -> oracle / upper bound only)."""
    order = np.unique(scores)
    if len(order) > 512:
        order = np.quantile(scores, np.linspace(0, 1, 512))
    best_t, best_f = 0.5, -1.0
    for t in order:
        f = f1_score(labels, (scores >= t).astype(int), zero_division=0)
        if f > best_f:
            best_f, best_t = f, float(t)
    return best_t


# ----------------------------------------------------------------- method 1: DM
def dm_threshold(source_scores, source_thr, target_scores):
    """Distribution matching: put the threshold at the same quantile of the
    target score distribution that source_thr occupied in the source."""
    q = float(np.mean(source_scores <= source_thr))
    return float(np.quantile(target_scores, q))


# ----------------------------------------------------------------- method 2: EM
def em_prior(target_scores, source_prior, n_iter=200, tol=1e-7):
    """Saerens-Latinne-Decaestecker EM: estimate target phishing prior from the
    source-trained posteriors on unlabelled target data.

    target_scores are P_source(phish | x) under source_prior. Returns pi_hat.
    """
    s = np.clip(target_scores, EPS, 1 - EPS)
    ps = float(np.clip(source_prior, EPS, 1 - EPS))
    pi = ps
    for _ in range(n_iter):
        a = (pi / ps) * s
        b = ((1 - pi) / (1 - ps)) * (1 - s)
        post = a / (a + b)
        new_pi = float(np.mean(post))
        if abs(new_pi - pi) < tol:
            break
        pi = new_pi
    return float(np.clip(pi, EPS, 1 - EPS))


def em_threshold(target_scores, source_prior, pi_hat=None):
    """Raw-score threshold equivalent to classifying adjusted posterior >= 0.5
    under the EM-estimated target prior."""
    if pi_hat is None:
        pi_hat = em_prior(target_scores, source_prior)
    ps = float(np.clip(source_prior, EPS, 1 - EPS))
    # solve (pi/ps)*s = ((1-pi)/(1-ps))*(1-s)  for s*
    r0 = (1 - pi_hat) / (1 - ps)
    r1 = pi_hat / ps
    thr = r0 / (r1 + r0)
    return float(np.clip(thr, EPS, 1 - EPS)), pi_hat


# ----------------------------------------------------------------- method 3: self-training
def selftrain_threshold(target_scores, source_thr, hi=0.85, lo=0.15,
                        n_iter=10, prior_lo=0.05, prior_hi=0.70):
    """Pseudo-label the confident tails, set the threshold at the midpoint of the
    two pseudo-class score means, iterate. Guarded by a prior-plausibility bound:
    if the implied phishing prevalence leaves [prior_lo, prior_hi] the step is
    rejected and the previous threshold kept (reported as a guard trigger)."""
    thr = float(source_thr)
    guard_hits = 0
    for _ in range(n_iter):
        hi_t = np.quantile(target_scores, hi)
        lo_t = np.quantile(target_scores, lo)
        pos = target_scores[target_scores >= hi_t]
        neg = target_scores[target_scores <= lo_t]
        if len(pos) == 0 or len(neg) == 0:
            break
        new_thr = float((pos.mean() + neg.mean()) / 2)
        prevalence = float(np.mean(target_scores >= new_thr))
        if not (prior_lo <= prevalence <= prior_hi):
            guard_hits += 1
            break                      # reject degenerate step, keep last thr
        if abs(new_thr - thr) < 1e-6:
            thr = new_thr
            break
        thr = new_thr
    return thr, guard_hits


# ----------------------------------------------------------------- orchestration
@dataclass
class CalibResult:
    source: str
    target: str
    f1_stale: float
    f1_oracle: float
    f1_dm: float
    f1_em: float
    f1_self: float
    pi_true: float
    pi_em: float
    guard_hits: int

    def recovery(self, which: str) -> float:
        base = self.f1_oracle - self.f1_stale
        if base <= 1e-6:
            return 1.0
        gained = getattr(self, f"f1_{which}") - self.f1_stale
        return float(np.clip(gained / base, -0.5, 1.5))


def _f1(scores, labels, thr):
    return f1_score(labels, (scores >= thr).astype(int), zero_division=0)


def calibrate_pair(source, target, source_scores, source_labels,
                   target_scores, target_labels):
    """Run all methods for one (source, target) pair. Labels on target are used
    ONLY for the oracle and for scoring - never by the label-free methods."""
    src_thr = threshold_at_recall(source_scores, source_labels)
    src_prior = float(np.mean(source_labels == 1))
    pi_true = float(np.mean(target_labels == 1))

    thr_dm = dm_threshold(source_scores, src_thr, target_scores)
    thr_em, pi_em = em_threshold(target_scores, src_prior)
    thr_self, guard = selftrain_threshold(target_scores, src_thr)
    thr_oracle = best_f1_threshold(target_scores, target_labels)

    return CalibResult(
        source=source, target=target,
        f1_stale=_f1(target_scores, target_labels, src_thr),
        f1_oracle=_f1(target_scores, target_labels, thr_oracle),
        f1_dm=_f1(target_scores, target_labels, thr_dm),
        f1_em=_f1(target_scores, target_labels, thr_em),
        f1_self=_f1(target_scores, target_labels, thr_self),
        pi_true=pi_true, pi_em=pi_em, guard_hits=guard,
    )
