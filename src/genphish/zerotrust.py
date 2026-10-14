"""S8 - confidence-graded zero-trust response layer.

Generator shift does not make the detector wrong, it makes it uncertain. A binary
allow/block decision throws that uncertainty away. The zero-trust principle of
graduated verification keeps a third option: deliver-but-defused (strip links,
hold attachments, external-sender banner, step-up auth). This module tests
whether that third tier is worth having.

Cost model (all ASSUMPTIONS, swept, never asserted as measured):
  phishing blocked        -> 0
  phishing defused        -> RESIDUAL * C_miss   (links stripped, harm reduced)
  phishing delivered      -> C_miss
  legitimate blocked      -> C_false_block
  legitimate defused      -> C_defused
  legitimate delivered    -> 0

We compare the best binary policy with the best graded policy, and report the
RANGE of cost ratios C_miss / C_false_block over which the graded policy wins.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from .config import (COST_MISS, COST_FALSE_BLOCK, COST_DEFUSED,
                     COST_RATIO_SWEEP, TARGET_RECALL)

RESIDUAL = 0.30     # fraction of miss-cost still incurred by a defused phishing email

# Default tier thresholds (used by dashboard and SOAR module)
DEFAULT_T_LO = 0.35   # below → DELIVER
DEFAULT_T_HI = 0.65   # above → BLOCK; between → DEFUSE


def classify(score: float, threshold: float = 0.5,
             t_lo: float = DEFAULT_T_LO, t_hi: float = DEFAULT_T_HI) -> str:
    """Assign a single score to a zero-trust tier.

    Returns
    -------
    "BLOCK"   — confident phishing, do not deliver
    "DEFUSE"  — uncertain; deliver stripped / sandboxed copy
    "DELIVER" — below detection threshold
    """
    if score >= t_hi:
        return "BLOCK"
    elif score >= t_lo:
        return "DEFUSE"
    return "DELIVER"


def classify_batch(scores: "np.ndarray",
                   t_lo: float = DEFAULT_T_LO,
                   t_hi: float = DEFAULT_T_HI) -> list[str]:
    """Vectorised tier assignment for an array of scores."""
    import numpy as np
    tiers = np.where(scores >= t_hi, "BLOCK",
                     np.where(scores >= t_lo, "DEFUSE", "DELIVER"))
    return tiers.tolist()


def _grid(scores, n=60):
    qs = np.linspace(0.01, 0.99, n)
    return np.unique(np.quantile(scores, qs))


def binary_cost(scores, labels, thr, c_miss, c_fb):
    phish = labels == 1
    legit = ~phish
    block = scores >= thr
    miss = np.sum(phish & ~block) * c_miss
    fb = np.sum(legit & block) * c_fb
    return (miss + fb) / len(labels)


def graded_cost(scores, labels, t_lo, t_hi, c_miss, c_fb, c_def):
    phish = labels == 1
    legit = ~phish
    block = scores >= t_hi
    defuse = (scores >= t_lo) & (scores < t_hi)
    deliver = scores < t_lo
    cost = (np.sum(phish & deliver) * c_miss
            + np.sum(phish & defuse) * RESIDUAL * c_miss
            + np.sum(legit & block) * c_fb
            + np.sum(legit & defuse) * c_def)
    return cost / len(labels)


def best_binary(scores, labels, c_miss, c_fb):
    g = _grid(scores)
    costs = [binary_cost(scores, labels, t, c_miss, c_fb) for t in g]
    i = int(np.argmin(costs))
    return g[i], costs[i]


def best_graded(scores, labels, c_miss, c_fb, c_def):
    g = _grid(scores)
    best = (None, None, np.inf)
    for i, lo in enumerate(g):
        for hi in g[i:]:
            c = graded_cost(scores, labels, lo, hi, c_miss, c_fb, c_def)
            if c < best[2]:
                best = (lo, hi, c)
    return best


@dataclass
class ZTPoint:
    ratio: float
    binary_cost: float
    graded_cost: float
    @property
    def graded_wins(self): return self.graded_cost < self.binary_cost - 1e-12


def sweep_cost_ratio(scores, labels, n=25,
                     ratio_range=COST_RATIO_SWEEP, c_fb=COST_FALSE_BLOCK,
                     c_def=COST_DEFUSED):
    """Sweep C_miss/C_false_block; at each ratio compare the best binary and
    best graded policy. Returns the list of points and the winning ratio range."""
    ratios = np.geomspace(ratio_range[0], ratio_range[1], n)
    pts = []
    for r in ratios:
        c_miss = r * c_fb
        _, bc = best_binary(scores, labels, c_miss, c_fb)
        _, _, gc = best_graded(scores, labels, c_miss, c_fb, c_def)
        pts.append(ZTPoint(float(r), float(bc), float(gc)))
    wins = [p.ratio for p in pts if p.graded_wins]
    win_range = (min(wins), max(wins)) if wins else None
    return pts, win_range


def legit_preserved_at_equal_containment(scores, labels, recall=TARGET_RECALL,
                                         c_def=COST_DEFUSED):
    """Hold phishing containment equal (same recall of block+defuse). How much
    MORE legitimate mail does the graded policy avoid blocking vs binary?"""
    phish = labels == 1
    legit = ~phish
    pos = np.sort(scores[phish])
    # binary: single threshold catching `recall` of phishing
    k = int(np.floor((1 - recall) * len(pos)))
    t_bin = pos[min(max(k, 0), len(pos) - 1)]
    legit_blocked_binary = np.mean(scores[legit] >= t_bin)
    # graded: block tier catches the confident phishing; defuse tier catches the
    # rest up to the same recall, so legit in the defuse band is NOT blocked.
    t_hi = np.quantile(scores[phish], 0.60)       # confident-phishing block gate
    # choose t_lo so that block+defuse recall == recall
    t_lo = pos[min(max(k, 0), len(pos) - 1)]
    legit_blocked_graded = np.mean(scores[legit] >= t_hi)
    legit_defused_graded = np.mean((scores[legit] >= t_lo) & (scores[legit] < t_hi))
    return {
        "legit_blocked_binary": float(legit_blocked_binary),
        "legit_blocked_graded": float(legit_blocked_graded),
        "legit_defused_graded": float(legit_defused_graded),
        "legit_rescued_from_block": float(legit_blocked_binary - legit_blocked_graded),
    }
