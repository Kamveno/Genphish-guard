"""Conformal-martingale drift monitor  (the cross-project differentiator).

Label-free calibration fixes the threshold, but something has to decide WHEN to
run it. Generator shift is a loss of exchangeability between the live score
stream and the source calibration scores - the same mathematical object this
group's conformal-martingale work monitors for alert suppression. So we reuse
that construction here, in a new domain: a Simple-Jumper conformal test
martingale watches the unlabelled target scores and raises an anytime-valid
alarm the instant the stream stops looking like the calibration data. The alarm
is the trigger that fires label-free recalibration.

Guarantee: under exchangeability the smoothed conformal p-values are uniform, so
the martingale is non-negative with unit expectation and, by Ville's inequality,
P(ever exceed 1/delta) <= delta. An alarm is therefore a delta-level event, not a
tuned heuristic.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from .config import DELTA, SEED


def conformal_pvalues(ref_scores, stream_scores, rng=None):
    """Smoothed conformal p-value of each stream score against the reference.

    p_i = ( #{ref > s_i} + U * (1 + #{ref == s_i}) ) / (n_ref + 1),  U ~ Uniform(0,1)

    Under exchangeability with the reference these are Uniform(0,1).
    """
    rng = rng or np.random.default_rng(SEED)
    ref = np.sort(np.asarray(ref_scores))
    n = len(ref)
    s = np.asarray(stream_scores)
    gt = n - np.searchsorted(ref, s, side="right")      # #{ref > s_i}
    ge = n - np.searchsorted(ref, s, side="left")        # #{ref >= s_i}
    eq = ge - gt                                          # #{ref == s_i}
    u = rng.random(len(s))
    return (gt + u * (1 + eq)) / (n + 1)


def simple_jumper(pvalues, eps_values=(-1.0, 0.0, 1.0), jump=0.02):
    """Simple-Jumper mixture martingale on conformal p-values. Betting functions
    f_eps(u)=1+eps(u-0.5). Grows when p-values depart from uniform. Kept for the
    graded-score regime; the deployed monitor uses the band-rate martingale below,
    which is robust to the score saturation of tree ensembles.
    """
    K = len(eps_values)
    cap = np.full(K, 1.0 / K)
    wealth = np.empty(len(pvalues))
    for i, p in enumerate(pvalues):
        S = cap.sum()
        cap = (1 - jump) * cap + jump * (S / K)
        for k, eps in enumerate(eps_values):
            cap[k] *= (1.0 + eps * (p - 0.5))
        wealth[i] = cap.sum()
    return wealth


def rate_martingale(indicators, r0, mults=(1.5, 2.0, 3.0, 5.0, 8.0), jump=0.01):
    """Mixture betting martingale testing whether the rate of a Bernoulli stream
    has risen above the null rate r0.

    Each step bets a likelihood ratio for several alternative rates r1 = m*r0 and
    mixes them. Under the null (P=r0) every bet has unit expectation, so the
    wealth is a non-negative martingale and Ville's inequality gives
    P(sup M_t >= 1/delta) <= delta. Grows when the stream's rate exceeds r0.
    """
    r0 = float(np.clip(r0, 1e-4, 0.5))
    rates = [min(m * r0, 0.97) for m in mults]
    K = len(rates)
    cap = np.full(K, 1.0 / K)
    wealth = np.empty(len(indicators))
    for i, x in enumerate(indicators):
        S = cap.sum()
        cap = (1 - jump) * cap + jump * (S / K)
        for k, r1 in enumerate(rates):
            lr = (r1 / r0) if x else ((1 - r1) / (1 - r0))
            cap[k] *= lr
        wealth[i] = cap.sum()
    return wealth


@dataclass
class DriftResult:
    wealth: np.ndarray
    alarm_index: int | None        # index into the stream where M_t first >= 1/delta
    switch_index: int              # where the stream switched source->target
    ttd: int | None                # time-to-detection = alarm_index - switch_index
    ceiling: float

    @property
    def fired(self) -> bool:
        return self.alarm_index is not None


def monitor_stream(cal_scores, pre_scores, post_scores, band, delta=DELTA,
                   seed=SEED, margin=1.4):
    """Monitor a stream of pre-shift then post-shift delivered mail for drift.

    ``band`` = (lo, hi) is the uncertain zone between the top of legitimate mail
    and the bottom of known phishing: [quantile(cal_ham, .99), quantile(cal_phish,
    .01)]. Under source calibration the detector is confident, so delivered mail
    almost never lands there; novel phishing the detector mis-scores floods it.
    We test, online, whether the RATE at which delivered mail lands in the band
    has risen above the calibration rate r0, with a mixture betting martingale.
    Ville's inequality makes an alarm (M_t >= 1/delta) a delta-level event.

    cal_scores sets r0 (the null band rate). Returns the alarm time and the
    time-to-detection measured from the source->target switch.
    """
    lo, hi = band
    in_band = lambda a: ((np.asarray(a) >= lo) & (np.asarray(a) < hi)).astype(float)
    # Null band rate from the source reference (same deployment composition),
    # with a small margin so ordinary sampling fluctuation does not trip the
    # alarm; shift multiplies the band rate many-fold, so power is unaffected.
    r0 = float(np.mean(in_band(cal_scores))) * margin
    r0 = max(r0, 1.0 / (len(cal_scores) + 1))        # floor so r0 > 0
    stream = np.concatenate([np.asarray(pre_scores), np.asarray(post_scores)])
    ind = in_band(stream)
    wealth = rate_martingale(ind, r0)
    ceiling = 1.0 / delta
    hits = np.where(wealth >= ceiling)[0]
    alarm = int(hits[0]) if len(hits) else None
    wealth = np.minimum(wealth, ceiling * 100)       # cap for display (alarm already set)
    switch = len(pre_scores)
    ttd = (alarm - switch) if (alarm is not None and alarm >= switch) else \
          (alarm if alarm is not None else None)
    res = DriftResult(wealth, alarm, switch, ttd, ceiling)
    res.r0 = r0
    return res


class ConformalMartingale:
    """Online, one-email-at-a-time form of :func:`rate_martingale`.

    Exactly the deployed batch monitor, stepped one score at a time so a
    dashboard or an API session can feed it a live stream. Each call applies
    one likelihood-ratio bet per alternative rate and mixes them with the same
    Simple-Jumper smoothing, so the wealth sequence this produces is identical
    to ``rate_martingale`` over the same indicator stream and keeps the same
    Ville guarantee: P(sup M_t >= 1/delta) <= delta.

    The band matters. Tree-ensemble scores saturate near 0 and 1, so a fixed
    mid-range band such as [0.35, 0.65] catches almost nothing and the monitor
    is blind. Pass the band computed from calibration data — the gap between
    the top of legitimate mail and the bottom of known phishing,
    ``[quantile(cal_ham, .99), quantile(cal_phish, .01)]`` — exactly as
    :func:`monitor_stream` does. :meth:`from_calibration` builds it for you.

    Parameters
    ----------
    band_lo, band_hi
        The uncertainty band. Defaults are placeholders for well-spread
        scores; always set them from calibration data for a tree ensemble.
    warmup
        Emails buffered to estimate the null band rate r0 before betting
        starts. Ignored if ``r0`` is supplied directly.
    r0
        Null band rate, if already known from calibration. Supplying it skips
        warm-up entirely, which is what you want when the calibration scores
        are in hand.
    """

    MULTS = (1.5, 2.0, 3.0, 5.0, 8.0)
    JUMP = 0.01

    def __init__(self, delta: float = DELTA, band_lo: float = 0.35,
                 band_hi: float = 0.65, margin: float = 1.4,
                 warmup: int = 30, r0: float | None = None) -> None:
        self.delta = delta
        self.band_lo = band_lo
        self.band_hi = band_hi
        self.margin = margin
        self.warmup = warmup
        self.ceiling = 1.0 / delta

        self._buf: list[float] = []
        self._wealth: float = 1.0
        self._r0: float | None = None
        self._alarmed: bool = False
        self._cap = np.full(len(self.MULTS), 1.0 / len(self.MULTS))
        self._rates: list[float] | None = None

        if r0 is not None:
            self._set_r0(float(r0))

    # ------------------------------------------------------------ builders
    @classmethod
    def from_calibration(cls, cal_ham_scores, cal_phish_scores,
                         delta: float = DELTA, margin: float = 1.4,
                         cal_all_scores=None) -> "ConformalMartingale":
        """Build a monitor whose band and null rate come from calibration data.

        This is the construction :func:`monitor_stream` uses and the one that
        works for saturating detectors.
        """
        lo = float(np.quantile(np.asarray(cal_ham_scores), 0.99))
        hi = float(np.quantile(np.asarray(cal_phish_scores), 0.01))
        if hi <= lo:                      # degenerate overlap; fall back wide
            lo, hi = 0.05, 0.95
        ref = (np.asarray(cal_all_scores) if cal_all_scores is not None
               else np.concatenate([np.asarray(cal_ham_scores),
                                    np.asarray(cal_phish_scores)]))
        rate = float(np.mean((ref >= lo) & (ref < hi))) * margin
        r0 = max(rate, 1.0 / (len(ref) + 1))
        return cls(delta=delta, band_lo=lo, band_hi=hi, margin=margin, r0=r0)

    # ------------------------------------------------------------ internals
    def _set_r0(self, r0: float) -> None:
        self._r0 = float(np.clip(r0, 1e-4, 0.5))
        self._rates = [min(m * self._r0, 0.97) for m in self.MULTS]

    def _in_band(self, s: float) -> float:
        return 1.0 if self.band_lo <= s < self.band_hi else 0.0

    # --------------------------------------------------------------- update
    def update(self, score: float) -> tuple[float, bool]:
        """Process one email score. Returns (current_wealth, alarm_raised)."""
        if self._r0 is None:
            self._buf.append(score)
            if len(self._buf) >= self.warmup:
                rate = float(np.mean([self._in_band(s) for s in self._buf]))
                self._set_r0(max(rate * self.margin,
                                 1.0 / (len(self._buf) + 1)))
            return self._wealth, False

        ind = self._in_band(score)
        r0 = self._r0
        K = len(self._rates)

        # Simple-Jumper smoothing, then one likelihood-ratio bet per rate.
        S = self._cap.sum()
        self._cap = (1 - self.JUMP) * self._cap + self.JUMP * (S / K)
        for k, r1 in enumerate(self._rates):
            lr = (r1 / r0) if ind else ((1 - r1) / (1 - r0))
            self._cap[k] *= lr

        wealth = float(self._cap.sum())
        if wealth > self.ceiling * 100:          # cap for display
            scale = (self.ceiling * 100) / wealth
            self._cap *= scale
            wealth = self.ceiling * 100

        self._wealth = wealth
        alarm = wealth >= self.ceiling
        if alarm:
            self._alarmed = True
        return wealth, alarm

    @property
    def r0(self) -> float | None:
        return self._r0

    @property
    def band(self) -> tuple[float, float]:
        return (self.band_lo, self.band_hi)

    def reset(self) -> None:
        self._buf.clear()
        self._wealth = 1.0
        self._r0 = None
        self._rates = None
        self._alarmed = False
        self._cap = np.full(len(self.MULTS), 1.0 / len(self.MULTS))
