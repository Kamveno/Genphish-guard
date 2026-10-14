"""S7 — robustness: does the correction survive a *moving* attacker?

Generator shift (S4/S5) is passive: a new model is released and its mail happens
to look different. This module adds the active case — the attacker rewrites the
mail *to evade the detector* — which is the half a security panel will press on.

All transforms act in the 17-dimensional stylometric space, because that is the
representation the detector consumes and the corpus ships. A transform is a
hypothesis about what a rewrite does to those coordinates, stated explicitly as
a multiplier or a shift toward the legitimate-mail centroid, so a reviewer can
read the assumption and argue with it. They are *not* claimed to be the output
of an actual paraphrasing model; when raw email bodies are available (see
neural.py) the same study reruns with real rewrites through features.extract.

Crucially the transform is applied to **held-out target phishing only, after
training**. The detector never sees a transformed email during fitting, and the
label-free calibration never sees a target label. That is the deployment
situation: the attacker changes, and you must re-place the threshold blind.

The question each row answers:

    the attacker moved the mail — did the DETECTOR break (AUC falls, the signal
    is gone) or did only the THRESHOLD break (AUC holds, F1 falls)?

If only the threshold breaks, label-free recalibration is still the right
response, and `rec_dm` says how much it recovers. If AUC collapses too, no
threshold can save it and the honest answer is that the attack works.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from .config import FEATURES, LABEL, SOURCE, HUMAN, SEED
from . import calibration as CAL, detectors as D, transfer as T
from .experiment import run_experiment, Transform

# Features an attacker can cheaply suppress without losing the lure's function.
# url_count is deliberately NOT fully suppressible: a phishing email that drops
# its payload link stops being phishing. That constraint is the defender's
# leverage and is modelled, not assumed away.
_SUPPRESSIBLE = ["urgency_term_count", "exclamation_count", "uppercase_ratio",
                 "digit_ratio", "punct_ratio", "question_count"]

_COUNTS = ["char_count", "word_count", "sentence_count", "exclamation_count",
           "question_count", "url_count", "urgency_term_count"]
_RATIOS = ["type_token_ratio", "uppercase_ratio", "digit_ratio", "punct_ratio",
           "stopword_ratio", "function_word_ratio", "personal_pronoun_ratio"]


def _clip_valid(X: pd.DataFrame, quantize_counts: bool = True) -> pd.DataFrame:
    """Keep transformed mail inside the physically possible feature space.

    Without this an 'attack' could win by producing emails that cannot exist.
    Counts are rounded back to integers because they *are* counts — you cannot
    send 2.4 exclamation marks — and because leaving them continuous silently
    confounds every scenario: see `make_dequantize`, which isolates that effect
    deliberately instead.
    """
    out = X.copy()
    for c in _COUNTS:
        out[c] = out[c].clip(lower=0)
        if quantize_counts:
            out[c] = out[c].round()
    for c in _RATIOS:
        out[c] = out[c].clip(0, 1)
    out["flesch_reading_ease"] = out["flesch_reading_ease"].clip(-20, 120)
    out["avg_word_len"] = out["avg_word_len"].clip(1, 20)
    out["avg_sentence_len"] = out["avg_sentence_len"].clip(1, 80)
    return out


def ham_profile(df: pd.DataFrame) -> pd.Series:
    """Centroid of legitimate mail — the point an evading attacker walks toward."""
    return df[df[SOURCE] == HUMAN][FEATURES].mean()


# ------------------------------------------------------------------ transforms
def make_paraphrase() -> Transform:
    """An LLM asked to 'rewrite this more naturally'. Surface markers soften;
    the structure and the payload stay."""
    mult = {"exclamation_count": 0.40, "uppercase_ratio": 0.60,
            "urgency_term_count": 0.70, "avg_sentence_len": 1.15,
            "punct_ratio": 0.85}
    add = {"type_token_ratio": 0.05, "flesch_reading_ease": -5.0}

    def f(X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        for c, m in mult.items():
            out[c] = out[c] * m
        for c, a in add.items():
            out[c] = out[c] + a
        return _clip_valid(out)
    return f


def make_register_shift() -> Transform:
    """The same lure written as corporate/legal correspondence: longer words and
    sentences, no exclamation, impersonal."""
    mult = {"avg_word_len": 1.10, "avg_sentence_len": 1.30,
            "exclamation_count": 0.10, "personal_pronoun_ratio": 0.70,
            "urgency_term_count": 0.80}
    add = {"flesch_reading_ease": -12.0, "function_word_ratio": 0.03}

    def f(X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        for c, m in mult.items():
            out[c] = out[c] * m
        for c, a in add.items():
            out[c] = out[c] + a
        return _clip_valid(out)
    return f


def make_length_inflation() -> Transform:
    """Padding with benign boilerplate (a long signature, a privacy footer) to
    dilute the per-character signal — a cheap, very common evasion."""
    def f(X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        k = 1.6
        for c in ("char_count", "word_count", "sentence_count"):
            out[c] = out[c] * k
        for c in ("uppercase_ratio", "digit_ratio", "punct_ratio"):
            out[c] = out[c] / k                      # same absolute count, more text
        out["stopword_ratio"] = out["stopword_ratio"] + 0.04
        out["type_token_ratio"] = out["type_token_ratio"] * 0.92
        return _clip_valid(out)
    return f


def make_evasion(ham_mu: pd.Series, eps: float,
                 cols: list[str] | None = None) -> Transform:
    """Targeted evasion: move the suppressible markers a fraction `eps` of the
    way to the legitimate-mail centroid. eps=0 is no attack, eps=1 erases those
    markers entirely. Everything else — including url_count — is untouched."""
    cols = cols or _SUPPRESSIBLE

    def f(X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        for c in cols:
            out[c] = out[c] + eps * (ham_mu[c] - out[c])
        return _clip_valid(out)
    return f


def make_noise(sd_frac: float = 0.10, seed: int = SEED) -> Transform:
    """Isotropic perturbation at `sd_frac` of each feature's spread — the
    reference point every targeted attack is judged against.

    It is not a no-op and is not presented as one: 17 independent nudges of
    sd_frac each displace a point by roughly sqrt(17)·sd_frac standard
    deviations in total. Its job is to price *undirected* movement, so that a
    targeted transform can be compared against an equal budget of aimless
    movement. A targeted attack that does no more damage than this is not an
    attack worth reporting.
    """
    def f(X: pd.DataFrame) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        out = X.copy()
        for c in FEATURES:
            s = float(out[c].std()) or 0.0
            out[c] = out[c] + rng.normal(0, sd_frac * s, len(out))
        return _clip_valid(out)
    return f


def make_dequantize(jitter: float = 1e-6, seed: int = SEED) -> Transform:
    """Diagnostic probe, not an attack: add a *negligible* sub-unit jitter to
    the integer count features and skip the re-rounding.

    Nothing about the mail changes — a count of 3 becomes 3.000001 — so any
    metric that moves is measuring the detector's dependence on those features
    arriving as exact integers, rather than anything about the email. Tree
    ensembles can place split points inside the gaps between integers and then
    rely on no sample ever landing there; this probe reveals that, and it is
    reported separately so it cannot be mistaken for, or confounded with, the
    paraphrase and evasion rows (which re-quantize).
    """
    def f(X: pd.DataFrame) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        out = X.copy()
        for c in _COUNTS:
            out[c] = out[c] + rng.normal(0, jitter, len(out))
        return _clip_valid(out, quantize_counts=False)
    return f


# ------------------------------------------------------------------ study
def attack_diagnostics(res) -> dict:
    """Split the damage into 'signal lost' vs 'threshold misplaced'."""
    return {
        "offdiag_auc": res.summary_auc["off_diagonal_mean"],
        "diag_f1": res.summary_f1["diagonal_mean"],
        "offdiag_f1": res.summary_f1["off_diagonal_mean"],
        "f1_stale": res.mean_f1["stale"],
        "f1_dm": res.mean_f1["dm"],
        "f1_oracle": res.mean_f1["oracle"],
        "rec_dm": res.recovery["dm"],
    }


def robustness_study(df, generators, kind="xgboost", seed=SEED,
                     eps_grid=(0.25, 0.50, 0.75, 1.00)) -> pd.DataFrame:
    """Every transform, one row each, plus the clean baseline."""
    mu = ham_profile(df)
    scenarios: dict[str, Transform | None] = {
        "clean (no transform)": None,
        "de-quantization probe (1e-6)": make_dequantize(seed=seed),
        "undirected noise 0.05σ": make_noise(0.05, seed=seed),
        "undirected noise 0.25σ": make_noise(0.25, seed=seed),
        "paraphrase": make_paraphrase(),
        "register shift": make_register_shift(),
        "length inflation": make_length_inflation(),
    }
    for e in eps_grid:
        scenarios[f"targeted evasion ε={e:.2f}"] = make_evasion(mu, e)

    rows = []
    for name, tf in scenarios.items():
        res = run_experiment(df, generators, label=name, kind=kind,
                             seed=seed, transform=tf)
        rows.append({"scenario": name, **attack_diagnostics(res)})

    out = pd.DataFrame(rows)
    clean = out.iloc[0]
    # Attribution. Two independent things can break, and the right response
    # differs: if RANKING survives (AUC holds) only the threshold moved, and
    # label-free recalibration is the fix. If ranking is gone, no threshold
    # helps and the honest answer is that the attack works.
    out["auc_drop_pp"] = (clean["offdiag_auc"] - out["offdiag_auc"]) * 100
    out["f1_drop_vs_clean_pp"] = (clean["f1_stale"] - out["f1_stale"]) * 100
    out["recoverable_by_recalib_pp"] = (out["f1_dm"] - out["f1_stale"]) * 100

    # Two axes, judged independently, because collapsing them to one verdict
    # misreads two of the four cases. In particular a transform can cost real
    # ranking power while F1 at the stale threshold *rises* — the blur happens
    # to drag scores toward a threshold that was already in the wrong place.
    # Calling that "no effect" (or "signal degraded") would both be wrong.
    hurt = out["f1_drop_vs_clean_pp"] > 3.0          # operational F1 actually fell
    signal_gone = out["auc_drop_pp"] > 5.0           # ranking power actually fell
    out["ranking_damaged"] = signal_gone
    out["operational_damage"] = hurt
    out["verdict"] = np.select(
        [signal_gone & hurt,
         ~signal_gone & hurt,
         signal_gone & ~hurt],
        ["signal degraded — recalibration cannot fix this",
         "threshold displaced — recalibration applies",
         "ranking degraded; F1 flat only because the stale threshold was "
         "already misplaced"],
        default="no material effect")
    return out


def evasion_curve(df, generators, kinds=("xgboost", "logreg"), seed=SEED,
                  eps_grid=np.linspace(0, 1, 9)) -> pd.DataFrame:
    """Finer sweep of the targeted attack, for the figure.

    Run for more than one detector family on purpose. A gradual attack does not
    have to produce a gradual failure: a tree ensemble is piecewise constant, so
    once suppression carries a feature past the splits the model relies on, its
    accuracy falls off a step and further suppression changes nothing until the
    next step. A linear model degrades smoothly over the same sweep. Which
    shape a detector has is an operational property worth knowing — a step means
    there is no early warning in the metric before the failure.
    """
    mu = ham_profile(df)
    rows = []
    for kind in kinds:
        for e in eps_grid:
            tf = None if e == 0 else make_evasion(mu, float(e))
            res = run_experiment(df, generators, label=f"ε={e:.2f}", kind=kind,
                                 seed=seed, transform=tf)
            rows.append({"model": kind, "eps": float(e), **attack_diagnostics(res)})
    return pd.DataFrame(rows)
