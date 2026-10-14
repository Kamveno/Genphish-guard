"""S1 - corpus assembly.

load_corpus() returns (DataFrame, is_real).

* If data/corpus_features.csv (the Zenodo release) is present, it is loaded and
  normalised to the project schema -> is_real = True. Panel-grade numbers.

* Otherwise a SYNTHETIC corpus is generated and cached. It is engineered so the
  known generator-shift phenomenon *emerges* (high AUC everywhere, but a fixed
  threshold collapses across generators), calibrated in magnitude to the real
  reference (Gutierrez et al., Frontiers 2026: ~28 pp transfer gap, worst cross
  cell ~0.46). The numbers the pipeline then reports are REAL computations on
  this synthetic data -> is_real = False. Every output is stamped SYNTHETIC so
  it is never mistaken for a measurement on the real corpus.

The generator-shift is not written into the scores by hand. Each generator is
given an 'aggressiveness' that scales how far its phishing sits from legitimate
mail on the signal features. A detector trained on an aggressive generator sets
its threshold high; applied to a milder generator it misses much of the phishing
(recall collapses) while still ranking it above ham (AUC preserved). That is the
mechanism the real paper observes, reproduced from first principles.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from .config import (DATA, REAL_CORPUS, SYNTH_CORPUS, FEATURES, LABEL, SOURCE,
                     HUMAN, GENERATORS, SEED)

# Each generator departs from legitimate mail along a SHARED phishiness
# direction (keeps cross-model AUC high) PLUS its own GENERATOR-SPECIFIC
# direction (a subset of signal features it leans on). A detector trained on one
# generator partly keys on that generator's specific axes; a different generator
# does not excite them, so its phishing scores lower and a fixed threshold misses
# it -> F1 collapses in BOTH directions while ranking (AUC) is largely preserved.
# This is the mechanism behind the real transfer gap, reproduced from scratch.
_AGGR = {"deepseek3.2": 1.10, "gpt-4.1": 1.00, "llama-3.3-70b": 0.92}

# signal features carry phishiness
_SIGNAL = ["urgency_term_count", "url_count", "exclamation_count",
           "uppercase_ratio", "digit_ratio", "punct_ratio", "question_count"]
# generator-specific signal subsets (the axis each generator leans on)
_SPECIFIC = {
    "deepseek3.2":   ["urgency_term_count", "uppercase_ratio"],
    "gpt-4.1":       ["url_count", "punct_ratio"],
    "llama-3.3-70b": ["exclamation_count", "digit_ratio"],
}
# style features carry generator identity (per-generator mean offset)
_STYLE = ["avg_sentence_len", "avg_word_len", "flesch_reading_ease",
          "type_token_ratio", "function_word_ratio"]


def _synthesise(n_ham: int = 5000, n_phish: int = 1660, seed: int = SEED,
                sd_mult: float = 0.66, push_scale: float = 0.63,
                extra_scale: float = 0.72, hard_frac: float = 0.28) -> pd.DataFrame:
    """Generate the synthetic corpus.

    sd_mult / push_scale / extra_scale / hard_frac are tuned (see tools) so the
    measured transfer matrix matches the real reference in shape and magnitude.
    hard_frac is the share of phishing drawn as 'mild' (weaker signal), which
    creates the genuine class overlap that makes F1 graded rather than binary.
    """
    rng = np.random.default_rng(seed)

    # plausible legitimate-mail centre and spread for all 17 features
    ham_mu = {
        "char_count": 850, "word_count": 150, "sentence_count": 9,
        "avg_word_len": 4.6, "avg_sentence_len": 16, "type_token_ratio": 0.62,
        "uppercase_ratio": 0.03, "digit_ratio": 0.02, "punct_ratio": 0.09,
        "exclamation_count": 0.4, "question_count": 0.8, "url_count": 0.9,
        "stopword_ratio": 0.42, "function_word_ratio": 0.48,
        "personal_pronoun_ratio": 0.06, "urgency_term_count": 0.3,
        "flesch_reading_ease": 62,
    }
    ham_sd = {k: abs(v) * sd_mult + 0.04 for k, v in ham_mu.items()}
    # shared push: every signal feature moves a moderate amount for all generators
    _push = {"urgency_term_count": 2.1, "url_count": 1.6, "exclamation_count": 1.7,
             "uppercase_ratio": 0.030, "digit_ratio": 0.026, "punct_ratio": 0.028,
             "question_count": 0.8}
    # extra push applied ONLY on a generator's specific axes (its fingerprint)
    _extra = {"urgency_term_count": 3.2, "url_count": 2.6, "exclamation_count": 2.8,
              "uppercase_ratio": 0.055, "digit_ratio": 0.048, "punct_ratio": 0.050,
              "question_count": 1.2}
    push = {k: v * push_scale for k, v in _push.items()}
    extra = {k: v * extra_scale for k, v in _extra.items()}
    # per-generator style offsets (in SDs) that give each generator a fingerprint
    style_off = {
        "deepseek3.2":   {"avg_sentence_len": -0.3, "flesch_reading_ease": -0.4, "type_token_ratio": 0.2},
        "gpt-4.1":       {"avg_sentence_len": 0.1, "flesch_reading_ease": 0.2, "function_word_ratio": 0.1},
        "llama-3.3-70b": {"avg_sentence_len": 0.6, "flesch_reading_ease": 0.6, "avg_word_len": 0.3},
    }

    def draw(n, mu_over=None):
        cols = {}
        for k in FEATURES:
            mu = ham_mu[k] + (mu_over.get(k, 0.0) if mu_over else 0.0)
            cols[k] = rng.normal(mu, ham_sd[k], n)
        return pd.DataFrame(cols)

    rows = []

    # legitimate mail (shared across the experiment; label 0, source=human)
    ham = draw(n_ham)
    ham[LABEL] = 0
    ham[SOURCE] = HUMAN
    rows.append(ham)

    # phishing per generator (label 1). A fraction hard_frac are 'mild' phishing
    # (signal scaled down), which creates real overlap with legitimate mail so
    # scores are graded and F1 is not an all-or-nothing step function.
    for g in GENERATORS:
        a = _AGGR[g]
        n_mild = int(hard_frac * n_phish)
        n_strong = n_phish - n_mild
        parts = []
        for strength, n in ((1.0, n_strong), (0.45, n_mild)):
            mu_over = {k: strength * a * push[k] for k in _SIGNAL}
            for k in _SPECIFIC[g]:
                mu_over[k] = mu_over.get(k, 0.0) + strength * extra[k]
            for k, v in style_off[g].items():
                mu_over[k] = mu_over.get(k, 0.0) + v * ham_sd[k]
            mu_over["personal_pronoun_ratio"] = -0.02
            mu_over["stopword_ratio"] = -0.05
            parts.append(draw(n, mu_over))
        ph = pd.concat(parts, ignore_index=True)
        ph[LABEL] = 1
        ph[SOURCE] = g
        rows.append(ph)

    df = pd.concat(rows, ignore_index=True)

    # enforce valid ranges
    nonneg = ["char_count", "word_count", "sentence_count", "exclamation_count",
              "question_count", "url_count", "urgency_term_count"]
    for k in nonneg:
        df[k] = df[k].clip(lower=0).round()
    ratios = ["type_token_ratio", "uppercase_ratio", "digit_ratio", "punct_ratio",
              "stopword_ratio", "function_word_ratio", "personal_pronoun_ratio"]
    for k in ratios:
        df[k] = df[k].clip(0, 1)
    df["flesch_reading_ease"] = df["flesch_reading_ease"].clip(-20, 120)
    df["avg_word_len"] = df["avg_word_len"].clip(1, 20)
    df["avg_sentence_len"] = df["avg_sentence_len"].clip(1, 80)

    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


def _normalise_real(df: pd.DataFrame) -> pd.DataFrame:
    """Map the Zenodo CSV onto the project schema (label + source + 17 features)."""
    cols = {c.lower().strip(): c for c in df.columns}
    out = pd.DataFrame()
    # label
    for cand in ("label", "is_phishing", "phishing", "y"):
        if cand in cols:
            out[LABEL] = (df[cols[cand]].astype(float) > 0.5).astype(int)
            break
    # source / generator
    for cand in ("source", "generator", "model", "llm", "origin"):
        if cand in cols:
            out[SOURCE] = df[cols[cand]].astype(str)
            break
    # features: take by name where present, else assume the 17 numeric columns
    present = [f for f in FEATURES if f in cols]
    if len(present) == len(FEATURES):
        for f in FEATURES:
            out[f] = df[cols[f]].astype(float)
    else:
        num = df.select_dtypes("number")
        num = num[[c for c in num.columns if c not in (out.columns)]]
        if num.shape[1] < len(FEATURES):
            raise ValueError(f"real corpus has {num.shape[1]} numeric cols, need 17")
        for f, c in zip(FEATURES, num.columns[:len(FEATURES)]):
            out[f] = df[c].astype(float)
    if SOURCE not in out:
        raise ValueError("real corpus: could not find a source/generator column")
    return out


def load_corpus(force_synth: bool = False):
    """Return (df, is_real). df has columns FEATURES + [LABEL, SOURCE]."""
    if REAL_CORPUS.exists() and not force_synth:
        raw = pd.read_csv(REAL_CORPUS)
        return _normalise_real(raw), True
    if SYNTH_CORPUS.exists() and not force_synth:
        return pd.read_csv(SYNTH_CORPUS), False
    df = _synthesise()
    df.to_csv(SYNTH_CORPUS, index=False)
    return df, False


if __name__ == "__main__":
    df, real = load_corpus()
    print(f"{'REAL' if real else 'SYNTHETIC'} corpus: {len(df)} rows")
    print(df.groupby([SOURCE, LABEL]).size())
