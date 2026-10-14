"""Central configuration: paths, feature schema, generators, seeds, cost model.

Everything the pipeline needs that is NOT data lives here, so a reviewer can
read one file and know exactly what the experiment assumes.
"""
from __future__ import annotations
from pathlib import Path

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
REAL_CORPUS = DATA / "corpus_features.csv"          # drop the Zenodo CSV here
SYNTH_CORPUS = DATA / "synthetic_corpus_features.csv"  # generated if real is absent
for d in (DATA, RESULTS, FIGURES):
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- reproducibility
SEED = 20261011
N_BOOTSTRAP = 2000          # resamples for every reported CI (10k in final run)

# ---------------------------------------------------------------- generators
# One "generator" == the source that produced a phishing email. HUMAN is the
# pre-LLM generator class. The LLM generators are the three in the Zenodo corpus.
HUMAN = "human"
GENERATORS = ["deepseek3.2", "gpt-4.1", "llama-3.3-70b"]   # order matches Zenodo
ALL_SOURCES = [HUMAN] + GENERATORS

# ---------------------------------------------------------------- 17 features
# Stylometric / linguistic features. Hand-engineered features are used because
# prior work (Gutierrez et al., Frontiers 2026) reports they transfer across
# generators better than learned representations. features.py extracts these
# from raw email text; the Zenodo CSV already carries the same 17 columns.
FEATURES = [
    "char_count", "word_count", "sentence_count",
    "avg_word_len", "avg_sentence_len", "type_token_ratio",
    "uppercase_ratio", "digit_ratio", "punct_ratio",
    "exclamation_count", "question_count", "url_count",
    "stopword_ratio", "function_word_ratio", "personal_pronoun_ratio",
    "urgency_term_count", "flesch_reading_ease",
]
assert len(FEATURES) == 17

LABEL = "label"          # 1 = phishing, 0 = legitimate
SOURCE = "source"        # one of ALL_SOURCES (provenance)

# ---------------------------------------------------------------- operating point
TARGET_RECALL = 0.90     # operating recall for precision@recall and threshold choice
DELTA = 0.01             # Ville alarm level for the drift monitor (ceiling 1/delta)

# ---------------------------------------------------------------- zero-trust cost model
# Costs are ASSUMPTIONS, not measurements. The zero-trust layer therefore sweeps
# the ratio C_miss / C_falseblock rather than resting on single values. These are
# only the centre of that sweep.
COST_MISS = 25.0         # a delivered phishing email that should have been caught
COST_FALSE_BLOCK = 1.0   # a legitimate email wrongly quarantined
COST_DEFUSED = 0.15      # a legitimate email delivered with friction (banner/link-strip)
COST_RATIO_SWEEP = (1.0, 100.0)   # range of C_miss / C_falseblock explored
