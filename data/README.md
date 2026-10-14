# data/

## Real corpus (for panel-grade numbers)

The experiment uses `corpus_features.csv` from the Zenodo release:

> Gutierrez, R., Villegas-Ch, W., & Govea, J. (2026). *Cross-model evaluation of
> phishing detectors against LLM-generated emails: dataset, code and results.*
> Frontiers in Big Data. Zenodo DOI **10.5281/zenodo.20250116**.

9,986 phishing emails (5,000 human-written + 4,986 from GPT-4.1, DeepSeek 3.2,
LLaMA 3.3 70B), 17 stylometric/linguistic features. Released under **CC BY 4.0** —
keep the attribution if you redistribute it.

**To run on the real data:**
1. Download `cross-model-phishing.zip` from the Zenodo record.
2. Extract and copy the feature table to `data/corpus_features.csv`
   (columns: the 17 features + a label column + a source/generator column;
   `corpus.py` auto-maps common column names).
3. `python -m genphish.run_all` — the pipeline detects the real CSV and every
   output is stamped **REAL** instead of SYNTHETIC.

## Synthetic corpus (default, runs anywhere)

If `corpus_features.csv` is absent, `corpus.py` generates
`synthetic_corpus_features.csv` deterministically (seed in `config.py`). Its
generator-shift is engineered to reproduce the *shape* of the real phenomenon
(high AUC, ~28 pp transfer gap, worst cross cell ~0.5). Numbers computed on it
are REAL computations on SYNTHETIC data and are stamped SYNTHETIC everywhere.
Use it to develop and demo; use the Zenodo CSV for reportable numbers.
