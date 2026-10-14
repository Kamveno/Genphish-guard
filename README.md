# genphish-guard

**Generator-agnostic detection of LLM-generated phishing — recalibrated without labels from the new generator, with a confidence-graded zero-trust response.**

M.Tech Cybersecurity (Industry-Integrated, 2025–27) · KIIT University · Industry partner: LTIMindtree

---

## The one claim

A phishing detector trained on emails from one LLM still **ranks** phishing from a
*different, unseen* LLM almost perfectly (AUC stays > 0.97). What breaks is the
**decision threshold**: at the old threshold, F1 collapses by ~28 points. The
detector isn't blind — its dial is in the wrong place.

> **This project moves the dial back using only *unlabelled* mail from the new generator** —
> because in reality, phishing from a newly released model arrives *before* anyone has labelled it.

A conformal-martingale monitor decides *when* to recalibrate (it fires an
anytime-valid alarm the moment the live stream stops matching calibration), and
a three-tier zero-trust layer consumes the recalibrated confidence instead of a
hard allow/block.

---

## Run the working system

```bash
pip install -r requirements.txt -r requirements-api.txt
python scripts/build_model.py        # train once (~60 s)
./scripts/serve.sh                   # serve
```

Open **http://localhost:8000** — a six-tab operator console (Detect, Transfer,
Calibration, Drift Monitor, SOAR, API) backed by a 15-endpoint REST API with
OpenAPI docs at `/docs`. Paste an email, get a confidence score, a zero-trust
tier, per-feature attribution and a SOAR incident with its playbook.

**[`STEPS.md`](STEPS.md)** has everything operational: running it, every
endpoint with `curl` examples, hosting it on Render / Hugging Face / Docker so
a reviewer can open a link, the demo running order, troubleshooting, and the
one-step switch from SYNTHETIC to REAL data.

---

## Quickstart (research pipeline)

```bash
# 1. install (uv recommended; pip works too)
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt        # or: pip install -r requirements.txt

# 2. the core claim — regenerates every headline table and figure (~15 s)
python -m genphish.run_all                # add --logreg for the LR baseline

# 3. the extended studies — ablations, adversary, neural arm (~2 min)
python -m genphish.run_extended
python -m genphish.run_extended --stability     # + 10-partition significance (~5 min)

# 4. run the tests
pytest -q
```

Core pipeline runs in ~15 s on a laptop CPU. No GPU, no cluster, no paid compute.

### Real vs synthetic data

The pipeline runs **out of the box on a synthetic corpus** engineered to
reproduce the *shape* of the real phenomenon (high AUC, ~28 pp transfer gap).
Every synthetic output is stamped **SYNTHETIC** so it is never mistaken for a
measurement. For **panel-grade numbers**, drop the real corpus in and re-run:

1. Download the Zenodo release (DOI `10.5281/zenodo.20250116`, CC BY 4.0).
2. Copy its feature table to `data/corpus_features.csv`.
3. `python -m genphish.run_all` → outputs now stamped **REAL**.

See `data/README.md`.

---

## What you get (one command → `results/`)

| Output | What it shows |
|---|---|
| `fig1_transfer_f1.png` + `transfer_f1_default.csv` | N×N F1 — the collapse off the diagonal |
| `fig2_transfer_auc.png` + `transfer_auc.csv` | N×N AUC — the signal **survives** the shift |
| `fig3_calibration_recovery.png` + `calibration_pairs.csv` | stale → 3 label-free methods → oracle |
| `fig5_drift_monitor.png` + `drift_monitor.csv` | martingale wealth crossing the Ville ceiling after the generator switch |
| `fig6_zerotrust.png` + `zerotrust_sweep.csv` | binary vs graded expected-cost curves |
| `fig7_error_profile.png` + `error_profile.csv` | which phishing is missed, and why |
| `fig8_score_shift.png` | the score distributions that move |
| `summary.json` | every headline number + bootstrap CIs + DATA_MODE |

### `python -m genphish.run_extended [--stability]` → more of `results/`

| Output | What it shows |
|---|---|
| `fig9_ablation_features.png` + `ablation_features.csv` | which features carry the gap, and which the correction needs |
| `fig13_detector_family.png` + `ablation_detector_family.csv` | four detector families — collapse severity vs correction |
| `ablation_shuffled_control.csv` | negative control: permuted training labels → AUC at chance |
| `ablation_provenance.csv` | the pooled upper bound, and how far label-free gets toward it |
| `ablation_calibration_sensitivity.csv` | every method's hyperparameter, swept |
| `fig10_robustness.png` + `robustness_scenarios.csv` | paraphrase, register shift, padding, evasion, de-quantization probe |
| `fig11_evasion_curve.png` + `robustness_evasion_curve.csv` | the evasion cliff, per detector family |
| `neural_arm_summary.csv` | learned-representation arm vs engineered features |
| `fig12_stability.png` + `stability_*.csv` | 10 partitions, mean ± sd ± CI, paired Holm-corrected tests |
| `extended_summary.json` | every extended number + verdicts + DATA_MODE |

---

## Pipeline (stages map to the Review-1 methodology)

```
S1 corpus.py        assemble corpus (real CSV or synthetic), per-email provenance
S2 features.py      17 stylometric/linguistic features (works on raw emails too)
S3 detectors.py     per-generator + pooled detectors (XGBoost primary, LogReg baseline)
S4 transfer.py      N×N F1 (default threshold) and threshold-free AUC
S5 calibration.py   label-free threshold correction  ← THE CONTRIBUTION
                      • distribution matching
                      • prior-shift EM (Saerens-Latinne-Decaestecker)
                      • guarded self-training
                      + labelled oracle (upper bound) + gap-recovery metric
   drift_monitor.py conformal / betting test martingale, Ville alarm  ← the trigger
S6 ablation.py      feature-group / shuffled-label control / detector-family /
                      provenance / hyperparameter-sensitivity studies
S7 robustness.py    paraphrase, register shift, padding, targeted evasion sweep,
                      de-quantization probe — the ACTIVE adversary
S8 zerotrust.py     three-tier policy, expected-cost sweep, legit-mail-rescued
   experiment.py    one reusable train→transfer→recalibrate driver; every study
                      above is that driver with a single knob changed
   stability.py     multi-partition sweep, paired Wilcoxon, Holm correction
   neural.py        learned-representation arm + text-encoder readiness check
   evaluate.py      F1 / AUC / precision@recall / FPR / ECE / bootstrap CIs
   error_analysis.py missed-vs-caught phishing profile
   plots.py         all figures
   run_all.py       one-command core pipeline
   run_extended.py  one-command S6 / S7 / neural / stability studies
```

---

## Headline results *(synthetic corpus — illustrative; re-run on Zenodo for real numbers)*

Over **10 independent partitions × 6 cross-generator pairs** (60 paired observations):

- **Transfer gap 29.02 ± 1.02 pp** (95% CI [28.43, 29.64]) at the default
  threshold, while off-diagonal **AUC holds at 0.9750 ± 0.0032**. The signal
  survives; the threshold dies.
- **Label-free calibration recovers 90.3% ± 2.3%** of the labelled-recalibration
  gap (stale F1 0.625 → DM 0.839, oracle 0.859). Distribution matching beats
  stale, EM and self-training at **p_Holm ≈ 6×10⁻¹¹, Cohen's d_z ≈ 2.1**, and
  wins on **60/60** observations. It is also **significantly below the oracle
  (−2.0 pp)** — the method buys back 90% of what labels would buy, not 100%.
- **Pooling is the upper bound** (F1 0.955) but needs target labels; label-free
  recalibration closes **67%** of the distance to it using none.
- **Four features carry the detector.** Dropping urgency/exclamation/question/URL
  counts takes off-diagonal AUC from 0.977 to **0.600**; those four alone reach
  0.992. The other 13 are nearly free.
- **The collapse is capacity-dependent, the correction is not.** Transfer gap
  ranges 7.5–29.7 pp across XGBoost / RF / LogReg / MLP, but label-free
  recalibration helps in **all four** (recovery 0.69–0.91).
- **Evasion is a cliff, not a slope.** Suppressing up to ~60% of the persuasion
  markers costs nothing (AUC 0.977 → 0.966); past that the detector collapses
  (0.892 → 0.390). No gradual warning in the accuracy metric before failure —
  which is the argument for the distribution-watching drift monitor.
- **A 1×10⁻⁶ jitter on the integer count features costs the XGBoost detector
  9.27 pp AUC**, independent of magnitude, and re-rounding restores it exactly.
  Logistic regression is unaffected. A representation brittleness no
  generator-shift experiment would surface.
- Drift monitor: fires after the generator switch (TTD a few hundred emails),
  **never fires under no shift** (0/40 seeds); Ville false-alarm bound δ = 0.01.
- Zero-trust: graded policy has lower expected cost than binary across cost ratio
  ∈ [1, 83]; rescues ~8% of legitimate mail from blocking at equal containment.

**Why EM underperforms is a real finding, not a bug:** EM assumes the source
classifier is well-calibrated, which generator shift violates. It estimates the
target phishing prior at **π̂ = 0.135 against a true 0.25**, and that estimate is
*invariant to the iteration budget* — it converges confidently to the wrong
answer rather than failing to converge. See `docs/REVIEW2.md`.

**The honest limit:** under paraphrase and register shift, *ranking itself*
degrades (−7.2 and −10.2 pp AUC). Label-free threshold correction answers
threshold displacement; it is not a defence against an adversary who degrades
the signal.

---

## Reproducibility

Fixed seed (`config.py`), pinned `requirements.txt`, deterministic synthetic
corpus, two-command regeneration, `pytest` invariant suite (20 checks, including
the shuffled-label negative control). Another person clones, installs, runs
`python -m genphish.run_all`, and gets identical tables.

Headline numbers are reported with a 10-partition spread and paired,
Holm-corrected significance tests rather than as single-partition point
estimates — see `stability.py` and `fig12`.

## Not yet done (→ Final Review)

- **Run on the real corpus** for reportable numbers (one file drop). The
  de-quantization probe should be re-run there first: real counts are integers
  too, so that brittleness may be real rather than synthetic.
- **Text-encoder arm.** Blocked on raw email bodies, not on code — the corpus
  ships feature tables. `neural.text_encoder_readiness()` checks this in code;
  `features.extract_frame` already ingests raw text, so the arm unblocks the day
  message bodies are in hand. The MLP arm covers the representation question as
  far as the available data allows.
- **Mitigation for the count-feature brittleness** (smoothing or monotone
  constraints), then re-test.
- Widen the matrix with 1–2 extra generators minted locally (Ollama).
- Cross-lingual subset of E-PhishLLM.

## References

- Gutierrez, Villegas-Ch & Govea (2026), *Cross-model evaluation of phishing detectors against LLM-generated emails*, Frontiers in Big Data. Dataset: Zenodo `10.5281/zenodo.20250116` (CC BY 4.0).
- Saerens, Latinne & Decaestecker (2002), *Adjusting the outputs of a classifier to new a priori probabilities*.
- Vovk, Nouretdinov & Gammerman (2003), *Testing exchangeability on-line*; Ho (2005), *A martingale framework for concept change detection*.

*Synthetic data is a development/demonstration stand-in only; it is not a dataset contribution and is not presented as real measurements.*
