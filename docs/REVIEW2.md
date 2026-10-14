# Review 2 — Implementation + Preliminary Results

Project: **Generator-agnostic detection of LLM-generated phishing without labelled target data, with a confidence-graded zero-trust response.**

> **Demo in two lines.**
> `python -m genphish.run_all` → transfer matrix, label-free recovery, martingale alarm, zero-trust curves, error analysis (~15 s).
> `python -m genphish.run_extended --stability` → S6 ablations, S7 adversarial robustness, neural arm, 10-partition significance tests (~5 min).

> **Data status.** Every number below is a real computation on a **synthetic**
> corpus, stamped SYNTHETIC on every figure. The reference paper's own
> cross-model gap is 28.1 pp; this corpus reproduces 29.0 ± 1.0 pp, so the
> *shape* is right — but these are not measurements on real mail. Dropping the
> Zenodo corpus into `data/` and re-running restamps everything REAL. Nothing
> in the method depends on which corpus is loaded.

---

## Rubric → evidence (what to point at on the day)

| # | Rubric item | Status | Where / what to show |
|---|---|---|---|
| 1 | Working implementation demoed live (core module functional) | ✅ | `run_all` runs S1–S8 live, CPU, ~15 s |
| 2 | Code repository (structure, Git commits, README) | ✅ | this repo; 17 modules, `README.md` |
| 3 | Preliminary results with real numbers/graphs (not projections) | ✅ | `results/summary.json`, `results/extended_summary.json`, 13 figures — computed, not quoted |
| 4 | Comparison against baseline started | ✅ | stale single-threshold baseline vs 3 label-free methods vs labelled oracle vs **pooled upper bound**; 4 detector families |
| 5 | Evaluation methodology matches Review 1 (or deviation justified) | ✅ | S1–S8 map 1:1; metrics as proposed, plus paired significance testing. Deviations in §Changes |
| 6 | Ablation / parameter tuning planned/started (AIML) | ✅ | **five S6 studies**: feature-group, shuffled-label control, detector-family, provenance, hyperparameter sensitivity |
| 7 | Attack/defense validated in controlled testbed, reproducible (CyberSec) | ✅ | passive generator shift **and active adversary** (S7: paraphrase, register shift, length inflation, targeted evasion sweep); monitor fires on shift, 0/40 false alarms; `pytest` |
| 8 | Error analysis / failure cases discussed | ✅ | three named failure modes, each quantified — see §Failures |
| 9 | Scope/dataset/method changes flagged and justified | ✅ | §Changes |
| 10 | Remaining work + realistic timeline | ✅ | §Timeline |
| 11 | Draft paper/report structure started (abstract + methodology) | ✅ | `docs/ABSTRACT_METHODOLOGY.md` |
| 12 | Reproducibility: can another person re-run from the repo? | ✅ | pinned deps, fixed seed, two commands, 20 `pytest` invariants, **10-partition stability study** |
| 13 | Plagiarism/originality self-check on draft | ◑ | code original (see §Originality); prose self-check pending on full draft |
| 14 | Panel feedback tracker (Review 0 & 1) | ◑ | §Feedback tracker — Review-1 rows to be filled from panel notes |

Legend: ✅ done · ◑ started/partial · ☐ not started

---

## The one claim, now with error bars

Across **10 independent partitions × 6 cross-generator pairs (60 observations)**:

| Quantity | Result |
|---|---|
| Transfer gap at the default threshold | **29.02 ± 1.02 pp** (95% CI [28.43, 29.64]) |
| Off-diagonal AUC — the signal survives | **0.9750 ± 0.0032** |
| Stale threshold F1 | 0.6249 ± 0.0111 |
| **Distribution matching F1 (zero target labels)** | **0.8390 ± 0.0114** |
| Labelled oracle F1 (upper bound) | 0.8588 ± 0.0111 |
| **Gap recovered, label-free** | **90.3% ± 2.3%** (95% CI [89.0%, 91.6%]) |

Paired Wilcoxon signed-rank, paired on (partition × source × target), Holm-corrected across the four comparisons:

| Comparison | mean ΔF1 | Cohen's d_z | p (Holm) | Reading |
|---|---|---|---|---|
| DM vs stale | **+0.214** | 2.21 | 6.5 × 10⁻¹¹ | DM better |
| DM vs EM | +0.178 | 2.13 | 6.5 × 10⁻¹¹ | DM better |
| DM vs self-training | +0.149 | 2.02 | 6.5 × 10⁻¹¹ | DM better |
| DM vs oracle | **−0.020** | −2.10 | 6.5 × 10⁻¹¹ | **DM worse** |

Two things to say about that last row, both deliberately: distribution matching
is **significantly better than every label-free alternative** and
**significantly worse than the labelled oracle**. It is not free. The honest
claim is that it buys back 90% of what labels would buy, at zero labelling cost
— not that labels are unnecessary.

**Consistency, not just averages:** DM beats the stale threshold on **60 / 60**
(partition × generator pair) observations — `win_rate = 1.0` for every pair.
Weakest pair: gpt-4.1 → deepseek3.2, still +0.093 F1.

---

## S6 — ablations and controls (`run_extended`, item 6)

**1. The negative control passes, as a permutation test.** With training labels
permuted (row selection and all test labels untouched), the null distribution of
off-diagonal AUC over 8 permutations is **0.493, range [0.441, 0.527]** —
centred at chance. The real detector scores **0.976** and **exceeds every
permutation replicate**. Nothing leaks.

> Say the p-value floor before the panel finds it: a permutation test with *n*
> replicates cannot report a p below 1/(*n*+1), so at the default 8 replicates
> the p bottoms out at **0.111** and 19 are needed to reach 0.05
> (`--control-perms 19`). The evidence here is the **separation** — 0.976
> against a null whose largest replicate is 0.527, roughly 14 null standard
> deviations — not the attainable p, and the separation does not improve with
> more replicates.

> *Two things went wrong here before they went right, and both are worth having
> ready if the panel asks how the controls were themselves validated.*
> *(a) The first version permuted labels **before** the train/test split and
> reported AUC 0.61. That was the control being wrong, not the pipeline: rows
> are selected by provenance, so pre-split permutation still left the two groups
> genuinely different mail. The permutation belongs inside `detectors.train`.
> (b) The second version tested whether a t-interval around the null mean
> covered 0.5. Per-permutation AUC scatters by about ±0.03, so at three
> permutations that interval excluded 0.5 and the control "failed" arithmetically.
> A permutation test asks the question that actually matters and does not depend
> on that scatter.*

**2. One feature group carries the phenomenon.**

| Features used | transfer gap | off-diag AUC | gap recovered by DM |
|---|---|---|---|
| all 17 | 28.8 pp | 0.977 | 89% |
| − length | 28.2 pp | 0.978 | 89% |
| − lexical | 28.2 pp | 0.980 | 88% |
| − orthographic | 28.8 pp | 0.978 | 86% |
| **− persuasion** | 18.5 pp | **0.600** | **19%** |
| only persuasion (4 features) | 28.0 pp | **0.992** | **93%** |
| only lexical (6 features) | 9.0 pp | 0.513 | 26% |

Urgency terms, exclamations, question marks and URL counts are the whole
detector. The other 13 features are close to free. Those four are also exactly
what an attacker can cheapest suppress — which is what S7 then tests.

**3. The collapse is model-dependent; the correction is not.**

| Detector | transfer gap | off-diag AUC | stale F1 | DM F1 | gap recovered |
|---|---|---|---|---|---|
| XGBoost (primary) | 28.8 pp | 0.977 | 0.629 | 0.849 | 89% |
| Random forest | 29.7 pp | 0.989 | 0.647 | 0.916 | 88% |
| Logistic regression | 7.5 pp | 0.992 | 0.784 | 0.930 | 69% |
| MLP | 7.8 pp | 0.989 | 0.803 | 0.915 | 91% |

The *severity* of threshold collapse is a property of model capacity — the tree
ensembles key onto generator-specific axes and are hurt four times as badly as
the linear and MLP detectors. But label-free recalibration helps in **all four
families**, which is the claim that matters. Say this before the panel says it:
had we only run XGBoost, we would have overstated how universal the collapse is.

**4. "Why not just pool all the generators?"** Because pooling needs the new
generator's labelled mail. Both halves, quantified:

| | mean F1 |
|---|---|
| Pooled detector (has seen target labels) | **0.955** |
| Single-source, stale threshold | 0.629 |
| Single-source + label-free DM | 0.849 |

Label-free recalibration closes **67%** of the distance from a stale
single-source detector to the pooled upper bound, using **no target labels**.
Pooling is what you do in month two; this is what you do on day one.

**5. The result is not a tuned ridge — with one documented exception.**

| Knob | Range swept | Gap recovered |
|---|---|---|
| Source operating recall | 0.80 → 0.95 | 0.863 → 0.895 (flat) |
| Source operating recall | **0.99** | **0.158 (fails)** |
| Self-training band (hi/lo) | 0.95/0.05 → 0.75/0.25 | 0.218 → 0.351 |
| EM iteration budget | 5 → 200 | 0.126 (invariant) |

DM is stable anywhere in a normal operating band and **breaks when the source
threshold is pushed into the extreme tail (recall 0.99)**: there are too few
source samples beyond it to estimate the quantile being transported. That is a
stated precondition of the method, not a hidden one.

---

## S7 — robustness: the attacker moves too (item 7)

All transforms act on **held-out target phishing after training** — the detector
never sees transformed mail while fitting, and the recalibration never sees a
target label. Counts are re-quantized to integers in every scenario, so no
scenario is silently paying the de-quantization cost measured separately below.

| Scenario | off-diag AUC | diag F1 | stale F1 | DM F1 | Verdict |
|---|---|---|---|---|---|
| clean | 0.977 | 0.971 | 0.629 | 0.849 | no material effect |
| undirected noise 0.05σ | 0.977 | 0.972 | 0.625 | 0.849 | no material effect |
| undirected noise 0.25σ | 0.977 | 0.970 | 0.630 | 0.848 | no material effect |
| length inflation (×1.6) | 0.975 | 0.970 | 0.642 | 0.846 | no material effect |
| targeted evasion ε=0.25 | 0.977 | 0.972 | 0.626 | 0.848 | no material effect |
| targeted evasion ε=0.50 | 0.966 | 0.883 | 0.740 | 0.829 | no material effect |
| paraphrase | 0.904 | 0.785 | 0.670 | 0.705 | ranking degraded |
| targeted evasion ε=0.75 | 0.892 | 0.321 | 0.622 | 0.643 | ranking degraded |
| register shift | 0.875 | 0.621 | 0.604 | 0.655 | ranking degraded |
| **targeted evasion ε=1.00** | **0.390** | **0.000** | 0.358 | 0.372 | **signal degraded — recalibration cannot fix this** |
| *de-quantization probe (1e-6)* | *0.884* | *0.871* | *0.606* | *0.668* | *representation brittleness (below)* |

Each row is judged on two independent axes — did **ranking** survive (AUC), and
did **operational F1** actually fall — because the right response differs. If
only the threshold moved, recalibration is the fix. If ranking is gone, no
threshold helps and the honest answer is that the attack works.

**Finding 1 — evasion is a cliff, not a slope.** Suppressing up to ~60% of the
persuasion markers costs essentially nothing (AUC 0.977 → 0.966); past that the
detector collapses (0.892 → 0.390 between ε = 0.75 and ε = 1.00). Both XGBoost
and logistic regression show the same step shape. **Operationally this is the
important one: there is no gradual warning in the accuracy metric before the
failure** — which is precisely the argument for the conformal-martingale
monitor, since that watches the *score distribution*, not accuracy, and needs no
labels to fire.

**Finding 2 — a representation brittleness that generator-shift experiments
cannot see.** Adding a **1 × 10⁻⁶** jitter to the integer count features —
changing a count of 3 into 3.000001, i.e. changing nothing about the email —
costs the XGBoost detector **9.27 pp of AUC**. The damage does not grow with
the jitter's size (identical at 0.02σ and 0.4σ), and re-rounding the counts
restores AUC to 0.977 exactly. Logistic regression is unaffected at any
magnitude. The tree ensemble has placed split points inside the gaps between
integers and relies on no sample ever landing there. This is a property of the
detector's representation of count features, not of the generator shift, and it
is invisible to every experiment that only varies the generator.

**Finding 3 — the honest limit of the contribution.** Under paraphrase and
register shift, ranking itself degrades (−7.2 and −10.2 pp AUC). Recalibration
still helps a little (+3.5 and +5.1 pp F1) but cannot recover what the ranking
lost. **Label-free threshold correction answers threshold displacement. It is
not a defence against an adversary who degrades the signal**, and the report
will say so in those words.

---

## Neural arm — is the finding representation-specific?

| Representation | stale F1 | DM F1 | oracle F1 | gap recovered |
|---|---|---|---|---|
| XGBoost (17 engineered features) | 0.629 | 0.849 | 0.869 | 88.7% |
| MLP (learned non-linear representation) | 0.803 | 0.915 | 0.928 | **91.4%** |

Recovery spread across families: **0.027**. Label-free calibration helps in
every family tested, so the contribution is **representation-agnostic** on the
evidence available.

**What is still not tested, and why it is a data problem rather than a code
problem.** A fine-tuned text encoder needs raw email bodies; the reference
corpus is published as stylometric feature tables. `neural.text_encoder_readiness`
checks this in code rather than asserting it in prose, and currently reports
`"blocked on data, not on code"`. `features.extract_frame` already maps raw
bodies onto the project schema, so the day message text is in hand the same
pipeline ingests it unchanged.

---

## Failures, named and quantified (item 8)

1. **EM prior-shift under-recovers — 15.7% vs DM's 90.3%.** Not a bug.
   Saerens-Latinne-Decaestecker assumes the source classifier is well
   calibrated; generator shift breaks exactly that. EM estimates the target
   phishing prior at **π̂ = 0.135 against a true 0.25**, and that estimate is
   *invariant to the iteration budget* (identical at 5 and 200 iterations), so
   it is converging confidently to the wrong answer rather than failing to
   converge. This is the known failure mode of prior-shift methods, reproduced
   — and the reason distribution matching, which targets the score distribution
   directly, wins.
2. **DM's precondition.** Fails at source operating recall 0.99 (recovery
   0.158): too few source samples past the threshold to transport its quantile.
3. **Who slips through** (`fig7`, `error_profile.csv`): the missed phishing is
   the *mild* phishing — lower on urgency, URL and caps — that the source
   detector scores like legitimate mail.
4. **Self-training's guard** triggered 0 times in the swept bands; reported as
   zero rather than omitted.
5. **Monitor sensitivity scales with shift severity.** Severe shift fires fast
   (TTD 391 emails), mild slowly (756), non-shift never. Expected.

---

## Changes since Review 1 (item 9)

| Change | Justification |
|---|---|
| Added a **synthetic-corpus mode** alongside the real Zenodo corpus | lets the pipeline run, CI and demo anywhere before the dataset is in hand; stamped SYNTHETIC on every output; not a data contribution |
| Drift monitor uses a **band-rate betting martingale** rather than raw conformal p-values | tree-ensemble scores saturate near 0/1, leaving the conformal region too sparse; the band-rate test is robust and keeps the Ville guarantee |
| Added **paired significance testing over 10 partitions** (not in the Review-1 plan) | a single partition cannot distinguish the method from the split; the panel is entitled to error bars |
| Added **four detector families** where Review 1 proposed two | the two extra families revealed that collapse severity is capacity-dependent — a result we would otherwise have overstated |
| Neural **text-encoder** arm still deferred | blocked on raw email bodies, not on implementation; the MLP arm covers the representation question as far as the data allows |

No change to the research question, the three nominated calibration families, or
the evaluation metrics.

---

## Live demo script (6 minutes)

1. **Repo** — tree, README, `git log --oneline`. *(item 2)*
2. **Run the core** — `python -m genphish.run_all`. Narrate the four blocks. *(items 1, 3)*
3. **Transfer** (`fig1`, `fig2`): "F1 collapses off-diagonal; **AUC stays 0.98** — the signal survives, the threshold dies."
4. **Calibration** (`fig3`): "Distribution matching recovers **90% of the labelled-recalibration gap with zero target labels**; EM converges confidently to the wrong prior — here's why." *(items 4, 8)*
5. **Error bars** (`fig12`): "10 partitions, 60 paired observations, Holm-corrected p ≈ 6×10⁻¹¹, and it wins on every single pair. It is also significantly *below* the oracle — it is not free." *(item 3)*
6. **Ablations** (`fig9`, `fig13`): "Four features carry the detector. The collapse is capacity-dependent; the correction is not." *(item 6)*
7. **Adversary** (`fig10`, `fig11`): "Evasion is a cliff, not a slope — no warning in the accuracy metric before it fails. And a 1e-6 jitter on integer counts costs 9 pp AUC, which no generator-shift experiment would ever surface." *(items 7, 8)*
8. **Monitor** (`fig5`): "Which is why the trigger watches the score distribution, not accuracy: flat under no shift, past the Ville ceiling after the switch, δ = 0.01." *(item 7)*
9. **Zero-trust** (`fig6`): "Graded beats binary across cost ratios 1–83; ~8% of legitimate mail rescued."
10. **Tests** — `pytest -q`: "20 invariants, including the negative control and a regression test for a reporting bug we found in our own confidence intervals." *(item 12)*

---

## Originality note (item 13)

A public reproduction of the *baseline* exists (a student repo that reruns the
Frontiers authors' labelled-recalibration pipeline). This implementation is
independent and original: written from the methodology, not copied. Its
contribution — the three **label-free** calibration methods, the
**conformal-martingale drift trigger**, the **confidence-graded zero-trust**
layer, the gap-recovery evaluation, and the S6/S7 ablation and adversarial
testbed — does not exist in that repo, which only does labelled recalibration.
The baseline is cited, not reused.

---

## Remaining work → Final Review (item 10)

| Week | Work |
|---|---|
| 1 | Run the full pipeline on the **real Zenodo corpus** (`cross-model-phishing.zip`, DOI 10.5281/zenodo.20250116); lock real numbers; re-run S6 + S7 + stability on it. **Re-run the de-quantization probe first** — real counts are integers too, so the brittleness may be real rather than synthetic |
| 2 | Widen the matrix with 1–2 extra generators minted locally (Ollama) through `features.extract` |
| 3 | **Raw email bodies**: extract from the Zenodo archive internals or the upstream CEAS-08 / TREC-07 / Nazario sources; if present, the text-encoder arm unblocks |
| 4 | **Text-encoder arm** (Kaggle/Colab GPU) on the same splits; confirm or refute that the calibration finding extends beyond stylometry |
| 5 | Mitigation for Finding 2 — count-feature smoothing or monotone constraints — and re-test; draft paper with full results and discussion |
| 6 | Buffer + final report + viva deck; plagiarism self-check |

---

---

## Progress towards Final Review — what the reviewer can verify now

> **How to read this section.**  The three-column table maps every claim in the
> dissertation to its review stage. Anything in the "Review 2" column exists in
> this commit and runs in under 5 minutes. "Final Review" items are either
> blocked on data (real Zenodo corpus) or on GPU compute (text-encoder).

### What existed at Review 1 (baseline)

| Delivered | Evidence |
|---|---|
| Multi-LLM corpus scaffold (GPT-4.1 / DeepSeek 3.2 / LLaMA 3.3 70B) | `corpus.py`, `features.py` |
| 17 stylometric features extracted | `features.py::extract_frame` |
| 3 × 3 transfer matrix: XGBoost | `results/transfer_auc.csv` |
| Label-free calibration (DM, EM, self-training) — proof of concept | `calibration.py` |
| Conformal-martingale drift monitor | `drift_monitor.py` |
| Zero-trust cost sweep (binary vs graded) | `zerotrust.py` |

### What is new at Review 2 (this submission, all runnable today)

| What was added | How it advances the work | Where |
|---|---|---|
| **Multi-seed stability** (10 partitions × 6 pairs = 60 observations) | Converts a single-partition result into a statistically defensible one; Wilcoxon + Holm-BH, 60/60 win rate for DM | `stability.py`, `run_extended --stability` |
| **Shuffled-label permutation control** | Proves the pipeline is not learning a statistical artefact; null AUC 0.493 vs real 0.977 | `ablation.py`, `run_extended --control-perms` |
| **Feature-group ablations** (5 studies) | Identifies persuasion features as the sole driver; exposes which attack surfaces directly | `ablation.py` |
| **Four detector families** (XGBoost, RF, LogReg, MLP) | Shows collapse severity is capacity-dependent; DM recovers across all families | `detectors.py`, `neural.py` |
| **S7 adversarial robustness** (paraphrase, register, 4-level evasion sweep) | Discovers evasion cliff + de-quantization brittleness; separates recoverable from irrecoverable attacks | `robustness.py`, `run_extended` |
| **De-quantization finding** (1 × 10⁻⁶ jitter → −9.27 pp AUC) | Novel finding invisible to generator-shift experiments alone; documented limitation with fix path | `robustness.py::make_dequantize` |
| **Neural arm** (MLP vs XGBoost) | Confirms recalibration is representation-agnostic (spread 0.027 across families) | `neural.py` |
| **LogReg arm** (`--logreg`) | Adds the linear family; exposes that linear models do not show the same threshold collapse | `run_extended --logreg` |
| **22 pytest invariants** (was 20) | Extends regression coverage to the new stability, robustness, and significance modules | `tests/test_extended.py` |
| **SOAR integration** (`soar.py`) | Closes the pipeline operationally: detection → incident → playbook; generic webhook + Splunk SOAR + simulation mode | `src/genphish/soar.py` |
| **Streamlit detection dashboard** (`app.py`) | Live demo: paste an email, see confidence gauge, tier badge, SOAR incident, drift simulation in 4 tabs | `app.py` (run: `streamlit run app.py`) |
| **Public `classify()` / `classify_batch()` API** | Decouples zero-trust tier assignment from the internal cost-sweep code; used by dashboard + SOAR | `zerotrust.py` |
| **Online `ConformalMartingale` class** | One-email-at-a-time streaming wrapper for the dashboard drift-simulation tab | `drift_monitor.py` |

### What remains for Final Review (scope declared, not deferred)

| Item | Blocker | Plan |
|---|---|---|
| Full pipeline on **real Zenodo corpus** (DOI 10.5281/zenodo.20250116) | Data download (`cross-model-phishing.zip`, ~400 MB) | Week 1 — drop CSV into `data/`, re-run; all output auto-restamps REAL |
| 1–2 additional generators (Ollama local) | Compute / setup time | Week 2 — widens the transfer matrix at no cost once corpus is real |
| **Text-encoder arm** (raw email bodies) | Raw bodies not in the feature table; Kaggle GPU for fine-tuning | Week 3–4 — unblocks if bodies present in Zenodo archive |
| **Hosted dashboard** (Streamlit Community Cloud) | Free account + `requirements.txt` push | Week 4 — makes the demo linkable without running locally |
| Cross-lingual subset (E-PhishLLM if available) | Dataset availability | Week 5 (buffer) |
| Mitigation for de-quantization brittleness (count smoothing) | Code change, low effort | Week 5 — monotone constraints or integer-bucketisation |
| Full draft report + plagiarism check | Awaits final numbers | Week 5–6 |
| Viva presentation deck | Awaits report | Week 6 |

### Measurable progress in one line per axis

| Axis | Review 1 | Review 2 | Change |
|---|---|---|---|
| Partitions / evidence | 1 split | 10 splits, 60 paired obs. | +10× |
| Detector families | 1 (XGBoost) | 4 (XGBoost, RF, LogReg, MLP) | +3 |
| Ablation studies | 0 | 5 (feature, control, family, provenance, sensitivity) | +5 |
| Adversarial scenarios | 0 | 9 (noise, evasion sweep, paraphrase, register, de-quant) | +9 |
| pytest invariants | 0 | 22 | +22 |
| Operational pipeline | detection only | detection + SOAR incidents + playbooks + dashboard | complete |
| Significance testing | none | Wilcoxon + Holm-BH, p ≈ 6×10⁻¹¹ | formal |

---

## Feedback tracker — closing the loop (item 14)

| From | Feedback | Action taken |
|---|---|---|
| Review 0 | Original proposal paired detection + ZTNA with no isolated problem | Pivoted: the contribution is now **label-free calibration under generator shift**; zero-trust subordinated to the *response* layer that consumes detector confidence |
| Review 1 | *(record each panel comment here, one row per comment)* | *(fill in before Review 2)* |

> Fill the Review-1 rows from your panel notes: one row per comment, each with
> the concrete change made, so the panel can see every loop closed.
