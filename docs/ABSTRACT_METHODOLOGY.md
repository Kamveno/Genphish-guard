# Draft: Abstract + Methodology

*(Review-2 deliverable: report structure started. Numbers are placeholders keyed
to `results/summary.json`; they resolve to real values once the pipeline is run
on the Zenodo corpus. Figures referenced are generated in `results/figures/`.)*

---

## Abstract (draft, ~180 words)

Content-based phishing detectors are trained on emails produced by one text
generator and deployed against emails produced by another. When an attacker
adopts a newer large language model, published detectors lose accuracy, and the
defender has no labelled phishing from that model with which to correct them. We
show that the discriminative signal survives this generator shift — ranking
quality (AUC) is preserved — while the decision threshold does not, producing a
≈28-point F1 collapse that recalibration with target labels would fix. Because
phishing from a new generator arrives before any labels exist, we study
**label-free** threshold correction and evaluate the three families the closest
prior work names as open: distribution matching, prior-shift estimation
(Saerens-Latinne-Decaestecker EM), and self-training. Distribution matching
recovers the majority of the labelled-recalibration gap with no target labels. A
conformal test martingale monitors the unlabelled score stream and raises an
anytime-valid alarm when exchangeability breaks, triggering recalibration; a
three-tier zero-trust layer consumes the recalibrated confidence. Ablations show
the effect is not a dataset artefact and that the correction holds across four
detector families, while an adversarial study bounds the claim: label-free
threshold correction answers threshold displacement, not an attacker who
degrades the ranking signal itself. All results use public corpora, fixed seeds,
and one-command regeneration.

---

## Methodology (draft outline → maps to `src/genphish/`)

**Problem.** Source generator *g_s*, target generator *g_t* (unseen). Detector
*f* trained on *g_s* phishing + legitimate mail. At deployment, target mail is
**unlabelled**. Goal: a decision rule on *g_t* that recovers the F1 a labelled
recalibration would achieve, using no *g_t* labels.

**S1 Corpus** (`corpus.py`). Multi-LLM phishing corpus (GPT-4.1, DeepSeek 3.2,
LLaMA 3.3 70B) + human-written phishing (Nazario) + legitimate mail
(SpamAssassin/Enron), unified with per-email provenance. *(Synthetic stand-in
until the Zenodo CSV is dropped in.)*

**S2 Features** (`features.py`). 17 hand-engineered stylometric/linguistic
features, chosen because prior work reports they transfer across generators
better than learned representations.

**S3 Detectors** (`detectors.py`). Per-generator and pooled detectors; XGBoost
primary, Logistic Regression as a transparent baseline.

**S4 Transfer matrix** (`transfer.py`). N×N F1 at the default threshold and
threshold-free AUC. Diagnostic contrast: AUC high off-diagonal, F1 collapsed →
*the threshold, not the signal, is the problem.* (fig1, fig2)

**S5 Label-free calibration** (`calibration.py`) — the contribution.
(i) **Distribution matching**: re-place the threshold at the quantile of the
target score distribution it held in the source. (ii) **Prior-shift EM**:
estimate the target prior from unlabelled scores (SLD), correct the boundary.
(iii) **Self-training**: pseudo-label the confident tails, iterate, guarded by a
prior-plausibility bound. Upper bound: labelled oracle. Headline metric:
**gap-recovery rate**. (fig3)

**Drift trigger** (`drift_monitor.py`). A Simple-Jumper / betting conformal test
martingale on the unlabelled stream; Ville's inequality makes an alarm a
δ-level event. Decides *when* to recalibrate. (fig5)

**S8 Zero-trust response** (`zerotrust.py`). Three tiers (block / deliver-defused
/ deliver) from calibrated confidence. Expected-cost sweep over the
miss-to-false-block cost ratio; report the ratio range where the graded policy
dominates, and legitimate mail rescued at equal containment. (fig6)

**S6 Controls and ablations** (`ablation.py`). Five studies, all implemented:
feature-group ablation (which features carry the gap, and which the correction
needs); **shuffled-label permutation test** (training labels permuted, null AUC
at chance, real detector exceeds every replicate); detector-family ablation over
four model families (the collapse is capacity-dependent, the correction is not);
provenance ablation (the pooled detector is the labelled upper bound, and
label-free recalibration closes ~67% of the distance to it); hyperparameter
sensitivity for all three calibration methods, including the operating-recall
range in which distribution matching is valid. (fig9, fig13)

**S7 Robustness to an active adversary** (`robustness.py`). Transforms applied
to held-out target phishing after training: paraphrase, register shift, length
inflation, and a targeted evasion sweep that moves the suppressible persuasion
markers a fraction ε toward the legitimate-mail centroid. Each scenario is
judged on two independent axes — did **ranking** survive (AUC) and did
**operational F1** fall — because only the second-order case (ranking intact,
threshold displaced) is one that recalibration answers. Two findings: evasion
produces a **cliff rather than a slope** (no metric warning before failure,
which is the argument for the distribution-watching drift trigger), and a
**de-quantization probe** (a 1e-6 jitter on the integer count features) costs
the tree detector ~9 pp AUC while leaving the linear detector untouched —
a representation brittleness invisible to generator-shift experiments.
(fig10, fig11)

**Learned-representation arm** (`neural.py`). An MLP over the same inputs,
through the identical protocol, to test whether the finding is specific to
hand-engineered features. A fine-tuned text encoder remains blocked on raw
email bodies rather than on implementation; `text_encoder_readiness()` checks
that precondition in code.

**Evaluation** (`evaluate.py`, `stability.py`). F1, AUC, gap-recovery,
precision@recall, FPR, ECE; every estimate with a bootstrap CI built around the
statistic actually reported. Headline comparisons are repeated over 10
independent partitions and tested with **paired Wilcoxon signed-rank on
(partition × source × target), Holm-corrected** across the comparison family,
with Cohen's d_z as the effect size and a per-pair win rate so that an average
win is not mistaken for a universal one. (fig12)
