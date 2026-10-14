"""Invariant tests for the extended studies (S6 ablations, S7 robustness,
the neural arm and the stability/significance machinery).

Same contract as test_pipeline.py: these are the checks a reviewer re-runs to
confirm the extended claims hold, not unit tests of implementation detail. They
use small seed counts so the suite stays runnable on a laptop; the reported
numbers come from the full sweeps in run_extended.
"""
import numpy as np
import pandas as pd
import pytest

from genphish.config import GENERATORS, FEATURES, LABEL, SOURCE, HUMAN
from genphish import corpus as C, detectors as D, transfer as T
from genphish import ablation as AB, robustness as RB, neural as NN
from genphish import stability as ST, evaluate as EV
from genphish.experiment import run_experiment

SEEDS2 = (20261011, 7)


@pytest.fixture(scope="module")
def df():
    d, _ = C.load_corpus()
    return d


@pytest.fixture(scope="module")
def baseline(df):
    return run_experiment(df, GENERATORS, label="baseline")


# ---------------------------------------------------------------- S6 controls
def test_shuffled_label_control_is_chance(df):
    """The negative control, as a permutation test. If a detector trained on
    permuted labels can still rank the test set, something leaks and every
    other number in the report is suspect.

    Asserted as 'the real detector beats every permutation replicate and the
    null sits near chance', not as 'an interval around the null mean covers
    0.5' — the null here centres a little below 0.5 for a documented reason,
    and the interval version failed for purely arithmetic reasons at n=3.
    """
    n = 3
    ctrl = AB.shuffled_label_control(df, GENERATORS, n_seeds=n)
    null = ctrl[ctrl["config"].str.startswith("shuffled")].iloc[0]
    real = ctrl[ctrl["config"] == "true labels"].iloc[0]

    assert bool(null["near_chance"]), null.to_dict()
    assert bool(null["real_exceeds_all_permutations"]), null.to_dict()
    assert null["permutation_p"] == pytest.approx(1 / (n + 1))
    assert real["offdiag_auc_min"] > 0.80            # the real detector does rank
    assert null["rec_dm_mean"] < 0.25                # 'recovery' is meaningless here


def test_feature_groups_partition_the_schema():
    flat = sum(AB.FEATURE_GROUPS.values(), [])
    assert sorted(flat) == sorted(FEATURES)
    assert len(flat) == len(set(flat)) == 17


def test_persuasion_features_carry_the_signal(df):
    """Dropping the persuasion group must cost real discriminative power —
    it is the group the phenomenon is built on."""
    full = run_experiment(df, GENERATORS, label="full")
    without = run_experiment(df, GENERATORS, label="-persuasion",
                             features=[f for f in FEATURES
                                       if f not in AB.FEATURE_GROUPS["persuasion"]])
    assert without.summary_auc["off_diagonal_mean"] < \
        full.summary_auc["off_diagonal_mean"] - 0.10


def test_label_free_calibration_helps_in_every_detector_family(df):
    """If the contribution only worked for one model family it would be a
    property of that model, not of the shift."""
    fam = AB.detector_family_ablation(df, GENERATORS)
    assert set(fam["config"]) == set(D.DETECTOR_KINDS)
    assert (fam["f1_dm"] > fam["f1_stale"]).all(), fam.to_string()
    assert (fam["f1_dm"] <= fam["f1_oracle"] + 1e-6).all()


def test_pooled_detector_is_an_upper_bound(df):
    """Pooling (which needs target labels) should beat single-source transfer
    (which does not). If it did not, the premise of the project is wrong."""
    prov = AB.provenance_ablation(df, GENERATORS)
    assert (prov["f1_pooled_seen_target"] > prov["f1_single_source_stale"]).all()
    assert (prov["f1_single_source_dm"] >= prov["f1_single_source_stale"]).all()


# ---------------------------------------------------------------- S7 robustness
def test_transforms_stay_in_the_valid_feature_space(df):
    """An 'attack' must not win by producing emails that cannot exist."""
    phish = df[(df[SOURCE] != HUMAN) & (df[LABEL] == 1)].head(400)
    mu = RB.ham_profile(df)
    for tf in (RB.make_paraphrase(), RB.make_register_shift(),
               RB.make_length_inflation(), RB.make_evasion(mu, 1.0),
               RB.make_noise(0.1)):
        out = tf(phish)
        assert list(out.columns) == list(phish.columns)
        assert len(out) == len(phish)
        for c in RB._RATIOS:
            assert out[c].between(0, 1).all(), c
        for c in RB._COUNTS:
            assert (out[c] >= 0).all(), c
        assert out[FEATURES].notna().all().all()


def test_transform_does_not_leak_into_training(df):
    """The transform must act only on held-out target mail. An identity
    transform must therefore reproduce the untransformed run exactly."""
    a = run_experiment(df, GENERATORS, label="clean")
    b = run_experiment(df, GENERATORS, label="identity",
                       transform=lambda X: X.copy())
    assert a.summary_f1 == b.summary_f1
    assert a.summary_auc == b.summary_auc


def test_full_evasion_degrades_ranking(df):
    """Erasing the suppressible markers entirely must cost ranking power —
    otherwise the transform is not doing anything."""
    mu = RB.ham_profile(df)
    clean = run_experiment(df, GENERATORS, label="clean")
    attacked = run_experiment(df, GENERATORS, label="eps=1",
                              transform=RB.make_evasion(mu, 1.0))
    assert attacked.summary_auc["off_diagonal_mean"] < \
        clean.summary_auc["off_diagonal_mean"] - 0.05


def test_robustness_verdicts_are_self_consistent(df):
    """Ranking damage and operational damage are independent, so all four
    combinations must get their own verdict. Collapsing them to two produced a
    row that said 'signal degraded' while F1 had gone up."""
    rob = RB.robustness_study(df, GENERATORS, eps_grid=(0.5, 1.0))
    assert rob.iloc[0]["scenario"].startswith("clean")
    for _, r in rob.iterrows():
        rank, op = bool(r["ranking_damaged"]), bool(r["operational_damage"])
        v = r["verdict"]
        if rank and op:
            assert "cannot fix" in v, r.to_dict()
        elif op:
            assert "recalibration applies" in v, r.to_dict()
        elif rank:
            assert "already misplaced" in v, r.to_dict()
        else:
            assert v == "no material effect", r.to_dict()


def test_dequantization_probe_isolates_representation_brittleness(df):
    """A 1e-6 jitter on integer counts changes nothing about the mail. Whatever
    it costs is the detector depending on those counts being exact integers —
    and the scenarios that re-quantize must not be paying that cost too."""
    mu = RB.ham_profile(df)
    phish = df[(df[SOURCE] != HUMAN) & (df[LABEL] == 1)].head(300)

    # every ordinary transform returns integer counts ...
    for tf in (RB.make_paraphrase(), RB.make_register_shift(),
               RB.make_length_inflation(), RB.make_evasion(mu, 0.5),
               RB.make_noise(0.1)):
        out = tf(phish)
        for c in RB._COUNTS:
            assert np.allclose(out[c], out[c].round()), c

    # ... and the probe deliberately does not
    probed = RB.make_dequantize()(phish)
    assert not np.allclose(probed["urgency_term_count"],
                           probed["urgency_term_count"].round())
    # the jitter is numerically negligible, so any metric change is brittleness
    assert (probed[RB._COUNTS] - phish[RB._COUNTS]).abs().max().max() < 1e-4


# ---------------------------------------------------------------- neural arm
def test_neural_arm_replicates_the_finding(df):
    out = NN.run_neural_arm(df, GENERATORS)
    s = out["neural_arm_summary"]
    assert set(s["model"]) == {"xgboost", "mlp"}
    assert (s["f1_dm"] > s["f1_stale"]).all()
    assert out["_verdict"]["label_free_helps_in_every_family"]


def test_text_encoder_readiness_is_honest_about_the_corpus(df):
    """The corpus ships features, not message bodies. The readiness check must
    say so rather than quietly claiming the arm is available."""
    r = NN.text_encoder_readiness(df)
    assert r["raw_text_column_present"] is False
    assert r["status"] == "blocked on data, not on code"
    assert r["unblocking_action"]
    # ...and must flip when raw text IS present
    with_text = df.head(5).copy()
    with_text["body"] = "URGENT: verify your account"
    assert NN.text_encoder_readiness(with_text)["status"] == "ready"


# ---------------------------------------------------------------- stability
def test_holm_correction_is_monotone_and_bounded():
    p = [0.001, 0.02, 0.04, 0.9]
    adj = ST._holm(p)
    assert all(a >= b - 1e-12 for a, b in zip(adj, p))      # never smaller
    assert all(0 <= a <= 1 for a in adj)
    assert adj == sorted(adj, key=lambda x: x)              # order preserved here
    assert ST._holm([0.5])[0] == pytest.approx(0.5)


def test_distribution_matching_beats_stale_significantly(df):
    """The headline comparison, paired on (seed, source, target)."""
    _, long_df = ST.multi_seed_study(df, GENERATORS, seeds=SEEDS2)
    sig = ST.paired_tests(long_df, against=("stale",))
    row = sig.iloc[0]
    assert row["n_pairs"] == len(SEEDS2) * 6
    assert row["mean_diff_f1"] > 0
    assert row["p_holm"] < 0.05
    assert row["reading"] == "dm better"


def test_stability_summary_brackets_the_point_estimates(df):
    per_seed, _ = ST.multi_seed_study(df, GENERATORS, seeds=SEEDS2)
    summ = ST.summarise_stability(per_seed).set_index("quantity")
    for q in ("transfer_gap_pp", "rec_dm", "f1_dm"):
        row = summ.loc[q]
        assert row["ci95_lo"] <= row["mean"] <= row["ci95_hi"]
        assert row["min"] <= row["mean"] <= row["max"]


def test_auc_ci_contains_the_reported_point(baseline):
    """Regression test for a real reporting bug: the CI used to be built on the
    pooled-score AUC while the abstract quoted the mean of per-pair AUCs, so
    the published interval did not contain the published number."""
    point, lo, hi = EV.mean_offdiag_auc_ci(baseline.cache, GENERATORS, n=200)
    assert lo <= point <= hi
    assert point == pytest.approx(baseline.summary_auc["off_diagonal_mean"], abs=5e-4)
