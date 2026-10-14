"""Multi-seed stability and paired significance testing.

A single train/test partition gives a single number. A panel is entitled to ask
whether that number is the method or the partition. This module reruns the whole
transfer-and-recalibrate experiment over K independent partitions and reports:

  * mean ± sd and a percentile bootstrap CI for every headline quantity;
  * **paired** significance tests comparing distribution matching against every
    alternative, pairing on (seed, source generator, target generator) — the
    same six generator pairs seen by every method on every seed, so the test is
    on the method difference and nothing else;
  * Holm–Bonferroni correction across the family of comparisons, because four
    tests on one table is four chances to be lucky;
  * Cohen's d_z as an effect size, because with enough pairs a trivial
    difference is still "significant".

Wilcoxon signed-rank is used rather than a paired t-test: F1 differences across
generator pairs are bounded, skewed and certainly not normal.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats

from .config import SEED, N_BOOTSTRAP
from .experiment import run_experiment, pair_long, METHODS, LABEL_FREE

DEFAULT_SEEDS = (20261011, 7, 101, 2024, 31337, 555, 84, 9001, 13, 404)


# ------------------------------------------------------------------ the sweep
def multi_seed_study(df, generators, kind="xgboost", seeds=DEFAULT_SEEDS):
    """Run the full experiment once per seed. Returns (per_seed_df, long_df)."""
    rows, longs = [], []
    for s in seeds:
        res = run_experiment(df, generators, label=f"seed={s}", kind=kind, seed=int(s))
        rows.append(res.row())
        longs.append(pair_long(res))
    return pd.DataFrame(rows), pd.concat(longs, ignore_index=True)


def _boot_ci(x, n=N_BOOTSTRAP, alpha=0.05, seed=SEED):
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), (n, len(x)))].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(x.mean()), float(lo), float(hi)


def summarise_stability(per_seed: pd.DataFrame) -> pd.DataFrame:
    """mean ± sd ± bootstrap CI for each headline quantity across seeds."""
    cols = (["transfer_gap_pp", "offdiag_auc"]
            + [f"f1_{m}" for m in METHODS] + [f"rec_{m}" for m in LABEL_FREE])
    rows = []
    for c in cols:
        m, lo, hi = _boot_ci(per_seed[c].values)
        rows.append({"quantity": c, "mean": m, "sd": float(per_seed[c].std(ddof=1)),
                     "ci95_lo": lo, "ci95_hi": hi,
                     "min": float(per_seed[c].min()), "max": float(per_seed[c].max()),
                     "n_seeds": int(len(per_seed))})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ significance
def _holm(pvals: list[float]) -> list[float]:
    """Holm–Bonferroni step-down adjusted p-values."""
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        val = (m - rank) * pvals[i]
        running = max(running, val)
        adj[i] = min(1.0, running)
    return adj.tolist()


def paired_tests(long_df: pd.DataFrame, reference: str = "dm",
                 against=("stale", "em", "self", "oracle")) -> pd.DataFrame:
    """Wilcoxon signed-rank, `reference` vs each comparator, paired on
    (seed, source, target). Holm-corrected across the family."""
    wide = long_df.pivot_table(index=["seed", "source", "target"],
                               columns="method", values="f1").dropna()
    rows = []
    for other in against:
        d = (wide[reference] - wide[other]).values
        n = len(d)
        nz = d[d != 0]
        if len(nz) == 0:
            stat, p = np.nan, 1.0
        else:
            stat, p = stats.wilcoxon(nz, alternative="two-sided",
                                     zero_method="wilcox")
        sd = float(np.std(d, ddof=1)) if n > 1 else 0.0
        rows.append({
            "comparison": f"{reference} vs {other}",
            "n_pairs": int(n),
            "mean_diff_f1": float(np.mean(d)),
            "median_diff_f1": float(np.median(d)),
            "cohens_dz": float(np.mean(d) / sd) if sd > 0 else np.inf,
            "wilcoxon_stat": float(stat) if stat == stat else np.nan,
            "p_raw": float(p),
        })
    out = pd.DataFrame(rows)
    out["p_holm"] = _holm(out["p_raw"].tolist())
    out["significant_05"] = out["p_holm"] < 0.05
    # direction, stated plainly so the table cannot be misread
    out["reading"] = np.where(
        ~out["significant_05"], "no detectable difference",
        np.where(out["mean_diff_f1"] > 0, f"{reference} better",
                 f"{reference} worse"))
    return out


def per_pair_consistency(long_df: pd.DataFrame, reference: str = "dm",
                         other: str = "stale") -> pd.DataFrame:
    """Does DM beat the stale threshold on *every* generator pair, or only on
    average? A method that wins on average while losing on some pairs is a
    different (weaker) claim, so it is reported separately."""
    wide = long_df.pivot_table(index=["seed", "source", "target"],
                               columns="method", values="f1").dropna().reset_index()
    wide["diff"] = wide[reference] - wide[other]
    g = wide.groupby(["source", "target"])["diff"]
    out = g.agg(mean_diff="mean", sd_diff="std", n="size",
                win_rate=lambda s: float((s > 0).mean())).reset_index()
    return out.sort_values("mean_diff")


def run_stability(df, generators, kind="xgboost", seeds=DEFAULT_SEEDS) -> dict:
    """Everything, as {name: DataFrame}."""
    per_seed, long_df = multi_seed_study(df, generators, kind, seeds)
    return {
        "stability_per_seed": per_seed,
        "stability_summary": summarise_stability(per_seed),
        "stability_significance": paired_tests(long_df),
        "stability_per_pair": per_pair_consistency(long_df),
        "_long": long_df,
    }
