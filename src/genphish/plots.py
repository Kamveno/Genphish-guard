"""All figures. Clean, colour-blind-safe, 150 dpi. Every figure that uses the
synthetic corpus is stamped SYNTHETIC in the title so it can never be mistaken
for a measurement on the real corpus.
"""
from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .config import FIGURES

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight",
    "font.size": 10, "axes.titlesize": 11, "axes.grid": True,
    "grid.alpha": 0.25, "axes.axisbelow": True,
})
BLUE, RED, GREEN, AMBER, GREY = "#2E6DB4", "#C0392B", "#2C8C5A", "#B5730F", "#7A7F8C"


def _stamp(fig, is_real):
    if not is_real:
        fig.text(0.01, 0.985, "SYNTHETIC DEMONSTRATION DATA", ha="left",
                 va="top", fontsize=7, color=RED, alpha=0.75, style="italic")


def heatmap(M, title, fname, is_real, vmin=0, vmax=1, cmap="RdYlGn"):
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    im = ax.imshow(M.values.astype(float), cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(M.columns)), M.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(M.index)), M.index)
    ax.set_xlabel("evaluated on (target)"); ax.set_ylabel("trained on (source)")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = float(M.values[i, j])
            ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                    color="black" if 0.35 < v < 0.85 else "white", fontsize=9)
    ax.set_title(title); ax.grid(False)
    fig.colorbar(im, fraction=0.046, pad=0.04)
    _stamp(fig, is_real); fig.savefig(FIGURES / fname); plt.close(fig)


def calibration_bars(mean_f1: dict, title, fname, is_real):
    order = ["stale", "em", "self", "dm", "oracle"]
    labels = ["Stale\n(no calib)", "EM\nprior-shift", "Self-\ntraining",
              "Distribution\nmatching", "Oracle\n(uses labels)"]
    colors = [GREY, AMBER, AMBER, GREEN, BLUE]
    vals = [mean_f1[k] for k in order]
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    bars = ax.bar(labels, vals, color=colors)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}",
                ha="center", fontsize=9)
    ax.axhline(mean_f1["stale"], ls="--", color=GREY, lw=1)
    ax.axhline(mean_f1["oracle"], ls="--", color=BLUE, lw=1)
    ax.set_ylabel("mean cross-generator F1"); ax.set_ylim(0, 1.0)
    ax.set_title(title); ax.grid(axis="x")
    _stamp(fig, is_real); fig.savefig(FIGURES / fname); plt.close(fig)


def drift_trajectories(trajs: dict, ceiling, title, fname, is_real):
    """trajs[label] = (wealth array, switch_index)."""
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    cmap = {"severe": RED, "mild": AMBER, "no shift": GREEN}
    switches = set()
    for label, (w, switch) in trajs.items():
        c = cmap.get(label.split(" (")[0], BLUE)
        ax.plot(np.arange(len(w)), w, color=c, lw=1.6, label=label)
        switches.add(switch)
    for sw in switches:
        ax.axvline(sw, ls=":", color=GREY, lw=1)
    ax.text(max(switches), ceiling*1.3, " generator switch", fontsize=7,
            color=GREY, ha="left", va="bottom")
    ax.axhline(ceiling, ls="--", color="black", lw=1, label=f"Ville ceiling 1/δ={ceiling:.0f}")
    ax.set_yscale("log"); ax.set_xlabel("emails processed in band")
    ax.set_ylabel("martingale wealth $M_t$ (log)")
    ax.set_title(title); ax.legend(fontsize=8, loc="upper left")
    _stamp(fig, is_real); fig.savefig(FIGURES / fname); plt.close(fig)


def zerotrust_curves(pts, win_range, title, fname, is_real):
    r = [p.ratio for p in pts]
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.plot(r, [p.binary_cost for p in pts], "-o", color=GREY, ms=3, label="binary allow/block")
    ax.plot(r, [p.graded_cost for p in pts], "-o", color=GREEN, ms=3, label="graded zero-trust")
    ax.set_xscale("log"); ax.set_xlabel("cost ratio  C_miss / C_false-block")
    ax.set_ylabel("expected cost per email")
    if win_range:
        ax.axvspan(win_range[0], win_range[1], color=GREEN, alpha=0.10)
        ax.set_title(f"{title}\ngraded wins for ratio ∈ [{win_range[0]:.1f}, {win_range[1]:.0f}]")
    else:
        ax.set_title(title)
    ax.legend(fontsize=9)
    _stamp(fig, is_real); fig.savefig(FIGURES / fname); plt.close(fig)


def error_profile(prof, title, fname, is_real):
    top = prof.head(6)[::-1]
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    y = np.arange(len(top)); h = 0.38
    # normalise each feature by caught mean for comparability
    denom = top["caught_mean"].replace(0, np.nan).abs()
    mm = (top["missed_mean"] / denom).fillna(0)
    cm = (top["caught_mean"] / denom).fillna(0)
    ax.barh(y + h/2, cm, h, color=GREEN, label="caught phishing")
    ax.barh(y - h/2, mm, h, color=RED, label="missed phishing")
    ax.set_yticks(y, top.index, fontsize=8)
    ax.set_xlabel("feature value (normalised to caught-phishing mean)")
    ax.set_title(title); ax.legend(fontsize=8); ax.grid(axis="y")
    _stamp(fig, is_real); fig.savefig(FIGURES / fname); plt.close(fig)


def ablation_bars(abl, title, fname, is_real):
    """S6 feature ablation: what each feature group contributes to the transfer
    gap, and whether label-free calibration still works without it."""
    d = abl.iloc[::-1]
    y = np.arange(len(d)); h = 0.38
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.0, 3.9), sharey=True)
    a1.barh(y, d["transfer_gap_pp"], h * 2, color=RED, alpha=0.85)
    a1.set_xlabel("transfer gap (pp)"); a1.set_title("the phenomenon")
    a1.set_yticks(y, d["config"], fontsize=8)
    for yi, v in zip(y, d["transfer_gap_pp"]):
        a1.text(v + 0.4, yi, f"{v:.1f}", va="center", fontsize=8)
    a2.barh(y, d["rec_dm"] * 100, h * 2, color=GREEN, alpha=0.9)
    a2.set_xlabel("gap recovered by distribution matching (%)")
    a2.set_title("the correction"); a2.set_xlim(0, 105)
    for yi, v in zip(y, d["rec_dm"] * 100):
        a2.text(min(v + 1.5, 97), yi, f"{v:.0f}%", va="center", fontsize=8)
    for ax in (a1, a2):
        ax.grid(axis="y")
    fig.suptitle(title, y=1.02)
    _stamp(fig, is_real); fig.savefig(FIGURES / fname); plt.close(fig)


def family_bars(fam, title, fname, is_real):
    """Is the collapse a property of the shift or of one model family?"""
    x = np.arange(len(fam)); w = 0.26
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.bar(x - w, fam["f1_stale"], w, color=GREY, label="stale threshold")
    ax.bar(x, fam["f1_dm"], w, color=GREEN, label="distribution matching")
    ax.bar(x + w, fam["f1_oracle"], w, color=BLUE, label="oracle (uses labels)")
    for xi, (st, auc) in enumerate(zip(fam["f1_stale"], fam["offdiag_auc"])):
        ax.text(xi, 0.02, f"AUC {auc:.3f}", ha="center", fontsize=7, color="white")
    ax.set_xticks(x, fam["config"]); ax.set_ylim(0, 1.05)
    ax.set_ylabel("mean cross-generator F1"); ax.set_title(title)
    ax.legend(fontsize=8, ncol=3, loc="upper center"); ax.grid(axis="x")
    _stamp(fig, is_real); fig.savefig(FIGURES / fname); plt.close(fig)


def robustness_panel(rob, title, fname, is_real):
    """S7: for each transform, did ranking survive (AUC) and did recalibration
    put the threshold back (stale -> DM)?"""
    d = rob.iloc[::-1]
    y = np.arange(len(d))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.4, 4.6), sharey=True)

    # Axis must start below the worst bar, or a catastrophic row renders as
    # nothing at all and reads as missing data.
    lo = min(0.45, float(d["offdiag_auc"].min()) - 0.06)
    a1.barh(y, d["offdiag_auc"], 0.6, color=BLUE, alpha=0.85)
    a1.axvline(0.5, ls="--", color=GREY, lw=1)
    a1.text(0.5, -0.45, " chance", fontsize=7, color=GREY, ha="left")
    a1.set_xlim(lo, 1.03)
    a1.set_xlabel("off-diagonal AUC (does the signal survive?)")
    a1.set_yticks(y, d["scenario"], fontsize=8); a1.grid(axis="y")
    span = 1.03 - lo
    for yi, v in zip(y, d["offdiag_auc"]):
        inside = (v - lo) > 0.22 * span
        a1.text(v - 0.01 * span if inside else v + 0.01 * span, yi, f"{v:.3f}",
                va="center", ha="right" if inside else "left",
                fontsize=8, color="white" if inside else "black")

    a2.barh(y + 0.19, d["f1_dm"], 0.36, color=GREEN,
            label="after label-free recalibration")
    a2.barh(y - 0.19, d["f1_stale"], 0.36, color=GREY, label="stale threshold")
    a2.set_xlim(0, 1.12); a2.set_xlabel("mean cross-generator F1")
    a2.legend(fontsize=8, loc="lower right", framealpha=0.95)
    a2.grid(axis="y")
    fig.suptitle(title, y=1.0)
    _stamp(fig, is_real); fig.savefig(FIGURES / fname); plt.close(fig)


def evasion_curve_plot(cur, title, fname, is_real):
    """The attacker turns one dial. One panel per detector family, because the
    *shape* of the failure differs: the tree ensemble steps, the linear model
    slides. A step means the metric gives no warning before the drop."""
    models = list(dict.fromkeys(cur["model"])) if "model" in cur else [None]
    fig, axes = plt.subplots(1, len(models), figsize=(5.0 * len(models), 3.9),
                             sharey=True, squeeze=False)
    for ax, m in zip(axes[0], models):
        d = cur[cur["model"] == m] if m is not None else cur
        e = d["eps"]
        ax.plot(e, d["offdiag_auc"], "-o", color=BLUE, ms=4, label="AUC (ranking)")
        ax.plot(e, d["f1_stale"], "-o", color=GREY, ms=4, label="F1, stale threshold")
        ax.plot(e, d["f1_dm"], "-o", color=GREEN, ms=4,
                label="F1, recalibrated (no labels)")
        ax.plot(e, d["f1_oracle"], ":", color=BLUE, lw=1.2,
                label="F1, oracle (uses labels)")
        ax.fill_between(e, d["f1_stale"], d["f1_dm"], color=GREEN, alpha=0.12)
        ax.set_xlabel("evasion strength ε")
        ax.set_ylim(0, 1.05)
        if m is not None:
            ax.set_title(m)
    axes[0][0].set_ylabel("score")
    axes[0][0].legend(fontsize=8, loc="lower left")
    fig.suptitle(f"{title}\nε = 0 untouched · ε = 1 persuasion markers erased",
                 y=1.06, fontsize=11)
    _stamp(fig, is_real); fig.savefig(FIGURES / fname); plt.close(fig)


def stability_box(long_df, title, fname, is_real):
    """Per-seed, per-pair spread of each method — the result is a distribution,
    not a single number."""
    order = ["stale", "em", "self", "dm", "oracle"]
    labels = ["stale", "EM", "self-train", "dist. matching", "oracle"]
    colors = [GREY, AMBER, AMBER, GREEN, BLUE]
    data = [long_df.loc[long_df["method"] == m, "f1"].values for m in order]
    fig, ax = plt.subplots(figsize=(6.6, 3.9))
    bp = ax.boxplot(data, patch_artist=True, widths=0.55, showfliers=False)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c); patch.set_alpha(0.65)
    for med in bp["medians"]:
        med.set_color("black")
    rng = np.random.default_rng(0)
    for i, d in enumerate(data, start=1):
        ax.scatter(rng.normal(i, 0.055, len(d)), d, s=5, color="black", alpha=0.25, zorder=3)
    ax.set_xticks(range(1, len(order) + 1), labels, fontsize=9)
    ax.set_ylabel("F1 per (seed × generator pair)"); ax.set_ylim(0, 1.05)
    n_seeds = long_df["seed"].nunique()
    ax.set_title(f"{title}\n{n_seeds} partitions × 6 generator pairs")
    ax.grid(axis="x")
    _stamp(fig, is_real); fig.savefig(FIGURES / fname); plt.close(fig)


def score_shift(dists: dict, title, fname, is_real):
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    cmap = {"legitimate": GREEN, "source phishing": BLUE, "target phishing (shifted)": RED}
    for label, s in dists.items():
        ax.hist(s, bins=40, alpha=0.55, density=True,
                color=cmap.get(label, GREY), label=label)
    ax.set_xlabel("detector score  P(phishing)"); ax.set_ylabel("density")
    ax.set_title(title); ax.legend(fontsize=8); ax.set_yscale("log")
    _stamp(fig, is_real); fig.savefig(FIGURES / fname); plt.close(fig)
