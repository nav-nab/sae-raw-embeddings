"""Make every figure of the paper from the logged results.

Run from the repository root:

    python paper/make_figures.py

It reads the CSV and JSON files under runs/ and writes PDF figures to
paper/figures/. A few numbers come from analyses that were not saved as CSV
(the Experiment 1 to 3 subset pilots, the margin bins of Experiment 6 and the
code spreads of Experiment 4). These are copied from the docs and marked
below.
"""

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
OUT = ROOT / "paper" / "figures"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 8,
    "axes.titlesize": 8.5,
    "axes.labelsize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#e6e6e6",
    "grid.linewidth": 0.6,
    "lines.linewidth": 1.4,
    "lines.markersize": 4,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,
})

# Two families: TopK in oranges and reds, ReLU in blues.
TOPK = "#e8590c"
TOPK_LIGHT = "#f4a261"
TOPK_DARK = "#9c2c00"
RELU = "#1c6dd0"
GREY = "#8a8a8a"
LAMBDA_COLOURS = {0.03: "#08306b", 0.1: "#2171b5", 0.2: "#4292c6", 0.3: "#6baed6",
                  0.5: "#9ecae1", 1.0: "#c6dbef"}
LAMBDA_COLOURS_STRONG = {0.03: "#253494", 0.1: "#2c7fb8", 0.3: "#41b6c4", 1.0: "#a1dab4"}
SIZE_STYLES = {2000: ("o", "-"), 5000: ("s", "--"), 10000: ("^", "-."), 20000: ("D", ":")}
SIZE_COLOURS = {2000: "#fdae6b", 5000: "#f16913", 10000: "#d94801", 20000: "#8c2d04"}

FULL_V = 50257
SUB_V = 2000
W_FULL = 6.3


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf")
    plt.close(fig)


def panel_label(ax, text):
    ax.set_title(text, loc="left")


def log2_ticks(ax, values, axis="x"):
    labels = [f"{v:,}" for v in values]
    if axis == "x":
        ax.set_xscale("log", base=2)
        ax.set_xticks(values)
        ax.set_xticklabels(labels)
        ax.minorticks_off()
    else:
        ax.set_yscale("log", base=2)
        ax.set_yticks(values)
        ax.set_yticklabels(labels)


# ----------------------------------------------------------------------------
# Data loading


def exp4():
    d = pd.read_csv(RUNS / "exp4" / "summary.csv")
    d["gao_steps"] = d.step * d.lr_factor
    d["epochs"] = d.tokens_m * 2 ** 20 / SUB_V  # the grid counts "millions" of 2^20 tokens
    return d


def exp7():
    d = pd.read_csv(RUNS / "exp7" / "summary.csv")
    d["frac"] = d.step / d.steps
    return d


def exp8(V):
    d = pd.read_csv(RUNS / "exp8" / f"V{V}" / "summary.csv")
    d["frac"] = d.step / d.steps
    return d


def exp9(V):
    d = pd.read_csv(RUNS / "exp9" / f"V{V}" / "summary.csv")
    d["epochs"] = d.step * d.batch_size / V
    return d


def end_of(d, cell, frac):
    x = d[d.cell == cell]
    return x.iloc[(x.frac - frac).abs().argmin()]


# ----------------------------------------------------------------------------
# Figure 1: the headline comparison on the 2k subset


def fig_headline():
    d4 = exp4()
    d7 = exp7()
    fig, ax = plt.subplots(figsize=(3.6, 2.5))
    for lam in (0.03, 0.3):
        colour = LAMBDA_COLOURS_STRONG[lam]
        x = d4[d4.cell == f"lam{lam:g}_lr1_b256"]
        ax.plot(x.epochs, 100 * x["n_recovered_cos0.9"] / SUB_V, "o-", color=colour,
                label=f"ReLU, $\\lambda = {lam:g}$")
    ends = d7[(d7.step == d7.steps) & (d7.lr_factor == 1.0)]
    means = ends.groupby("steps")["n_recovered_cos0.9"].mean()
    epochs = means.index * 512 / SUB_V
    ax.plot(epochs, 100 * means / SUB_V, "s-", color=TOPK, label="TopK, $k$ annealed to 1")
    # Top1 from the start (2k subset pilot, 4,000 steps at batch 512; docs/experiment_3).
    ax.plot([1024], [100 * 227 / SUB_V], "x", color=TOPK_DARK, markersize=6, mew=1.5,
            label="Top1 throughout")
    log2_ticks(ax, [256, 1024, 4096, 16384])
    ax.set_xlabel("training (epochs' worth of tokens)")
    ax.set_ylabel("tokens recovered (%)")
    ax.set_ylim(-3, 103)
    ax.legend(loc="lower right", frameon=False)
    save(fig, "headline")


# ----------------------------------------------------------------------------
# Figure 2: fixed k, and the starting k of the annealing schedule


def fig_topk_k():
    fig, (a, b) = plt.subplots(1, 2, figsize=(W_FULL, 2.3))
    # Fixed-k pilots on the 2k subset, 3,000 steps at batch 512 (docs/experiment_3).
    ks = [1, 2, 4, 8, 16, 32]
    alive = [382, 558, 792, 1111, 1478, 1778]
    rec = [222, 162, 73, 7, 0, 0]
    a.plot(ks, 100 * np.array(alive) / SUB_V, "o-", color=GREY, label="features alive")
    a.plot(ks, 100 * np.array(rec) / SUB_V, "s-", color=TOPK, label="tokens recovered")
    log2_ticks(a, ks)
    a.set_xlabel("$k$ (fixed throughout training)")
    a.set_ylabel("% of 2,000")
    a.set_ylim(-3, 100)
    a.legend(frameon=False, loc="upper left")
    panel_label(a, "(a) Fixed $k$")

    # Annealing pilots, 4,000 steps at batch 512 (docs/experiment_3, finer grid).
    halving = {8: 851, 16: 1171, 32: 1362, 64: 1454, 128: 1438}
    long2 = {2: 295, 4: 506, 32: 1402, 64: 1501, 128: 1499}
    b.plot(list(halving), 100 * np.array(list(halving.values())) / SUB_V, "o-",
           color=TOPK_LIGHT, label="halving schedule")
    b.plot(list(long2), 100 * np.array(list(long2.values())) / SUB_V, "s-", color=TOPK,
           label="long $k = 2$ stage")
    init = pd.read_csv(RUNS / "exp5" / "alive_at_init_freq.csv")
    init = init[init.V == SUB_V].groupby("k").share.mean()
    kk = [2, 4, 8, 16, 32, 64, 128]
    b.plot(kk, [100 * init[k] for k in kk], ":", color=GREY, label="alive at initialisation")
    log2_ticks(b, kk)
    b.set_xlabel("starting $k_0$")
    b.set_ylabel("% of 2,000")
    b.set_ylim(-3, 103)
    b.legend(frameon=False, loc="lower right")
    panel_label(b, "(b) Annealing from $k_0$ to 1")
    fig.tight_layout(w_pad=2.5)
    save(fig, "topk_k")


# ----------------------------------------------------------------------------
# Figure 3: where features die during annealing


def shade_stages(ax, d, ymax):
    ks = d.k.values
    fr = d.frac.values
    start = 0
    for i in range(1, len(ks) + 1):
        if i == len(ks) or ks[i] != ks[start]:
            # Each eval logs the k in force after it, so the stage ends at the next eval.
            lo = fr[start]
            hi = fr[i] if i < len(ks) else 1.0
            if ks[start] in (2, 1):
                ax.axvspan(lo, hi, color="#f3f3f3" if ks[start] == 2 else "#e9e9e9", zorder=0,
                           lw=0)
            if hi - lo > 0.05:
                ax.text((lo + hi) / 2, ymax, f"$k={ks[start]}$", ha="center", va="bottom",
                        fontsize=6.5, color="#555555")
            start = i


def shade_fixed(ax):
    """Shade the k = 2 and k = 1 stages of the standard schedule and label all three."""
    ax.axvspan(0.4, 0.8, color="#f3f3f3", lw=0, zorder=0)
    ax.axvspan(0.8, 1.02, color="#e9e9e9", lw=0, zorder=0)
    hi = ax.get_ylim()[1]
    for x0, x1, t in [(0, 0.4, "$k_0 \\to 4$"), (0.4, 0.8, "$k = 2$"), (0.8, 1.0, "$k = 1$")]:
        ax.text((x0 + x1) / 2, hi, t, ha="center", va="bottom", fontsize=6.5, color="#555555")


def fig_anneal_dynamics():
    d = exp7()
    fig, axes = plt.subplots(1, 2, figsize=(W_FULL, 2.4), sharey=True)
    for ax, cell, title in [(axes[0], "steps4000_lr1", "(a) 4,000 steps"),
                            (axes[1], "steps128000_lr1", "(b) 128,000 steps")]:
        x = d[d.cell == cell]
        ax.set_ylim(0, 2000)
        shade_fixed(ax)
        ax.plot(x.frac, x.n_alive, "-", color=GREY, label="features alive")
        ax.plot(x.frac, x.n_top1, "--", color=TOPK_LIGHT,
                label="features that are some token's top feature")
        ax.plot(x.frac, x["n_recovered_cos0.9"], "-", color=TOPK, label="tokens recovered")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 2000)
        ax.set_xlabel("fraction of training")
        ax.set_title(title, loc="left", pad=11)
    axes[0].set_ylabel("count (of 2,000)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(w_pad=1.5, rect=(0, 0.06, 1, 1))
    save(fig, "anneal_dynamics")


# ----------------------------------------------------------------------------
# Figure 4: more training helps TopK annealing


def fig_topk_training():
    d = exp7()
    ends = d[d.step == d.steps].copy()
    ends["gao"] = ends.steps * ends.lr_factor
    fig, (a, b) = plt.subplots(1, 2, figsize=(W_FULL, 2.4))
    base = ends[ends.lr_factor == 1.0]
    for col, style, colour, lab in [("n_recovered_cos0.9", "s-", TOPK, "cosine > 0.9"),
                                    ("n_recovered_cos0.99", "o--", TOPK_DARK, "cosine > 0.99")]:
        m = base.groupby("gao")[col].mean()
        a.plot(m.index, 100 * m / SUB_V, style, color=colour, label=lab)
        a.plot(base.gao, 100 * base[col] / SUB_V, ".", color=colour, alpha=0.5, markersize=3)
    other = ends[ends.lr_factor != 1.0]
    a.plot(other.gao, 100 * other["n_recovered_cos0.9"] / SUB_V, "^", color="black",
           markersize=4.5, label="4,000 steps at ½× and 2× rate")
    log2_ticks(a, [2000, 4000, 16000, 64000], "x")
    a.set_xticks([2000, 4000, 8000, 16000, 32000, 64000, 128000])
    a.set_xticklabels(["2k", "4k", "8k", "16k", "32k", "64k", "128k"])
    a.set_xlabel("learning rate × steps (Gao-rate steps)")
    a.set_ylabel("tokens recovered (%)")
    a.set_ylim(45, 92)
    a.legend(frameon=False, loc="lower right")
    panel_label(a, "(a) Recovery")

    rows = []
    for cell, x in d[d.lr_factor == 1.0].groupby("cell"):
        steps = x.steps.iloc[0]
        e2 = x.iloc[(x.frac - 0.8).abs().argmin()]
        e = x.iloc[-1]
        rows.append((steps, e2.n_alive, e2.n_top1, e.n_alive))
    r = pd.DataFrame(rows, columns=["steps", "alive2", "top1", "alive1"]).groupby("steps").mean()
    b.plot(r.index, r.alive2, "o-", color=GREY, label="alive at end of $k = 2$")
    b.plot(r.index, r.top1, "s--", color=TOPK_LIGHT, label="top feature of some token\nat end of $k = 2$")
    b.plot(r.index, r.alive1, "D-", color=TOPK, label="alive at end ($k = 1$)")
    b.set_xscale("log", base=2)
    b.set_xticks([4000, 8000, 16000, 32000, 64000, 128000])
    b.set_xticklabels(["4k", "8k", "16k", "32k", "64k", "128k"])
    b.minorticks_off()
    b.set_xlabel("steps (at the Gao rate)")
    b.set_ylabel("features (of 2,000)")
    b.set_ylim(1250, 2000)
    b.legend(frameon=False, loc="lower right", fontsize=6.5)
    panel_label(b, "(b) Survival of the drop to $k = 1$")
    fig.tight_layout(w_pad=2.5)
    save(fig, "topk_training")


# ----------------------------------------------------------------------------
# Figure 5: separation margin predicts recovery (Experiment 6 with Experiment 2)


def fig_margin():
    # Margin bins and recovery from docs/experiment_6_hull_vertices.md.
    labels = ["≤ 0", "0–0.1", "0.1–0.2", "0.2–0.3", "0.3–0.4", "> 0.4"]
    tokens = [172, 1431, 7467, 13107, 13480, 14600]
    rec = [37, 64, 77, 91, 99.3, 100]
    fig, a = plt.subplots(figsize=(3.6, 2.2))
    bars = a.bar(range(6), rec, color=RELU, width=0.7)
    for i, (bar, n) in enumerate(zip(bars, tokens)):
        a.text(bar.get_x() + bar.get_width() / 2, rec[i] + 1.5, f"{n:,}", ha="center",
               va="bottom", fontsize=6, color="#555555")
    a.set_xticks(range(6))
    a.set_xticklabels(labels)
    a.set_xlabel("own-direction margin (fraction of the token's norm)")
    a.set_ylabel("recovered at full scale (%)")
    a.set_ylim(0, 112)
    a.grid(axis="x", visible=False)
    save(fig, "margin")


# ----------------------------------------------------------------------------
# Figure 6: the ReLU grid


def fig_relu_grid():
    d = exp4()
    fig, axes = plt.subplots(1, 3, figsize=(W_FULL, 2.3))
    final = d[d.step == d.groupby("cell").step.transform("max")]
    healthy = set(final[final.n_alive >= 1800].cell)
    for cell, x in d.groupby("cell"):
        lam = x.l1_coef.iloc[0]
        if x.lr_factor.iloc[0] == 4.0 or lam not in (0.03, 0.1, 0.3, 1.0):
            continue
        col = LAMBDA_COLOURS_STRONG[lam]
        axes[0].plot(x.gao_steps, x.mean_l0, "-", color=col, alpha=0.85, lw=1)
        axes[1].plot(x.gao_steps, 100 * x["n_recovered_cos0.9"] / SUB_V, "-", color=col,
                     alpha=0.85, lw=1)
    for lam, col in LAMBDA_COLOURS_STRONG.items():
        axes[0].plot([], [], "-", color=col, label=f"$\\lambda = {lam:g}$")
    for ax in axes[:2]:
        ax.set_xscale("log", base=2)
        ax.set_xticks([2 ** 9, 2 ** 11, 2 ** 13, 2 ** 15])
        ax.set_xticklabels(["512", "2k", "8k", "32k"])
        ax.minorticks_off()
        ax.set_xlabel("Gao-rate steps")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("active features per token (mean)")
    axes[0].legend(frameon=False, loc="lower left")
    panel_label(axes[0], "(a) Sparsification")
    axes[1].set_ylabel("tokens recovered (%)")
    axes[1].set_ylim(-3, 103)
    panel_label(axes[1], "(b) Recovery")

    a = axes[2]
    for lrf, marker, col in [(0.25, "o", "#a1dab4"), (1.0, "s", "#2c7fb8"), (4.0, "^", "#253494")]:
        x = final[final.lr_factor == lrf].groupby("l1_coef").n_alive.mean()
        a.plot(x.index, 100 * x / SUB_V, marker + "-", color=col, label=f"{lrf:g}× Gao rate")
    a.set_xscale("log")
    a.set_xticks([0.03, 0.1, 0.3, 1])
    a.set_xticklabels(["0.03", "0.1", "0.3", "1"])
    a.minorticks_off()
    a.set_xlabel("$\\lambda$")
    a.set_ylabel("features alive at end (%)")
    a.set_ylim(-3, 103)
    a.legend(frameon=False, loc="lower left")
    panel_label(a, "(c) Deaths from over-pressure")
    fig.tight_layout(w_pad=1.6)
    save(fig, "relu_grid")


# ----------------------------------------------------------------------------
# Figure 7: how many features each token uses


def fig_codes():
    # Active features per token at the end (docs/write_up_plan.md, from the best cells).
    runs = ["$\\lambda = 0.03$", "$\\lambda = 0.1$", "$\\lambda = 0.3$", "$\\lambda = 1$",
            "full table,\n$\\lambda = 0.3$"]
    shares = np.array([[90.1, 5.9, 2.7, 1.4], [88.5, 5.0, 2.1, 4.5], [78.4, 13.1, 5.0, 3.6],
                       [68.1, 28.2, 3.3, 0.5], [83.2, 8.3, 3.0, 5.4]])
    cols = [RELU, "#6baed6", "#c6dbef", "#e6eef7"]
    labs = ["1 feature", "2", "3", "4 or more"]
    fig, a = plt.subplots(figsize=(3.6, 2.1))
    left = np.zeros(len(runs))
    for j in range(4):
        a.barh(range(len(runs)), shares[:, j], left=left, color=cols[j], label=labs[j],
               edgecolor="white", lw=0.5)
        left += shares[:, j]
    a.set_yticks(range(len(runs)))
    a.set_yticklabels(runs)
    a.invert_yaxis()
    a.set_xlim(0, 100)
    a.set_xlabel("share of tokens (%)")
    a.grid(axis="y", visible=False)
    a.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.45, 1.2), fontsize=6.5,
             handlelength=1)
    save(fig, "codes")


# ----------------------------------------------------------------------------
# Figure 8: the penalty must be weak, but not zero


def time_to_sparsify(x, target=2.0):
    """Gao-rate steps at which the mean L0 first falls to the target (log interpolation)."""
    g = x.gao_steps.values
    l0 = x.mean_l0.values
    for i in range(1, len(g)):
        if l0[i] <= target < l0[i - 1]:
            t = (np.log(l0[i - 1]) - np.log(target)) / (np.log(l0[i - 1]) - np.log(l0[i]))
            return float(np.exp(np.log(g[i - 1]) + t * (np.log(g[i]) - np.log(g[i - 1]))))
    return None


def fig_lambda():
    """The trade-off in lambda: final recovery, and the training needed to sparsify."""
    d = exp4()
    final = d[d.step == d.groupby("cell").step.transform("max")]
    healthy = set(final[final.n_alive >= 1800].cell)
    fig, (b, a) = plt.subplots(1, 2, figsize=(W_FULL, 2.4))
    best = final.loc[final.groupby("l1_coef")["n_recovered_cos0.9"].idxmax()]
    b.plot(best.l1_coef, 100 * best["n_recovered_cos0.9"] / SUB_V, "s-", color=RELU,
           label="tokens recovered")
    b.plot(best.l1_coef, 100 * best.n_alive / SUB_V, "o--", color=GREY, label="features alive")
    b.set_xscale("log")
    b.set_xticks([0.03, 0.1, 0.3, 1])
    b.set_xticklabels(["0.03", "0.1", "0.3", "1"])
    b.minorticks_off()
    b.set_xlabel("$\\lambda$")
    b.set_ylabel("% of 2,000 (best run)")
    b.set_ylim(55, 102)
    b.legend(frameon=False, loc="lower left")
    panel_label(b, "(a) Recovery at the end")

    pts = []
    for cell, x in d.groupby("cell"):
        if cell not in healthy:
            continue
        t = time_to_sparsify(x)
        if t is not None:
            pts.append((x.l1_coef.iloc[0], t))
    pts = pd.DataFrame(pts, columns=["lam", "t"])
    a.plot(pts.lam, pts.t, "o", color=RELU, alpha=0.7, markersize=3.5)
    lam = np.array([0.03, 0.3])
    fit = pts[pts.lam <= 0.3]
    slope, icpt = np.polyfit(np.log(fit.lam), np.log(fit.t), 1)
    a.plot(lam, np.exp(icpt) * lam ** slope, "--", color=GREY,
           label=f"$\\propto \\lambda^{{{slope:.1f}}}$")
    a.set_xscale("log")
    a.set_yscale("log")
    a.set_xticks([0.03, 0.1, 0.3, 1])
    a.set_xticklabels(["0.03", "0.1", "0.3", "1"])
    a.set_yticks([1000, 4000, 16000])
    a.set_yticklabels(["1k", "4k", "16k"])
    a.minorticks_off()
    a.set_xlabel("$\\lambda$")
    a.set_ylabel("Gao-rate steps until 2 active\nfeatures per token (mean)")
    a.legend(frameon=False, loc="upper right")
    panel_label(a, "(b) Time to sparsify")
    fig.tight_layout(w_pad=2.5)
    save(fig, "lambda")
    return pts


# The two runs share their initialisation (seed 0), which the logs do not evaluate. These values
# come from evaluating build_model("l1", 768, 2000, seed=0) on the 2k subset with
# sae_raw.evaluation (the same code reproduces the logged step-0 values of the TopK grid).
INIT_2K_RELU = {"step": 0, "mean_l0": 987.2, "n_recovered_cos0.9": 0, "rel_loss": 1.319}


def fig_no_penalty():
    """lambda = 0.03 against lambda = 0, same run otherwise, from initialisation."""
    d = exp4()
    fig, axes = plt.subplots(1, 3, figsize=(W_FULL, 2.3))
    z = pd.read_csv(RUNS / "exp10" / "summary.csv")
    p = d[d.cell == "lam0.03_lr1_b256"]
    init = pd.DataFrame([INIT_2K_RELU])
    for x, col, lab in [(p, LAMBDA_COLOURS_STRONG[0.03], "$\\lambda = 0.03$"),
                        (z, "#d62728", "$\\lambda = 0$")]:
        x = pd.concat([init, x[list(INIT_2K_RELU)]], ignore_index=True)
        axes[0].plot(x.step, x.mean_l0, "o-", color=col, label=lab)
        axes[1].plot(x.step, 100 * x["n_recovered_cos0.9"] / SUB_V, "o-", color=col, label=lab)
        axes[2].plot(x.step, x.rel_loss, "o-", color=col, label=lab)
    for ax in axes:
        ax.set_xticks([0, 8192, 16384, 24576, 32768])
        ax.set_xticklabels(["0", "8k", "16k", "24k", "32k"])
        ax.set_xlim(-1000, 34000)
        ax.set_xlabel("steps (Gao rate, batch 256)")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("active features per token (mean)")
    axes[0].legend(frameon=False, loc="lower left")
    panel_label(axes[0], "(a) Sparsity")
    axes[1].set_ylabel("tokens recovered (%)")
    axes[1].set_ylim(-3, 103)
    panel_label(axes[1], "(b) Recovery")
    axes[2].set_yscale("log")
    axes[2].minorticks_off()
    axes[2].set_yticks([1e-4, 1e-3, 1e-2, 1e-1, 1])
    axes[2].set_yticklabels(["0.0001", "0.001", "0.01", "0.1", "1"])
    axes[2].set_ylabel("relative reconstruction error")
    panel_label(axes[2], "(c) Reconstruction")
    fig.tight_layout(w_pad=1.6)
    save(fig, "no_penalty")


# ----------------------------------------------------------------------------
# Figure 9: ReLU recovery does not depend on the vocabulary size


def fig_relu_sizes():
    d4 = exp4()
    fig, a = plt.subplots(figsize=(W_FULL * 0.62, 2.7))
    for lam in (1.0, 0.3, 0.1, 0.03):
        col = LAMBDA_COLOURS_STRONG[lam]
        x = d4[d4.cell == f"lam{lam:g}_lr1_b256"]
        m, ls = SIZE_STYLES[2000]
        a.plot(x.step / SUB_V, 100 * x["n_recovered_cos0.9"] / SUB_V, marker=m, ls=ls, color=col,
               markersize=3)
        for V in (5000, 10000, 20000):
            y = exp9(V)
            y = y[y.cell == f"V{V}_lam{lam:g}"]
            if y.empty:
                continue
            m, ls = SIZE_STYLES[V]
            a.plot(y.step / V, 100 * y["n_recovered_cos0.9"] / V, ls=ls, color=col)
    # Experiment 2 at full scale (lambda = 0.3, batch 1,024).
    full = []
    for name in ["analysis_step18500.json", "analysis_step55500.json", "analysis.json"]:
        r = json.load(open(RUNS / "exp2" / name))
        full.append(r["n_recovered_cos0.9"] / FULL_V * 100)
    a.plot(np.array([18500, 55500, 85100]) / FULL_V, full, "*", color=LAMBDA_COLOURS_STRONG[0.3],
           markersize=8, markeredgecolor="black", markeredgewidth=0.5, ls="none")
    for lam, col in LAMBDA_COLOURS_STRONG.items():
        a.plot([], [], "-", color=col, lw=3, label=f"$\\lambda = {lam:g}$")
    for V, (m, ls) in SIZE_STYLES.items():
        a.plot([], [], ls=ls, color="black", lw=1, label=f"$V = {V:,}$")
    a.plot([], [], "*", color=LAMBDA_COLOURS_STRONG[0.3], markersize=8, markeredgecolor="black",
           markeredgewidth=0.5, ls="none", label="$V = 50{,}257$")
    a.set_xscale("log", base=2)
    a.set_xticks([1 / 16, 1 / 4, 1, 4, 16])
    a.set_xticklabels(["1/16", "1/4", "1", "4", "16"])
    a.minorticks_off()
    a.set_xlabel("training steps per token of vocabulary (steps / $V$)")
    a.set_xlim(0.15, 22)
    a.set_ylabel("tokens recovered (%)")
    a.set_ylim(-3, 103)
    a.legend(frameon=False, loc="center left", bbox_to_anchor=(1.0, 0.5))
    save(fig, "relu_sizes")


# ----------------------------------------------------------------------------
# Figure 10: features alive at initialisation, and the starting k


def k_for_share(curve, target):
    """Interpolated k at which the share alive reaches the target."""
    k = curve.index.values.astype(float)
    s = curve.values
    i = np.argmax(s >= target)
    if i == 0:
        return k[0]
    return k[i - 1] + (target - s[i - 1]) / (s[i] - s[i - 1]) * (k[i] - k[i - 1])


def fig_init():
    f = pd.read_csv(RUNS / "exp5" / "alive_at_init_freq.csv")
    r = pd.read_csv(RUNS / "exp5" / "alive_at_init_random.csv")
    fig, (a, b) = plt.subplots(1, 2, figsize=(W_FULL, 2.4))
    cmap = plt.get_cmap("Oranges")
    sizes = [2000, 5000, 10000, 20000, 50257]
    for i, V in enumerate(sizes):
        c = f[f.V == V].groupby("k").share.mean()
        a.plot(c.index, 100 * c, "-", color=cmap(0.35 + 0.15 * i), label=f"$V = {V:,}$")
    a.axhline(93, color=GREY, ls=":", lw=1)
    a.set_xscale("log", base=2)
    a.set_xticks([1, 4, 16, 64, 256])
    a.set_xticklabels(["1", "4", "16", "64", "256"])
    a.minorticks_off()
    a.set_xlabel("$k$")
    a.set_ylabel("features alive at initialisation (%)")
    a.legend(frameon=False, loc="lower right")
    panel_label(a, "(a) Share alive at initialisation")

    for data, col, marker, lab, (c0, p) in [
            (f, TOPK, "o", "most frequent tokens", (1.37, 0.405)),
            (r, GREY, "s", "random tokens", (2.90, 0.345))]:
        Vs, ks = [], []
        for V, x in data.groupby("V"):
            Vs.append(V)
            ks.append(k_for_share(x.groupby("k").share.mean(), 0.93))
        b.plot(Vs, ks, marker, color=col, markersize=3.5, label=lab)
        vv = np.geomspace(2000, 50257, 50)
        b.plot(vv, c0 * vv ** p, "-", color=col, lw=1,
               label=f"${c0:.2f}\\,V^{{{p}}}$")
    b.set_xscale("log")
    b.set_yscale("log")
    b.set_xticks([2000, 5000, 10000, 20000, 50000])
    b.set_xticklabels(["2k", "5k", "10k", "20k", "50k"])
    b.set_yticks([32, 64, 128])
    b.set_yticklabels(["32", "64", "128"])
    b.minorticks_off()
    b.set_xlabel("vocabulary size $V$")
    b.set_ylabel("$k$ for 93% alive")
    b.legend(frameon=False, loc="upper left")
    panel_label(b, "(b) The starting $k_0$ for 93% alive")
    fig.tight_layout(w_pad=2.5)
    save(fig, "init")


# ----------------------------------------------------------------------------
# Figure 11: TopK annealing is the same at every size


def fig_topk_sizes():
    fig, (a, b) = plt.subplots(1, 2, figsize=(W_FULL, 2.4), sharex=True)
    d7 = exp7()
    curves = {2000: d7[d7.cell == "steps4000_lr1"]}
    for V in (5000, 10000, 20000):
        curves[V] = exp8(V)
    for V, x in curves.items():
        m, ls = SIZE_STYLES[V]
        a.plot(x.frac, 100 * x.n_alive / V, ls=ls, color=SIZE_COLOURS[V], label=f"$V = {V:,}$")
        b.plot(x.frac, 100 * x["n_recovered_cos0.9"] / V, ls=ls, color=SIZE_COLOURS[V])
    e3 = pd.read_csv(RUNS / "exp3" / "eval_log.csv")
    a.plot(e3.step / e3.step.max(), 100 * e3.n_alive / FULL_V, "-", color="black", lw=1,
           label="$V = 50{,}257$")
    r3 = json.load(open(RUNS / "exp3" / "analysis.json"))
    b.plot([1.0], [100 * r3["n_recovered_cos0.9"] / FULL_V], "*", color="black", markersize=8,
           ls="none")
    for ax in (a, b):
        ax.set_xlim(0, 1.02)
        ax.set_xlabel("fraction of training")
    a.set_ylabel("features alive (%)")
    a.set_ylim(60, 101)
    a.legend(frameon=False, loc="lower left")
    panel_label(a, "(a) Features alive")
    b.set_ylabel("tokens recovered (%)")
    b.set_ylim(-3, 100)
    panel_label(b, "(b) Recovery")
    for ax in (a, b):
        shade_fixed(ax)
        ax.set_title(ax.get_title(loc="left"), loc="left", pad=11)
    fig.tight_layout(w_pad=2.5)
    save(fig, "topk_sizes")


# ----------------------------------------------------------------------------
# Appendix figure: one-hot codes and the convex hull


def fig_hull():
    rng = np.random.default_rng(3)
    ang = np.sort(rng.uniform(0, 2 * np.pi, 9))
    pts = np.c_[np.cos(ang) * rng.uniform(0.8, 1.0, 9), np.sin(ang) * rng.uniform(0.8, 1.0, 9)]
    inner = np.array([0.15, -0.1])
    from scipy.spatial import ConvexHull
    allp = np.vstack([pts, inner])
    hull = ConvexHull(allp)
    fig, (a, b) = plt.subplots(1, 2, figsize=(5.0, 2.3))
    for ax in (a, b):
        for s in hull.simplices:
            ax.plot(allp[s, 0], allp[s, 1], ":", color=GREY, lw=1)
        ax.plot(allp[:, 0], allp[:, 1], "o", color="black", markersize=3.5)
        ax.set_aspect("equal")
        ax.set_xlim(-1.35, 1.35)
        ax.set_ylim(-1.25, 1.5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(False)
    def margin(v):
        w = allp[v] / np.linalg.norm(allp[v])
        return allp[v] @ w - np.delete(allp, v, axis=0).dot(w).max()
    v = max(hull.vertices, key=margin)
    xv = allp[v]
    w = xv / np.linalg.norm(xv)
    others = np.delete(allp, v, axis=0) @ w
    cut = (others.max() + xv @ w) / 2
    perp = np.array([-w[1], w[0]])
    line = np.array([cut * w - 1.4 * perp, cut * w + 1.4 * perp])
    a.plot(line[:, 0], line[:, 1], "-", color=RELU, lw=1.2)
    a.plot(*xv, "o", color=RELU, markersize=6, mfc="none", mew=1.4)
    a.text(*(xv + 0.13 * w), "$x_j$", color=RELU, ha="center", va="center")
    a.set_title("(a) A vertex can be cut off\nby a hyperplane", loc="left")
    b.plot(*inner, "o", color="#d62728", markersize=6, mfc="none", mew=1.4)
    b.set_title("(b) An interior point cannot", loc="left")
    fig.tight_layout()
    save(fig, "hull")


def main():
    fig_headline()
    fig_topk_k()
    fig_anneal_dynamics()
    fig_topk_training()
    fig_margin()
    fig_relu_grid()
    fig_codes()
    pts = fig_lambda()
    fig_no_penalty()
    fig_relu_sizes()
    fig_init()
    fig_topk_sizes()
    fig_hull()
    print("time to sparsify (Gao-rate steps to mean L0 = 2):")
    print(pts.groupby("lam").t.describe()[["count", "min", "50%", "max"]].round(0).to_string())


if __name__ == "__main__":
    main()
