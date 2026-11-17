"""
The four figures of the thesis. Reads only result files; draws no number
that is not in one.

run (from code/):  python scripts/make_figures.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
RES = HERE.parents[1] / "results"
RANGE = RES / "range_vs_degree"
ROT = RES / "spread_and_rotation"
EXP4 = RES / "grouping_time"
OUT = HERE.parents[1] / "thesis" / "figures"
OUT.mkdir(exist_ok=True)

INK, ACCENT, SOFT, WARN, GOOD = "#171B24", "#24406E", "#8A93A6", "#B5442E", "#2E7D4F"
plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": SOFT, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK, "axes.spines.top": False,
    "axes.spines.right": False, "savefig.bbox": "tight", "savefig.dpi": 200,
    "legend.frameon": False,
})


def read(p):
    return pd.read_csv(p, comment="#")


def fig_blowup():
    C = np.polynomial.chebyshev
    t = np.linspace(-2, 2, 2001)
    relu = np.maximum(0, t)
    fig, ax = plt.subplots(figsize=(5.2, 2.6))
    ax.axvspan(-1, 1, color="#EEF1F6", lw=0)
    for d, col in zip((7, 15, 31, 63), ("#9AA8C2", "#5E77A3", ACCENT, INK)):
        c = C.Chebyshev.interpolate(lambda x: np.maximum(0, x), d,
                                    domain=[-1, 1]).coef
        err = np.abs(C.chebval(t, c) - relu)
        ax.semilogy(t, np.maximum(err, 1e-6), color=col, lw=1.2,
                    label=f"degree {d}")
    ax.set_xlabel("normalized input  t = (x - center) / half-width")
    ax.set_ylabel("|polynomial - ReLU|")
    ax.text(0, 3e8, "fitted interval", ha="center", color=SOFT, fontsize=8)
    ax.set_ylim(1e-6, 1e31)
    ax.legend(loc="upper center", fontsize=7.5, ncol=2)
    fig.savefig(OUT / "fig_blowup.pdf")
    plt.close(fig)


def fig_range_degree():
    g = read(RANGE / "exp3_range_grid_train.csv")
    names = {"cifar10_resnet20": "ResNet-20, CIFAR-10",
             "cifar10_resnet56": "ResNet-56, CIFAR-10",
             "cifar100_resnet20": "ResNet-20, CIFAR-100"}
    style = {"minmax": (ACCENT, "-", "observed range"),
             "sigma6": (SOFT, "--", "mean \u00b1 6 std"),
             "sigma3": (WARN, "-", "mean \u00b1 3 std"),
             "pct0.1": ("#C9A227", ":", "0.1 to 99.9 percentile")}
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4), sharey=False)
    for ax, (net, title) in zip(axes, names.items()):
        s = g[g.target == net]
        base = s.baseline_top1.iloc[0]
        ax.axhline(base, color=GOOD, lw=0.8, ls="--")
        for rule, (col, ls, lab) in style.items():
            r = s[s.rule == rule].sort_values("degree")
            ax.plot(r.degree, r.top1, color=col, ls=ls, marker="o", ms=3,
                    lw=1.2, label=lab)
        ax.set_xscale("log", base=2)
        ax.set_xticks([7, 15, 31, 63], ["7", "15", "31", "63"])
        ax.set_title(title, fontsize=8.5, color=INK)
        ax.set_xlabel("degree")
        ax.set_ylim(0, 100)
    axes[0].set_ylabel("top-1 accuracy (%)")
    h, l = axes[0].get_legend_handles_labels()
    h.append(plt.Line2D([], [], color=GOOD, lw=0.8, ls="--"))
    l.append("unmodified network")
    fig.legend(h, l, loc="lower center", ncol=5, fontsize=7.5,
               bbox_to_anchor=(0.5, -0.07))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(OUT / "fig_range_degree.pdf")
    plt.close(fig)


def fig_rotation():
    r = read(ROT / "exp2_rotation.csv")
    r = r[r.delta_model == "B"]
    nets = {"imagenet_mobilenetv2": "MobileNetV2\n(ImageNet)",
            "imagenet_mnasnet1_0": "MNASNet\n(ImageNet)",
            "cifar10_mobilenetv2_x1_0": "MobileNetV2\n(CIFAR-10)"}
    fig, ax = plt.subplots(figsize=(4.6, 2.3))
    w = 0.25
    for i, (G, col) in enumerate(zip((2, 4, 8), ("#9AA8C2", "#5E77A3", ACCENT))):
        vals = [100 * r[(r.net == n) & (r.G == G)].rotates.mean() for n in nets]
        x = np.arange(len(nets)) + (i - 1) * w
        ax.bar(x, vals, w, color=col, label=f"G = {G}")
        for xi, v in zip(x, vals):
            ax.text(xi, v + 1.2, f"{v:.1f}", ha="center", fontsize=6.5)
    ax.axhline(50, color=WARN, lw=0.8, ls="--")
    ax.text(2.45, 51.5, "unrelated channels, G = 2", color=WARN, fontsize=7, ha="right")
    ax.set_xticks(range(len(nets)), list(nets.values()), fontsize=7.5)
    ax.set_ylabel("worst group changes (%)")
    ax.set_ylim(0, 75)
    ax.legend(fontsize=7, loc="upper left", ncol=3)
    fig.savefig(OUT / "fig_rotation.pdf")
    plt.close(fig)


def fig_time():
    """Left: time per degree, mean and standard deviation over the five
    repeated runs (12-level chain) and the single run on the 24-level chain.
    Right: saving from grouping, mean and standard deviation over the same
    five runs, next to the saving in the multiplication count."""
    rep = EXP4 / "repeat_runs"
    m = read(rep / "summary_per_degree.csv")
    sv = read(rep / "summary_time_saving.csv")
    t = read(EXP4 / "exp4_degree_timing.csv")
    tg = read(EXP4 / "exp4_time_gain.csv")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 2.6))
    ms, sd = 1000 * m.seconds_mean, 1000 * m.seconds_std
    a1.fill_between(m.degree, ms - sd, ms + sd, color="#C9D3E6", lw=0)
    a1.plot(m.degree, ms, color=ACCENT, lw=1.1,
            label="12-level chain, mean of 5 runs (band: 1 std)")
    s24 = t[t.chain_depth == t.chain_depth.max()]
    a1.plot(s24.degree, 1000 * s24.seconds_median, color=SOFT, lw=1.0, ls="--",
            label=f"{int(t.chain_depth.max())}-level chain, one run")
    ax2 = a1.twinx()
    ax2.step(m.degree, m.levels_used, where="post", color=WARN, lw=0.9)
    ax2.set_ylabel("levels used", color=WARN)
    ax2.tick_params(axis="y", colors=WARN)
    ax2.spines["right"].set_visible(True)
    ax2.set_ylim(0, 8)
    a1.set_xlabel("polynomial degree")
    a1.set_ylabel("ms per ciphertext")
    a1.set_ylim(0, None)
    a1.legend(fontsize=6.5, loc="upper left")
    mult = tg[(tg.chain_depth == tg.chain_depth.min())
              & (tg.scope == "activations")].set_index("budget_B").mult_saving_pct_model
    sv = sv.sort_values("budget_B")
    x = np.arange(len(sv))
    a2.bar(x - 0.2, [mult[b] for b in sv.budget_B], 0.4, color="#9AA8C2",
           label="multiplications (count)")
    a2.bar(x + 0.2, sv["mean"], 0.4, yerr=sv["std"], color=ACCENT,
           error_kw={"elinewidth": 0.8, "capsize": 2, "ecolor": INK},
           label="activation time (mean of 5 runs, 1 std)")
    a2.set_xticks(x, [f"{b:g}" for b in sv.budget_B], fontsize=7.5)
    a2.set_xlabel("error budget B  (tight to loose)")
    a2.set_ylabel("saving against layerwise (%)")
    a2.set_ylim(0, 95)
    a2.legend(fontsize=6.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "fig_time.pdf")
    plt.close(fig)


if __name__ == "__main__":
    fig_blowup()
    fig_range_degree()
    fig_rotation()
    if (EXP4 / "repeat_runs" / "summary_time_saving.csv").exists():
        fig_time()
    print("figures in", OUT)
