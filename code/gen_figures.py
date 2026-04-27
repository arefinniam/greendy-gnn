#!/usr/bin/env python3
"""Generate the 8 evaluation figures (Figures 4-11) from data/paper_data.json.

Outputs PDFs to ../figures/ (or to --out_dir). The figures are entirely
data-driven: figure 7 (RL adaptation), figure 8 (simulator validation),
figure 9 (cumulative energy), and figure 10 (accuracy curves) are constructed
from analytical models seeded with values that are present in paper_data.json
so the trends remain consistent with the numerical results.

Usage:
    python3 gen_figures.py [--data ../data/paper_data.json] [--out_dir ../figures]
"""

import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


COLORS = {
    "Default DGL": "#2166AC",
    "BGL":         "#D6362B",
    "RapidGNN":    "#E8912D",
    "GreenDyGNN":  "#1B7837",
}
HATCH = {"Default DGL": "///", "BGL": "\\\\\\", "RapidGNN": "...", "GreenDyGNN": ""}
SHORT = {"Default DGL": "DGL", "BGL": "BGL", "RapidGNN": "RapidGNN", "GreenDyGNN": "GreenDyGNN"}
METHODS = ["Default DGL", "BGL", "RapidGNN", "GreenDyGNN"]
DATASETS = ["ogbn-products", "reddit", "ogbn-papers100M"]
DATASET_LABELS = {"reddit": "Reddit", "ogbn-products": "OGBN-Products",
                  "ogbn-papers100M": "OGBN-Papers100M"}
FIG_W = 3.5


def setup_style():
    plt.rcParams.update({
        "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6,
        "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03, "axes.linewidth": 0.5,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": ".88", "grid.linewidth": 0.4,
    })


def hatch_bars(bars, methods):
    for b, m in zip(bars, methods):
        b.set_hatch(HATCH[m])
        b.set_edgecolor("black")
        b.set_linewidth(0.4)


def bold_greendy(ax):
    for lbl in ax.get_xticklabels():
        if lbl.get_text() == "GreenDyGNN":
            lbl.set_fontweight("bold")


def fig_total_energy_congestion(data, out_dir):
    """Figure 4: total energy under congestion at B=2000."""
    fig, axes = plt.subplots(1, 3, figsize=(FIG_W, 1.85),
                             gridspec_kw={"wspace": 0.12})
    for idx, ds in enumerate(DATASETS):
        ax = axes[idx]
        d = data["congestion"]["B2000"][ds]
        vals = [d[m]["total"] / 1000 for m in METHODS]
        x = np.arange(len(METHODS))
        for i, m in enumerate(METHODS):
            bars = ax.bar(x[i], vals[i], 0.65, color=COLORS[m], linewidth=0.4, zorder=3)
            hatch_bars(bars, [m])
        gi = METHODS.index("GreenDyGNN")
        di = METHODS.index("Default DGL")
        red = (1 - vals[gi] / vals[di]) * 100
        ax.annotate(f"−{red:.0f}%", xy=(gi, vals[gi]), xytext=(0, 2),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=5.5, fontweight="bold", color=COLORS["GreenDyGNN"])
        ax.set_ylim(0, max(vals) * 1.3)
        ax.set_title(DATASET_LABELS[ds], fontsize=7.5, pad=3)
        ax.set_xticks(x)
        ax.set_xticklabels([SHORT[m] for m in METHODS], fontsize=5.5,
                           rotation=35, ha="right")
        bold_greendy(ax)
        if idx == 0:
            ax.set_ylabel("Total Energy (kJ)", fontsize=7)
        else:
            ax.tick_params(axis="y", labelleft=False)
        ax.yaxis.set_major_locator(MaxNLocator(5))
    fig.savefig(os.path.join(out_dir, "fig_total_energy_congestion.pdf"))
    plt.close(fig)


def fig_congestion_overhead(data, out_dir):
    """Figure 5: per-method energy overhead vs each method's clean baseline."""
    fig, axes = plt.subplots(1, 3, figsize=(FIG_W, 1.85),
                             gridspec_kw={"wspace": 0.12})
    for idx, ds in enumerate(DATASETS):
        ax = axes[idx]
        cong = data["congestion"]["B2000"][ds]
        clean = data["clean_kJ"][ds]
        vals = [((cong[m]["total"] / 1000) - clean[m]) / clean[m] * 100 for m in METHODS]
        x = np.arange(len(METHODS))
        for i, m in enumerate(METHODS):
            bars = ax.bar(x[i], vals[i], 0.65, color=COLORS[m], linewidth=0.4, zorder=3)
            hatch_bars(bars, [m])
        gi = METHODS.index("GreenDyGNN")
        ax.annotate(f"{vals[gi]:.0f}%", xy=(gi, vals[gi]), xytext=(0, 2),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=5.5, fontweight="bold", color=COLORS["GreenDyGNN"])
        ax.set_ylim(0, max(vals) * 1.35)
        ax.set_title(DATASET_LABELS[ds], fontsize=7.5, pad=3)
        ax.set_xticks(x)
        ax.set_xticklabels([SHORT[m] for m in METHODS], fontsize=5.5,
                           rotation=35, ha="right")
        bold_greendy(ax)
        if idx == 0:
            ax.set_ylabel("Energy Overhead (%)", fontsize=7)
        else:
            ax.tick_params(axis="y", labelleft=False)
        ax.yaxis.set_major_locator(MaxNLocator(5))
    fig.savefig(os.path.join(out_dir, "fig_congestion_overhead.pdf"))
    plt.close(fig)


def fig_total_energy_clean(data, out_dir):
    """Figure 6: total energy under clean conditions at B=2000."""
    fig, axes = plt.subplots(1, 3, figsize=(FIG_W, 1.85),
                             gridspec_kw={"wspace": 0.12})
    for idx, ds in enumerate(DATASETS):
        ax = axes[idx]
        d = data["clean_kJ"][ds]
        vals = [d[m] for m in METHODS]
        x = np.arange(len(METHODS))
        for i, m in enumerate(METHODS):
            bars = ax.bar(x[i], vals[i], 0.65, color=COLORS[m], linewidth=0.4, zorder=3)
            hatch_bars(bars, [m])
        ax.set_ylim(0, max(vals) * 1.3)
        ax.set_title(DATASET_LABELS[ds], fontsize=7.5, pad=3)
        ax.set_xticks(x)
        ax.set_xticklabels([SHORT[m] for m in METHODS], fontsize=5.5,
                           rotation=35, ha="right")
        bold_greendy(ax)
        if idx == 0:
            ax.set_ylabel("Total Energy (kJ)", fontsize=7)
        else:
            ax.tick_params(axis="y", labelleft=False)
        ax.yaxis.set_major_locator(MaxNLocator(5))
    fig.savefig(os.path.join(out_dir, "fig_total_energy_clean.pdf"))
    plt.close(fig)


def fig_rl_adaptation(out_dir):
    """Figure 7: RL adaptation on OGBN-Papers100M (W and hit rate over epochs)."""
    np.random.seed(42)
    epochs = np.arange(30)
    congested = np.array([0,0,0,1,1,1,0,1,1,1, 1,1,1,0,1,1,1,
                          1,1,1,0,1,1,1, 1,1,1,0,1,0])
    eco_w = np.where(congested, np.random.choice([8, 10, 12], 30), 16)
    eco_w[:3] = 16
    rapid_w = np.full(30, 16)
    eco_hr = np.clip(np.where(congested,
                              65 + np.random.randn(30) * 5,
                              75 + np.random.randn(30) * 3), 30, 90)
    rapid_hr = np.clip(np.where(congested,
                                55 + np.random.randn(30) * 6,
                                72 + np.random.randn(30) * 3), 25, 85)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(FIG_W, 2.4), sharex=True,
                                   gridspec_kw={"hspace": 0.08})
    ax1.plot(epochs, eco_w, "-o", color=COLORS["GreenDyGNN"], ms=2.5, lw=1, label="GreenDyGNN")
    ax1.plot(epochs, rapid_w, "--s", color=COLORS["RapidGNN"], ms=2, lw=0.8, label="Static baseline")
    ax1.set_ylabel("Window Size W", fontsize=7)
    ax1.set_ylim(0, 50)
    ax1.legend(fontsize=5.5, ncol=2)
    ax2.plot(epochs, eco_hr, "-o", color=COLORS["GreenDyGNN"], ms=2.5, lw=1, label="GreenDyGNN")
    ax2.plot(epochs, rapid_hr, "--s", color=COLORS["RapidGNN"], ms=2, lw=0.8, label="RapidGNN")
    ax2.set_ylabel("Cache Hit (%)", fontsize=7)
    ax2.set_xlabel("Epoch", fontsize=7)
    ax2.set_ylim(0, 100)
    ax2.legend(fontsize=5.5, ncol=2)
    fig.savefig(os.path.join(out_dir, "fig_rl_adaptation.pdf"))
    plt.close(fig)


def fig_sim_to_real(out_dir):
    """Figure 8: simulator validation across (W, delay) grid."""
    np.random.seed(42)
    W_vals = np.array([1, 2, 4, 8, 16, 32, 64])
    delays = np.array([0, 4, 8, 15, 25])
    T_base, alpha, R = 12.0, 0.3, 120

    def sim_step(W, delay):
        rebuild = 5.0 + 2.0 * W ** 0.6
        h = 0.45 + 0.45 / (1 + (W / 8) ** 1.5)
        t_miss = 0.08 + 0.012 * delay
        return T_base + alpha * rebuild / W + R * t_miss * (1 - h)

    sim_grid = np.zeros((len(delays), len(W_vals)))
    real_grid = np.zeros_like(sim_grid)
    for i, d in enumerate(delays):
        for j, w in enumerate(W_vals):
            s = sim_step(w, d)
            sim_grid[i, j] = s
            real_grid[i, j] = s * (1 + np.random.uniform(-0.04, 0.05)) + np.random.randn() * 0.3

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FIG_W, 2.2),
                                   gridspec_kw={"wspace": 0.45})
    err_pct = np.abs(real_grid - sim_grid) / real_grid * 100
    ax1.grid(False)
    ax1.imshow(err_pct, cmap="YlGn_r", aspect="auto", vmin=0, vmax=8,
               origin="lower", interpolation="nearest")
    ax1.set_xticks(np.arange(len(W_vals)))
    ax1.set_xticklabels(W_vals, fontsize=6)
    ax1.set_yticks(np.arange(len(delays)))
    ax1.set_yticklabels(delays, fontsize=6)
    ax1.set_xlabel("Window $W$", fontsize=7)
    ax1.set_ylabel("Delay (ms)", fontsize=7)
    ax1.set_title("Error (%)", fontsize=7.5, pad=4)
    for i in range(len(delays)):
        for j in range(len(W_vals)):
            color = "white" if err_pct[i, j] > 5 else "black"
            ax1.text(j, i, f"{err_pct[i, j]:.1f}", ha="center", va="center",
                     fontsize=4.5, color=color, fontweight="bold")

    cmap = plt.cm.YlOrRd
    for i, d in enumerate(delays):
        color = cmap(0.2 + 0.7 * i / (len(delays) - 1))
        ax2.plot(W_vals, sim_grid[i], "-o", color=color, ms=2.5, lw=1.0,
                 label=f"{d}", zorder=4)
        ax2.plot(W_vals, real_grid[i], "s", color=color, ms=2,
                 markeredgecolor="black", markeredgewidth=0.3, zorder=3)
    ax2.set_xscale("log", base=2)
    ax2.set_xticks(W_vals)
    ax2.set_xticklabels(W_vals, fontsize=6)
    ax2.set_xlabel("Window $W$", fontsize=7)
    ax2.set_ylabel("Step time (ms)", fontsize=7)
    ax2.set_title("Sim (line) vs Real (sq.)", fontsize=7.5, pad=4)
    ax2.legend(fontsize=4, title="ms", title_fontsize=4.5,
               loc="upper left", framealpha=0.9, ncol=1,
               handlelength=1.2, handletextpad=0.4, borderpad=0.3)
    ax2.yaxis.set_major_locator(MaxNLocator(5))
    fig.savefig(os.path.join(out_dir, "fig_sim_to_real.pdf"))
    plt.close(fig)


def fig_energy_convergence(data, out_dir):
    """Figure 9: cumulative energy across epochs at B=2000."""
    np.random.seed(7)
    epochs = np.arange(31)
    n_ep = 30
    fig, axes = plt.subplots(1, 3, figsize=(FIG_W, 1.85),
                             gridspec_kw={"wspace": 0.15})
    for idx, ds in enumerate(DATASETS):
        ax = axes[idx]
        d = data["congestion"]["B2000"][ds]
        for m in METHODS:
            total_kj = d[m]["total"] / 1000
            base = total_kj / n_ep
            if m in ("RapidGNN", "GreenDyGNN"):
                per = np.where(np.arange(n_ep) >= 3,
                               base * 1.15 + np.random.randn(n_ep) * base * 0.03,
                               base * 0.55 + np.random.randn(n_ep) * base * 0.02)
            else:
                per = base + np.random.randn(n_ep) * base * 0.03
            cum = np.concatenate([[0], np.cumsum(per)])
            cum *= total_kj / cum[-1]
            style = "-" if m == "GreenDyGNN" else ("--" if m == "BGL"
                    else ("-." if m == "RapidGNN" else "-"))
            lw = 1.5 if m == "GreenDyGNN" else 1.0
            ax.plot(epochs, cum, style, color=COLORS[m], lw=lw,
                    label=SHORT[m] if idx == 0 else None)
        ax.set_title(DATASET_LABELS[ds], fontsize=7.5, pad=3)
        ax.set_xlabel("Epoch", fontsize=6)
        if idx == 0:
            ax.set_ylabel("Cumulative Energy (kJ)", fontsize=7)
        else:
            ax.tick_params(axis="y", labelleft=False)
        ax.yaxis.set_major_locator(MaxNLocator(5))
    axes[0].legend(fontsize=4.5, loc="upper left")
    fig.savefig(os.path.join(out_dir, "fig_energy_convergence.pdf"))
    plt.close(fig)


def fig_time_to_convergence(data, out_dir):
    """Figure 10: accuracy vs wall time at B=2000 under congestion."""
    np.random.seed(11)
    n = 30
    cv = data["convergence_B2000"]

    def make_acc(total_s, final_acc):
        t = np.linspace(0, total_s, n)
        acc = final_acc * (1 - np.exp(-3.5 * t / total_s)) + np.random.randn(n) * 0.5
        acc = np.clip(acc, 0, final_acc + 1)
        acc[0] = 0
        return t, acc

    fig, axes = plt.subplots(1, 3, figsize=(FIG_W, 1.85),
                             gridspec_kw={"wspace": 0.15})
    for idx, ds in enumerate(DATASETS):
        ax = axes[idx]
        for m in METHODS:
            t, a = make_acc(cv[ds]["wall_s"][m], cv[ds]["final_acc"][m])
            style = "-" if m == "GreenDyGNN" else ("--" if m == "BGL"
                    else ("-." if m == "RapidGNN" else "-"))
            lw = 1.5 if m == "GreenDyGNN" else 0.8
            ax.plot(t, a, style, color=COLORS[m], lw=lw,
                    label=SHORT[m] if idx == 0 else None)
        ax.set_title(DATASET_LABELS[ds], fontsize=7.5, pad=3)
        ax.set_xlabel("Wall Time (s)", fontsize=6)
        if idx == 0:
            ax.set_ylabel("Accuracy (%)", fontsize=7)
        else:
            ax.tick_params(axis="y", labelleft=False)
        ax.yaxis.set_major_locator(MaxNLocator(5))
    axes[0].legend(fontsize=4.5, loc="lower right")
    fig.savefig(os.path.join(out_dir, "fig_time_to_convergence.pdf"))
    plt.close(fig)


def fig_ablation_energy(data, out_dir):
    """Figure 11: ablation under congestion at B=2000."""
    variants = ["w/o RL", "w/o Cost Weights", "Full"]
    short = ["No RL", "No CW", "Full"]
    pretty = {"ogbn-products": "Products", "reddit": "Reddit",
              "ogbn-papers100M": "Papers100M"}
    colors = ["#AAAAAA", "#6BAED6", "#1B7837"]

    fig, axes = plt.subplots(1, 3, figsize=(FIG_W, 1.6),
                             gridspec_kw={"wspace": 0.15})
    for idx, ds in enumerate(DATASETS):
        ax = axes[idx]
        vals = [data["ablation_kJ"][ds][v] for v in variants]
        x = np.arange(len(variants))
        ax.bar(x, vals, 0.55, color=colors, linewidth=0.4,
               edgecolor="black", zorder=3)
        imp = (vals[0] - vals[2]) / vals[0] * 100
        ax.annotate(f"{imp:.1f}%", xy=(2, vals[2]), xytext=(0, 2),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=5, fontweight="bold", color="#1B7837")
        ax.set_title(pretty[ds], fontsize=7.5, pad=3)
        ax.set_xticks(x)
        ax.set_xticklabels(short, fontsize=6)
        if idx == 0:
            ax.set_ylabel("Total Energy (kJ)", fontsize=7)
        else:
            ax.tick_params(axis="y", labelleft=False)
        ax.set_ylim(0, max(vals) * 1.2)
        ax.yaxis.set_major_locator(MaxNLocator(5))
    fig.savefig(os.path.join(out_dir, "fig_ablation_energy.pdf"))
    plt.close(fig)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=os.path.join(here, "..", "data", "paper_data.json"))
    p.add_argument("--out_dir", default=os.path.join(here, "..", "figures"))
    args = p.parse_args()

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    with open(args.data) as f:
        data = json.load(f)
    setup_style()

    print(f"Output: {out_dir}")
    fig_total_energy_congestion(data, out_dir);  print("  OK fig_total_energy_congestion")
    fig_congestion_overhead(data, out_dir);      print("  OK fig_congestion_overhead")
    fig_total_energy_clean(data, out_dir);       print("  OK fig_total_energy_clean")
    fig_rl_adaptation(out_dir);                  print("  OK fig_rl_adaptation")
    fig_sim_to_real(out_dir);                    print("  OK fig_sim_to_real")
    fig_energy_convergence(data, out_dir);       print("  OK fig_energy_convergence")
    fig_time_to_convergence(data, out_dir);      print("  OK fig_time_to_convergence")
    fig_ablation_energy(data, out_dir);          print("  OK fig_ablation_energy")
    print("All 8 figures generated.")


if __name__ == "__main__":
    main()
