#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
mpl.rcParams["svg.fonttype"] = "none"
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

THRESHOLD_ORDER = ["-50%", "-25%", "Original", "+25%", "+50%"]
SITE_ORDER = ["Shiode", "Yamanokami"]
MODEL_ORDER = ["GLM", "LSTM", "rSSM", "PGNN"]
COLORS = {"GLM": "#A6A6A6", "LSTM": "#6BAED6", "rSSM": "#FDAE6B", "PGNN": "#74C476"}
EDGES = {"GLM": "#777777", "LSTM": "#3182BD", "rSSM": "#E6550D", "PGNN": "#31A354"}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=Path, default=Path(__file__).resolve().parent / "data" / "FigureC2_event_level_ratios.csv")
    p.add_argument("--outdir", type=Path, default=Path.cwd() / "revision_figures" / "FigureC2_threshold_sensitivity_v2")
    p.add_argument("--dpi", type=int, default=500)
    p.add_argument("--show-points", action="store_true")
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    args.outdir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.2), sharey=True)
    group_centers = np.arange(len(THRESHOLD_ORDER), dtype=float)
    width = 0.16
    offsets = np.array([-1.5, -0.5, 0.5, 1.5]) * width

    for ax, site, letter in zip(axes, SITE_ORDER, ["(a)", "(b)"]):
        sdf = df[df["site"] == site].copy()
        counts = sdf.groupby("threshold_setting")["event_id"].nunique().reindex(THRESHOLD_ORDER)
        for mi, model in enumerate(MODEL_ORDER):
            for ti, th in enumerate(THRESHOLD_ORDER):
                vals = sdf[(sdf["model"] == model) & (sdf["threshold_setting"] == th)]["drop_ratio_mean"].dropna().to_numpy()
                x = group_centers[ti] + offsets[mi]
                bp = ax.boxplot(
                    [vals], positions=[x], widths=width*0.88, patch_artist=True, showfliers=True,
                    medianprops=dict(color="black", linewidth=1.0),
                    boxprops=dict(facecolor=COLORS[model], edgecolor=EDGES[model], linewidth=0.9),
                    whiskerprops=dict(color=EDGES[model], linewidth=0.9),
                    capprops=dict(color=EDGES[model], linewidth=0.9),
                    flierprops=dict(marker="o", markerfacecolor=COLORS[model], markeredgecolor=EDGES[model], markersize=2.5, alpha=0.65),
                )
                if args.show_points and len(vals):
                    rng = np.random.default_rng(1000 + mi*100 + ti)
                    jitter = rng.uniform(-width*0.28, width*0.28, size=len(vals))
                    ax.scatter(np.full(len(vals), x)+jitter, vals, s=7, color=EDGES[model], alpha=0.25, linewidths=0, zorder=2)
        ax.axhline(1.0, color="black", linestyle="--", linewidth=0.9, alpha=0.7)
        ax.axvline(group_centers[2], color="black", linestyle=":", linewidth=0.9, alpha=0.6)
        ax.set_xticks(group_centers)
        ax.set_xticklabels(THRESHOLD_ORDER)
        ax.set_xlabel("Threshold change")
        ax.set_title(f"{letter} {site}", loc="left")
        ax.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.3)
        ymax = ax.get_ylim()[1]
        for ti, th in enumerate(THRESHOLD_ORDER):
            n = int(counts.loc[th])
            ax.text(group_centers[ti], ymax*0.97, f"n={n}", ha="center", va="top", fontsize=8)
    axes[0].set_ylabel("Conditional drop ratio\n(modeled drop / observed drop)")
    handles = [plt.Rectangle((0,0),1,1,facecolor=COLORS[m],edgecolor=EDGES[m]) for m in MODEL_ORDER]
    fig.legend(handles, MODEL_ORDER, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.01))
    fig.subplots_adjust(left=0.08, right=0.995, bottom=0.16, top=0.82, wspace=0.05)

    base = args.outdir / "Figure_C2_threshold_sensitivity_final"
    fig.savefig(base.with_suffix(".png"), dpi=args.dpi)
    fig.savefig(base.with_suffix(".pdf"))
    fig.savefig(base.with_suffix(".svg"))
    plt.close(fig)
    df.to_csv(args.outdir / "Figure_C2_plot_data_used.csv", index=False, encoding="utf-8-sig")
    print("Saved", base.with_suffix(".png"))
    print("Saved", base.with_suffix(".pdf"))
    print("Saved", base.with_suffix(".svg"))


if __name__ == "__main__":
    main()
