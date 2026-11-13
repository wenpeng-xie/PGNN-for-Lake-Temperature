#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
mpl.rcParams["svg.fonttype"] = "none"  # keep SVG text editable
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SITE_ORDER = ["Yamanokami", "Shiode"]
MODEL_ORDER = ["GLM", "LSTM", "rSSM", "PGNN"]
COLORS = {
    "GLM": "#9e9e9e",
    "LSTM": "#4C78A8",
    "rSSM": "#F58518",
    "PGNN": "#54A24B",
}


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description="Plot revised JoHRS Figure 8.")
    p.add_argument(
        "--event-csv",
        type=Path,
        default=script_dir / "data" / "Figure8_event_level_drop_ratios.csv",
        help="Event-level CSV produced by rebuild_figure8_data.py.",
    )
    p.add_argument(
        "--outdir",
        type=Path,
        default=Path.cwd() / "revision_figures" / "Figure8_drop_ratio_v1",
    )
    p.add_argument("--dpi", type=int, default=500)
    p.add_argument("--ymax", type=float, default=None, help="Optional fixed upper y limit.")
    p.add_argument(
        "--use-legacy-unordered-range",
        action="store_true",
        help="Audit-only: plot the old unordered max-min ratio instead of the revised ordered drop ratio.",
    )
    return p.parse_args()


def deterministic_jitter(n: int, width: float = 0.16) -> np.ndarray:
    if n <= 1:
        return np.zeros(n)
    # Even deterministic spread with a mild alternating permutation.
    base = np.linspace(-width, width, n)
    order = np.r_[np.arange(0, n, 2), np.arange(1, n, 2)]
    return base[order]


def draw_panel(ax: plt.Axes, df: pd.DataFrame, site: str, value_col: str, panel_label: str) -> None:
    site_df = df[df["site"] == site].copy()
    n_events = site_df["event_id"].nunique()
    data = []

    for model in MODEL_ORDER:
        values = (
            site_df[site_df["model"] == model]
            .sort_values("event_id")[value_col]
            .to_numpy(dtype=float)
        )
        values = values[np.isfinite(values)]
        data.append(values)

    bp = ax.boxplot(
        data,
        positions=np.arange(1, len(MODEL_ORDER) + 1),
        widths=0.52,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color="black", linewidth=1.25),
        whiskerprops=dict(color="black", linewidth=0.9),
        capprops=dict(color="black", linewidth=0.9),
        boxprops=dict(color="black", linewidth=0.8),
    )
    for patch, model in zip(bp["boxes"], MODEL_ORDER):
        patch.set_facecolor(COLORS[model])
        patch.set_alpha(0.48)

    for xpos, model, values in zip(np.arange(1, 5), MODEL_ORDER, data):
        jitter = deterministic_jitter(len(values))
        ax.scatter(
            xpos + jitter,
            values,
            s=16 if site == "Yamanokami" else 11,
            color=COLORS[model],
            edgecolor="none",
            alpha=0.72 if site == "Yamanokami" else 0.52,
            zorder=3,
        )

    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0, alpha=0.75)
    ax.set_xticks(np.arange(1, 5))
    ax.set_xticklabels(MODEL_ORDER)
    ax.set_title(f"{panel_label} {site}\nn = {n_events} observed events", loc="left", fontsize=11)
    ax.grid(axis="y", linewidth=0.45, alpha=0.32)
    ax.set_axisbelow(True)


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.event_csv)
    required = {
        "site", "event_id", "model", "n_members",
        "ordered_drop_ratio_mean", "unordered_range_ratio_mean",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {args.event_csv}: {sorted(missing)}")

    expected = {"Yamanokami": 13, "Shiode": 91}
    for site, n in expected.items():
        actual = int(df.loc[df["site"] == site, "event_id"].nunique())
        if actual != n:
            raise ValueError(f"{site}: expected {n} events, found {actual}")
    if not (df["n_members"] == 10).all():
        raise ValueError("Not all event/model rows contain 10 members")

    value_col = "unordered_range_ratio_mean" if args.use_legacy_unordered_range else "ordered_drop_ratio_mean"
    ylabel = (
        "Legacy unordered range ratio\n(model range / observed drop)"
        if args.use_legacy_unordered_range
        else "Conditional drop ratio\n(ordered modeled drop / observed drop)"
    )

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.8), sharey=True)
    draw_panel(axes[0], df, "Yamanokami", value_col, "(a)")
    draw_panel(axes[1], df, "Shiode", value_col, "(b)")
    axes[0].set_ylabel(ylabel)

    all_values = df[value_col].to_numpy(dtype=float)
    all_values = all_values[np.isfinite(all_values)]
    ymax = args.ymax
    if ymax is None:
        ymax = max(1.8, float(np.nanmax(all_values)) * 1.10)
    for ax in axes:
        ax.set_ylim(bottom=-0.03, top=ymax)

    title = (
        "Conditional reproduction of observed STD(t)-drop magnitude"
        if not args.use_legacy_unordered_range
        else "Audit view: legacy unordered STD(t) range ratios"
    )
    fig.suptitle(title, fontsize=13, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.94], w_pad=2.2)

    args.outdir.mkdir(parents=True, exist_ok=True)
    suffix = "legacy_unordered" if args.use_legacy_unordered_range else "ordered"
    base = args.outdir / f"Figure8_conditional_drop_ratio_{suffix}"
    fig.savefig(base.with_suffix(".png"), dpi=args.dpi)
    fig.savefig(base.with_suffix(".pdf"))
    fig.savefig(base.with_suffix(".svg"))
    plt.close(fig)

    used = df[[
        "site", "event_id", "model", "obs_drop_degC", "n_members",
        "ordered_drop_ratio_mean", "ordered_drop_ratio_sd", "ordered_drop_ratio_median",
        "unordered_range_ratio_mean", "unordered_range_ratio_sd", "unordered_range_ratio_median",
    ]].copy()
    used.to_csv(args.outdir / "Figure8_plot_data_used.csv", index=False, encoding="utf-8-sig")

    print("Saved:")
    print(base.with_suffix(".png"))
    print(base.with_suffix(".pdf"))
    print(base.with_suffix(".svg"))


if __name__ == "__main__":
    main()
