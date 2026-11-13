#!/usr/bin/env python3
"""Plot JoHRS Figure 6: error–variability trade-off of STD and Gmean."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
mpl.rcParams["svg.fonttype"] = "none"  # keep SVG text editable
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SITE_ORDER = ["Yamanokami", "Shiode", "Shin"]
MODEL_ORDER = ["GLM", "LSTM", "rSSM", "PGNN"]
TARGET_ORDER = ["STD", "Gmean"]

MODEL_COLORS = {
    "GLM": "#9e9e9e",
    "LSTM": "#4C78A8",
    "rSSM": "#F58518",
    "PGNN": "#54A24B",
}
MODEL_MARKERS = {
    "GLM": "s",
    "LSTM": "o",
    "rSSM": "^",
    "PGNN": "D",
}

TARGET_XLABEL = {
    "STD": "STD RMSE (°C)",
    "Gmean": r"$G_{mean}$ RMSE (°C m$^{-1}$)",
}
TARGET_YLABEL = {
    "STD": "STD variability ratio",
    "Gmean": r"$G_{mean}$ variability ratio",
}
TARGET_ROW_LABEL = {
    "STD": "(a) Vertical-temperature variability, STD",
    "Gmean": r"(b) Mean vertical-gradient structure, $G_{mean}$",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--csv",
        type=Path,
        default=Path(__file__).with_name("figure6_structure_metrics.csv"),
    )
    p.add_argument(
        "--outdir",
        type=Path,
        default=Path.cwd() / "revision_figures" / "Figure6_structure_tradeoff_v1",
    )
    p.add_argument("--dpi", type=int, default=500)
    p.add_argument("--width", type=float, default=13.4)
    p.add_argument("--height", type=float, default=7.2)
    p.add_argument(
        "--label-points",
        action="store_true",
        help="Add model labels beside markers. The default relies on the shared legend.",
    )
    p.add_argument(
        "--independent-axes",
        action="store_true",
        help="Use panel-specific axis limits rather than common limits within each row.",
    )
    return p.parse_args()


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {
        "site", "model", "target", "rmse_mean", "rmse_sd", "vr_mean", "vr_sd"
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")

    expected = {(s, m, t) for s in SITE_ORDER for m in MODEL_ORDER for t in TARGET_ORDER}
    found = set(zip(df["site"], df["model"], df["target"]))
    if found != expected:
        missing_rows = sorted(expected - found)
        extra_rows = sorted(found - expected)
        raise ValueError(
            f"CSV design mismatch. Missing={missing_rows}; extra={extra_rows}"
        )

    numeric = ["rmse_mean", "rmse_sd", "vr_mean", "vr_sd"]
    df[numeric] = df[numeric].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(df[numeric].to_numpy()).all():
        raise ValueError("Non-finite values found in Figure 6 input CSV")
    if (df[["rmse_sd", "vr_sd"]] < 0).any().any():
        raise ValueError("Standard deviations must be non-negative")
    return df


def common_limits(df: pd.DataFrame, target: str) -> tuple[tuple[float, float], tuple[float, float]]:
    sub = df[df["target"] == target]
    x_low = float((sub["rmse_mean"] - sub["rmse_sd"]).min())
    x_high = float((sub["rmse_mean"] + sub["rmse_sd"]).max())
    y_low = float((sub["vr_mean"] - sub["vr_sd"]).min())
    y_high = float((sub["vr_mean"] + sub["vr_sd"]).max())

    x_span = max(x_high - x_low, 0.2)
    y_span = max(y_high - y_low, 0.2)
    xlim = (max(0.0, x_low - 0.08 * x_span), x_high + 0.10 * x_span)
    ylim = (max(0.0, y_low - 0.08 * y_span), max(1.10, y_high + 0.10 * y_span))
    return xlim, ylim


def panel_limits(sub: pd.DataFrame) -> tuple[tuple[float, float], tuple[float, float]]:
    x_low = float((sub["rmse_mean"] - sub["rmse_sd"]).min())
    x_high = float((sub["rmse_mean"] + sub["rmse_sd"]).max())
    y_low = float((sub["vr_mean"] - sub["vr_sd"]).min())
    y_high = float((sub["vr_mean"] + sub["vr_sd"]).max())
    x_span = max(x_high - x_low, 0.1)
    y_span = max(y_high - y_low, 0.2)
    return (
        (max(0.0, x_low - 0.12 * x_span), x_high + 0.14 * x_span),
        (max(0.0, y_low - 0.12 * y_span), max(1.10, y_high + 0.14 * y_span)),
    )


def draw_panel(
    ax,
    sub: pd.DataFrame,
    target: str,
    site: str,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    label_points: bool,
) -> None:
    for model in MODEL_ORDER:
        row = sub[sub["model"] == model].iloc[0]
        x = float(row["rmse_mean"])
        y = float(row["vr_mean"])
        xerr = float(row["rmse_sd"])
        yerr = float(row["vr_sd"])
        ax.errorbar(
            x,
            y,
            xerr=xerr,
            yerr=yerr,
            fmt=MODEL_MARKERS[model],
            markersize=8,
            markerfacecolor=MODEL_COLORS[model],
            markeredgecolor="black",
            markeredgewidth=0.55,
            color=MODEL_COLORS[model],
            ecolor=MODEL_COLORS[model],
            elinewidth=1.1,
            capsize=3,
            capthick=1.0,
            alpha=0.98,
            zorder=4,
        )
        if label_points:
            ax.annotate(
                model,
                (x, y),
                xytext=(5, 4),
                textcoords="offset points",
                fontsize=7.5,
            )

    ax.axhline(1.0, color="0.25", linestyle="--", linewidth=1.0, zorder=1)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.grid(True, linestyle="--", linewidth=0.45, alpha=0.35)
    ax.set_axisbelow(True)
    ax.set_title(site, fontsize=11, pad=7)
    ax.set_xlabel(TARGET_XLABEL[target], fontsize=10)
    if site == SITE_ORDER[0]:
        ax.set_ylabel(TARGET_YLABEL[target], fontsize=10)


def main() -> None:
    args = parse_args()
    df = load_data(args.csv)
    args.outdir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        nrows=2,
        ncols=3,
        figsize=(args.width, args.height),
        sharex=False,
        sharey=False,
    )

    shared_limits = {target: common_limits(df, target) for target in TARGET_ORDER}

    for i, target in enumerate(TARGET_ORDER):
        for j, site in enumerate(SITE_ORDER):
            sub = df[(df["target"] == target) & (df["site"] == site)].copy()
            if args.independent_axes:
                xlim, ylim = panel_limits(sub)
            else:
                xlim, ylim = shared_limits[target]
            draw_panel(
                axes[i, j],
                sub,
                target,
                site,
                xlim,
                ylim,
                args.label_points,
            )

        axes[i, 0].text(
            -0.22,
            1.15,
            TARGET_ROW_LABEL[target],
            transform=axes[i, 0].transAxes,
            fontsize=11.5,
            fontweight="bold",
            ha="left",
            va="bottom",
        )

    # Minimal explanatory notes outside the data regions.
    fig.text(
        0.012,
        0.505,
        "VR < 1: under-variable\nVR > 1: over-variable",
        fontsize=8,
        color="0.35",
        va="center",
        ha="left",
    )
    fig.text(
        0.985,
        0.505,
        "Preferred direction:\nlower RMSE; VR closer to 1",
        fontsize=8,
        color="0.35",
        va="center",
        ha="right",
    )

    handles = [
        plt.Line2D(
            [0], [0],
            marker=MODEL_MARKERS[m],
            linestyle="none",
            markersize=8,
            markerfacecolor=MODEL_COLORS[m],
            markeredgecolor="black",
            markeredgewidth=0.55,
            label=m,
        )
        for m in MODEL_ORDER
    ]
    fig.legend(
        handles=handles,
        labels=MODEL_ORDER,
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 1.015),
        fontsize=10,
    )

    fig.subplots_adjust(left=0.08, right=0.92, bottom=0.10, top=0.88, hspace=0.47, wspace=0.30)

    base = args.outdir / "Figure6_structure_error_variability_tradeoff"
    fig.savefig(base.with_suffix(".png"), dpi=args.dpi)
    fig.savefig(base.with_suffix(".pdf"))
    fig.savefig(base.with_suffix(".svg"))
    plt.close(fig)

    df.to_csv(args.outdir / "Figure6_plot_data_used.csv", index=False, encoding="utf-8-sig")
    print("Saved:")
    print(base.with_suffix(".png"))
    print(base.with_suffix(".pdf"))
    print(base.with_suffix(".svg"))


if __name__ == "__main__":
    main()
