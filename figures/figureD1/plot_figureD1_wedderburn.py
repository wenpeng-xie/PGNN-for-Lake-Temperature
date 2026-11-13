#!/usr/bin/env python3
"""Plot revised Supplementary Figure D1 (event-scale Wedderburn number).

Input data are the already-verified final fixed-depth stability results:
- Yamanokami: 13 final events, 12 finite positive W values at observed STD peak
- Shiode: 91 final events, 91 finite positive W values at observed STD peak

The script only plots the verified results. It does not alter event detection or
recalculate the stability physics.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

mpl.rcParams["svg.fonttype"] = "none"
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42

SITE_ORDER = ["yamanokami", "shiode"]
SITE_LABELS = {"yamanokami": "Yamanokami", "shiode": "Shiode"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--csv",
        type=Path,
        default=Path(__file__).parent / "data" / "TableS_event_stability_fixeddepth.csv",
    )
    p.add_argument(
        "--summary-csv",
        type=Path,
        default=Path(__file__).parent / "data" / "TableS_event_stability_summary_fixeddepth.csv",
    )
    p.add_argument(
        "--outdir",
        type=Path,
        default=Path.cwd() / "revision_figures" / "FigureD1_wedderburn_v1",
    )
    p.add_argument("--dpi", type=int, default=500)
    p.add_argument("--show-points", action="store_true", help="Overlay individual event values.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.csv)
    summary = pd.read_csv(args.summary_csv)

    required = {"site", "event_id", "Wedderburn_peak"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    values = []
    labels = []
    audit_rows = []
    for site in SITE_ORDER:
        site_df = df[df["site"] == site].copy()
        w = pd.to_numeric(site_df["Wedderburn_peak"], errors="coerce").to_numpy(float)
        finite_positive = np.isfinite(w) & (w > 0)
        valid = w[finite_positive]
        if valid.size == 0:
            raise ValueError(f"{site}: no finite positive Wedderburn values")
        values.append(valid)
        labels.append(SITE_LABELS[site])
        audit_rows.append(
            {
                "site": site,
                "site_label": SITE_LABELS[site],
                "final_event_count": int(len(site_df)),
                "valid_positive_W_count": int(valid.size),
                "excluded_nonfinite_or_nonpositive_count": int(len(site_df) - valid.size),
                "median_W": float(np.median(valid)),
                "q25_W": float(np.quantile(valid, 0.25)),
                "q75_W": float(np.quantile(valid, 0.75)),
                "min_W": float(np.min(valid)),
                "max_W": float(np.max(valid)),
            }
        )

    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    bp = ax.boxplot(
        values,
        labels=labels,
        widths=0.48,
        patch_artist=True,
        showfliers=True,
        medianprops={"linewidth": 1.6, "color": "black"},
        boxprops={"linewidth": 1.1, "edgecolor": "black"},
        whiskerprops={"linewidth": 1.0, "color": "black"},
        capprops={"linewidth": 1.0, "color": "black"},
        flierprops={"marker": "o", "markersize": 4, "markerfacecolor": "none", "markeredgecolor": "black"},
    )
    for patch in bp["boxes"]:
        patch.set_facecolor("#d9d9d9")
        patch.set_alpha(0.8)

    if args.show_points:
        rng = np.random.default_rng(20261031)
        for i, vals in enumerate(values, start=1):
            jitter = rng.uniform(-0.08, 0.08, size=len(vals))
            ax.scatter(np.full(len(vals), i) + jitter, vals, s=13, alpha=0.45, edgecolors="none", color="black", zorder=3)

    ax.set_yscale("log")
    ax.axhline(1.0, linestyle="--", linewidth=1.0, color="black", alpha=0.7)
    ax.set_ylabel("Wedderburn number at observed STD(t) peak")
    ax.grid(axis="y", which="both", linewidth=0.45, alpha=0.30)

    # Report valid-event counts without placing dense annotations inside the data region.
    counts = [f"n = {len(v)} valid" for v in values]
    for i, text in enumerate(counts, start=1):
        ax.text(i, 1.025, text, transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=9)

    ax.set_title("Event-scale Wedderburn number", pad=20)
    fig.tight_layout()

    args.outdir.mkdir(parents=True, exist_ok=True)
    base = args.outdir / "Figure_D1_event_wedderburn_final"
    fig.savefig(base.with_suffix(".png"), dpi=args.dpi, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)

    audit = pd.DataFrame(audit_rows)
    audit.to_csv(args.outdir / "Figure_D1_plot_audit.csv", index=False, encoding="utf-8-sig")

    provenance = {
        "input_csv": str(args.csv.resolve()),
        "summary_csv": str(args.summary_csv.resolve()),
        "event_catalog": "final fixed-depth, complete-window event catalog",
        "Yamanokami_final_events": 13,
        "Yamanokami_valid_W_at_peak": 12,
        "Shiode_final_events": 91,
        "Shiode_valid_W_at_peak": 91,
        "plot_exclusion": "Only non-finite or non-positive W values are omitted from the logarithmic plot.",
        "interpretation": "Relative stability diagnostic under simplified fetch and morphometry assumptions; not proof of complete mixing or a unique mechanism.",
    }
    (args.outdir / "Figure_D1_provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")

    print("Saved:")
    for suffix in [".png", ".pdf", ".svg"]:
        print(base.with_suffix(suffix))
    print(audit.to_string(index=False))


if __name__ == "__main__":
    main()
