"""Final C1 count plot extracted from the corrected R2-3 plotting package.
The superseded C2 plotting branch is intentionally excluded.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

THRESHOLD_ORDER = ["-50%", "-25%", "Original", "+25%", "+50%"]

SITE_ORDER_C1 = ["Yamanokami", "Shiode", "Shin"]

def require(condition, message):
    if not bool(condition):
        raise ValueError(message)

def plot_c1(counts: pd.DataFrame, out_dir: Path):
    """
    Keep the original Figure-S.ipynb line-plot style.
    """
    fig, ax = plt.subplots(figsize=(6.2, 4.2))

    x = np.arange(len(THRESHOLD_ORDER))

    for site in SITE_ORDER_C1:
        sub = (
            counts[counts["site"] == site]
            .copy()
            .assign(
                threshold_setting=lambda d:
                pd.Categorical(
                    d["threshold_setting"],
                    categories=THRESHOLD_ORDER,
                    ordered=True,
                )
            )
            .sort_values("threshold_setting")
        )

        require(len(sub) == len(THRESHOLD_ORDER),
                f"{site}: expected five threshold rows.")

        ax.plot(
            x,
            sub["detected_events"].to_numpy(),
            marker="o",
            linewidth=1.8,
            label=site,
        )

    original_x = THRESHOLD_ORDER.index("Original")
    ax.axvline(
        original_x,
        color="black",
        linestyle="--",
        linewidth=1.0,
        alpha=0.65,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(THRESHOLD_ORDER)
    ax.set_xlabel("Threshold change")
    ax.set_ylabel("Number of detected events")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(frameon=False)

    fig.tight_layout()

    png = out_dir / "Figure_C1_threshold_sensitivity_updated.png"
    pdf = out_dir / "Figure_C1_threshold_sensitivity_updated.pdf"

    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    return png, pdf

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv',type=Path,required=True)
    parser.add_argument('--outdir',type=Path,default=Path('outputs/figures/C1'))
    args=parser.parse_args();args.outdir.mkdir(parents=True,exist_ok=True)
    plot_c1(pd.read_csv(args.csv),args.outdir)

if __name__=='__main__':main()
