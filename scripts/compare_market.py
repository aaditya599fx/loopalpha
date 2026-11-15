"""Plot held-out daily IC distributions across the two frozen market universes."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from loopalpha.evaluation import block_interval


METHODS = ["sgd", "adamw", "muon", "depth", "depth_moments", "shuffle", "noise", "momentum"]


def distributions(run):
    fold = run / "fold-2"
    boxplot = []
    statistics = {}
    for method in METHODS:
        root = fold / method
        frames = [pd.read_csv(f) for f in sorted(root.rglob("daily_ic.csv"))]
        averaged = pd.concat(frames).groupby("date", sort=True).rank_ic.mean()
        values = averaged.to_numpy()
        boxplot.append(values)
        statistics[method] = {"dates": len(values), "mean": float(values.mean()),
                              "median": float(np.median(values)),
                              "quantiles": np.quantile(values, [.05, .25, .75, .95]).tolist(),
                              "date_block_95": block_interval(values, block=5)}
    return boxplot, statistics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("wiki", "etf", "out"):
        p.add_argument(f"--{name}", type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(exist_ok=True, parents=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.3), layout="constrained", sharex=True, sharey=True)
    summary = {}
    for ax, (label, run) in zip(axes, [("Frozen WIKI stock signals", a.wiki), ("Archived ETF basket", a.etf)]):
        values, summary[label] = distributions(run)
        boxes = ax.boxplot(values, orientation="horizontal", tick_labels=METHODS,
                           showfliers=False, whis=(5, 95), patch_artist=True)
        for b in boxes["boxes"]:
            b.set(facecolor="#d3eae3", edgecolor="#167268")
        for median in boxes["medians"]:
            median.set(color="#172d35")
        ax.axvline(0, color="#6f7b85", linewidth=1)
        ax.set(title=label, xlabel="Daily Rank IC (mean across seeds)")
    axes[0].invert_yaxis()
    fig.savefig(a.out / "daily_ic_distributions.png", dpi=165)
    plt.close(fig)
    (a.out / "distribution_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
