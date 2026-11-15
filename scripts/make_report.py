"""Regenerate the evidence summary and figures from completed runs."""

import argparse
import json
from pathlib import Path
from statistics import mean, median, stdev

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from loopalpha.experiment import OPTIMIZERS


def load(path):
    return json.loads(path.read_text())


def values(rows, key):
    return [row[key] for row in rows if row.get(key) is not None]


def fmt(numbers, *, center=mean, digits=4, spread=False):
    if not numbers:
        return "—"
    result = f"{center(numbers):.{digits}f}"
    if spread and len(numbers) > 1:
        result += f" ± {stdev(numbers):.{digits}f}"
    return result


def summarize_toy(fold, config):
    steps = config["steps"]
    budget_ms = config["time_budget"] * 1000
    rows = fold["runs"]
    lines = [
        f"| Method | LR | Step-{steps} MSE ± SD | {budget_ms:g} ms MSE ± SD | Median checkpoint | Median threshold ms | Threshold hits | Run failures |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in dict.fromkeys(row["method"] for row in rows):
        subset = [row for row in rows if row["method"] == method]
        times = [1000 * t for t in values(subset, "threshold_seconds")]
        failures = sum(row["failed"] for row in subset)
        fixed = values(subset, "time_budget_test_mse")
        lines.append(f"| {method} | {fold['chosen'].get(method, '—')} | "
                     f"{fmt(values(subset, 'test_mse'), spread=True)} | "
                     f"{fmt(fixed, spread=True)} ({len(fixed)}/{len(subset)}) | "
                     f"{fmt(values(subset, 'time_budget_step'), center=median, digits=0)} | "
                     f"{fmt(times, center=median, digits=1)} | {len(times)}/{len(subset)} | "
                     f"{failures}/{len(subset)} |")
    return lines


def summarize_market(fold):
    rows = fold["runs"]
    lines = [
        "| Method | Test MSE ± SD | Rank IC ± SD | Rank IC > 0 | Median IC dates | Net return at 10 bps | Priced runs |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in dict.fromkeys(row["method"] for row in rows):
        subset = [row for row in rows if row["method"] == method]
        costs = [row.get("cost_stress", {}).get("10", {}) for row in subset]
        returns = [100 * v for v in values(costs, "net_return")]
        lines.append(f"| {method} | {fmt(values(subset, 'test_mse'), spread=True)} | "
                     f"{fmt(values(subset, 'rank_ic'), spread=True)} | "
                     f"{fmt(values(subset, 'rank_ic_positive'), digits=3)} | "
                     f"{fmt(values(subset, 'ic_days'), center=median, digits=0)} | "
                     f"{fmt(returns, digits=2) + '%' if returns else '—'} | {len(returns)}/{len(subset)} |")
    return lines


def convergence_figure(rows, output, config):
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), layout="constrained")
    colors = ["#405e87", "#1b776c", "#a14824", "#704399", "#b86f05", "#5b6573", "#ba4373"]
    for method, color in zip(OPTIMIZERS, colors):
        traces = [row.get("trace", []) for row in rows if row["method"] == method]
        complete = [trace for trace in traces if len(trace) == max(map(len, traces))]
        if not complete or not complete[0]:
            continue
        losses = np.array([[point["val_mse"] for point in trace] for trace in complete])
        for axis, key, multiplier in zip(axes, ["step", "seconds"], [1, 1000]):
            x = np.array([[point[key] * multiplier for point in trace] for trace in complete])
            axis.plot(np.median(x, axis=0), np.median(losses, axis=0),
                      color=color, linewidth=1.8, label=method)
    for axis, xlabel in zip(axes, ["Optimizer steps", "Training wall time (ms)"]):
        axis.axhline(config["threshold"], color="#475569", linestyle="--", linewidth=.9)
        axis.set(xlabel=xlabel, ylabel="Validation MSE", yscale="log")
    axes[1].axvline(config["time_budget"] * 1000, color="#475569", linestyle=":", linewidth=.9)
    axes[1].legend(frameon=False, fontsize=8)
    fig.savefig(output, dpi=170)
    plt.close(fig)


def market_figure(folds, output):
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, len(folds), figsize=(11, 5), layout="constrained", sharex=True, sharey=True)
    for i, (axis, fold) in enumerate(zip(np.atleast_1d(axes), folds), 1):
        runs = fold["runs"]
        methods = list(dict.fromkeys(row["method"] for row in runs))
        for j, method in enumerate(methods):
            for k, row in enumerate(r for r in runs if r["method"] == method and "rank_ic" in r):
                y = j + .14 * (k - 1)
                lo, hi = row["rank_ic_95"]
                if lo is not None:
                    axis.hlines(y, lo, hi, color="#91a7af", linewidth=1)
                axis.plot(row["rank_ic"], y, "o", color="#176f63", markersize=3.5)
        axis.axvline(0, color="#6b7280", linewidth=1)
        axis.set(xlabel="Mean daily Rank IC", title=f"Fold {i}", yticks=range(len(methods)), yticklabels=methods)
        axis.grid(axis="x", alpha=.2)
    axes[0].invert_yaxis()
    fig.savefig(output, dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--toy", type=Path, help="Include the sequence benchmark")
    parser.add_argument("--market", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("reports"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    folds = [load(p) for p in sorted(args.market.glob("fold-*/results.json"))]
    manifest = load(args.market / "manifest.json")
    market_task = manifest["task"]
    market_seeds = len(manifest["config"]["seeds"])
    if not folds:
        parser.error("No completed market fold found")
    market_figure(folds, args.out / "market_rank_ic.png")
    lines = ["# LoopAlpha experiment results", "",
             f"Market input type: `{market_task}`. Market results use {market_seeds} training seeds "
             "unless the method is deterministic."]
    if args.toy:
        toy = load(args.toy / "fold-1/results.json")
        toy_config = load(args.toy / "manifest.json")["config"]
        convergence_figure(toy["runs"], args.out / "toy_convergence.png", toy_config)
        lines.extend(["", "## Controlled sequence task", "", *summarize_toy(toy, toy_config), "",
                      "![Toy validation convergence](toy_convergence.png)", "",
                      f"Median validation MSE across up to {len(toy_config['seeds'])} seeds at each checkpoint. "
                      "The wall-clock panel uses median checkpoint times. The horizontal line marks the loss "
                      "threshold and the vertical line marks the time budget."])
    lines.extend(["", "## Market panel"])
    for i, fold in enumerate(folds, 1):
        lines.extend(["", f"### Fold {i}", "",
                      f"Session boundaries {fold['boundaries']}. Labeled train, validation and test counts "
                      f"{[fold['counts'][k] for k in ('train', 'validation', 'labeled_test')]}.",
                      "", *summarize_market(fold)])
    lines.extend(["", "![Rank IC by seed and fold](market_rank_ic.png)", "",
                  "Each point is one training seed or the deterministic momentum rule. Intervals are "
                  "95% circular date-block bootstrap intervals within that seed and fold.", "",
                  "Full per-run JSON contains elapsed time, parameter counts, probe tensor bytes, "
                  "daily-IC summary statistics and the complete cost sweep. Tuning traces give trial-level failures.",
                  ""])
    (args.out / "results.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
