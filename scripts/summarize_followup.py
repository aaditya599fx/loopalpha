"""Create extended benchmark tables and figures from the stored evidence."""

import argparse
import json
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from loopalpha.experiment import OPTIMIZERS


TASKS = {"nonlinear_recurrence": "Nonlinear recurrence", "delayed_interaction": "Delayed interaction"}
COLORS = ["#5676a3", "#087569", "#a33d22", "#774c99", "#9a650f", "#596371", "#a73570"]


def load(path):
    return json.loads(path.read_text())


def summarize(unit):
    table = {}
    for method in OPTIMIZERS:
        runs = [row for row in unit["runs"] if row["method"] == method]
        budgets = sorted({key for row in runs for key in row.get("checkpoints", {})})
        summary = {}
        for key in budgets:
            observations = [row["checkpoints"][key] for row in runs
                            if row.get("checkpoints", {}).get(key, {}).get("test_mse") is not None]
            vals = [point["test_mse"] for point in observations]
            if vals:
                summary[key] = {"mean": mean(vals), "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else None,
                                "n": len(vals), "median_updates": float(np.median([x["step"] for x in observations]))}
        table[method] = summary
    differences = {}
    for key in ("step:1000", "time:0.12", "time:1.2"):
        paired = {}
        for method in ("muon", "depth_moments", "adamw"):
            paired[method] = {r["seed"]: r["checkpoints"][key]["test_mse"] for r in unit["runs"]
                              if r["method"] == method and key in r.get("checkpoints", {})
                              and r["checkpoints"][key]["test_mse"] is not None}
        for a, b in (("depth_moments", "muon"), ("depth_moments", "adamw")):
            vals = np.array([paired[a][seed] - paired[b][seed] for seed in sorted(paired[a].keys() & paired[b].keys())])
            if len(vals):
                rng = np.random.default_rng(8425)
                interval = np.quantile(rng.choice(vals, size=(5000, len(vals))).mean(axis=1), [.025, .975])
                differences[f"{key} {a} minus {b}"] = {"mean": vals.mean(), "paired_seed_95": interval.tolist(),
                                                       "n": len(vals)}
    return table, differences


def figures(evidence, output):
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
    summaries = {}
    profiles = load(evidence / "optimizer-profile.json")
    flops = {r["method"]: r["profile_total_registered_flops"] / r["profile_steps"] for r in profiles}
    common = 1000 * min(flops.values())
    for index, (name, title) in enumerate(TASKS.items()):
        unit = load(evidence / f"{name}.json")
        summaries[name] = dict(zip(("checkpoints", "paired_differences"), summarize(unit)))
        compute = {}
        for fraction in (.25, .50, .90):
            budget = common * fraction
            record = {}
            for method in OPTIMIZERS:
                points = [eligible[-1] for row in unit["runs"] if row["method"] == method
                          if (eligible := [p for p in row["trace"] if p["step"] * flops[method] <= budget])]
                if points:
                    record[method] = {"mean_validation_mse": mean(p["val_mse"] for p in points),
                                      "median_updates": float(np.median([p["step"] for p in points])),
                                      "n": len(points)}
            compute[f"{fraction:g}"] = {"estimated_registered_flops": budget, "methods": record}
        summaries[name]["registered_flop_budgets"] = compute
        for method, color in zip(OPTIMIZERS, COLORS):
            traces = [r["trace"] for r in unit["runs"] if r["method"] == method and not r["failed"]]
            losses = np.array([[v["val_mse"] for v in trace] for trace in traces])
            for j, key in enumerate(("step", "seconds")):
                x = np.median([[v[key] for v in t] for t in traces], axis=0)
                med = np.median(losses, axis=0)
                low, high = np.quantile(losses, [.25, .75], axis=0)
                axes[index][j].fill_between(x, low, high, color=color, alpha=.08)
                axes[index][j].plot(x, med, color=color, label=method, linewidth=1.6)
                axes[index][j].set(yscale="log", xlabel="Updates" if j == 0 else "CPU wall time (s)",
                                  ylabel="Validation MSE", title=title)
    axes[0][1].legend(fontsize=7, ncol=2, frameon=False)
    fig.savefig(output / "extended_convergence.png", dpi=165)
    plt.close(fig)
    (output / "extended_summary.json").write_text(json.dumps(summaries, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    figures(args.evidence, args.out)


if __name__ == "__main__":
    main()
