"""Summarize dated IC and portfolio exposure by ex-ante SPY regime."""

import argparse
import json
from pathlib import Path

import pandas as pd

from loopalpha.evaluation import block_interval, cost_metrics, portfolio
from prepare_etfs import GROUPS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["panel", "calendar", "run", "out"]:
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    panel = pd.read_csv(args.panel)
    calendar = pd.Index(pd.read_csv(args.calendar).date)
    opens = panel.pivot(index="date", columns="symbol", values="open").reindex(calendar)
    spy = panel.loc[panel.symbol.eq("SPY")].set_index("date").close.reindex(calendar)
    trailing = spy / spy.shift(60) - 1
    regimes = pd.Series("unavailable", index=calendar)
    regimes.loc[trailing.ge(0)] = "SPY trailing 60-session return nonnegative"
    regimes.loc[trailing.lt(0)] = "SPY trailing 60-session return negative"
    groups = {symbol: key for key, symbols in GROUPS.items() for symbol in symbols.split()}
    manifest = json.loads((args.run / "manifest.json").read_text())
    horizon = manifest["config"]["horizon"]
    folds = []
    for fold in sorted(args.run.glob("fold-*")):
        summary = json.loads((fold / "results.json").read_text())
        start, stop = summary["boundaries"][1:]
        runs = []
        for row in summary["runs"]:
            directory = fold / row["method"]
            if "seed" in row:
                directory /= str(row["seed"])
            daily = pd.read_csv(directory / "daily_ic.csv")
            daily["regime"] = daily.date.map(regimes)
            by_regime = {}
            for name, group in daily.groupby("regime"):
                by_regime[name] = {"days": len(group), "rank_ic": float(group.rank_ic.mean()),
                                   "rank_ic_95": block_interval(group.rank_ic, 5)}
            predictions = pd.read_csv(directory / "predictions.csv")
            ledgers = []
            for fraction in [.1, .2]:
                ledger = portfolio(predictions, opens, horizon, start, stop, fraction)
                ledger["asset_group"] = ledger.symbol.map(groups)
                cohort_count = len(range(start, stop - horizon - 1, horizon))
                allocation = ledger.groupby("asset_group").weight
                sector_weights = ledger.loc[ledger.asset_group.eq("US sectors")].groupby("symbol").weight
                ledgers.append({
                    "fraction": fraction,
                    "mean_gross_group_weight": (allocation.apply(lambda x: x.abs().sum()) / cohort_count).to_dict(),
                    "mean_net_group_weight": (allocation.sum() / cohort_count).to_dict(),
                    "mean_net_sector_etf_weight": (sector_weights.sum() / cohort_count).to_dict(),
                    "cost_stress": {str(bps): cost_metrics(ledger, horizon, bps, start, stop)
                                    for bps in manifest["config"]["cost_bps"]}})
            runs.append({"method": row["method"], "seed": row.get("seed"), "regime_ic": by_regime,
                         "portfolios": ledgers})
        folds.append({"fold": fold.name, "boundaries": summary["boundaries"], "runs": runs})
    with args.out.open("x") as handle:
        json.dump({"groups": GROUPS, "folds": folds}, handle, indent=2, allow_nan=False)
        handle.write("\n")


if __name__ == "__main__":
    main()
