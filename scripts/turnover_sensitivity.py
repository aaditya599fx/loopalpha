"""Replay signed position notionals with simultaneous exit/entry netting."""

import argparse
import json
from pathlib import Path

import pandas as pd

from loopalpha.evaluation import portfolio
from zipfile import ZipFile


def netted(ledger, bps):
    active = ledger.loc[ledger.weight.ne(0)]
    if active.exit_open.isna().any():
        return {"status": "unpriced_exit"}
    entries = {date: df for date, df in active.groupby("entry_date")}
    exits = {date: df for date, df in active.groupby("exit_date")}
    equity, traded = 1., 0.
    shares = {}
    for date in sorted(entries.keys() | exits.keys()):
        old_values = {}
        if date in exits:
            for row in exits[date].itertuples():
                old_values[row.symbol] = shares[row.symbol] * row.exit_open
                equity += shares[row.symbol] * (row.exit_open - row.entry_open)
        if equity <= 0:
            return {"status": "capital_exhausted"}
        new_values = {}
        if date in entries:
            for row in entries[date].itertuples():
                new_values[row.symbol] = (equity * row.weight, row.entry_open)
        notional = sum(abs(new_values.get(sym, (0., 0.))[0] - old_values.get(sym, 0.))
                       for sym in old_values.keys() | new_values.keys())
        traded += notional
        equity -= notional * bps / 10_000
        if equity <= 0:
            return {"status": "capital_exhausted"}
        shares = {sym: value / opening for sym, (value, opening) in new_values.items()}
    return {"status": "ok", "net_return": equity - 1., "traded_notional_per_initial_dollar": traded}


def run(args):
    bars = pd.read_csv(args.panel)
    schedule = pd.Index(pd.read_csv(args.calendar).date)
    opens = bars.pivot(index="date", columns="symbol", values="open").reindex(schedule)
    if args.recoveries:
        from reprice_wiki import alternate_bars, validated_open

        if not args.alternate:
            raise ValueError("An alternate ZIP is required with recoveries")
        recovered = json.loads(args.recoveries.read_text())["recoveries"]
        with ZipFile(args.alternate) as archive:
            for record in recovered:
                symbol, date = record["symbol"], record["date"]
                primary = bars.loc[bars.symbol.eq(symbol)].set_index("date").sort_index()
                price, check = validated_open(primary, alternate_bars(archive, symbol), date)
                if check != {k: v for k, v in record.items() if k not in ("symbol", "date")}:
                    raise ValueError("Recovered observation differs from verified input")
                if price is not None:
                    opens.loc[date, symbol] = price
    manifest = json.loads((args.run / "manifest.json").read_text())
    horizon = manifest["config"]["horizon"]
    results = []
    for fold in sorted(args.run.glob("fold-*")):
        base = json.loads((fold / "results.json").read_text())
        start, stop = base["boundaries"][1:]
        runs = []
        for row in base["runs"]:
            p = fold / row["method"]
            if "seed" in row:
                p /= str(row["seed"])
            pred = pd.read_csv(p / "predictions.csv")
            ledger = portfolio(pred, opens, horizon, start, stop)
            costs = {str(bps): netted(ledger, bps) for bps in manifest["config"]["cost_bps"]}
            runs.append({"method": row["method"], "seed": row.get("seed"), "cost_stress": costs})
        results.append({"fold": fold.name, "runs": runs})
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("panel", "calendar", "run", "out"):
        p.add_argument(f"--{name}", type=Path, required=True)
    p.add_argument("--alternate", type=Path)
    p.add_argument("--recoveries", type=Path)
    args = p.parse_args()
    with args.out.open("x") as out:
        json.dump(run(args), out, indent=2, allow_nan=False)
        out.write("\n")


if __name__ == "__main__":
    main()
