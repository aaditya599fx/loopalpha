"""Replay frozen portfolio signals with observed alternate opening prints."""

import argparse
from collections import defaultdict
from hashlib import file_digest
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

from loopalpha.evaluation import cost_metrics, portfolio


def digest(path):
    with path.open("rb") as handle:
        return file_digest(handle, "sha256").hexdigest()


def alternate_bars(archive, symbol, folder="Stocks"):
    member = f"Data/{folder}/{symbol.lower().replace('_', '-')}.us.txt"
    with archive.open(member) as handle:
        bars = pd.read_csv(handle).rename(columns=str.lower)
    return bars.set_index("date", verify_integrity=True).sort_index()


def validated_open(primary, alternate, date):
    if date not in alternate.index:
        return None, {"status": "no_alternate_bar"}
    row = alternate.loc[date, ["open", "high", "low", "close", "volume"]]
    if (not np.isfinite(row).all() or row.min() <= 0
            or row.high < max(row.open, row.close) or row.low > min(row.open, row.close)):
        return None, {"status": "invalid_alternate_bar"}
    history = primary.loc[primary.index < date, ["open", "close"]].dropna().tail(20).join(
        alternate[["open", "close"]], how="inner", lsuffix="_primary", rsuffix="_alternate")
    if len(history) != 20 or not np.isfinite(history).all().all() or history.le(0).any().any():
        return None, {"status": "insufficient_overlap"}
    ratios = np.r_[history.open_primary / history.open_alternate,
                   history.close_primary / history.close_alternate]
    # Apply one adjustment scale, calibrated only on overlapping dates before the missing bar.
    scale = float(np.median(ratios))
    deviation = float(np.max(np.abs(ratios / scale - 1)))
    details = {"last_overlap": history.index.max(), "overlap_sessions": 20,
               "adjustment_scale": scale, "max_relative_deviation": deviation}
    if deviation > .005:
        return None, {"status": "overlap_disagreement", **details}
    return float(row.open * scale), {"status": "observed_rebased_open", **details}


def replay(panel_path, calendar_path, alternate_path, run, output):
    manifest = json.loads((run / "manifest.json").read_text())
    if digest(panel_path) != manifest["data"]["sha256"] or digest(calendar_path) != manifest["data"]["calendar_sha256"]:
        raise ValueError("Input bars/calendar do not match the frozen run")
    calendar = pd.Index(pd.read_csv(calendar_path).date)
    bars = pd.read_csv(panel_path)
    opens = bars.pivot(index="date", columns="symbol", values="open").reindex(calendar)
    candidates = defaultdict(set)
    for path in sorted(run.glob("fold-*/**/trade_ledger.csv")):
        ledger = pd.read_csv(path)
        missing = ledger.loc[ledger.weight.ne(0) & ledger.exit_open.isna()]
        for date, symbol in missing[["exit_date", "symbol"]].itertuples(index=False, name=None):
            candidates[symbol].add(date)
    records = []
    with ZipFile(alternate_path) as source:
        for symbol, dates in sorted(candidates.items()):
            prices = bars.loc[bars.symbol.eq(symbol)].set_index("date").sort_index()
            try:
                alternate = alternate_bars(source, symbol)
            except KeyError:
                alternate = pd.DataFrame()
            for date in sorted(dates):
                price, audit = validated_open(prices, alternate, date)
                records.append({"symbol": symbol, "date": date, **audit})
                if price is not None:
                    opens.loc[date, symbol] = price
    folds = []
    horizon = manifest["config"]["horizon"]
    for directory in sorted(run.glob("fold-*")):
        baseline = json.loads((directory / "results.json").read_text())
        start, stop = baseline["boundaries"][1:]
        rows = []
        for entry in baseline["runs"]:
            location = directory / entry["method"]
            if "seed" in entry:
                location /= str(entry["seed"])
            predictions_path = location / "predictions.csv"
            predictions = pd.read_csv(predictions_path)
            portfolios = []
            # Primary protocol plus a broader-basket sensitivity.
            for fraction in [.1, .2]:
                ledger = portfolio(predictions, opens, horizon, start, stop, fraction=fraction)
                costs = {str(bps): cost_metrics(ledger, horizon, bps, start, stop)
                         for bps in manifest["config"]["cost_bps"]}
                portfolios.append({"fraction": fraction, "cost_stress": costs})
            rows.append({"method": entry["method"], "seed": entry.get("seed"),
                         "rank_ic_frozen": entry.get("rank_ic"), "rank_ic_95_frozen": entry.get("rank_ic_95"),
                         "predictions_sha256": digest(predictions_path),
                         "prior_portfolio_status": entry["cost_stress"]["10"]["status"],
                         "portfolios": portfolios})
        folds.append({"fold": directory.name, "boundaries": baseline["boundaries"], "runs": rows})
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        json.dump({"frozen_run_manifest_sha256": digest(run / "manifest.json"),
                   "alternate_archive_sha256": digest(alternate_path),
                   "alternate_source": "https://www.kaggle.com/datasets/borismarjanovic/price-volume-data-for-all-us-stocks-etfs",
                   "calendar_sha256": digest(calendar_path), "recoveries": records, "folds": folds},
                  handle, indent=2, allow_nan=False)
        handle.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["panel", "calendar", "alternate", "run", "out"]:
        parser.add_argument(f"--{name}", type=Path, required=True)
    options = parser.parse_args()
    replay(options.panel, options.calendar, options.alternate, options.run, options.out)


if __name__ == "__main__":
    main()
