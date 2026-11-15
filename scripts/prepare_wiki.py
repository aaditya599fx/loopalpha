"""Build the fixed historical panel from a Quandl WIKI/PRICES CSV or ZIP."""

import argparse
from datetime import datetime, timezone
from hashlib import file_digest
import json
from pathlib import Path

import numpy as np
import pandas as pd

from loopalpha.data import FIELDS, validate_panel


REFERENCE_START = "2012-01-01"
START = "2012-02-01"
END = "2017-12-31"
UNIVERSE_SIZE = 30
ADJUSTED = [f"adj_{field}" for field in FIELDS]


def read_archive(path, fields):
    return pd.read_csv(path, usecols=["date", "ticker", *fields],
                       dtype={"ticker": "string", "date": "string"}, chunksize=250_000)


def prepare(path):
    reference = pd.concat([
        chunk.loc[chunk.date.ge(REFERENCE_START) & chunk.date.lt(START)]
        for chunk in read_archive(path, ["close", "volume"])
    ], ignore_index=True)
    if reference.duplicated(["date", "ticker"]).any():
        raise ValueError("Duplicate date/ticker key in the universe formation period")
    reference["dollar_volume"] = reference.close * reference.volume
    reference = reference.loc[np.isfinite(reference.dollar_volume) & reference.dollar_volume.gt(0)]
    liquidity = reference.groupby("ticker").dollar_volume.agg(["median", "size"]).reset_index()
    liquid = liquidity.loc[liquidity["size"].ge(18)].sort_values(
        ["median", "ticker"], ascending=[False, True]).head(UNIVERSE_SIZE)
    if len(liquid) != UNIVERSE_SIZE:
        raise ValueError("Insufficient coverage in the pre-period formation window")

    panel = pd.concat([
        chunk.loc[chunk.date.ge(START) & chunk.date.le(END) & chunk.ticker.isin(liquid.ticker)]
        for chunk in read_archive(path, ADJUSTED)
    ], ignore_index=True)
    panel = panel.rename(columns={"ticker": "symbol", **dict(zip(ADJUSTED, FIELDS))})
    if panel.duplicated(["date", "symbol"]).any():
        raise ValueError("Duplicate date/ticker key in the evaluation panel")
    # Preserve malformed vendor bars as gaps. Never repair a price using a later bar.
    invalid = (~np.isfinite(panel[list(FIELDS)]).all(axis=1)
               | panel[list(FIELDS[:-1])].le(0).any(axis=1)
               | panel.volume.lt(0)
               | panel.high.lt(panel[["open", "close", "low"]].max(axis=1))
               | panel.low.gt(panel[["open", "close"]].min(axis=1)))
    rejected = panel.loc[invalid, ["date", "symbol"]].sort_values(["date", "symbol"])
    panel = validate_panel(panel.loc[~invalid])
    if panel.date.dt.dayofweek.ge(5).any():
        raise ValueError("Empty panel or unexpected weekend bar")
    per_day = panel.groupby("date").symbol.size()
    per_asset = panel.groupby("symbol").date.agg(["min", "max", "size"])
    per_asset[["min", "max"]] = per_asset[["min", "max"]].astype(str)
    audit = {"formation_start": REFERENCE_START, "formation_end_exclusive": START,
             "formation_min_positive_volume_days": 18,
             "formation_rule": "Top 30 by median raw close times raw volume; ticker breaks ties",
             "universe": liquid.ticker.tolist(), "first_bar": str(panel.date.min().date()),
             "last_bar": str(panel.date.max().date()), "rows": len(panel),
             "sessions": len(per_day), "minimum_daily_assets": int(per_day.min()),
             "missing_date_symbol_bars": int(len(per_day) * UNIVERSE_SIZE - len(panel)),
             "zero_volume_bars": int(panel.volume.eq(0).sum()),
             "rejected_invalid_bars": rejected.to_dict(orient="records"),
             "per_ticker": per_asset.to_dict(orient="index"),
             "prices": "vendor split/dividend-adjusted OHLC and vendor adjusted volume",
             "source": "https://data.nasdaq.com/databases/WIKIP",
             "archive": "https://www.kaggle.com/datasets/marketneutral/quandl-wiki-prices-us-equites"}
    return panel[["date", "symbol", *FIELDS]], audit


def sha256(path):
    with path.open("rb") as payload:
        return file_digest(payload, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()
    panel, audit = prepare(args.archive)
    with args.csv.open("x") as output:
        panel.to_csv(output, index=False)
    audit.update({"archive_sha256": sha256(args.archive), "csv_sha256": sha256(args.csv),
                  "prepared_utc": datetime.now(timezone.utc).isoformat()})
    with args.audit.open("x") as output:
        json.dump(audit, output, indent=2, allow_nan=False)
        output.write("\n")


if __name__ == "__main__":
    main()
