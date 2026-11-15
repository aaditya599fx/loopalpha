"""Create a fixed pre-2010 US ETF basket and a real SPY regime reference."""

import argparse
from pathlib import Path
from zipfile import ZipFile

import pandas as pd

from loopalpha.data import FIELDS, validate_panel
from reprice_wiki import alternate_bars


GROUPS = {
    "US broad/style": "SPY DIA QQQ IWM IWD IWF",
    "US sectors": "XLB XLE XLF XLI XLK XLP XLU XLV XLY",
    "International equity": "EFA EEM",
    "Fixed income": "AGG TLT",
    "Gold": "GLD",
}
START, END = "2010-01-04", "2017-11-10"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    frames = []
    with ZipFile(args.archive) as archive:
        for symbols in GROUPS.values():
            for symbol in symbols.split():
                bars = alternate_bars(archive, symbol, folder="ETFs")
                if bars.index.min() >= START:
                    raise ValueError(f"{symbol} lacks pre-period history")
                selected = bars.loc[START:END, list(FIELDS)].reset_index()
                frames.append(selected.assign(symbol=symbol))
    panel = validate_panel(pd.concat(frames, ignore_index=True))
    if panel.date.dt.dayofweek.ge(5).any():
        raise ValueError("ETF panel has an unexpected weekend session")
    with args.out.open("x") as handle:
        panel[["date", "symbol", *FIELDS]].to_csv(handle, index=False)


if __name__ == "__main__":
    main()
