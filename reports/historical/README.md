# LoopAlpha on historical U.S. equities

This benchmark ran the same three-seed optimizer comparison on 30 U.S.-listed equities from February 2012 through December 2017. The panel came from the public-domain Quandl WIKI/PRICES archive. The fixed cohort was selected using January 2012 trading activity, before any dates used for training or evaluation. [Nasdaq Data Link catalog](https://data.nasdaq.com/databases/WIKIP) · [Archive mirror](https://www.kaggle.com/datasets/marketneutral/quandl-wiki-prices-us-equites)

[Historical protocol](../historical_protocol.md) · [Detailed results and figure](results.md) · [Panel audit](panel_audit.json) · [Run manifest](evidence/manifest.json)

## Data and execution

The importer retained 44,627 bars. The explicit NYSE calendar has 1,489 sessions. There were 43 missing ticker/session pairs, including seven bars that failed OHLC checks and a full missing session on November 8, 2017. Missing prices stayed missing. The session calendar comes from `exchange-calendars` 4.13.3 and is pinned in the repository. Features use only known historical windows. Results use vendor-adjusted opens as hypothetical execution proxies, with next-open entries and five-session holding periods. Portfolio summaries are unavailable for any run that selects a position with no observable exit. When a summary is available, its cost sweep covers 0, 5, 10 and 25 basis points per traded dollar. Return calculations omit borrowing, market impact, financing and empirical slippage.

The archive was discontinued in 2018. Its ticker history can have coverage and survivorship gaps. This experiment measures one liquidity-selected cohort during one historical period. Results apply to that sample and are not estimates of current trading performance. The repository contains the importer, provenance hashes and derived metrics. Reproduction requires the independently downloaded archive.

## Findings

The first test fold had 217 days with valid cross-sectional rankings. The untied model had the highest mean daily Rank IC at 0.0362. AdamW and the equal-weight depth variant both averaged 0.0354.

On the terminal holdout, 335 days had valid rankings. Momentum had the highest Rank IC at 0.0164. Momentum SGD led the trained models at 0.0005. AdamW averaged −0.0124, and the strongest depth-aware variant, separate moments, averaged −0.0081. Every terminal-fold per-run 95% date-block interval included zero. Both negative controls scored close to zero. This sample provides no evidence of robust historical ranking gains for the depth-aware optimizers.

All 37 portfolios in the first fold had priced exits. The leading mean net return at 10 bps was the MLP's 13.09%, which fell to −0.91% at 25 bps. In the terminal fold, only seven of 37 portfolios had prices for every selected exit. There were 53 affected trade rows across the other 30 portfolios, concentrated on three exit dates. Every fully priced terminal portfolio lost money after 10 bps costs. Missing exits prevent a complete portfolio comparison across optimizers.

## Verification

All 16 tests passed. The independent recalculation verified the dataset and calendar fingerprints, mapped held-out predictions to their forward returns, and recalculated gross P&L and all four cost totals for fully priced portfolios. It checked 23,754 trade rows across the two folds. [Verification counts](evidence/verification.json)
