# LoopAlpha follow-up

[Technical research report (PDF)](technical_report.pdf)

## Equity exits recovered

A frozen-signal replay recovered all seven stock/date opening observations responsible for the 53 unpriced trades in the original terminal fold. The dates were November 16 and 23, 2016, and August 7, 2017. The alternate archive reports an adjusted open for every required pair. For each pair, the importer validates the alternate bar and rebases its observed open using a single median ratio estimated from 20 prior overlapping opens and closes. The 0.5% overlap limit was applied uniformly. The worst accepted deviation was 0.282%. The replay verifies the original panel and calendar hashes and hashes each prediction file. It changes execution pricing only. Original training, predictions, historical IC labels and IC metrics stay frozen.

Both folds now price all 37 portfolios at the original decile width and the alternative quintile width. In the terminal test at 10 bps, the momentum rule returned +0.85% in the decile replay and −6.14% in the quintile replay. All trained methods had negative mean terminal returns at both widths. Netting overlapping positions at the same opening reduced compulsory round trips without changing that conclusion about mean learned-method returns. The momentum baseline returned +2.73% under netting. The alternate archive is a published adjusted-price source whose collection lineage and execution-price fidelity are incompletely documented. Agreement with earlier WIKI prices is a consistency check.

## Broader optimizer benchmark

The extended study compares seven methods across two distinct target-generating rules, using two tuning seeds and ten paired evaluation seeds per task. It reports held-out MSE at 100, 300, 600 and 1,000 updates and the last completed checkpoints at or before 0.12, 0.30, 0.60 and 1.20 seconds. The update cap can end a run before a time budget. Muon led both tasks at every wall-time cutoff. At 1,000 updates, separate per-depth moments nearly matched Muon on the recurrence task. The paired seed interval on their MSE difference included zero.

The benchmark ran on an Apple M3 with one CPU thread. Resource profiles measured each method in a fresh process. Registered-operation FLOP estimates support an additional comparison at three common compute budgets using validation checkpoints. They only count operations covered by the profiler, and do not measure memory movement or all optimizer arithmetic. GPU performance was outside these experiments.

[Convergence curves](extended_convergence.png) · [Checkpoint summary and paired intervals](extended_summary.json) · [Resource profiles](evidence/optimizer-profile.json)

## ETF validation and robustness

The second universe is a fixed basket of 20 ETFs with observations before 2010. It includes U.S. equity styles and sector ETFs, international equity, bonds and gold. It covers January 2010 to November 2017 across two walk-forward folds, with 37 priced portfolios in each fold and both basket widths. This basket is selected retrospectively and cannot substitute for a point-in-time constituent and delisting feed.

In the terminal ETF fold, AdamW mean daily Rank IC was +0.0422 over 80 negative-trailing-SPY sessions and −0.0248 over 409 nonnegative-trailing-SPY sessions. Separate moments averaged +0.0220 and −0.0232 over those same sessions. Regimes are determined by SPY’s trailing 60-session return available at the signal close. AdamW averaged about 45% gross exposure to sector ETFs in the terminal decile portfolios. Reported sector exposures refer to ETF positions, without a look-through into broader fund holdings. Regime comparisons are descriptive. Date-block intervals computed inside a regime operate on that subset’s dated sequence.

The turnover sensitivity is a signed-dollar simulation. At a shared exit and entry timestamp it marks existing positions, sizes the next weights using pre-cost equity and charges fees on the difference in signed dollar exposure. It settles the final cohort in full. The base simulator charges separately for each cohort exit and entry. Both use the recorded opening observations and omit financing, borrow and market impact.

[IC distributions](daily_ic_distributions.png) · [Date-block intervals and quantiles](distribution_summary.json) · [ETF regimes, basket-width stress and exposures](evidence/etf-robustness.json) · [Equity netting](evidence/wiki-netting.json) · [ETF netting](evidence/etf-netting.json)

## Reproduce

Set up the repository environment as described in the main README. Download the [Quandl WIKI mirror](https://www.kaggle.com/datasets/marketneutral/quandl-wiki-prices-us-equites) and the [Boris Marjanovic adjusted daily archive](https://www.kaggle.com/datasets/borismarjanovic/price-volume-data-for-all-us-stocks-etfs), both as ZIP files. The project publishes derived results and archive hashes.

```sh
python scripts/prepare_wiki.py data/wiki-prices.zip --csv data/wiki-panel.csv --audit data/wiki-audit.json
loopalpha market --csv data/wiki-panel.csv --calendar configs/wiki_xnys_sessions.csv \
  --prices adjusted --source 'Quandl WIKI/PRICES public-domain archive via marketneutral Kaggle mirror; vendor split/dividend-adjusted OHLCV' \
  --out runs/wiki-market
python scripts/reprice_wiki.py --panel data/wiki-panel.csv --calendar configs/wiki_xnys_sessions.csv \
  --alternate data/alternate-prices.zip --run runs/wiki-market --out runs/wiki-repricing.json
python scripts/prepare_etfs.py data/alternate-prices.zip --out data/etf-panel.csv
loopalpha market --csv data/etf-panel.csv --calendar configs/etf_xnys_sessions.csv \
  --prices adjusted --source 'Boris Marjanovic Huge Stock Market Dataset; Kaggle published adjusted daily OHLCV' \
  --out runs/etf-market
python scripts/etf_robustness.py --panel data/etf-panel.csv --calendar configs/etf_xnys_sessions.csv \
  --run runs/etf-market --out runs/etf-robustness.json
python scripts/turnover_sensitivity.py --panel data/wiki-panel.csv --calendar configs/wiki_xnys_sessions.csv \
  --run runs/wiki-market --recoveries runs/wiki-repricing.json --alternate data/alternate-prices.zip \
  --out runs/wiki-netting.json
python scripts/turnover_sensitivity.py --panel data/etf-panel.csv --calendar configs/etf_xnys_sessions.csv \
  --run runs/etf-market --out runs/etf-netting.json
python scripts/extended_benchmark.py --out runs/extended
python scripts/profile_benchmark.py --out runs/extended/optimizer-profile.json
python scripts/summarize_followup.py --evidence runs/extended --out runs/followup
python scripts/compare_market.py --wiki runs/wiki-market --etf runs/etf-market --out runs/followup
```

Use fresh output paths and run timed work sequentially. Settings, data and seeds are pinned. Wall times, timing checkpoints and run manifests can vary with machine load and runtime versions.
