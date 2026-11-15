# LoopAlpha

LoopAlpha tests whether an optimizer can benefit from seeing the individual gradient contributions of a shared block that runs repeatedly. It benchmarks convergence on synthetic sequence regression, then measures transfer to cross-sectional return ranking on a chronological market panel.

## Latest Research Findings

LoopAlpha investigates whether per-depth gradient information improves optimization in shared-weight neural networks.

**Key findings**

- Verified that the gradients from each use of a shared parameter sum to the ordinary tied-weight gradient.
- Benchmarked seven optimization methods across two synthetic tasks and ten evaluation seeds.
- Muon achieved the lowest mean held-out error at every tested CPU time budget on both tasks.
- Recovered the missing exit prices and reran all 37 final-period equity portfolios.
- The historical experiments found no statistically reliable ranking advantage for the depth-aware methods. Every trained method had negative mean net returns in the final equity period at 10 basis points per dollar traded.

**Research materials**

- [Technical Research Report (PDF)](reports/followup/technical_report.pdf)
- [Experiments, Evidence & Reproduction](reports/followup/README.md)

[Experimental protocol](reports/research_plan.md) · [Generated results](reports/results.md) · [Research interpretation](reports/research_report.md) · [Market CSV contract](data/README.md) · [Historical U.S. equities benchmark](reports/historical/README.md) · [Follow-up validation and ablations](reports/followup/README.md)

## Reproduce

Use Python 3.13 on macOS ARM64 for the supplied environment lock. Other PyTorch-supported systems can install the project's pinned direct dependencies using `python -m pip install -e '.[test]'`.

```sh
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock
python -m pip install --no-build-isolation --no-deps -e .
python -m pytest -q

loopalpha toy --out runs/toy
loopalpha synthetic-market --out runs/synthetic-market
python scripts/make_report.py --toy runs/toy --market runs/synthetic-market --out reports
python scripts/audit_depth.py --out reports/depth_audit.json
```

Use fresh output paths for new runs. Run benchmarks sequentially on an idle machine for wall-clock comparisons. Retraining reproduces the seeded data and experiment design. Timing and checkpoint selection may vary with the machine and CPU load.

Run the same chronological pipeline on your own permitted historical feed.

```sh
loopalpha market \
  --csv data/daily.csv \
  --source 'Vendor, license, extraction date and price convention' \
  --prices adjusted \
  --out runs/historical
```

To reproduce the historical benchmark, download the [Quandl WIKI archive mirror](https://www.kaggle.com/datasets/marketneutral/quandl-wiki-prices-us-equites) and save it as `data/wiki-prices.zip`.

```sh
python scripts/prepare_wiki.py data/wiki-prices.zip --csv data/wiki-panel.csv --audit data/wiki-audit.json
loopalpha market --csv data/wiki-panel.csv --calendar configs/wiki_xnys_sessions.csv --prices adjusted \
  --source 'Quandl WIKI/PRICES public-domain archive via marketneutral Kaggle mirror; vendor split/dividend-adjusted OHLCV' \
  --out runs/wiki-market
python scripts/make_report.py --market runs/wiki-market --out runs/wiki-market/report
```

## Experiment design

The sequence model embeds a lookback window and repeatedly applies one LayerNorm residual MLP. The untied baseline has the same depth, width and initial forward function with independently trainable blocks. Additional baselines include momentum, linear regression and a shallow MLP.

Ordinary autograd adds the gradients from each use of the shared block. The probe substitutes equal-valued differentiable copies in a single forward pass and reads each copy's gradient after backpropagation. Tests verify that their sum reconstructs the tied gradient for every parameter. The default candidate applies AdamW to their mean and preserves ordinary gradients for the encoder and head. Ablations apply depthwise norm clipping, increasing fixed weights or independent Adam moments per occurrence. Muon operates on residual hidden matrices, with AdamW covering all remaining parameters.

The market experiment fits scaling on each expanding training window, selects learning rates on validation only, and trims labels whose exits cross partition boundaries. It reports per-date Pearson IC and Spearman Rank IC. The backtest enters at the next open, closes each cohort after five sessions, and charges costs on actual traded notional. It ignores borrow fees, financing and market impact. The included historical-data interface assumes the supplied opens are usable execution proxies.

## Outputs and verification

Each run saves input provenance, code and configuration hashes, per-seed validation traces, selected learning rates, and inference checkpoints containing the training feature transform. Market runs also save per-date IC, predictions and position-level trade ledgers. The checked-in evidence includes the synthetic benchmark and a separate historical U.S. equities benchmark. The synthetic sample ledger can be audited independently of PyTorch.

This is an independent portfolio project inspired by the CaCTüS looped-optimizer problem. The historical results describe a fixed stock cohort in a single archived data source. The historical report documents the source, coverage, results and execution assumptions.
