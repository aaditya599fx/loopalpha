"""Daily cross-sectional statistics and a non-overlapping trade ledger."""

import numpy as np
import pandas as pd
from scipy.stats import rankdata


def correlations(predictions):
    rows = []
    for date, group in predictions.groupby("date", sort=True):
        group = group.dropna(subset=["prediction", "realized"])
        x, y = group.prediction.to_numpy(), group.realized.to_numpy()
        if len(group) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
            continue
        rows.append((date, len(group), np.corrcoef(x, y)[0, 1],
                     np.corrcoef(rankdata(x), rankdata(y))[0, 1]))
    return pd.DataFrame(rows, columns=["date", "assets", "ic", "rank_ic"])


def block_interval(values, block, seed=73, draws=1000):
    values = np.asarray(values, dtype=float)
    if len(values) < block * 2:
        return [None, None]
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, len(values), size=(draws, int(np.ceil(len(values) / block))))
    indices = (starts[:, :, None] + np.arange(block)) % len(values)
    means = values[indices.reshape(draws, -1)[:, :len(values)]].mean(axis=1)
    return np.quantile(means, [.025, .975]).tolist()


def portfolio(predictions, opens, horizon, start, stop, fraction=.1):
    """Close each cohort in full. Retain unfilled entry weight in cash."""
    if horizon < 1 or not 0 < fraction <= .5:
        raise ValueError("Invalid horizon or portfolio fraction")
    ledger = []
    for day in range(start, stop - horizon - 1, horizon):
        group = predictions.loc[predictions.session.eq(day)].dropna(subset=["prediction"])
        group = group.sort_values(["prediction", "symbol"])
        if len(group) < 4 or group.prediction.nunique() < 2:
            continue
        count = max(1, int(len(group) * fraction))
        for side, selected in [(-1, group.head(count)), (1, group.tail(count))]:
            for symbol in selected.symbol:
                entry, exit_ = opens.iloc[[day + 1, day + horizon + 1]][symbol].to_numpy()
                filled = bool(np.isfinite(entry))
                weight = side * .5 / count if filled else 0.
                position_return = 0.
                if filled:
                    position_return = weight * (exit_ / entry - 1)
                traded = 0. if not filled else abs(weight) * (1 + exit_ / entry)
                ledger.append((day, opens.index[day], opens.index[day + 1],
                               opens.index[day + horizon + 1], symbol, side,
                               entry, exit_, weight, position_return, traded))
    return pd.DataFrame(ledger, columns=["session", "signal_date", "entry_date", "exit_date",
                                         "symbol", "side", "entry_open", "exit_open", "weight",
                                         "gross", "traded_notional"])


def cost_metrics(ledger, horizon, bps, start, stop):
    schedule = pd.Index(range(start, stop - horizon - 1, horizon), name="session")
    if len(schedule) == 0:
        return {"status": "no_holding_periods"}
    if not ledger.empty and ledger.gross.isna().any():
        return {"status": "unpriced_exit", "affected_trades": int(ledger.gross.isna().sum())}
    groups = ledger.groupby("session")[["gross", "traded_notional"]].sum()
    cohorts = groups.reindex(schedule, fill_value=0.)
    gross = cohorts.gross
    net = gross - cohorts.traded_notional * bps / 10_000
    if (net <= -1).any():
        return {"status": "capital_exhausted"}
    equity = np.r_[1., (1 + net).cumprod()]
    volatility = net.std()
    return {"status": "ok", "cohorts": len(cohorts),
            "gross_return": float((1 + gross).prod() - 1),
            "net_return": float(equity[-1] - 1),
            "net_sharpe_rf0": float(net.mean() / volatility * np.sqrt(252 / horizon))
            if np.isfinite(volatility) and volatility > 0 else None,
            "max_drawdown": float(np.min(equity / np.maximum.accumulate(equity) - 1)),
            "mean_traded_notional": float(cohorts.traded_notional.mean())}
