"""Calendar-aligned OHLCV preparation and deterministic synthetic inputs."""

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view


FIELDS = ("open", "high", "low", "close", "volume")
FEATURES = ("ret1", "ret5", "ret20", "vol20", "relvolume", "close_ma", "range")


@dataclass
class Samples:
    x: np.ndarray
    y: np.ndarray
    date: np.ndarray
    symbol: np.ndarray
    signal: np.ndarray | None = None
    realized: np.ndarray | None = None
    dates: pd.DatetimeIndex | None = None
    opens: pd.DataFrame | None = None
    horizon: int = 0

    def split(self, start, stop):
        # Require both signal and the final label observation to lie inside the split.
        return np.flatnonzero((self.date >= start) & (self.date + self.horizon + 1 < stop))

    @property
    def sessions(self):
        return len(self.dates) if self.dates is not None else int(self.date.max()) + 1


def read_panel(path, *, source, adjusted):
    if not source.strip():
        raise ValueError("Supply the data source and price convention")
    payload = Path(path).read_bytes()
    from io import BytesIO
    frame = pd.read_csv(BytesIO(payload), dtype={"symbol": "string"})
    frame = validate_panel(frame)
    audit = {"source": source, "adjusted": adjusted, "sha256": sha256(payload).hexdigest(),
             "rows": len(frame), "assets": int(frame.symbol.nunique()),
             "first_session": str(frame.date.min().date()), "last_session": str(frame.date.max().date())}
    return frame, audit


def validate_panel(frame):
    missing = set(("date", "symbol", *FIELDS)) - set(frame)
    if missing:
        raise ValueError(f"Missing fields: {sorted(missing)}")
    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame.date, format="%Y-%m-%d", errors="raise")
    if frame.empty or frame.date.isna().any() or frame.symbol.isna().any():
        raise ValueError("Nonempty date and symbol keys required")
    if not frame.date.eq(frame.date.dt.normalize()).all():
        raise ValueError("Dates must represent session dates without a time")
    if frame.symbol.str.strip().eq("").any() or frame.duplicated(["date", "symbol"]).any():
        raise ValueError("Blank symbol or duplicate date-symbol key")
    for name in FIELDS:
        frame[name] = pd.to_numeric(frame[name], errors="raise")
    if not np.isfinite(frame[list(FIELDS)]).all().all():
        raise ValueError("OHLCV values must be finite")
    if (frame[list(FIELDS[:-1])] <= 0).any().any() or (frame.volume < 0).any():
        raise ValueError("Prices must be positive and volumes nonnegative")
    if ((frame.high < frame[["open", "close", "low"]].max(axis=1)) |
        (frame.low > frame[["open", "close"]].min(axis=1))).any():
        raise ValueError("Inconsistent OHLC bounds")
    if "eligible" in frame:
        if not frame.eligible.isin([True, False, 0, 1]).all():
            raise ValueError("eligible must contain only boolean values")
        frame["eligible"] = frame.eligible.astype(bool)
    return frame.sort_values(["date", "symbol"])


def read_sessions(path, frame):
    from io import BytesIO

    payload = Path(path).read_bytes()
    values = pd.read_csv(BytesIO(payload), usecols=["date"])["date"]
    sessions = pd.DatetimeIndex(pd.to_datetime(values, format="%Y-%m-%d", errors="raise"))
    if (sessions.empty or sessions.hasnans or sessions.has_duplicates
            or not sessions.is_monotonic_increasing or not sessions.equals(sessions.normalize())):
        raise ValueError("Calendar must contain unique, ascending session dates")
    observed = pd.DatetimeIndex(frame.date.unique()).sort_values()
    if not observed.isin(sessions).all():
        raise ValueError("Bar dates must belong to the supplied calendar")
    if sessions.min() != observed.min() or sessions.max() != observed.max():
        raise ValueError("Calendar and bars must cover the same first and last sessions")
    audit = {"calendar": "explicit session dates", "calendar_sha256": sha256(payload).hexdigest(),
             "sessions_without_bars": sessions.difference(observed).strftime("%Y-%m-%d").tolist()}
    return sessions, audit


def market_samples(frame, lookback, horizon, *, sessions=None):
    if lookback < 1 or horizon < 1:
        raise ValueError("Lookback and horizon must be positive")
    pivots = {name: frame.pivot(index="date", columns="symbol", values=name)
              for name in FIELDS}
    if sessions is not None:
        pivots = {name: values.reindex(sessions) for name, values in pivots.items()}
    opening, high, low, close, volume = (pivots[name] for name in FIELDS)
    if len(close) < lookback + 20 + horizon + 1:
        raise ValueError("Panel has too few sessions for the lookback and horizon")
    # End-of-day volume cannot be used to decide whether an opening order fills.
    tradable_open = opening
    eligible = volume.gt(0)
    if "eligible" in frame:
        eligible &= frame.pivot(index="date", columns="symbol", values="eligible").reindex(eligible.index).eq(True)
    close = close.where(volume.gt(0))
    returns = np.log(close).diff()
    momentum = np.log(close).diff(5)
    features = np.stack([returns, momentum, np.log(close).diff(20),
                         returns.rolling(20).std(), np.log1p(volume) - np.log1p(volume.rolling(20).mean()),
                         np.log(close / close.rolling(20).mean()), np.log(high / low)], axis=-1)
    windows = sliding_window_view(features, lookback, axis=0).transpose(0, 1, 3, 2)
    windows = np.pad(windows, ((lookback - 1, 0), (0, 0), (0, 0), (0, 0)), constant_values=np.nan)
    candidate = eligible.to_numpy() & np.isfinite(windows).all(axis=(2, 3))
    raw = np.log(tradable_open.shift(-(horizon + 1)) / tradable_open.shift(-1))
    raw = raw.where(candidate)
    centered = raw.sub(raw.median(axis=1), axis=0)
    targets = centered.div(raw.std(axis=1, ddof=0).replace(0, np.nan), axis=0)
    dates, tickers = np.where(candidate)
    if not len(dates):
        raise ValueError("No eligible complete feature windows")
    return Samples(windows[dates, tickers].astype(np.float32),
                   targets.to_numpy()[dates, tickers].astype(np.float32), dates,
                   close.columns.to_numpy()[tickers], momentum.to_numpy()[dates, tickers],
                   raw.to_numpy()[dates, tickers], close.index, tradable_open, horizon)


def synthetic_panel(seed, sessions, assets):
    rng = np.random.default_rng(seed)
    regimes = np.repeat([0.62, -0.2, 0.45], [sessions // 3, sessions // 3, sessions - 2 * (sessions // 3)])
    shocks = rng.normal(0, 0.008, (sessions, assets))
    moves = shocks.copy()
    for day in range(1, sessions):
        moves[day] += regimes[day] * moves[day - 1] + rng.normal(0, 0.002)
    close = 70 * np.exp(np.cumsum(moves, axis=0))
    opening = np.vstack([np.full(assets, 70.), close[:-1]]) * np.exp(rng.normal(0, .002, close.shape))
    dates = pd.bdate_range("2018-01-02", periods=sessions)
    return pd.DataFrame({"date": np.repeat(dates, assets),
                         "symbol": np.tile([f"S{i:03}" for i in range(assets)], sessions),
                         "open": opening.ravel(), "close": close.ravel(),
                         "high": np.maximum(opening, close).ravel() * 1.003,
                         "low": np.minimum(opening, close).ravel() * .997,
                         "volume": rng.integers(200_000, 2_000_000, close.size)})


def toy_samples(seed, sessions, assets, length):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(sessions * assets, length, 3)).astype(np.float32)
    state = x[:, 0, 0]
    for period in range(1, length):
        state = np.tanh(0.8 * state + x[:, period, 0] * x[:, period, 1]) + .2 * x[:, period, 2]
    y = state + rng.normal(0, .03, size=len(state))
    return Samples(x, y.astype(np.float32), np.repeat(np.arange(sessions), assets),
                   np.tile(np.arange(assets), sessions))
