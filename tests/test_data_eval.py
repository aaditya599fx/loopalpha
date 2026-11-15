import numpy as np
import pandas as pd
import pytest

from loopalpha.data import market_samples, read_sessions, synthetic_panel, validate_panel
from loopalpha.evaluation import block_interval, correlations, cost_metrics, portfolio


def test_causal_windows_calendar_label_and_embargo():
    frame = synthetic_panel(13, 90, 8)
    cutoff = frame.date.sort_values().unique()[46]
    baseline = market_samples(validate_panel(frame), 8, 5)
    mutated = frame.copy()
    mutated.loc[mutated.date > cutoff, ["open", "high", "low", "close"]] *= 2
    perturbed = market_samples(validate_panel(mutated), 8, 5)
    idx = baseline.date <= 46
    np.testing.assert_array_equal(baseline.x[idx], perturbed.x[idx])
    row = np.flatnonzero((baseline.date == 46) & (baseline.symbol == "S000"))[0]
    expected = np.log(baseline.opens.S000.iloc[52] / baseline.opens.S000.iloc[47])
    assert baseline.realized[row] == pytest.approx(expected)
    assert perturbed.realized[row] == pytest.approx(expected)
    for start, stop in [(0, 47), (47, 65), (65, 90)]:
        train = baseline.split(start, stop)
        assert baseline.date[train].min() >= start
        assert baseline.date[train].max() + 6 < stop


def test_missing_exit_stays_a_candidate_with_missing_target():
    frame = synthetic_panel(10, 70, 5)
    dates = frame.date.unique()
    frame = frame[~(frame.date.eq(dates[46]) & frame.symbol.eq("S000"))]
    samples = market_samples(validate_panel(frame), 5, 5)
    subset = (samples.date == 40) & (samples.symbol == "S000")
    assert subset.sum() == 1
    assert np.isnan(samples.realized[subset]).all()
    assert np.isfinite(samples.x[subset]).all()


def test_duplicate_ohlcv_rejected():
    frame = synthetic_panel(10, 5, 5)
    with pytest.raises(ValueError, match="duplicate"):
        validate_panel(pd.concat([frame, frame.iloc[[0]]]))


def test_rank_ic_uses_daily_ties_and_uncertainty():
    frame = pd.DataFrame({"date": [1] * 4 + [2] * 4, "prediction": [1, 2, 2, 3] * 2,
                          "realized": [1, 2, 2, 4, 4, 2, 2, 1]})
    daily = correlations(frame)
    np.testing.assert_allclose(daily.rank_ic, [1, -1])
    assert block_interval([2] * 20, 5) == [2., 2.]


def test_backtest_entry_exit_cash_and_costs():
    prediction = pd.DataFrame({"session": [0] * 4, "symbol": list("ABCD"),
                               "prediction": [-2, -1, 1, 2]})
    opens = pd.DataFrame([[99, 99, 99, 99], [10, 10, 10, 10], [8, 10, 10, 12]],
                         index=pd.bdate_range("2020-03-02", periods=3), columns=list("ABCD"))
    ledger = portfolio(prediction, opens, 1, 0, 3)
    assert ledger.weight.abs().sum() == 1.
    assert ledger.gross.sum() == pytest.approx(.2)
    assert ledger.traded_notional.sum() == pytest.approx(2.)
    assert cost_metrics(ledger, 1, 10, 0, 3)["net_return"] == pytest.approx(.198)
    reversed_scores = prediction.assign(prediction=-prediction.prediction)
    assert portfolio(reversed_scores, opens, 1, 0, 3).gross.sum() == pytest.approx(-.2)
    assert portfolio(prediction.assign(prediction=0), opens, 1, 0, 3).empty
    opens.iloc[2, 3] = np.nan
    assert cost_metrics(portfolio(prediction, opens, 1, 0, 3), 1, 10, 0, 3)["status"] == "unpriced_exit"
    opens.iloc[1, 3] = np.nan
    assert portfolio(prediction, opens, 1, 0, 3).weight.tolist() == [-.5, 0.]



def test_explicit_calendar_preserves_a_session_missing_from_every_asset(tmp_path):
    frame = synthetic_panel(11, 85, 8).assign(eligible=True)
    dates = pd.DatetimeIndex(frame.date.unique())
    missing_open = dates[51]
    frame = validate_panel(frame.loc[frame.date.ne(missing_open)])
    calendar = tmp_path / "dates.csv"
    pd.DataFrame({"date": dates}).to_csv(calendar, index=False)
    sessions, audit = read_sessions(calendar, frame)

    samples = market_samples(frame, 5, 5, sessions=sessions)
    assert audit["sessions_without_bars"] == [str(missing_open.date())]
    assert samples.sessions == len(dates)
    assert samples.opens.iloc[51].isna().all()
    assert np.isnan(samples.realized[samples.date == 45]).all()
    assert np.isfinite(samples.realized[samples.date == 44]).all()
    assert np.isfinite(samples.realized[samples.date == 46]).all()


def test_explicit_calendar_rejects_excluded_bar_dates(tmp_path):
    frame = validate_panel(synthetic_panel(11, 85, 8))
    dates = pd.DatetimeIndex(frame.date.unique()).delete(51)
    calendar = tmp_path / "dates.csv"
    pd.DataFrame({"date": dates}).to_csv(calendar, index=False)
    with pytest.raises(ValueError, match="must belong"):
        read_sessions(calendar, frame)
