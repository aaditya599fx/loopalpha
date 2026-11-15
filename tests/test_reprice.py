import pandas as pd
import pytest

from scripts.reprice_wiki import validated_open


def test_alternate_open_rebases_on_prior_sessions_only():
    dates = pd.bdate_range("2017-01-02", periods=25).strftime("%Y-%m-%d")
    alternate = pd.DataFrame({"open": 100., "high": 103., "low": 99., "close": 102.,
                              "volume": 100.}, index=dates)
    original = alternate[["open", "close"]] * 2.
    missing = dates[21]
    original = original.drop(missing)
    # A future scale change must not affect the selected open.
    original.loc[dates[22]:] *= 5
    alternate.loc[missing, "open"] = 101.
    recovered, audit = validated_open(original, alternate, missing)
    assert recovered == pytest.approx(202.)
    assert audit["last_overlap"] == dates[20]
    assert audit["status"] == "observed_rebased_open"
    original.loc[dates[20], "open"] *= 1.02
    assert validated_open(original, alternate, missing)[1]["status"] == "overlap_disagreement"


def test_netting_avoids_round_trips_for_same_position():
    from scripts.turnover_sensitivity import netted

    ledger = pd.DataFrame([
        ("2017-01-02", "2017-01-03", "A", 1., 100., 100.),
        ("2017-01-03", "2017-01-04", "A", 1., 100., 100.),
    ], columns=["entry_date", "exit_date", "symbol", "weight", "entry_open", "exit_open"])
    outcome = netted(ledger, 10)
    assert outcome["status"] == "ok"
    assert outcome["traded_notional_per_initial_dollar"] == pytest.approx(2.)
    assert outcome["net_return"] == pytest.approx(-.002)
