import pandas as pd

from scripts import prepare_wiki


def test_wiki_universe_is_formed_before_evaluation(tmp_path, monkeypatch):
    monkeypatch.setattr(prepare_wiki, "UNIVERSE_SIZE", 2)
    rows = []
    for date in pd.bdate_range("2012-01-02", "2012-02-03"):
        for symbol, close in [("A", 20), ("B", 40), ("C", 10)]:
            # C becomes the most liquid ticker only after formation.
            volume = 10_000 if symbol == "C" and date.month == 2 else 100
            if date.day == 2 and date.month == 2 and symbol == "A":
                continue
            rows.append((date.strftime("%Y-%m-%d"), symbol, close, volume, close * .5))
    frame = pd.DataFrame(rows, columns=["date", "ticker", "close", "volume", "adj_close"])
    frame["adj_open"] = frame.adj_close
    frame["adj_high"] = frame.adj_close + 1
    frame["adj_low"] = frame.adj_close - 1
    frame["adj_volume"] = frame.volume * 2
    frame.loc[frame.date.eq("2012-02-03") & frame.ticker.eq("B"), "adj_high"] = 15
    archive = tmp_path / "source.csv"
    frame.to_csv(archive, index=False)

    panel, audit = prepare_wiki.prepare(archive)

    assert audit["universe"] == ["B", "A"]
    assert audit["missing_date_symbol_bars"] == 2
    assert audit["rejected_invalid_bars"] == [{"date": "2012-02-03", "symbol": "B"}]
    assert audit["first_bar"] == "2012-02-01"
    assert panel.loc[panel.symbol.eq("B"), "close"].eq(20).all()
