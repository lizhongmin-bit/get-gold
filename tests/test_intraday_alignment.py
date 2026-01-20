import pandas as pd

from tail30_selector.indicators.intraday import intraday_strength


def test_intraday_outperformance_alignment():
    timestamps = pd.date_range("2024-01-02 14:30", periods=5, freq="1min")
    stock_df = pd.DataFrame(
        {
            "datetime": timestamps,
            "close": [10, 10.2, 10.3, 10.5, 10.6],
            "volume": [100, 120, 110, 130, 125],
            "amount": [1000, 1224, 1133, 1365, 1325],
        }
    )
    index_df = pd.DataFrame(
        {
            "datetime": timestamps,
            "close": [3000, 3001, 3001.5, 3002, 3002.5],
            "volume": [1000, 1100, 1050, 1150, 1200],
            "amount": [3000000, 3301100, 3151575, 3452300, 3603000],
        }
    )
    result = intraday_strength(stock_df, index_df)
    assert result.outperformance_last_tail >= 0
