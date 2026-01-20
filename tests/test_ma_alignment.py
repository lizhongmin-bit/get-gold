import pandas as pd

from tail30_selector.indicators.ma import build_ma_snapshot, is_bullish_alignment


def test_bullish_alignment():
    close = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    snapshot = build_ma_snapshot(close, [2, 3, 4, 5])
    assert is_bullish_alignment(snapshot)
