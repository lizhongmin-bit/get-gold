import pandas as pd

from tail30_selector.indicators.volume_shape import analyze_volume_shape


def test_volume_shape_stepwise_vs_ecg():
    stepwise = pd.Series([100, 110, 120, 130, 150, 170, 190, 210])
    result_stepwise = analyze_volume_shape(stepwise, segments=3)
    assert result_stepwise.is_stepwise

    ecg = pd.Series([100, 300, 90, 310, 80, 320, 70, 330])
    result_ecg = analyze_volume_shape(ecg, segments=3)
    assert not result_ecg.is_stepwise
