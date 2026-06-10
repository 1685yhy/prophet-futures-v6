"""Unit tests for tools/next_day_predictor.py"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import numpy as np
import pandas as pd
from tools.next_day_predictor import predict_next_day


def _make_df(closes, oi_trend="flat", n=60):
    """Build a test DataFrame."""
    closes = list(closes)
    if len(closes) < n:
        closes = [closes[0]] * (n - len(closes)) + closes
    highs  = [c * 1.005 for c in closes]
    lows   = [c * 0.995 for c in closes]
    opens  = [closes[i-1] if i > 0 else closes[0] for i in range(len(closes))]
    vols   = [50000.0] * len(closes)

    if oi_trend == "up":
        oi = [200000 + i * 500 for i in range(len(closes))]
    elif oi_trend == "down":
        oi = [250000 - i * 500 for i in range(len(closes))]
    else:
        oi = [200000.0] * len(closes)

    return pd.DataFrame({
        "open": opens, "high": highs, "low": lows,
        "close": closes, "volume": vols, "oi": oi,
    })


class TestBullishSignals:
    def test_uptrend_with_macd_positive_gives_up(self):
        # 持续上涨趋势，MACD应为正值
        closes = [5000 + i * 20 for i in range(80)]
        df = _make_df(closes, oi_trend="up")
        result = predict_next_day(df)
        assert result["direction"] == "UP"
        assert result["confidence"] > 0.2

    def test_score_above_5_for_bull_setup(self):
        closes = [5000 + i * 15 for i in range(80)]
        df = _make_df(closes, oi_trend="up")
        result = predict_next_day(df)
        assert result["score"] > 5

    def test_support_below_current_price(self):
        closes = [5000 + i * 10 for i in range(70)]
        df = _make_df(closes)
        result = predict_next_day(df)
        assert result["support"] < closes[-1]

    def test_resistance_above_current_price(self):
        closes = [5000 + i * 10 for i in range(70)]
        df = _make_df(closes)
        result = predict_next_day(df)
        assert result["resistance"] > closes[-1]


class TestBearishSignals:
    def test_downtrend_with_macd_negative_gives_down_or_neutral(self):
        # 下降趋势得分应 <= 5（中性或偏空）
        closes = [12000 - i * 20 for i in range(80)]
        df = _make_df(closes, oi_trend="down")
        result = predict_next_day(df)
        assert result["direction"] in ("DOWN", "NEUTRAL")
        # 空头得分不应高于6
        assert result["score"] <= 6

    def test_score_below_or_equal_5_for_bear_setup(self):
        closes = [12000 - i * 15 for i in range(80)]
        df = _make_df(closes, oi_trend="down")
        result = predict_next_day(df)
        assert result["score"] <= 5

    def test_bear_market_oi_reducing_adds_to_bear_score(self):
        closes_down  = _make_df([12000 - i*15 for i in range(80)], oi_trend="down")
        closes_flat  = _make_df([12000 - i*15 for i in range(80)], oi_trend="flat")
        r_down = predict_next_day(closes_down)
        r_flat = predict_next_day(closes_flat)
        assert r_down["score"] <= r_flat["score"]


class TestNeutralSignals:
    def test_flat_market_neutral(self):
        import math
        closes = [5000 + math.sin(i * 0.3) * 50 for i in range(80)]
        df = _make_df(closes)
        result = predict_next_day(df)
        # 震荡市得分应接近5
        assert 2 <= result["score"] <= 8

    def test_insufficient_data_returns_neutral(self):
        # 少于20行数据触发不足保护
        import pandas as pd
        df = pd.DataFrame({
            "open": [5000.0]*5, "high": [5010.0]*5,
            "low":  [4990.0]*5, "close":[5000.0]*5,
            "volume":[1000.0]*5, "oi":[10000.0]*5,
        })
        result = predict_next_day(df)
        assert result["direction"] == "NEUTRAL"


class TestOutputStructure:
    def test_all_required_keys_present(self):
        df = _make_df([5000 + i * 10 for i in range(70)])
        result = predict_next_day(df)
        required = ["direction", "confidence", "score", "key_signals",
                    "support", "resistance", "action_advice", "risk_note"]
        for k in required:
            assert k in result, f"Missing key: {k}"

    def test_confidence_in_range(self):
        df = _make_df([5000 + i * 10 for i in range(70)])
        result = predict_next_day(df)
        assert 0.0 <= result["confidence"] <= 1.0

    def test_score_in_range(self):
        df = _make_df([5000 + i * 10 for i in range(70)])
        result = predict_next_day(df)
        assert 0 <= result["score"] <= 10

    def test_direction_is_valid(self):
        df = _make_df([5000 + i * 10 for i in range(70)])
        result = predict_next_day(df)
        assert result["direction"] in ("UP", "DOWN", "NEUTRAL")
