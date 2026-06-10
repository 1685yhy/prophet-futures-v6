"""Unit tests for tools/position_manager.py"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from tools.position_manager import get_position_advice, format_position_report


def _make_pred(direction="NEUTRAL", confidence=0.5, support=11500, resistance=12200):
    return {
        "direction":     direction,
        "confidence":    confidence,
        "score":         7 if direction == "UP" else (3 if direction == "DOWN" else 5),
        "support":       support,
        "resistance":    resistance,
        "action_advice": "test",
        "risk_note":     "当前无特别风险提示",
        "atr":           200.0,
        "bb_pos":        0.5,
        "key_signals":   [],
    }


def _make_pos(direction="SHORT", entry=11910, stop=12115, target=11295,
              entry_date="2026-06-09"):
    return {
        "direction":  direction,
        "entry":      entry,
        "stop":       stop,
        "target":     target,
        "entry_date": entry_date,
    }


class TestShortPositionManagement:
    def test_short_next_up_high_confidence_partial_exit(self):
        pos    = _make_pos("SHORT", entry=11910, stop=12115, target=11295)
        pred   = _make_pred("UP", confidence=0.70)
        result = get_position_advice(pos, 11800, pred, {}, atr=200)
        assert result["action"] == "PARTIAL_EXIT"
        assert result["urgency"] == "HIGH"

    def test_short_next_up_mid_confidence_tighten_stop(self):
        pos    = _make_pos("SHORT", entry=11910, stop=12115, target=11295)
        pred   = _make_pred("UP", confidence=0.45)
        result = get_position_advice(pos, 11800, pred, {}, atr=200)
        assert result["action"] == "TIGHTEN_STOP"
        assert result["urgency"] == "MEDIUM"
        assert result["new_stop"] is not None
        # 空单止损应下移（比原止损更低）
        assert result["new_stop"] < 12115

    def test_short_next_down_in_profit_can_add(self):
        pos    = _make_pos("SHORT", entry=12200, stop=12500, target=11600)
        pred   = _make_pred("DOWN", confidence=0.55)
        # 当前11800，浮盈400点 > ATR200
        result = get_position_advice(pos, 11800, pred, {}, atr=200)
        assert result["action"] in ("ADD", "TIGHTEN_STOP", "HOLD")
        assert result["pnl_now"] == pytest.approx(400.0, abs=1)

    def test_short_next_neutral_hold(self):
        pos    = _make_pos("SHORT", entry=11910, stop=12115, target=11295)
        pred   = _make_pred("NEUTRAL", confidence=0.0)
        result = get_position_advice(pos, 11850, pred, {}, atr=200)
        assert result["action"] in ("HOLD", "TIGHTEN_STOP")  # NEUTRAL时可能追踪止损

    def test_short_pnl_calculated_correctly(self):
        pos    = _make_pos("SHORT", entry=12000, stop=12300, target=11400)
        pred   = _make_pred("NEUTRAL")
        result = get_position_advice(pos, 11700, pred, {}, atr=200)
        assert result["pnl_now"] == pytest.approx(300.0, abs=1)  # 12000 - 11700
        assert result["pnl_pct"] > 0


class TestLongPositionManagement:
    def test_long_next_down_high_confidence_partial_exit(self):
        pos    = _make_pos("LONG", entry=11500, stop=11200, target=12200)
        pred   = _make_pred("DOWN", confidence=0.70)
        result = get_position_advice(pos, 11800, pred, {}, atr=200)
        assert result["action"] == "PARTIAL_EXIT"

    def test_long_next_down_mid_confidence_tighten_stop(self):
        pos    = _make_pos("LONG", entry=11500, stop=11200, target=12200)
        pred   = _make_pred("DOWN", confidence=0.45)
        result = get_position_advice(pos, 11800, pred, {}, atr=200)
        assert result["action"] == "TIGHTEN_STOP"
        assert result["new_stop"] > 11200  # 多单止损上移

    def test_long_pnl_negative_when_price_below_entry(self):
        pos    = _make_pos("LONG", entry=12000, stop=11700, target=12600)
        pred   = _make_pred("NEUTRAL")
        result = get_position_advice(pos, 11800, pred, {}, atr=200)
        assert result["pnl_now"] == pytest.approx(-200.0, abs=1)
        assert result["pnl_pct"] < 0


class TestGapForcedExit:
    def test_gap_up_forces_full_exit_for_short(self):
        pos    = _make_pos("SHORT", entry=11910, stop=12115, target=11295)
        pred   = _make_pred("UP", confidence=0.8)
        # 跳空3%
        result = get_position_advice(pos, 12280, pred, {}, atr=200, prev_close=11900)
        assert result["action"] == "FULL_EXIT"
        assert result["urgency"] == "HIGH"


class TestMaxHoldDays:
    def test_old_position_prepares_exit(self):
        pos = _make_pos("SHORT", entry_date="2026-06-01")  # 很久以前
        pred = _make_pred("NEUTRAL")
        result = get_position_advice(pos, 11800, pred, {}, atr=200)
        assert result["hold_more"] is False or result["action"] == "FULL_EXIT"


class TestFormatReport:
    def test_report_contains_key_sections(self):
        pos    = _make_pos("SHORT", entry=11910, stop=12115, target=11295)
        pred   = _make_pred("UP", confidence=0.45)
        advice = get_position_advice(pos, 11850, pred, {}, atr=200)
        report = format_position_report("LH2609", pos, 11850, pred, advice)
        assert "持仓状态" in report
        assert "今日盘面分析" in report
        assert "持仓管理建议" in report
