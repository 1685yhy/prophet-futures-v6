"""
次日方向预测模块 — 纯规则评分，不依赖 LLM。

评分体系（0-10分）：
  5分基准（NEUTRAL），>5偏多(UP)，<5偏空(DOWN)

偏多加分：MA多头(+1)、MACD正(+2)、MACD正扩大(+1)、
          RSI健康区(+1)、OI积累(+1)、今日阳线未破低(+1)、尾盘未减仓(+1)
偏空减分：MA空头(-1)、MACD负不收窄(-2)、RSI超买(-1)、
          OI减少(-1)、今日阴线且尾盘减仓(-1)、BB上轨阻力(-1)

用途：
  - 无持仓：辅助判断是否是入场时机
  - 有持仓：每日更新，决定是否收紧止损/减仓/加仓
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List

from tools.indicators import calc_indicators, _calc_macd


def predict_next_day(
    df_window: pd.DataFrame,
    ind: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """
    预测次日价格方向。

    Args:
        df_window: 含 open/high/low/close/volume/oi 列，至少 30 行
        ind:       已计算的指标字典；若 None 则内部计算

    Returns:
        {
            "direction":     "UP" | "DOWN" | "NEUTRAL",
            "confidence":    float,        # 0-1
            "score":         int,          # 0-10 (5=中性)
            "key_signals":   List[str],    # 触发的信号描述
            "support":       float,        # 关键支撑
            "resistance":    float,        # 关键压力
            "action_advice": str,          # 操作建议
            "risk_note":     str,          # 风险提示
        }
    """
    if len(df_window) < 20:
        return _neutral_result(0.0, "数据不足")

    if ind is None:
        ind = calc_indicators(df_window)

    closes = df_window["close"].values.astype(float)
    opens  = df_window["open"].values.astype(float)
    highs  = df_window["high"].values.astype(float)
    lows   = df_window["low"].values.astype(float)

    _, _, h0 = _calc_macd(closes)
    _, _, h1 = _calc_macd(closes[:-1]) if len(closes) > 1 else (0, 0, 0)
    _, _, h2 = _calc_macd(closes[:-2]) if len(closes) > 2 else (0, 0, 0)
    macd_improving = bool(h0 < 0 and h1 < 0 and abs(h0) < abs(h1) < abs(h2))
    macd_expanding = bool(h0 > 0 and h1 > 0 and h0 > h1)  # 正值扩大

    # OI 趋势
    oi_vals = df_window["oi"].values.astype(float) if "oi" in df_window.columns else np.zeros(10)
    oi_3d   = float(oi_vals[-1] - oi_vals[-4]) if len(oi_vals) >= 4 else 0
    oi_5d   = float(oi_vals[-1] - oi_vals[-6]) if len(oi_vals) >= 6 else oi_3d
    if oi_3d > 0 and oi_5d > 0:   oi_trend = "ACCUMULATING"
    elif oi_3d < 0 and oi_5d < 0: oi_trend = "REDUCING"
    else:                           oi_trend = "FLAT"

    # 尾盘OI变化（判断主力是否离场）
    oi_today_chg = float(oi_vals[-1] - oi_vals[-2]) if len(oi_vals) >= 2 else 0

    close  = ind["current_close"]
    ma5    = ind["ma5"];  ma20 = ind["ma20"];  ma60 = ind["ma60"]
    rsi    = ind["rsi14"]; adx = ind["adx14"]
    bb_upper = ind["bb_upper"]; bb_lower = ind["bb_lower"]
    atr    = ind["atr14"]
    prev_close = float(closes[-2]) if len(closes) >= 2 else close

    # 今日K线性质
    today_is_bull = closes[-1] >= opens[-1]
    today_new_low = lows[-1] < lows[-2] if len(lows) >= 2 else False

    # ── 评分 ────────────────────────────────────────────────────────────────
    score    = 5   # 基准中性
    signals: List[str] = []

    # 均线
    if ma5 > ma20:
        score += 1; signals.append("MA5>MA20 均线偏多(+1)")
    elif ma5 < ma20:
        score -= 1; signals.append("MA5<MA20 均线偏空(-1)")

    # MACD 柱（最重要，权重2）
    if h0 > 0 and not macd_improving:
        score += 2; signals.append(f"MACD柱正({h0:.1f}) 多头动能(+2)")
    elif h0 < 0 and not macd_improving:
        score -= 2; signals.append(f"MACD柱负({h0:.1f})不收窄 空头动能(-2)")
    elif macd_improving:
        score += 1; signals.append(f"MACD柱负值收窄 动能好转(+1)")

    # MACD 正值扩大
    if macd_expanding:
        score += 1; signals.append("MACD柱正值扩大 多头加速(+1)")

    # RSI
    if 40 <= rsi <= 60:
        prev_rsi = _calc_prev_rsi(closes)
        if prev_rsi is not None and rsi > prev_rsi:
            score += 1; signals.append(f"RSI={rsi:.0f}健康区上行(+1)")
    elif rsi > 72:
        score -= 1; signals.append(f"RSI={rsi:.0f}超买(-1)")
    elif rsi < 28:
        score += 1; signals.append(f"RSI={rsi:.0f}超卖反弹概率增(+1)")

    # OI 趋势
    if oi_trend == "ACCUMULATING":
        score += 1; signals.append(f"OI 3日持续积累(+1)")
    elif oi_trend == "REDUCING":
        score -= 1; signals.append(f"OI 3日持续减少(-1)")

    # 今日K线 + 尾盘OI
    if today_is_bull and not today_new_low:
        score += 1; signals.append("今日阳线未破前低 多头守护(+1)")
    elif not today_is_bull and oi_today_chg < -abs(oi_today_chg) * 0.1:
        score -= 1; signals.append("今日阴线+尾盘减仓 空头压制(-1)")

    # 布林带位置
    bb_pos = (close - bb_lower) / (bb_upper - bb_lower + 1e-8)
    if bb_pos > 0.88:
        score -= 1; signals.append(f"价格接近布林上轨({bb_pos:.0%}) 阻力区(-1)")
    elif bb_pos < 0.12:
        score += 1; signals.append(f"价格接近布林下轨({bb_pos:.0%}) 支撑区(+1)")

    score = max(0, min(10, score))

    # ── 方向和置信度 ────────────────────────────────────────────────────────
    if score > 5:
        direction  = "UP"
        confidence = min(0.90, (score - 5) / 5)
    elif score < 5:
        direction  = "DOWN"
        confidence = min(0.90, (5 - score) / 5)
    else:
        direction  = "NEUTRAL"
        confidence = 0.0

    # ── 关键价位 ────────────────────────────────────────────────────────────
    recent_lows  = lows[-10:]
    recent_highs = highs[-10:]
    support     = round(max(min(recent_lows), bb_lower, close - 2 * atr), 2)
    resistance  = round(min(max(recent_highs), bb_upper, close + 2 * atr), 2)
    # 取最近的支撑（不要离当前价太远）
    support    = round(max(support, close - 3 * atr), 2)
    resistance = round(min(resistance, close + 3 * atr), 2)

    # ── 操作建议 ────────────────────────────────────────────────────────────
    if direction == "UP" and confidence >= 0.5:
        action_advice = f"偏多信号（置信{confidence:.0%}），若无持仓可关注支撑{support}附近做多机会"
    elif direction == "DOWN" and confidence >= 0.5:
        action_advice = f"偏空信号（置信{confidence:.0%}），若无持仓可关注压力{resistance}附近做空机会"
    else:
        action_advice = f"信号偏弱（得分{score}/10），建议观望，等信号更明确"

    risk_note = _get_risk_note(rsi, adx, oi_trend, bb_pos, atr, close)

    return {
        "direction":     direction,
        "confidence":    round(confidence, 3),
        "score":         score,
        "key_signals":   signals,
        "support":       support,
        "resistance":    resistance,
        "action_advice": action_advice,
        "risk_note":     risk_note,
        "atr":           round(atr, 2),
        "bb_pos":        round(bb_pos, 3),
    }


def _calc_prev_rsi(closes: np.ndarray) -> float:
    """计算前一日RSI用于判断RSI方向。"""
    if len(closes) < 16:
        return None
    from tools.indicators import _calc_rsi
    return _calc_rsi(closes[:-1], 14)


def _get_risk_note(rsi, adx, oi_trend, bb_pos, atr, close) -> str:
    notes = []
    if rsi > 70:
        notes.append(f"RSI={rsi:.0f}偏高，追多需谨慎")
    if rsi < 30:
        notes.append(f"RSI={rsi:.0f}偏低，追空需谨慎")
    if adx < 18:
        notes.append("ADX偏低，趋势不明，方向信号可靠性下降")
    if oi_trend == "FLAT":
        notes.append("OI方向不明，资金面信号较弱")
    if bb_pos > 0.85:
        notes.append("接近布林上轨，注意短期回调风险")
    if bb_pos < 0.15:
        notes.append("接近布林下轨，注意超卖反弹风险")
    return "；".join(notes) if notes else "当前无特别风险提示"


def _neutral_result(confidence: float, reason: str) -> Dict[str, Any]:
    return {
        "direction": "NEUTRAL", "confidence": confidence, "score": 5,
        "key_signals": [reason], "support": 0.0, "resistance": 0.0,
        "action_advice": reason, "risk_note": "数据不足，无法评估",
        "atr": 0.0, "bb_pos": 0.5,
    }
