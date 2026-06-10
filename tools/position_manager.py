"""
持仓管理模块 — 结合次日方向给出每日持仓操作建议。

决策逻辑：
  持多仓：
    次日DOWN 高置信(>0.6) → PARTIAL_EXIT（减半仓锁利）
    次日DOWN 中置信(0.3-0.6) → TIGHTEN_STOP（止损上移）
    次日UP + 浮盈>1ATR → ADD（可加仓）
    次日NEUTRAL → HOLD
  持空仓（对称）：
    次日UP 高置信(>0.6) → PARTIAL_EXIT
    次日UP 中置信 → TIGHTEN_STOP（止损下移）
    次日DOWN + 浮盈>1ATR → ADD
    次日NEUTRAL → HOLD
  通用：
    持仓≥MAX_HOLD_DAYS-1 → 提示准备离场
    跳空>2% → FULL_EXIT
"""

from typing import Dict, Any, Optional
from datetime import datetime, date


MAX_HOLD_DAYS = 8


def get_position_advice(
    position: Dict[str, Any],
    current_price: float,
    next_day_pred: Dict[str, Any],
    ind: Dict[str, Any],
    atr: float,
    prev_close: Optional[float] = None,
) -> Dict[str, Any]:
    """
    根据当前持仓和次日预测给出管理建议。

    Args:
        position:      持仓信息字典，字段：
                         direction  "LONG"|"SHORT"
                         entry      入场价
                         stop       当前止损价
                         target     目标价
                         entry_date 入场日期字符串 'YYYY-MM-DD'
        current_price: 当前价格
        next_day_pred: predict_next_day() 的返回值
        ind:           技术指标字典
        atr:           当前 ATR14
        prev_close:    前一交易日收盘价（用于跳空检测）

    Returns:
        {
            "action":      "HOLD"|"TIGHTEN_STOP"|"ADD"|"PARTIAL_EXIT"|"FULL_EXIT",
            "new_stop":    float | None,
            "reasoning":   str,
            "urgency":     "LOW"|"MEDIUM"|"HIGH",
            "pnl_now":     float,   # 浮盈点数
            "pnl_pct":     float,   # 浮盈百分比
            "hold_more":   bool,
            "exit_signal": str | None,
            "hold_days":   int,
        }
    """
    d          = position.get("direction", "LONG")
    entry      = float(position.get("entry", current_price))
    stop       = float(position.get("stop",  entry * (0.97 if d == "LONG" else 1.03)))
    target     = float(position.get("target", entry * (1.03 if d == "LONG" else 0.97)))
    entry_date = position.get("entry_date", str(date.today()))

    # 持仓天数
    try:
        ed         = datetime.strptime(entry_date[:10], "%Y-%m-%d").date()
        hold_days  = (date.today() - ed).days
    except Exception:
        hold_days  = 0

    # 浮盈
    if d == "LONG":
        pnl_pts = current_price - entry
    else:
        pnl_pts = entry - current_price
    pnl_pct = pnl_pts / (entry + 1e-8) * 100

    next_dir  = next_day_pred.get("direction", "NEUTRAL")
    next_conf = next_day_pred.get("confidence", 0.0)
    support   = next_day_pred.get("support",    current_price - atr)
    resistance= next_day_pred.get("resistance", current_price + atr)

    action      = "HOLD"
    new_stop    = None
    urgency     = "LOW"
    exit_signal = None
    hold_more   = True
    reasons     = []

    # ── 跳空强制平仓 ────────────────────────────────────────────────────────
    if prev_close and abs(current_price - prev_close) / (prev_close + 1e-8) > 0.025:
        action      = "FULL_EXIT"
        urgency     = "HIGH"
        exit_signal = f"跳空>{2.5:.0f}%，强制平仓规避极端风险"
        hold_more   = False
        return _build_result(action, new_stop, exit_signal, urgency,
                             pnl_pts, pnl_pct, hold_more, hold_days, reasons)

    # ── 超过最大持仓天数 ────────────────────────────────────────────────────
    if hold_days >= MAX_HOLD_DAYS - 1:
        reasons.append(f"已持仓{hold_days}天（建议最长{MAX_HOLD_DAYS}天），准备离场")
        urgency   = "MEDIUM"
        hold_more = False
        if hold_days >= MAX_HOLD_DAYS:
            action      = "FULL_EXIT"
            exit_signal = f"达到最大持仓天数{MAX_HOLD_DAYS}天"

    # ── 核心逻辑：结合次日方向 ───────────────────────────────────────────────
    if action != "FULL_EXIT":
        if d == "LONG":
            if next_dir == "DOWN":
                if next_conf > 0.60:
                    action      = "PARTIAL_EXIT"
                    urgency     = "HIGH"
                    exit_signal = f"次日偏空信号强（置信{next_conf:.0%}），减半仓锁利"
                    hold_more   = pnl_pts > 0  # 有盈利才减仓
                    reasons.append(exit_signal)
                elif next_conf > 0.30:
                    action   = "TIGHTEN_STOP"
                    new_stop = round(max(stop, current_price - 1.0 * atr), 2)
                    urgency  = "MEDIUM"
                    reasons.append(f"次日偏空（置信{next_conf:.0%}），止损上移至{new_stop}")
            elif next_dir == "UP":
                if pnl_pts > atr and next_conf > 0.40:
                    action  = "ADD"
                    urgency = "LOW"
                    reasons.append(f"次日偏多（置信{next_conf:.0%}）且浮盈{pnl_pts:.0f}点>{atr:.0f}ATR，可加仓")
                else:
                    reasons.append(f"次日偏多，继续持有，追踪止损至{round(current_price - 1.5*atr,2)}")
                    new_stop = round(max(stop, current_price - 1.5 * atr), 2)
                    action   = "TIGHTEN_STOP" if new_stop > stop else "HOLD"
            else:  # NEUTRAL
                reasons.append(f"次日信号中性，继续持有")
                new_stop = round(max(stop, current_price - 2.0 * atr), 2)
                action   = "TIGHTEN_STOP" if new_stop > stop else "HOLD"

        else:  # SHORT
            if next_dir == "UP":
                if next_conf > 0.60:
                    action      = "PARTIAL_EXIT"
                    urgency     = "HIGH"
                    exit_signal = f"次日偏多信号强（置信{next_conf:.0%}），减半仓锁利"
                    hold_more   = pnl_pts > 0
                    reasons.append(exit_signal)
                elif next_conf > 0.30:
                    action   = "TIGHTEN_STOP"
                    new_stop = round(min(stop, current_price + 1.0 * atr), 2)
                    urgency  = "MEDIUM"
                    reasons.append(f"次日偏多（置信{next_conf:.0%}），止损下移至{new_stop}")
            elif next_dir == "DOWN":
                if pnl_pts > atr and next_conf > 0.40:
                    action  = "ADD"
                    urgency = "LOW"
                    reasons.append(f"次日偏空（置信{next_conf:.0%}）且浮盈{pnl_pts:.0f}点>{atr:.0f}ATR，可加仓")
                else:
                    reasons.append(f"次日偏空，继续持有，追踪止损至{round(current_price + 1.5*atr,2)}")
                    new_stop = round(min(stop, current_price + 1.5 * atr), 2)
                    action   = "TIGHTEN_STOP" if new_stop < stop else "HOLD"
            else:
                reasons.append("次日信号中性，继续持有")
                new_stop = round(min(stop, current_price + 2.0 * atr), 2)
                action   = "TIGHTEN_STOP" if new_stop < stop else "HOLD"

    # 已亏损超过目标的50%，提示
    if pnl_pts < -atr * 0.8:
        reasons.append(f"浮亏已达{-pnl_pts:.0f}点，止损{stop}附近需注意")
        urgency = max(urgency, "MEDIUM") if isinstance(urgency, str) else "MEDIUM"

    reasoning = "；".join(reasons) if reasons else "持仓正常，无特别提示"

    return _build_result(action, new_stop, exit_signal, urgency,
                         pnl_pts, pnl_pct, hold_more, hold_days, [reasoning])


def _build_result(action, new_stop, exit_signal, urgency,
                  pnl_pts, pnl_pct, hold_more, hold_days, reasons):
    return {
        "action":      action,
        "new_stop":    new_stop,
        "reasoning":   "；".join(r for r in reasons if r),
        "urgency":     urgency,
        "pnl_now":     round(pnl_pts, 2),
        "pnl_pct":     round(pnl_pct, 3),
        "hold_more":   hold_more,
        "exit_signal": exit_signal,
        "hold_days":   hold_days,
    }


def format_position_report(
    symbol: str,
    position: Dict[str, Any],
    current_price: float,
    next_day_pred: Dict[str, Any],
    advice: Dict[str, Any],
    trend_signal: Dict[str, Any] = None,
) -> str:
    """生成每日更新的格式化报告字符串。"""
    lines = []
    d     = position.get("direction", "LONG")
    entry = float(position.get("entry", current_price))
    stop  = float(position.get("stop", 0))
    tgt   = float(position.get("target", 0))

    pnl   = advice["pnl_now"]
    pct   = advice["pnl_pct"]
    pnl_str = f"+{pnl:.0f}点(+{pct:.2f}%)" if pnl >= 0 else f"{pnl:.0f}点({pct:.2f}%)"

    lines.append(f"【持仓状态】")
    lines.append(f"  {symbol} {'空单' if d=='SHORT' else '多单'} @ {entry:.0f}  "
                 f"当前 {current_price:.0f}  浮{'盈' if pnl>=0 else '亏'} {pnl_str}")
    lines.append(f"  持仓 {advice['hold_days']} 天 | 止损 {stop:.0f} | 目标 {tgt:.0f}")
    lines.append("")

    lines.append(f"【今日盘面分析】")
    if trend_signal:
        lines.append(f"  趋势信号: {trend_signal.get('cycle','N/A')}周期，"
                     f"{'做空信号' if d=='SHORT' else '做多信号'}{'持续有效' if trend_signal.get('signal','')==d else '已减弱'}")
    nd = next_day_pred
    dir_cn = {"UP":"偏多↑","DOWN":"偏空↓","NEUTRAL":"中性→"}.get(nd["direction"],"?")
    lines.append(f"  次日方向: {dir_cn} 置信度{nd['confidence']:.0%}  "
                 f"得分{nd['score']}/10")
    if nd.get("key_signals"):
        lines.append(f"  信号详情: {' | '.join(nd['key_signals'][:3])}")
    lines.append(f"  关键支撑: {nd['support']:.0f}  关键压力: {nd['resistance']:.0f}")
    lines.append("")

    action_cn = {
        "HOLD":         "继续持有",
        "TIGHTEN_STOP": "收紧止损",
        "ADD":          "可加仓",
        "PARTIAL_EXIT": "减半仓",
        "FULL_EXIT":    "全部平仓",
    }.get(advice["action"], advice["action"])

    urgency_cn = {"LOW":"低","MEDIUM":"中","HIGH":"高"}.get(advice["urgency"],"?")
    lines.append(f"【持仓管理建议】")
    lines.append(f"  操作: ⚡ {action_cn}  紧迫度: {urgency_cn}")
    if advice.get("new_stop"):
        lines.append(f"  新止损: {advice['new_stop']:.0f}  "
                     f"（原止损: {stop:.0f}，{'上移' if d=='LONG' else '下移'}保护利润）")
    lines.append(f"  原因: {advice['reasoning']}")
    if nd.get("risk_note") and nd["risk_note"] != "当前无特别风险提示":
        lines.append(f"  风险: ⚠ {nd['risk_note']}")
    lines.append("")

    return "\n".join(lines)
