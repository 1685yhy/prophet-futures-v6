"""
事件驱动趋势跟踪回测引擎 v4

设计原则：
- 不预测次日方向，而是跟踪趋势持仓5日
- 止损：入场价 ± 1.5×ATR（历史数据覆盖75%分位逆向波动）
- 止盈：分批止盈，1/2仓在1:1时止盈，剩余在2.5×ATR或最大持仓日止盈
- 突发事件：单日涨跌>2%强制平仓
- 以LH（生猪）为主，兼容JD（鸡蛋）、FU/BU（燃油）、MA（甲醇）
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

from tools.indicators import calc_indicators
from tools.cycle_detector import get_lh_signal_conditions, get_generic_signal_conditions

logger = logging.getLogger(__name__)

SLIPPAGE_BPS  = 2       # 滑点 bps
COMMISSION_RT = 0.0001  # 单边手续费率
MAX_HOLD_DAYS  = 20     # 最大持仓天数（追踪止损模式下延长到20日）
TRAIL_ATR_MULT = 2.0    # 追踪止损：收盘价 ± N×ATR（做空用+，做多用-）

# 合约规格：每手对应的实物数量（吨/桶/克）
# 价格单位：元/吨（或元/桶、元/克）
# PnL = 价差(元/吨) × 合约规格(吨/手) × 手数
CONTRACT_LOT_SIZE: Dict[str, float] = {
    "lh": 16.0,    # 生猪：16吨/手
    "jd": 5.0,     # 鸡蛋：5吨/手（价格元/500克，需注意单位）
    "bu": 10.0,    # 沥青：10吨/手
    "fu": 10.0,    # 燃料油：10吨/手
    "ma": 10.0,    # 甲醇：10吨/手
    "rb": 10.0,    # 螺纹钢：10吨/手
    "i":  100.0,   # 铁矿石：100吨/手
    "sc": 1000.0,  # 原油：1000桶/手
    "cu": 5.0,     # 铜：5吨/手
    "au": 1000.0,  # 黄金：1000克/手
    "ag": 15000.0, # 白银：15000克/手
    "zn": 5.0,     # 锌：5吨/手
    "al": 5.0,     # 铝：5吨/手
}

def get_lot_size(symbol: str) -> float:
    """获取品种合约乘数（吨或桶或克/手）。"""
    return CONTRACT_LOT_SIZE.get(symbol.lower().rstrip("0123456789"), 10.0)


# ── 数据获取 ────────────────────────────────────────────────────────────────

def _fetch_history(symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
    try:
        import akshare as ak
        df = ak.futures_main_sina(
            symbol=symbol.upper() + "0",
            start_date=start_date,
            end_date=end_date,
        )
        df.columns = ["date", "open", "high", "low", "close", "volume", "oi", "settle"]
        for c in ["open", "high", "low", "close", "volume", "oi"]:
            df[c] = df[c].astype(float)
        return df.reset_index(drop=True)
    except Exception as e:
        logger.warning("Failed to fetch %s: %s", symbol, e)
        return None


# ── 主回测函数 ───────────────────────────────────────────────────────────────

def run_backtest(
    date: str,
    symbols: Optional[List[str]] = None,
    backtest_days: int = 180,
    capital: float = 1_000_000.0,
    system=None,
) -> Dict[str, Any]:
    """
    趋势跟踪回测。

    Args:
        date:          回测结束日期 'YYYY-MM-DD'
        symbols:       品种列表，默认 ['lh', 'jd', 'bu', 'ma']
        backtest_days: 回测天数
        capital:       初始资金
    """
    if symbols is None:
        symbols = ["lh", "jd", "bu", "ma"]

    try:
        end_dt = datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        return {"error": f"Invalid date: {date}"}

    start_dt  = end_dt - timedelta(days=backtest_days + 130)
    start_str = start_dt.strftime("%Y%m%d")
    end_str   = end_dt.strftime("%Y%m%d")

    all_trades: List[Dict] = []
    daily_pnl: Dict[str, float] = {}

    for symbol in symbols:
        logger.info("Backtesting %s (%s→%s)...", symbol, start_str, end_str)
        df = _fetch_history(symbol, start_str, end_str)
        if df is None or len(df) < 70:
            logger.warning("Insufficient data for %s", symbol)
            continue
        trades = _backtest_symbol(df, symbol, capital / len(symbols))
        all_trades.extend(trades)
        for t in trades:
            d = t["exit_date"]
            daily_pnl[d] = daily_pnl.get(d, 0.0) + t["pnl"]

    return _compute_stats(all_trades, daily_pnl, capital, date, symbols)


# ── 单品种回测 ───────────────────────────────────────────────────────────────

def _backtest_symbol(df: pd.DataFrame, symbol: str, capital: float) -> List[Dict]:
    trades   = []
    pos      = None
    WINDOW   = 60
    is_lh    = symbol.lower() == "lh"
    lot_size = get_lot_size(symbol)   # 合约乘数（吨/桶/手）

    for i in range(WINDOW, len(df) - 1):
        today    = df.iloc[i]
        tomorrow = df.iloc[i + 1]
        date_str = str(today["date"])
        window   = df.iloc[i - WINDOW: i + 1].copy()
        ind      = calc_indicators(window)
        atr      = ind["atr14"]
        close    = float(today["close"])
        high_    = float(today["high"])
        low_     = float(today["low"])

        # ── 持仓管理 ──
        if pos is not None:
            d     = pos["direction"]
            entry = pos["entry"]
            hold  = i - pos["entry_idx"]

            # 突发事件：跳空>2%强制平仓
            gap_pct = abs(close - float(df.iloc[i - 1]["close"])) / float(df.iloc[i - 1]["close"])
            force_exit = gap_pct > 0.025 or hold >= MAX_HOLD_DAYS

            hit_stop   = (d == "LONG"  and low_  <= pos["stop"]) or \
                         (d == "SHORT" and high_ >= pos["stop"])
            hit_target = (d == "LONG"  and high_ >= pos["target_full"]) or \
                         (d == "SHORT" and low_  <= pos["target_full"])

            # ── 追踪止损更新（每日收盘后，止损只能朝有利方向移动）──
            if not hit_stop and not hit_target:
                if d == "LONG":
                    new_trail = close - TRAIL_ATR_MULT * atr
                    pos["stop"] = max(pos["stop"], new_trail)   # 多单止损只能上移
                else:
                    new_trail = close + TRAIL_ATR_MULT * atr
                    pos["stop"] = min(pos["stop"], new_trail)   # 空单止损只能下移

            # 分批止盈：到达1:1时减半（保留，但不是主要退出机制）
            if not pos["half_done"]:
                hit_half = (d == "LONG"  and high_ >= pos["target_half"]) or \
                           (d == "SHORT" and low_  <= pos["target_half"])
                if hit_half:
                    pos["half_done"] = True
                    half_price = pos["target_half"]
                    # 半仓PnL = 价差 × 合约规格 × 手数/2
                    half_pnl   = (half_price - entry) * lot_size * (pos["qty"] / 2) * (1 if d == "LONG" else -1)
                    half_pnl  -= abs(half_price * lot_size * pos["qty"] / 2 * COMMISSION_RT)
                    pos["stop"] = entry   # 减半后止损移至成本保本

            if hit_stop or hit_target or force_exit:
                remain_qty = pos["qty"] / 2 if pos["half_done"] else pos["qty"]
                if hit_stop:
                    exit_price = pos["stop"]
                    reason     = "STOP"
                elif hit_target:
                    exit_price = pos["target_full"]
                    reason     = "TP"
                else:
                    exit_price = close * (1 - SLIPPAGE_BPS/10000 if d == "LONG" else 1 + SLIPPAGE_BPS/10000)
                    reason     = "GAP" if gap_pct > 0.025 else "MAX_HOLD"

                # PnL = 价差(元/吨) × 合约规格(吨/手) × 手数
                pnl = (exit_price - entry) * lot_size * remain_qty * (1 if d == "LONG" else -1)
                pnl -= abs(exit_price * lot_size * remain_qty * COMMISSION_RT)

                trades.append({
                    "symbol":     symbol,
                    "direction":  d,
                    "entry_date": pos["entry_date"],
                    "exit_date":  date_str,
                    "entry_price":round(entry, 2),
                    "exit_price": round(exit_price, 2),
                    "pnl_pts":    round((exit_price - entry) * (1 if d == "LONG" else -1), 2),
                    "pnl":        round(pnl, 2),
                    "reason":     reason,
                    "hold_days":  hold,
                    "half_done":  pos["half_done"],
                })
                pos = None

        # ── 无持仓时生成信号 ──
        if pos is None and i < len(df) - 2:
            if is_lh:
                sig = get_lh_signal_conditions(window, ind)
            else:
                sig = get_generic_signal_conditions(symbol, window, ind)

            if sig["signal"] in ("LONG", "SHORT"):
                d          = sig["signal"]
                slippage   = close * SLIPPAGE_BPS / 10000
                entry      = close + slippage * (1 if d == "LONG" else -1)
                stop_dist  = atr * sig["stop_atr_mult"]
                target_dist= atr * sig["target_atr_mult"]

                if stop_dist < atr * 0.5:
                    stop_dist = atr  # 防止止损过近

                stop_price  = entry - stop_dist if d == "LONG" else entry + stop_dist
                target_half = entry + stop_dist if d == "LONG" else entry - stop_dist   # 1:1 减半
                target_full = entry + target_dist if d == "LONG" else entry - target_dist  # 2.5:1 满仓

                # 手数 = 最大风险金额 / (止损点数 × 合约规格)
                max_risk_per_trade = capital * 0.02
                qty = round(max(1.0, min(20.0, max_risk_per_trade / (stop_dist * lot_size))), 1)

                pos = {
                    "direction":   d,
                    "entry":       round(entry, 2),
                    "stop":        round(stop_price, 2),
                    "target_half": round(target_half, 2),
                    "target_full": round(target_full, 2),
                    "entry_date":  date_str,
                    "entry_idx":   i,
                    "qty":         qty,
                    "half_done":   False,
                    "confidence":  sig["confidence"],
                }

    return trades


# ── 统计函数 ─────────────────────────────────────────────────────────────────

def _compute_stats(
    trades: List[Dict],
    daily_pnl: Dict[str, float],
    capital: float,
    date: str,
    symbols: List[str],
) -> Dict[str, Any]:
    if not trades:
        return {
            "date_range": date, "symbols": symbols,
            "total_trades": 0, "win_rate": 0.0,
            "sharpe_ratio": 0.0, "max_drawdown_pct": 0.0,
            "total_pnl": 0.0, "equity_curve": [capital],
            "trades": [], "message": "No trades generated",
        }

    pnls     = [t["pnl"] for t in trades]
    wins     = [p for p in pnls if p > 0]
    losses   = [p for p in pnls if p <= 0]
    win_rate = len(wins) / len(pnls)
    avg_win  = float(np.mean(wins))          if wins   else 0.0
    avg_loss = abs(float(np.mean(losses)))   if losses else 1.0
    pl_ratio = avg_win / (avg_loss + 1e-8)

    sorted_dates  = sorted(daily_pnl.keys())
    daily_returns = [daily_pnl[d] / capital for d in sorted_dates]
    sharpe        = calculate_sharpe(daily_returns)

    equity = [capital]
    for r in daily_returns:
        equity.append(equity[-1] * (1 + r))
    dd_stats  = drawdown_analysis(equity)

    return {
        "date_range":      date,
        "symbols":         symbols,
        "total_trades":    len(trades),
        "win_rate":        round(win_rate, 3),
        "avg_win":         round(avg_win, 2),
        "avg_loss":        round(avg_loss, 2),
        "pl_ratio":        round(pl_ratio, 3),
        "sharpe_ratio":    round(sharpe, 3),
        "max_drawdown_pct":dd_stats["max_drawdown_pct"],
        "calmar_ratio":    dd_stats["calmar_ratio"],
        "total_pnl":       round(sum(pnls), 2),
        "total_return_pct":round(sum(pnls) / capital * 100, 2),
        "equity_curve":    [round(e, 2) for e in equity],
        "trades":          trades,
    }


def calculate_sharpe(returns: List[float], risk_free: float = 0.02) -> float:
    if len(returns) < 2:
        return 0.0
    arr      = np.array(returns)
    daily_rf = risk_free / 252
    excess   = arr - daily_rf
    std      = np.std(excess, ddof=1)
    return float(np.mean(excess) / std * np.sqrt(252)) if std > 0 else 0.0


def drawdown_analysis(equity_curve: List[float]) -> Dict[str, Any]:
    if len(equity_curve) < 2:
        return {"max_drawdown": 0.0, "max_drawdown_pct": 0.0,
                "current_drawdown": 0.0, "calmar_ratio": 0.0}
    equity      = np.array(equity_curve)
    running_max = np.maximum.accumulate(equity)
    drawdown    = (equity - running_max) / (running_max + 1e-8)
    max_dd      = float(drawdown.min())
    return {
        "max_drawdown":     round(max_dd, 4),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "current_drawdown": round(float(drawdown[-1]), 4),
        "calmar_ratio":     round(-1 / (max_dd * 252) if max_dd < -1e-6 else 0, 3),
    }
