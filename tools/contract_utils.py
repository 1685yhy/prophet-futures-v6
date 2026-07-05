"""
合约工具模块 — 处理生猪等多合约品种的信号汇总和主力合约判断。

行业标准做法（参考 Man AHL / Winton / 九坤）：
1. 主力合约判断：OI最大的合约为主力，并计算2609在全品种的占比
2. 换仓期检测：OI跷跷板结构（主力减 + 次主力增）+ 到期日窗口
3. 全品种OI加权动量：Σ(各合约OI变化 × 方向) / Σ(OI)，验证主力信号
4. 价格基准：结算价（settlement price）用于信号计算，收盘价用于显示
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# 生猪各合约到期月份对应规律（大商所，每双月为合约月）
LH_CONTRACT_MONTHS = [1, 3, 5, 7, 9, 11]

# 主力合约切换窗口：到期前N天内视为换仓期，信号权重衰减
ROLLOVER_WINDOW_DAYS = 55


def get_dominant_contract(symbol_base: str = "LH") -> str:
    """获取当前主力合约代码（OI最大的合约）。"""
    try:
        import akshare as ak
        # 枚举当前和未来6个月的合约
        candidates = []
        now = datetime.now()
        for delta_month in range(0, 8):
            m = now.month + delta_month
            y = now.year + (m - 1) // 12
            m = ((m - 1) % 12) + 1
            # 只取双月合约
            if m % 2 == 0:
                m += 1
            if m > 12:
                m = 1; y += 1
            sym = f"{symbol_base}{str(y)[2:]}{m:02d}"
            candidates.append(sym)

        best_sym, best_oi = "", 0
        for sym in candidates:
            try:
                df = ak.futures_main_sina(
                    symbol=sym,
                    start_date=(now - timedelta(days=3)).strftime("%Y%m%d"),
                    end_date=now.strftime("%Y%m%d"),
                )
                df.columns = ["date","open","high","low","close","volume","oi","settle"]
                oi = float(df.iloc[-1]["oi"])
                if oi > best_oi:
                    best_oi = oi
                    best_sym = sym
            except Exception:
                continue
        return best_sym or f"{symbol_base}2609"
    except Exception as e:
        logger.warning("get_dominant_contract failed: %s", e)
        return f"{symbol_base}2609"


def get_multi_contract_oi(
    symbol_base: str = "LH",
    lookback_days: int = 10,
) -> Dict[str, Any]:
    """
    汇总全品种（所有月份合约）的OI动态。

    Returns:
        {
            "total_oi":          int,    # 全品种OI合计
            "total_oi_chg_1d":   int,    # 全品种昨日净变化
            "total_oi_chg_3d":   int,    # 全品种3日净变化
            "dominant_contract": str,    # 主力合约代码
            "dominant_pct":      float,  # 主力占全品种比例
            "dominant_oi_chg":   int,    # 主力合约昨日净变化
            "rollover_detected": bool,   # 是否检测到换仓（跷跷板结构）
            "rollover_intensity":float,  # 换仓强度（0-1）
            "all_contracts":     dict,   # 各合约明细
            "oi_trend_full":     str,    # 全品种OI趋势 ACCUMULATING/REDUCING/FLAT
        }
    """
    try:
        import akshare as ak
        now = datetime.now()
        start = (now - timedelta(days=lookback_days + 5)).strftime("%Y%m%d")
        end   = now.strftime("%Y%m%d")

        contracts = []
        for delta in range(0, 10):
            m = now.month + delta
            y = now.year + (m - 1) // 12
            m = ((m - 1) % 12) + 1
            if m % 2 == 0: m += 1
            if m > 12: m = 1; y += 1
            contracts.append(f"{symbol_base}{str(y)[2:]}{m:02d}")

        frames = []
        for sym in contracts:
            try:
                df = ak.futures_main_sina(symbol=sym, start_date=start, end_date=end)
                df.columns = ["date","open","high","low","close","volume","oi","settle"]
                df["contract"] = sym
                if len(df) >= 3:
                    frames.append(df[["date","close","settle","oi","contract"]])
            except Exception:
                continue

        if not frames:
            return _empty_multi_oi()

        all_df = pd.concat(frames).sort_values(["date","contract"])
        last_date = all_df["date"].max()
        prev_dates = sorted(all_df["date"].unique())

        # 最新一日各合约OI
        latest = all_df[all_df["date"] == last_date].set_index("contract")

        def _safe_int(val) -> int:
            """安全转为 int，兼容 Series/DataFrame/np.array 返回值。"""
            import numpy as np
            try:
                if hasattr(val, "item"):
                    val = val.item()
            except (ValueError, TypeError):
                pass
            if hasattr(val, "sum"):
                val = val.sum()
            if isinstance(val, (np.floating, float)):
                return int(round(float(val)))
            if isinstance(val, (np.ndarray, list)):
                val = sum(val)
            return int(val)

        total_oi = _safe_int(latest["oi"].sum())

        # 主力合约 = OI最大
        dominant = latest["oi"].idxmax() if len(latest) > 0 else ""
        dominant_oi = _safe_int(latest.loc[dominant, "oi"]) if dominant else 0

        # 全品种昨日OI
        prev_1d = _safe_int(all_df[all_df["date"] == prev_dates[-2]].set_index("contract")["oi"].sum()) if len(prev_dates) >= 2 else total_oi
        prev_3d = _safe_int(all_df[all_df["date"] == prev_dates[-4]].set_index("contract")["oi"].sum()) if len(prev_dates) >= 4 else total_oi

        total_chg_1d = _safe_int(total_oi - prev_1d)
        total_chg_3d = _safe_int(total_oi - prev_3d)

        # 主力合约昨日OI变化
        dominant_prev_series = all_df[(all_df["date"] == prev_dates[-2]) & (all_df["contract"] == dominant)]["oi"]
        dominant_prev = _safe_int(dominant_prev_series.sum()) if (dominant and len(prev_dates) >= 2 and len(dominant_prev_series) > 0) else dominant_oi
        dominant_chg = dominant_oi - dominant_prev

        # 换仓检测：主力减少 + 次主力增加（跷跷板结构）
        rollover_detected = False
        rollover_intensity = 0.0
        if dominant and len(prev_dates) >= 2 and len(latest) >= 2:
            non_dominant = latest.drop(dominant)["oi"]
            non_dominant_prev = all_df[(all_df["date"] == prev_dates[-2]) & (all_df["contract"] != dominant)].set_index("contract")["oi"]
            non_dom_chg = non_dominant.sum() - non_dominant_prev.sum()
            # 跷跷板：主力减 + 其他增
            if dominant_chg < -500 and non_dom_chg > 200:
                rollover_detected = True
                rollover_intensity = min(1.0, abs(dominant_chg) / (total_oi * 0.03 + 1))

        # 全品种OI趋势
        if total_chg_3d > 0 and total_chg_1d >= 0:
            oi_trend_full = "ACCUMULATING"
        elif total_chg_3d < 0 and total_chg_1d <= 0:
            oi_trend_full = "REDUCING"
        else:
            oi_trend_full = "FLAT"

        return {
            "total_oi":           total_oi,
            "total_oi_chg_1d":    total_chg_1d,
            "total_oi_chg_3d":    total_chg_3d,
            "dominant_contract":  dominant,
            "dominant_pct":       round(dominant_oi / (total_oi + 1e-8), 3),
            "dominant_oi_chg":    dominant_chg,
            "rollover_detected":  rollover_detected,
            "rollover_intensity": round(rollover_intensity, 3),
            "all_contracts":      latest["oi"].to_dict(),
            "oi_trend_full":      oi_trend_full,
        }

    except Exception as e:
        logger.warning("get_multi_contract_oi failed: %s", e)
        return _empty_multi_oi()


def get_settle_price(symbol: str) -> Optional[float]:
    """获取品种最新结算价（用于信号计算基准）。"""
    try:
        import akshare as ak
        now = datetime.now()
        for d in range(0, 5):
            check = (now - timedelta(days=d)).strftime("%Y%m%d")
            df = ak.futures_main_sina(
                symbol=symbol.upper() + "0",
                start_date=check, end_date=check,
            )
            if not df.empty:
                df.columns = ["date","open","high","low","close","volume","oi","settle"]
                settle = float(df.iloc[-1]["settle"])
                if settle > 0:
                    return settle
    except Exception as e:
        logger.warning("get_settle_price failed for %s: %s", symbol, e)
    return None


def is_rollover_period(symbol: str, days_before_expiry: int = ROLLOVER_WINDOW_DAYS) -> Dict[str, Any]:
    """
    判断当前是否处于合约换仓窗口期。

    Returns:
        {
            "in_rollover":    bool,
            "days_to_expiry": int,
            "signal_weight":  float,  # 换仓期内信号衰减系数 0.5-1.0
        }
    """
    try:
        # 从合约代码解析到期日（大商所生猪：交割月第三个周五前一个交易日）
        contract_year  = int("20" + symbol[2:4])
        contract_month = int(symbol[4:6])

        # 简化估算：合约月最后一个交易日约为该月第26日左右
        import calendar
        last_day = calendar.monthrange(contract_year, contract_month)[1]
        expiry = datetime(contract_year, contract_month, min(26, last_day))
        days_left = (expiry - datetime.now()).days

        in_rollover = 0 <= days_left <= days_before_expiry
        # 信号权重：临近交割线性衰减（45天时0.8，15天时0.5）
        weight = max(0.5, min(1.0, days_left / days_before_expiry)) if in_rollover else 1.0

        return {
            "in_rollover":    in_rollover,
            "days_to_expiry": max(0, days_left),
            "signal_weight":  round(weight, 2),
        }
    except Exception:
        return {"in_rollover": False, "days_to_expiry": 999, "signal_weight": 1.0}


def _empty_multi_oi() -> Dict[str, Any]:
    return {
        "total_oi": 0, "total_oi_chg_1d": 0, "total_oi_chg_3d": 0,
        "dominant_contract": "", "dominant_pct": 0.0, "dominant_oi_chg": 0,
        "rollover_detected": False, "rollover_intensity": 0.0,
        "all_contracts": {}, "oi_trend_full": "FLAT",
    }
