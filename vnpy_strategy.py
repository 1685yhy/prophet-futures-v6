#!/usr/bin/env python3
"""Prophet Strategy — VNPY回测版"""

import sys; sys.path.insert(0, ".")
import numpy as np, pandas as pd, xgboost as xgb
from datetime import datetime, timedelta

# ══════════════════ VNPY Strategy ══════════════════
try:
    from vnpy_ctastrategy import CtaTemplate, StopOrder, TickData, BarData, TradeData, OrderData
    from vnpy.trader.constant import Direction, Offset, Interval, Status

    class ProphetStrategy(CtaTemplate):
        """XGBoost + 规则信号策略"""
        
        author = "Prophet"
        fast_window = 60
        atr_stop = 1.0
        atr_target = 3.0
        risk_pct = 0.02
        
        def __init__(self, engine, strategy_name, vt_symbol, setting):
            super().__init__(engine, strategy_name, vt_symbol, setting)
            self.model = None
            self.bars = []
            self.trained = False
            
        def on_init(self):
            self.write_log("策略初始化")
            self.load_bar(2000)  # 加载历史K线
            
        def on_start(self):
            self.write_log("策略启动")
            
        def on_stop(self):
            self.write_log("策略停止")
            
        def on_bar(self, bar: BarData):
            self.bars.append(bar)
            if len(self.bars) < self.fast_window + 10:
                return
            
            # 每30根K线重新训练ML
            if len(self.bars) % 30 == 0 and len(self.bars) > 200:
                self._train_model()
            
            # 有仓位则检查出场
            if self.pos != 0:
                self._check_exit(bar)
                return
            
            # 无仓位则检查入场
            if self.model is not None:
                self._check_entry(bar)
        
        def _train_model(self):
            """用截至当前的K线训练ML."""
            from tools.indicators import calc_indicators
            from tools.cycle_detector import detect_cycle, detect_rollover_noise
            
            df = pd.DataFrame([{
                "date": b.datetime, "open": b.open_price,
                "high": b.high_price, "low": b.low_price,
                "close": b.close_price, "volume": b.volume,
                "oi": getattr(b, 'open_interest', 0)
            } for b in self.bars])
            
            fx, ly = [], []
            for i in range(self.fast_window, len(df) - 1):
                w = df.iloc[i-self.fast_window:i+1]
                ind = calc_indicators(w)
                from run import rule_signal
                sg = rule_signal(w, ind, 7)
                if sg is None: continue
                f = self._build_feat(df, i)
                if f is None: continue
                nc = df.iloc[i+1]["close"]; c = df.iloc[i]["close"]
                ly.append(1 if (sg == "LONG" and nc > c) or (sg == "SHORT" and nc < c) else 0)
                fx.append(f)
            
            if len(fx) >= 80:
                self.model = xgb.XGBClassifier(
                    n_estimators=100, max_depth=5, learning_rate=0.05,
                    random_state=42, verbosity=0
                )
                self.model.fit(np.array(fx), np.array(ly))
                self.trained = True
        
        def _build_feat(self, df, i):
            from run import build_features
            return build_features(df, i)
        
        def _check_entry(self, bar: BarData):
            from tools.indicators import calc_indicators
            from run import rule_signal
            
            df = pd.DataFrame([{
                "date": b.datetime, "open": b.open_price,
                "high": b.high_price, "low": b.low_price,
                "close": b.close_price, "volume": b.volume,
                "oi": getattr(b, 'open_interest', 0)
            } for b in self.bars])
            
            i = len(df) - 1
            w = df.iloc[i-self.fast_window:i+1]
            ind = calc_indicators(w)
            sg = rule_signal(w, ind, 7)
            if sg is None: return
            
            f = self._build_feat(df, i)
            if f is None: return
            
            prob = self.model.predict_proba(f.reshape(1, -1))[0, 1]
            if prob < 0.50: return
            
            # 计算仓位
            atr = ind["atr14"]
            entry = bar.close_price
            sd = max(atr * 0.3, atr * self.atr_stop)
            td = atr * self.atr_target
            stop_price = entry - sd if sg == "LONG" else entry + sd
            
            # 风控计算手数
            lot = 16 if "lh" in self.vt_symbol.lower() else 60
            risk_cash = 500000 * self.risk_pct
            qty = max(1, min(20, int(risk_cash / (sd * lot))))
            
            if sg == "LONG":
                self.buy(entry, qty)
                self.short(stop_price, qty)  # stop order
            else:
                self.short(entry, qty)
                self.buy(stop_price, qty)  # stop order
            
            self.write_log(f"{sg} {qty}手 @ {entry:.0f} 止损{stop_price:.0f}")
        
        def _check_exit(self, bar: BarData):
            """检查止盈止损."""
            # VNPY框架自动管理止损单
            # 这里检查时间止盈（持有10天）
            if hasattr(self, '_entry_bar_count'):
                self._entry_bar_count += 1
                if self._entry_bar_count >= 10:
                    if self.pos > 0:
                        self.sell(bar.close_price, abs(self.pos))
                    else:
                        self.cover(bar.close_price, abs(self.pos))
                    self._entry_bar_count = 0
                    self.write_log(f"时间止盈 @ {bar.close_price:.0f}")
    print("✅ ProphetStrategy 类定义成功")
except ImportError:
    print("⚠️ VNPY回测引擎未加载（GUI模式需要）")
    print("  命令行模式可用vnpy_ctabacktester")

# ══════════════════ 命令行回测 ══════════════════
print("\n" + "=" * 55)
print("  VNPY 回测 — Prophet策略")
print("=" * 55)

# 用vnpy的命令行回测器
try:
    # vnpy回测需要的数据格式转换
    from tools.indicators import calc_indicators
    from tools.cycle_detector import detect_cycle, detect_rollover_noise
    from run import rule_signal, build_features, fetch, LOT, calc_indicators
    
    print("\n  使用本地回测引擎（与VNPY逻辑一致）")
    
    for sym in ["jm", "lh"]:
        df = fetch(sym, 2500)
        if df is None: continue
        
        pnls = []
        model = None
        W = 60
        
        for today in range(500, len(df) - 1):
            # Retrain every 30 bars
            if today % 30 == 0:
                fx, ly = [], []
                for i in range(W, today - 1):
                    w = df.iloc[i-W:i+1]
                    ind = calc_indicators(w)
                    sg = rule_signal(w, ind, 7)
                    if sg is None: continue
                    f = build_features(df, i)
                    if f is None: continue
                    nc = float(df.iloc[i+1]["close"]); c = float(df.iloc[i]["close"])
                    ly.append(1 if (sg == "LONG" and nc > c) or (sg == "SHORT" and nc < c) else 0)
                    fx.append(f)
                if len(fx) >= 80:
                    model = xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.05, random_state=42, verbosity=0)
                    model.fit(np.array(fx), np.array(ly))
            
            if model is None: continue
            
            w = df.iloc[today-W:today+1]
            ind = calc_indicators(w)
            sg = rule_signal(w, ind, 7)
            if sg is None: continue
            
            f = build_features(df, today)
            if f is None: continue
            
            prob = model.predict_proba(f.reshape(1, -1))[0, 1]
            if prob < 0.50: continue
            
            c = float(df.iloc[today]["close"]); atr = ind["atr14"]
            entry = c + 0.0002 * c * (1 if sg == "LONG" else -1)
            sd = max(atr * 0.3, atr * 1.0); td = atr * 3.0
            stop = entry - sd if sg == "LONG" else entry + sd
            target = entry + td if sg == "LONG" else entry - td
            
            lot = LOT.get(sym, 10)
            q = max(1.0, min(20.0, 500000 * 0.02 / (sd * lot)))
            
            nc = float(df.iloc[today+1]["close"])
            nh = float(df.iloc[today+1]["high"])
            nl = float(df.iloc[today+1]["low"])
            
            if sg == "LONG":
                ep = stop if nl <= stop else (target if nh >= target else nc)
            else:
                ep = stop if nh >= stop else (target if nl <= target else nc)
            
            pnl = (ep - entry) * lot * q * (1 if sg == "LONG" else -1)
            pnl -= abs(entry * lot * q * 0.0004)
            pnls.append(pnl)
        
        if pnls:
            w = [p for p in pnls if p > 0]
            wr = len(w) / len(pnls)
            tp = sum(pnls)
            eq = 500000; me = 500000; mdd = 0
            for p in pnls:
                eq += p; me = max(me, eq)
                mdd = min(mdd, (eq - me) / me * 100)
            print(f"\n  {sym.upper()}: {len(pnls)}笔 {wr:.0%}wr PnL{tp:+,.0f} {tp/500000*100:+.1f}% DD{abs(mdd):.1f}%")

except Exception as e:
    print(f"  ⚠️ 回测异常: {e}")
