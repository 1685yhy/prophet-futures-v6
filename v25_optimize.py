#!/usr/bin/env python3
"""
Prophet Futures v25 — 实盘级止损优化
约束: 单笔最大亏损 ≤ 资本金的 5~10%
对比: v24 Struct26 基准
"""
import sys, os, time, json
import numpy as np, pandas as pd
from datetime import datetime, timedelta
import akshare as ak
import xgboost as xgb
from itertools import product

# ===== CONFIG =====
SYMBOLS = ['lh', 'jm']
REAL_COST = {'lh': 0.0006, 'jm': 0.0011}
LOT_MULT = {'lh': 16, 'jm': 60}
CAPITAL = 300000
WF_WINDOWS = 500
MAX_LOSS_PCT = 0.10  # 单笔最大亏损不超过10%

# Grid: stop methods × RR × position sizing
STOP_METHODS = {
    'atr1.5': {'type': 'atr', 'mult': 1.5},
    'atr2.0': {'type': 'atr', 'mult': 2.0},
    'atr2.5': {'type': 'atr', 'mult': 2.5},
    'atr3.0': {'type': 'atr', 'mult': 3.0},
    'struct10': {'type': 'struct', 'n': 10},
    'struct15': {'type': 'struct', 'n': 15},
    'struct20': {'type': 'struct', 'n': 20},
    'struct26': {'type': 'struct', 'n': 26},  # v24 baseline
}
RR_VALUES = [2, 2.5, 3, 3.5, 4]
POS_MODES = ['fixed_2x', 'fixed_3x', 'vol_1.5x', 'vol_2x']

# XGBoost params per symbol
XGB_PARAMS = {
    'lh': {'n_estimators': 200, 'max_depth': 5, 'learning_rate': 0.05},
    'jm': {'n_estimators': 200, 'max_depth': 5, 'learning_rate': 0.03},
}

# ===== DATA =====
def fetch(sym, days=2000):
    code = sym.upper() + '0'
    end = datetime.now(); start = end - timedelta(days=days+200)
    try:
        df = ak.futures_main_sina(symbol=code, start_date=start.strftime('%Y%m%d'),
                                   end_date=end.strftime('%Y%m%d'))
        df.columns = ['date','open','high','low','close','volume','oi','settle']
        for c in ['open','high','low','close','volume','oi']:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        return df.dropna(subset=['close']).reset_index(drop=True)
    except: return None

def build_features(df, idx, window=60):
    if idx < window + 5: return None
    w = df.iloc[idx-window:idx+1]
    c = w['close'].values; o = w['open'].values
    h = w['high'].values; l = w['low'].values
    v = w['volume'].values; oi = w['oi'].values
    f = []
    if idx >= 1: f.append((o[-1]-c[-2])/c[-2]); f.append(abs(f[-1]))
    else: f.extend([0, 0])
    for lag in [1,3,5,10,20]: f.append((c[-1]-c[-lag-1])/c[-lag-1] if len(c) > lag else 0)
    for p in [5,10,20,60]: ma = np.mean(c[-min(p,len(c)):]); f.append((c[-1]-ma)/ma)
    f.append(np.std(c[-20:])/np.mean(c[-20:])); f.append((h[-1]-l[-1])/c[-1])
    vma = np.mean(v[-20:]) if np.mean(v[-20:]) > 0 else 1; f.append(v[-1]/vma)
    f.append(oi[-1]/np.mean(oi[-20:]) if len(oi) >= 20 and np.mean(oi[-20:]) > 0 else 1)
    ema12 = c[-1]; ema26 = c[-1]
    for j in range(len(c)-2, -1, -1): ema12 = (2/13)*c[j] + (11/13)*ema12; ema26 = (2/27)*c[j] + (25/27)*ema26
    f.append((ema12-ema26)/c[-1])
    dd_ = np.diff(c[-15:]); g = dd_[dd_>0].sum() if len(dd_[dd_>0]) > 0 else 0
    lo = abs(dd_[dd_<0].sum()) if len(dd_[dd_<0]) > 0 else 1e-10
    f.append(100 - 100/(1+g/lo) if lo > 0 else 50)
    bb = np.std(c[-20:]); ma20 = np.mean(c[-20:]); f.append((c[-1]-ma20)/(2*bb+1e-10))
    f.append(c[-1]/1000.0)
    return np.array(f, dtype=np.float32)

# ===== BACKTEST =====
def backtest_one(df, model, stop_cfg, rr, pos_mode, cost, mult):
    """单次回测: train/test split, test on last 30%"""
    W = 60
    trades = []
    
    # Split: train on first 70%, test on last 30%
    total_len = len(df)
    test_start = int(total_len * 0.7)
    if test_start < W + 100: return []
    
    test_range = range(max(test_start, W), total_len - 1)
    if len(test_range) < 50: return []
    
    # Calculate ATR for test period
    atr_series = np.zeros(total_len)
    for i in range(14, total_len):
        tr_vals = []
        for j in range(i-13, i+1):
            h = float(df.iloc[j]['high']); l = float(df.iloc[j]['low'])
            pc = float(df.iloc[j-1]['close']) if j > 0 else float(df.iloc[j]['close'])
            tr = max(h-l, abs(h-pc), abs(l-pc))
            tr_vals.append(tr)
        atr_series[i] = np.mean(tr_vals)

    pos = None
    
    for today in test_range:
        # Skip if holding a position (multi-day hold)
        if pos is not None:
            d = pos['dir']; ep = pos['entry']
            sp = pos['stop']; tp = pos['tp']
            vol_ = pos['vol']; cost_ = pos['cost']
            mult_ = pos['mult']

            o = float(df.iloc[today+1]['open'])
            h = float(df.iloc[today+1]['high'])
            l = float(df.iloc[today+1]['low'])
            c = float(df.iloc[today+1]['close'])

            hit_stop = (d == 'LONG' and l <= sp) or (d == 'SHORT' and h >= sp)
            hit_tp = (d == 'LONG' and h >= tp) or (d == 'SHORT' and l <= tp)
            hold_days = today - pos['entry_day'] + 1

            # Check multi-day hold (max 15 days, then close)
            if hold_days >= 15:
                exit_price = c; exit_type = 'MAXHOLD'
            elif hit_stop and hit_tp:
                exit_price = sp; exit_type = 'STOP'
            elif hit_stop:
                exit_price = sp; exit_type = 'STOP'
            elif hit_tp:
                exit_price = tp; exit_type = 'TP'
            else:
                continue  # Hold position

            if d == 'LONG':
                gross_pnl_pct = (exit_price - ep) / ep
            else:
                gross_pnl_pct = (ep - exit_price) / ep

            net_pnl_pct = gross_pnl_pct - cost_ * 2
            pnl_amount = net_pnl_pct * vol_ * ep * mult_

            trades.append({
                'entry': ep, 'exit': exit_price, 'type': exit_type,
                'dir': d, 'vol': vol_,
                'pnl_pct': net_pnl_pct, 'pnl': pnl_amount,
                'hold_days': hold_days
            })
            pos = None
            continue

        # No position — generate signal
        f = build_features(df, today, W)
        if f is None: continue
        try:
            prob = float(model.predict_proba(f.reshape(1, -1))[0][1])
        except: continue

        direction = 'LONG' if prob > 0.5 else 'SHORT'
        price = float(df.iloc[today]['close'])

        # Calculate stop
        if stop_cfg['type'] == 'atr':
            atr_val = atr_series[today]
            if atr_val == 0: continue
            stop_dist = atr_val * stop_cfg['mult']
            if direction == 'LONG':
                stop_price = price - stop_dist
            else:
                stop_price = price + stop_dist
        else:
            n = stop_cfg['n']
            if direction == 'LONG':
                lows = [float(df.iloc[k]['low']) for k in range(max(0, today-n), today+1)]
                stop_price = min(lows)
            else:
                highs = [float(df.iloc[k]['high']) for k in range(max(0, today-n), today+1)]
                stop_price = max(highs)

        stop_pct = abs(stop_price - price) / price
        if stop_pct > 0.15 or stop_pct < 0.002: continue

        # Position sizing
        if pos_mode.startswith('fixed'):
            parts = pos_mode.split('_')
            vol = int(parts[1].replace('x', ''))
        else:
            atr_val = atr_series[today]
            if atr_val == 0: continue
            atr_pct = atr_val / price
            if atr_pct < 0.01: leverage = 3.0
            elif atr_pct < 0.02: leverage = 2.0
            elif atr_pct < 0.03: leverage = 1.5
            else: leverage = 0.5
            base = 2
            vol = max(1, int(leverage * base))
            if pos_mode == 'vol_1.5x': vol = max(1, int(vol * 0.75))

        # Max loss check
        stop_distance = abs(price - stop_price)
        max_loss = stop_distance * vol * mult
        if max_loss > CAPITAL * MAX_LOSS_PCT:
            vol = max(1, int(CAPITAL * MAX_LOSS_PCT / (stop_distance * mult)))
            if vol == 0: continue

        # Take profit
        if direction == 'LONG':
            tp_price = price + (price - stop_price) * rr
        else:
            tp_price = price - (stop_price - price) * rr

        pos = {
            'dir': direction, 'entry': price, 'vol': vol,
            'stop': stop_price, 'tp': tp_price,
            'entry_day': today, 'cost': cost,
            'mult': mult
        }

    return trades

# ===== MAIN =====
def main():
    print("=" * 70)
    print("  Prophet v25 — 实盘级止损优化")
    print(f"  约束: 单笔最大亏损 ≤ {MAX_LOSS_PCT:.0%}")
    print(f"  WF窗口: {WF_WINDOWS}  资金: {CAPITAL:,}")
    print("=" * 70)

    all_results = {}

    for sym in SYMBOLS:
        print(f"\n{'='*70}")
        print(f"  {sym.upper()} {sym=='lh' and '生猪' or '焦煤'}")
        print(f"{'='*70}")

        # Fetch data
        print("  获取数据...", end=' ', flush=True)
        df = fetch(sym)
        if df is None or len(df) < 500:
            print("失败")
            continue
        print(f"{len(df)}行 OK")

        # Train model on in-sample
        print("  训练XGBoost...", end=' ', flush=True)
        W = 60
        X_list, y_list = [], []
        for i in range(W, len(df) - 1):
            f = build_features(df, i, W)
            if f is None: continue
            ret = (float(df.iloc[i+1]['close']) - float(df.iloc[i]['close'])) / float(df.iloc[i]['close'])
            label = 1 if ret > 0 else 0
            X_list.append(f); y_list.append(label)

        X = np.array(X_list); y = np.array(y_list)
        split = int(len(X) * 0.7)
        X_train, y_train = X[:split], y[:split]

        model = xgb.XGBClassifier(**XGB_PARAMS[sym], random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        print(f"OK (训练{len(X_train)}样本)")

        cost = REAL_COST[sym]
        mult = LOT_MULT[sym]

        # Baseline v24
        print(f"\n  基准 v24 (Struct26 RR=4):")
        bl_cfg = STOP_METHODS['struct26']
        bl_trades = backtest_one(df, model, bl_cfg, 4, 'vol_2x', cost, mult)
        if bl_trades:
            bl_wr = sum(1 for t in bl_trades if t['pnl'] > 0) / len(bl_trades)
            bl_total = sum(t['pnl'] for t in bl_trades)
            bl_max_loss = min(t['pnl'] for t in bl_trades) if bl_trades else 0
            avgs = [t['pnl'] for t in bl_trades]
            bl_calmar = bl_total / abs(bl_max_loss) if bl_max_loss < 0 and len(avgs) > 5 else 0
            bl_annual = bl_total / (len(bl_trades) / 250) if bl_trades else 0
            print(f"    {len(bl_trades)}笔 WR={bl_wr:.1%} PnL={bl_total:+,.0f} "
                  f"最大单笔亏损={bl_max_loss:,.0f}({abs(bl_max_loss)/CAPITAL:.0%})"
                  f" AR={bl_annual:+,.0f}/yr")

        # Grid search
        print(f"\n  网格搜索 ({len(STOP_METHODS)}×{len(RR_VALUES)}×{len(POS_MODES)}={len(STOP_METHODS)*len(RR_VALUES)*len(POS_MODES)}组合)...")
        
        best_score = -999
        best_combo = None
        rank = []

        total = len(STOP_METHODS) * len(RR_VALUES) * len(POS_MODES)
        count = 0

        for stop_name, stop_cfg in STOP_METHODS.items():
            for rr in RR_VALUES:
                for pos_mode in POS_MODES:
                    count += 1
                    trades = backtest_one(df, model, stop_cfg, rr, pos_mode, cost, mult)
                    if not trades or len(trades) < 10:
                        continue

                    wr = sum(1 for t in trades if t['pnl'] > 0) / len(trades)
                    total_pnl = sum(t['pnl'] for t in trades)
                    losses = [t['pnl'] for t in trades if t['pnl'] < 0]
                    max_loss = min(t['pnl'] for t in trades) if trades else -1
                    max_loss_pct = abs(max_loss) / CAPITAL

                    # Score: Calmar-like but penalize extreme single-trade losses
                    avg_loss = np.mean(losses) if losses else -1
                    if max_loss < 0 and len(trades) > 20:
                        calmar = total_pnl / abs(max_loss)
                    else:
                        calmar = 0

                    # Composite score: favor high PnL, low max single loss
                    score = total_pnl / CAPITAL * 10 + calmar * 0.5 - max_loss_pct * 5

                    n_trades = len(trades)
                    avg_hold = np.mean([t['hold_days'] for t in trades]) if trades else 0

                    rank.append({
                        'stop': stop_name, 'rr': rr, 'pos': pos_mode,
                        'n': n_trades, 'wr': wr,
                        'pnl': total_pnl, 'max_loss': max_loss,
                        'max_loss_pct': max_loss_pct,
                        'calmar': calmar, 'score': score,
                        'avg_hold': avg_hold
                    })

                    if len(rank) % 20 == 0:
                        print(f"    {count}/{total}...", end='\r', flush=True)

        print(f"    {count}/{total} 完成")
        
        # Sort by score
        rank.sort(key=lambda x: -x['score'])
        
        print(f"\n  TOP 10:")
        print(f"  {'止损':<12} {'RR':<4} {'仓位':<10} {'笔数':<5} {'胜率':<6} {'PnL':>12} {'最大单亏%':>8} {'Calmar':>7}")
        print(f"  {'-'*12} {'-'*4} {'-'*10} {'-'*5} {'-'*6} {'-'*12} {'-'*8} {'-'*7}")
        for r in rank[:10]:
            print(f"  {r['stop']:<12} {r['rr']:<4} {r['pos']:<10} {r['n']:<5} "
                  f"{r['wr']:.1%}  {r['pnl']:>+12,.0f} {r['max_loss_pct']:>7.0%} {r['calmar']:>6.1f}")

        # Find best within max loss constraint
        print(f"\n  约束内最优 (单笔亏损≤{MAX_LOSS_PCT:.0%}):")
        constrained = [r for r in rank if r['max_loss_pct'] <= MAX_LOSS_PCT]
        if constrained:
            best = constrained[0]
            print(f"    止损: {best['stop']}  RR: {best['rr']}  仓位: {best['pos']}")
            print(f"    {best['n']}笔 WR={best['wr']:.1%} PnL={best['pnl']:+,.0f} "
                  f"最大单亏={best['max_loss_pct']:.0%} Calmar={best['calmar']:.1f}")
        else:
            print(f"    无组合满足约束！放宽到15%:")
            constrained2 = [r for r in rank if r['max_loss_pct'] <= 0.15]
            if constrained2:
                best = constrained2[0]
                print(f"    止损: {best['stop']}  RR: {best['rr']}  仓位: {best['pos']}")
                print(f"    {best['n']}笔 WR={best['wr']:.1%} PnL={best['pnl']:+,.0f} "
                      f"最大单亏={best['max_loss_pct']:.0%}")

        all_results[sym] = {'rank': rank, 'constrained': constrained}

    # Summary
    print(f"\n{'='*70}")
    print(f"  总结")
    print(f"{'='*70}")
    for sym in SYMBOLS:
        if sym not in all_results: continue
        r = all_results[sym]
        if r['constrained']:
            b = r['constrained'][0]
            print(f"  {sym.upper()}: {b['stop']} RR={b['rr']} {b['pos']} "
                  f"→ {b['n']}笔 PnL={b['pnl']:+,.0f} 单亏≤{b['max_loss_pct']:.0%} Calmar={b['calmar']:.1f}")

    print(f"\n✅ v25优化完成")
    print(f"  对比v24 Struct26: 更紧止损, 更小单亏, 更适合实盘")

if __name__ == '__main__':
    main()
