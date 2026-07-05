#!/usr/bin/env python3
"""V17 vs V25 head-to-head: same data, same model, different stops."""
import numpy as np, pandas as pd, pickle, os, sys
from datetime import datetime, timedelta
import akshare as ak

MODEL_DIR = '/home/a/prophet_futures/prophet_futures/models'

SYMBOLS = {
    'lh2609': {
        'code': 'LH0', 'name': 'LH', 'multiplier': 16, 'cost': 0.0006,
        'max_pos': 6, 'struct_period': 26,
        'atr_stop_mult': 1.5, 'atr_period': 20, 'rr': 4.0,
    },
    'jm2609': {
        'code': 'JM0', 'name': 'JM', 'multiplier': 60, 'cost': 0.0011,
        'max_pos': 4, 'struct_period': 20,
        'atr_stop_mult': 2.0, 'atr_period': 20, 'rr': 3.5,
    },
}

def build_features(df, idx, window=60):
    if idx < window + 5: return None
    w = df.iloc[idx-window:idx+1]
    c = w['close'].values; o = w['open'].values
    h = w['high'].values; l = w['low'].values
    v = w['volume'].values; oi = w['oi'].values
    f = []
    if idx >= 1:
        f.append((o[-1]-c[-2])/c[-2]); f.append(abs(f[-1]))
    else:
        f.extend([0, 0])
    for lag in [1,3,5,10,20]:
        f.append((c[-1]-c[-lag-1])/c[-lag-1] if len(c) > lag else 0)
    for p in [5,10,20,60]:
        ma = np.mean(c[-min(p,len(c)):]); f.append((c[-1]-ma)/ma)
    f.append(np.std(c[-20:])/np.mean(c[-20:]))
    f.append((h[-1]-l[-1])/c[-1])
    vma = np.mean(v[-20:]) if np.mean(v[-20:]) > 0 else 1
    f.append(v[-1]/vma)
    f.append(oi[-1]/np.mean(oi[-20:]) if len(oi) >= 20 and np.mean(oi[-20:]) > 0 else 1)
    ema12 = c[-1]; ema26 = c[-1]
    for j in range(len(c)-2, -1, -1):
        ema12 = (2/13)*c[j] + (11/13)*ema12; ema26 = (2/27)*c[j] + (25/27)*ema26
    f.append((ema12-ema26)/c[-1])
    dd_ = np.diff(c[-15:])
    g = dd_[dd_>0].sum() if len(dd_[dd_>0]) > 0 else 0
    lo = abs(dd_[dd_<0].sum()) if len(dd_[dd_<0]) > 0 else 1e-10
    f.append(100 - 100/(1+g/lo) if lo > 0 else 50)
    bb = np.std(c[-20:]); ma20 = np.mean(c[-20:])
    f.append((c[-1]-ma20)/(2*bb+1e-10))
    f.append(c[-1]/1000.0)
    return np.array(f, dtype=np.float32)

def calc_atr(df, idx, period=20):
    if idx < period: return None
    vals = [abs(float(df.iloc[i]['high']) - float(df.iloc[i]['low']))
            for i in range(idx-period+1, idx+1)]
    return np.mean(vals)

def struct_stop(df, idx, period, dir_long):
    if idx < period: return None
    start = idx - period
    if dir_long:
        return float(df.iloc[start:idx]['low'].min())
    else:
        return float(df.iloc[start:idx]['high'].max())

def run_backtest(df, model, cfg, mode='v17'):
    trades = []
    pos = None
    warmup = max(cfg.get('struct_period', 26), cfg.get('atr_period', 20)) + 10
    
    for i in range(warmup, len(df)):
        feats = build_features(df, i, 60)
        if feats is None: continue
        try:
            prob = float(model.predict_proba(feats.reshape(1, -1))[0][1])
        except: continue
        
        price = float(df.iloc[i]['close'])
        high = float(df.iloc[i]['high'])
        low = float(df.iloc[i]['low'])
        atr = calc_atr(df, i, cfg['atr_period'])
        if atr is None or price <= 0: continue
        
        # Check existing position stop
        if pos:
            d, entry, stop, entry_i, vol = pos
            if d == 'LONG':
                if low <= stop:
                    pnl = ((stop-entry)/entry) - cfg['cost']*2
                    trades.append({'dir': d, 'entry': entry, 'exit': stop, 
                        'pnl_pct': pnl, 'bars': i-entry_i, 'type': 'STOP'})
                    pos = None
            else:
                if high >= stop:
                    pnl = ((entry-stop)/entry) - cfg['cost']*2
                    trades.append({'dir': d, 'entry': entry, 'exit': stop,
                        'pnl_pct': pnl, 'bars': i-entry_i, 'type': 'STOP'})
                    pos = None
        
        # Check TP (V25 only)
        if pos and mode == 'v25':
            d, entry, stop, entry_i, vol = pos
            stop_dist = abs(price - stop)
            tp = price + stop_dist * cfg['rr'] if d == 'LONG' else price - stop_dist * cfg['rr']
            if d == 'LONG' and high >= tp:
                pnl = ((tp-entry)/entry) - cfg['cost']*2
                trades.append({'dir': d, 'entry': entry, 'exit': tp, 'pnl_pct': pnl,
                    'bars': i-entry_i, 'type': 'TP'})
                pos = None
            elif d == 'SHORT' and low <= tp:
                pnl = ((entry-tp)/entry) - cfg['cost']*2
                trades.append({'dir': d, 'entry': entry, 'exit': tp, 'pnl_pct': pnl,
                    'bars': i-entry_i, 'type': 'TP'})
                pos = None
        
        # New entry
        if pos is None:
            atr_pct = atr / price
            if atr_pct < 0.01: lev = 3.0
            elif atr_pct < 0.02: lev = 2.0
            elif atr_pct < 0.03: lev = 1.5
            elif atr_pct < 0.05: lev = 0.5
            else: lev = 0
            pos_size = max(1, int(lev * (cfg['max_pos'] // 2))) if lev > 0 else 0
            
            if pos_size > 0:
                sd = 'LONG' if prob > 0.5 else 'SHORT'
                
                if mode == 'v17':
                    stop_val = struct_stop(df, i, cfg['struct_period'], sd=='LONG')
                    if stop_val is None: continue
                    if sd == 'LONG' and low <= stop_val: continue
                    if sd == 'SHORT' and high >= stop_val: continue
                else:
                    stop_dist = atr * cfg['atr_stop_mult']
                    if sd == 'LONG':
                        stop_val = price - stop_dist
                        if low <= stop_val: continue
                    else:
                        stop_val = price + stop_dist
                        if high >= stop_val: continue
                
                pos = (sd, price, stop_val, i, pos_size)
    
    # EOD close
    if pos:
        d, entry, stop, entry_i, vol = pos
        lp = float(df.iloc[-1]['close'])
        pnl = ((lp-entry)/entry if d=='LONG' else (entry-lp)/entry) - cfg['cost']*2
        trades.append({'dir': d, 'entry': entry, 'exit': lp, 'pnl_pct': pnl,
            'bars': len(df)-1-entry_i, 'type': 'EOD'})
    
    return trades

def stats(name, trades):
    if not trades: return f"{name}: 无交易", None
    wins = [t for t in trades if t['pnl_pct'] > 0]
    losses = [t for t in trades if t['pnl_pct'] <= 0]
    wr = len(wins)/len(trades)
    total_pnl = sum(t['pnl_pct'] for t in trades)
    gw = sum(t['pnl_pct'] for t in wins)
    gl = abs(sum(t['pnl_pct'] for t in losses))
    pf = gw/gl if gl>0 else 99
    
    eq = 1.0; peak = 1.0; mdd = 0
    for t in trades:
        eq *= (1+t['pnl_pct'])
        peak = max(peak, eq)
        mdd = min(mdd, (eq-peak)/peak)
    
    avg_win = np.mean([t['pnl_pct'] for t in wins]) if wins else 0
    avg_loss = np.mean([t['pnl_pct'] for t in losses]) if losses else 0
    avg_bars = np.mean([t['bars'] for t in trades])
    
    # Daily compound return
    bars_total = sum(t['bars'] for t in trades)
    if bars_total > 0:
        final_eq = eq
        # approximate from total pnl
    
    return f"{name}: {len(trades)}笔 WR={wr:.0%} 收益={total_pnl*100:+.1f}% PF={pf:.1f} MDD={mdd*100:.1f}% 均盈={avg_win*100:+.1f}% 均亏={avg_loss*100:+.1f}%", {
        'trades': len(trades), 'wr': wr, 'pnl': total_pnl, 'pf': pf, 'mdd': mdd,
        'avg_win': avg_win, 'avg_loss': avg_loss, 'avg_bars': avg_bars,
    }

# ===== MAIN =====
print("=" * 55)
print("  V17 vs V25 — 同数据/模型/特征，只比止损逻辑")
print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
print("=" * 55)

for sym_key, cfg in SYMBOLS.items():
    mp = os.path.join(MODEL_DIR, f'{sym_key}_xgb.pkl')
    if not os.path.exists(mp):
        print(f"\n{sym_key}: 模型不存在")
        continue
    
    with open(mp, 'rb') as f:
        model = pickle.load(f)
    
    print(f"\n{'─'*55}")
    print(f"  {sym_key} ({cfg['name']})")
    print(f"  V17: Struct{cfg['struct_period']}止损 | V25: ATR{cfg['atr_stop_mult']}× RR{cfg['rr']}")
    print(f"{'─'*55}")
    
    try:
        end = datetime.now(); start = end - timedelta(days=1200)
        df = ak.futures_main_sina(symbol=cfg['code'], start_date=start.strftime('%Y%m%d'), end_date=end.strftime('%Y%m%d'))
        df.columns = ['date','open','high','low','close','volume','oi','settle']
        for c in ['open','high','low','close','volume','oi']:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        df = df.dropna(subset=['close']).reset_index(drop=True)
        print(f"  数据: {len(df)}行 {df.iloc[0]['date']} ~ {df.iloc[-1]['date']}")
    except Exception as e:
        print(f"  数据失败: {e}")
        continue
    
    t17 = run_backtest(df.copy(), model, cfg, 'v17')
    s17, d17 = stats('V17', t17)
    
    t25 = run_backtest(df.copy(), model, cfg, 'v25')
    s25, d25 = stats('V25', t25)
    
    print(f"  {s17}")
    print(f"  {s25}")
    
    if d17 and d25:
        pnl_diff = (d25['pnl'] - d17['pnl']) * 100
        wr_diff = (d25['wr'] - d17['wr']) * 100
        mdd_diff = (d25['mdd'] - d17['mdd']) * 100
        winner = 'V25 ✅' if d25['pnl'] > d17['pnl'] else 'V17 ✅'
        print(f"  {'─'*40}")
        print(f"  {winner} 收益差{pnl_diff:+.1f}pp | 胜率差{wr_diff:+.1f}pp | 回撤差{mdd_diff:+.1f}pp")

print(f"\n{'='*55}")
print(f"  PK完成")
print(f"{'='*55}")
