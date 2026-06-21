#!/usr/bin/env python3
"""Quantitative stop optimization: 3 types, WF, Calmar-ranked"""
import numpy as np, pandas as pd, json, time
import akshare as ak, xgboost as xgb

SYM = {
    'LH': ('LH0', 0.0006, False, 200, 5, 0.05, 60, 16, 3),
    'JM': ('JM0', 0.0011, False, 100, 4, 0.03, 60, 60, 5),
    'RM': ('RM0', 0.0011, True,  100, 5, 0.03, 120, 10, 20),
}
N_WF = 1000; CAP = 300000
STOP_VALS = [0.5, 0.7, 1.0, 1.3, 1.5, 1.8, 2.0, 2.5, 3.0]  # R = entry ATR
STRUCT_N = [5, 8, 10, 12, 15, 20]  # N-bar lookback for structural stop

def precompute(df, win=60):
    n = len(df); F = np.zeros((n, 20), dtype=np.float32)
    A = np.zeros(n, dtype=np.float32); V = np.zeros(n, dtype=bool)
    for idx in range(win+5, n):
        w = df.iloc[idx-win:idx+1]; c = w['close'].values; o = w['open'].values
        h = w['high'].values; l = w['low'].values; v = w['volume'].values; oi = w['oi'].values
        f = F[idx]
        if idx >= 1: f[0] = (o[-1]-c[-2])/c[-2]; f[1] = abs(f[0])
        for li, lag in enumerate([1,3,5,10,20], 2): f[li] = (c[-1]-c[-lag-1])/c[-lag-1] if len(c) > lag else 0
        for pi, p in enumerate([5,10,20,60], 7): ma = np.mean(c[-min(p,len(c)):]); f[pi] = (c[-1]-ma)/ma
        f[11] = np.std(c[-20:])/np.mean(c[-20:]); f[12] = (h[-1]-l[-1])/c[-1]
        vma = np.mean(v[-20:]) if np.mean(v[-20:]) > 0 else 1; f[13] = v[-1]/vma
        f[14] = oi[-1]/np.mean(oi[-20:]) if len(oi) >= 20 and np.mean(oi[-20:]) > 0 else 1
        ema12 = c[-1]; ema26 = c[-1]
        for j in range(len(c)-2, -1, -1): ema12 = (2/13)*c[j] + (11/13)*ema12; ema26 = (2/27)*c[j] + (25/27)*ema26
        f[15] = (ema12-ema26)/c[-1]
        dd_ = np.diff(c[-15:]); g = dd_[dd_>0].sum() if len(dd_[dd_>0]) > 0 else 0
        lo = abs(dd_[dd_<0].sum()) if len(dd_[dd_<0]) > 0 else 1e-10
        f[16] = 100 - 100/(1+g/lo) if lo > 0 else 50
        bb = np.std(c[-20:]); ma20 = np.mean(c[-20:])
        f[17] = (c[-1]-ma20)/(2*bb+1e-10)
        try: m = int(str(df.iloc[idx]['date'])[5:7]); f[18] = np.sin(2*np.pi*m/12); f[19] = np.cos(2*np.pi*m/12)
        except: pass
        V[idx] = True
        A[idx] = np.mean([abs(df.iloc[k]['high']-df.iloc[k]['low']) for k in range(max(0,idx-20),idx+1)])
    L = np.zeros(n, dtype=int)
    for i in range(n-1):
        if V[i]: L[i] = 1 if df.iloc[i+1]['close'] > df.iloc[i]['close'] else 0
    return F, A, V, L

def stop_price(entry, atr, pred, mode, val, df_vals=None):
    """Calculate stop price based on mode and value"""
    if mode == 'fixed_points':
        dist = val  # val = points
    elif mode == 'atr_mult':
        dist = val * atr  # val = ATR multiplier
    elif mode == 'structure':
        # val = N (bars), use lowest low (LONG) or highest high (SHORT) of last N bars
        n_bars = int(val)
        if df_vals is not None and len(df_vals) >= n_bars:
            if pred == 1: dist = entry - min(df_vals[-n_bars:])  # LONG: stop below recent low
            else: dist = max(df_vals[-n_bars:]) - entry  # SHORT: stop above recent high
        else: dist = atr  # fallback
    else:
        dist = atr
    
    if pred == 1: return entry - dist  # LONG stop
    else: return entry + dist  # SHORT stop

def run_backtest(df, F, A, V, L, ne, d, lr, win, cost, rev, mult, base, mode, val, struct_n=None):
    n = len(df); ts = int(n * 0.7)  # 70/30 split → ~920 training samples
    pnl_seq = []; wins = 0; total = 0; cap = CAP; peak = CAP; max_dd = 0
    noise_stops = 0  # stops hit but trade would have won
    
    for run in range(min(N_WF, (n-ts)//10)):
        sp = ts + run * 10
        if sp + 10 > n: break
        t1 = sp; t2 = min(sp + 10, n)
        tr = [i for i in range(win, t1) if V[i]]
        if len(tr) < 100: continue
        X_tr = F[tr]; y_tr = L[tr]
        if len(np.unique(y_tr)) < 2: continue
        m = xgb.XGBClassifier(n_estimators=ne, max_depth=d, learning_rate=lr,
                               verbosity=0, random_state=42)
        m.fit(X_tr, y_tr)
        
        for j in range(t1, t2-1):
            if not V[j]: continue
            p = m.predict_proba(F[j].reshape(1, -1))[0]
            pred = 1 if p[1] > 0.5 else 0
            if rev: pred = 1 - pred
            
            entry = df.iloc[j]['close']
            atr = A[j]
            
            # Multi-day hold: track forward until stop/target or 15 days
            if mode == 'structure' and struct_n:
                struct_vals_low = [df.iloc[k]['low'] for k in range(max(0,j-struct_n), j+1)]
                struct_vals_high = [df.iloc[k]['high'] for k in range(max(0,j-struct_n), j+1)]
                stp = stop_price(entry, atr, pred, mode, struct_n, struct_vals_low if pred==1 else struct_vals_high)
            else:
                stp = stop_price(entry, atr, pred, mode, val)
            
            tgt = entry + 3*abs(stp-entry) if pred==1 else entry - 3*abs(stp-entry)  # RR=3
            
            trade_pnl = 0; stopped = False; noise = False
            for k in range(j+1, min(j+16, n)):
                px_high = df.iloc[k]['high']; px_low = df.iloc[k]['low']; px_close = df.iloc[k]['close']
                
                hit = False
                if pred == 1:  # LONG
                    if px_high >= tgt: trade_pnl = (tgt-entry)*mult*base - 2*cost*entry*mult*base; hit=True
                    elif px_low <= stp: trade_pnl = (stp-entry)*mult*base - 2*cost*entry*mult*base; stopped=True; hit=True
                else:  # SHORT
                    if px_low <= tgt: trade_pnl = (entry-tgt)*mult*base - 2*cost*entry*mult*base; hit=True
                    elif px_high >= stp: trade_pnl = (entry-stp)*mult*base - 2*cost*entry*mult*base; stopped=True; hit=True
                
                if hit:
                    if stopped:
                        would_win = px_close > entry if pred==1 else px_close < entry
                        if would_win: noise = True
                    break
            else:
                # Held to end, exit at last close
                last_close = df.iloc[min(j+15, n-1)]['close']
                trade_pnl = (last_close-entry)*mult*base - 2*cost*entry*mult*base if pred==1 else (entry-last_close)*mult*base - 2*cost*entry*mult*base
            
            pnl_seq.append(trade_pnl); cap += trade_pnl
            if cap > peak: peak = cap
            dd_pct = (peak - cap) / peak * 100
            if dd_pct > max_dd: max_dd = dd_pct
            wins += (1 if trade_pnl > 0 else 0); total += 1
            if noise: noise_stops += 1
    
    if total == 0: return None
    tp = sum(pnl_seq); wr = wins / total * 100
    gp = sum(p for p in pnl_seq if p > 0); gl = abs(sum(p for p in pnl_seq if p < 0))
    pf = gp / gl if gl > 0 else 999
    avg_rr = (gp / wins) / (gl / (total - wins)) if wins > 0 and (total - wins) > 0 else 0
    yrs = (n - ts) / 252
    ann_ret = ((CAP + tp) / CAP) ** (1 / yrs) - 1 if tp > -CAP and yrs > 0 else tp / CAP / yrs
    calmar = (ann_ret * 100) / max_dd if max_dd > 0 else 0
    sharpe = (ann_ret) / (np.std(pnl_seq) / CAP * np.sqrt(252)) if np.std(pnl_seq) > 0 else 0
    noise_pct = noise_stops / total * 100
    
    return {'trades': total, 'wr': round(wr, 1), 'pf': round(pf, 2), 'pnl': round(tp, 0),
            'ann': round(ann_ret*100, 1), 'dd': round(max_dd, 1), 'avg_rr': round(avg_rr, 2),
            'calmar': round(calmar, 2), 'sharpe': round(sharpe, 2),
            'noise_pct': round(noise_pct, 1), 'mode': mode, 'val': val}

# Run all combos
all_results = []
print(f'{"Sym":<4} {"Stop":<15} {"Trd":<7} {"WR":<6} {"PF":<6} {"PnL":<9} {"Ann":<8} {"DD":<7} {"Calmar":<7} {"Noise%":<7} {"RRavg":<6}')

for sk, (sc, cost, rev, ne, d, lr, win, mult, base) in SYM.items():
    e = pd.Timestamp.now(); s = e - pd.Timedelta(days=2500)
    df = ak.futures_main_sina(symbol=sc, start_date=s.strftime('%Y%m%d'), end_date=e.strftime('%Y%m%d'))
    df.columns = ['date','open','high','low','close','volume','oi','settle']
    for x in ['open','high','low','close','volume','oi']: df[x] = pd.to_numeric(df[x], errors='coerce')
    df = df.dropna(subset=['close']).reset_index(drop=True)
    F, A, V, L = precompute(df, win)
    
    # ATR-based stops
    for v in STOP_VALS:
        r = run_backtest(df, F, A, V, L, ne, d, lr, win, cost, rev, mult, base, 'atr_mult', v)
        if r:
            all_results.append({**r, 'symbol': sk, 'label': f'ATR{v:.1f}'})
            sign = '+' if r['pnl'] > 0 else ''
            print(f'{sk:<4} ATR{v:<11.1f} {r["trades"]:<7} {r["wr"]:<5.1f}% {r["pf"]:<6.2f} {r["pnl"]/10000:<+8.1f}万 {r["ann"]:<+7.1f}% {r["dd"]:<6.1f}% {r["calmar"]:<7.2f} {r["noise_pct"]:<6.1f}% {r["avg_rr"]:<6.2f} {sign}')
    
    # Fixed points (in ATR units for comparison, converted to points)
    avg_atr = np.mean(A[V])
    for pts_factor in [0.3, 0.5, 1.0]:
        pts = avg_atr * pts_factor
        r = run_backtest(df, F, A, V, L, ne, d, lr, win, cost, rev, mult, base, 'fixed_points', pts)
        if r:
            all_results.append({**r, 'symbol': sk, 'label': f'Fix{pts:.0f}pt'})
            print(f'{sk:<4} Fix{pts:<10.0f}pt {r["trades"]:<7} {r["wr"]:<5.1f}% {r["pf"]:<6.2f} {r["pnl"]/10000:<+8.1f}万 {r["ann"]:<+7.1f}% {r["dd"]:<6.1f}% {r["calmar"]:<7.2f} {r["noise_pct"]:<6.1f}% {r["avg_rr"]:<6.2f}')
    
    # Structural stops
    for sn in STRUCT_N:
        r = run_backtest(df, F, A, V, L, ne, d, lr, win, cost, rev, mult, base, 'structure', sn, struct_n=sn)
        if r:
            all_results.append({**r, 'symbol': sk, 'label': f'Struct{sn}'})
            print(f'{sk:<4} Struct{sn:<9} {r["trades"]:<7} {r["wr"]:<5.1f}% {r["pf"]:<6.2f} {r["pnl"]/10000:<+8.1f}万 {r["ann"]:<+7.1f}% {r["dd"]:<6.1f}% {r["calmar"]:<7.2f} {r["noise_pct"]:<6.1f}% {r["avg_rr"]:<6.2f}')

# Rank by Calmar (higher = better risk-adjusted return)
# Filter: DD < 200%, noise < 30%
all_results.sort(key=lambda x: -x['calmar'])
filtered = [r for r in all_results if r['dd'] < 200 and r['noise_pct'] < 30]

print(f'\n===== TOP BY CALMAR (DD<200% noise<30%) =====')
print(f'{"Rank":<5} {"Sym":<4} {"Stop":<15} {"Calmar":<7} {"PnL":<9} {"Ann":<8} {"DD":<7} {"WR":<6} {"Noise%":<7} {"PF":<6}')
for i, r in enumerate(filtered[:10]):
    print(f'{i+1:<5} {r["symbol"]:<4} {r["label"]:<15} {r["calmar"]:<7.2f} {r["pnl"]/10000:<+8.1f}万 {r["ann"]:<+7.1f}% {r["dd"]:<6.1f}% {r["wr"]:<5.1f}% {r["noise_pct"]:<6.1f}% {r["pf"]:<6.2f}')

with open('/home/a/prophet_futures/prophet_futures/stop_opt_results.json', 'w') as f:
    json.dump(all_results, f, indent=2, default=str)
print('\nSaved: stop_opt_results.json')
