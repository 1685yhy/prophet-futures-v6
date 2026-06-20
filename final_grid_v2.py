#!/usr/bin/env python3
"""Final Grid v2 — Dynamic position sizing based on current capital"""
import sys, json, time
import numpy as np, pandas as pd
from datetime import datetime, timedelta
import akshare as ak
import xgboost as xgb

SYMBOLS = {
    'LH': ('LH0', 0.0006, False, 200, 5, 0.05, 60, 16, 3),
    'JM': ('JM0', 0.0011, False, 100, 4, 0.03, 60, 60, 5),
    'RM': ('RM0', 0.0011, True,  100, 5, 0.03, 120, 10, 20),
}
START_CAPITAL = 300000
N_WF = 500
SM_LIST = [3, 4, 5]
RR_LIST = [4.0, 4.5, 5.0]

def fetch(c):
    e = datetime.now(); s = e - timedelta(days=2500)
    df = ak.futures_main_sina(symbol=c, start_date=s.strftime('%Y%m%d'), end_date=e.strftime('%Y%m%d'))
    df.columns = ['date','open','high','low','close','volume','oi','settle']
    for x in ['open','high','low','close','volume','oi']:
        df[x] = pd.to_numeric(df[x], errors='coerce')
    return df.dropna(subset=['close']).reset_index(drop=True)

def feats(df, idx, L=60):
    if idx < L + 5: return None
    w = df.iloc[idx-L:idx+1]
    c = w['close'].values; o = w['open'].values
    h = w['high'].values; l = w['low'].values
    v = w['volume'].values; oi = w['oi'].values
    f = []
    if idx >= 1: f.append((o[-1]-c[-2])/c[-2]); f.append(abs(f[-1]))
    else: f.extend([0, 0])
    for lag in [1,3,5,10,20]:
        f.append((c[-1]-c[-lag-1])/c[-lag-1] if len(c) > lag else 0)
    for p in [5,10,20,60]:
        ma = np.mean(c[-min(p, len(c)):]); f.append((c[-1]-ma)/ma)
    f.append(np.std(c[-20:])/np.mean(c[-20:])); f.append((h[-1]-l[-1])/c[-1])
    vma = np.mean(v[-20:]) if np.mean(v[-20:]) > 0 else 1; f.append(v[-1]/vma)
    f.append(oi[-1]/np.mean(oi[-20:]) if len(oi) >= 20 and np.mean(oi[-20:]) > 0 else 1)
    ema12 = c[-1]; ema26 = c[-1]
    for i in range(len(c)-2, -1, -1):
        ema12 = (2/13)*c[i] + (11/13)*ema12; ema26 = (2/27)*c[i] + (25/27)*ema26
    f.append((ema12-ema26)/c[-1])
    d = np.diff(c[-15:]); g = d[d>0].sum() if len(d[d>0]) > 0 else 0
    lo = abs(d[d<0].sum()) if len(d[d<0]) > 0 else 1e-10
    f.append(100 - 100/(1+g/lo) if lo > 0 else 50)
    bb = np.std(c[-20:]); ma20 = np.mean(c[-20:]); f.append((c[-1]-ma20)/(2*bb+1e-10))
    try:
        m = int(str(df.iloc[idx]['date'])[5:7])
        f.append(np.sin(2*np.pi*m/12)); f.append(np.cos(2*np.pi*m/12))
    except:
        f.extend([0, 0])
    f.append(c[-1]/1000.0)
    return np.array(f, dtype=np.float32)

def wf(df, ne, d, lr, win, cost, rev, mult, base, sm, rr):
    n = len(df); ts = int(n * 0.6)
    if ts < 200: return None
    
    wins = 0; total = 0; pnl_rmb = []
    current_capital = START_CAPITAL  # Dynamic! Changes with PnL
    max_capital = START_CAPITAL
    peak_capital = START_CAPITAL
    max_dd = 0
    
    for run in range(min(N_WF, (n - ts) // 10)):
        sp = ts + run * 10
        if sp + 10 > n: break
        td = df.iloc[:sp]; t1 = sp; t2 = min(sp + 10, n)
        X, y = [], []
        for i in range(win, len(td) - 1):
            f = feats(td, i, win)
            if f is None: continue
            y.append(1 if td.iloc[i+1]['close'] > td.iloc[i]['close'] else 0); X.append(f)
        if len(X) < 100: continue
        ya = np.array(y)
        if len(np.unique(ya)) < 2: continue
        m = xgb.XGBClassifier(n_estimators=ne, max_depth=d, learning_rate=lr,
                               use_label_encoder=False, eval_metric='logloss',
                               verbosity=0, random_state=42)
        m.fit(np.array(X), ya)
        for j in range(t1, t2 - 1):
            f = feats(df, j, win)
            if f is None: continue
            p = m.predict_proba(f.reshape(1, -1))[0]
            pred = 1 if p[1] > 0.5 else 0
            if rev: pred = 1 - pred
            entry = df.iloc[j]['close']; fp = df.iloc[j+1:min(j+15, n)]['close'].values
            if len(fp) == 0: continue
            av = [abs(df.iloc[k]['high'] - df.iloc[k]['low']) for k in range(max(0, j-20), j+1)]
            atr = np.mean(av) if av else entry * 0.02; ap = atr / entry
            if ap > 0.05: continue
            
            # Dynamic position based on CURRENT capital
            cap_ratio = max(0.3, current_capital / START_CAPITAL)  # floor 30%
            
            if ap < 0.01: lev = 3.0
            elif ap < 0.02: lev = 2.0
            elif ap < 0.03: lev = 1.5
            elif ap < 0.05: lev = 0.5
            else: continue
            
            lots = max(1, int(lev * base * cap_ratio))
            if current_capital < 100000: lots = max(1, lots // 2)  # Half size below 10万
            
            stop_pts = sm * cost * entry; target_pts = rr * stop_pts
            rtc = 2 * cost * entry * lots * mult
            
            for px in fp:
                if pred == 1: chg = px - entry
                else: chg = entry - px
                if chg >= target_pts:
                    trade_pnl = target_pts * lots * mult - rtc; wins += 1; total += 1; break
                elif chg <= -stop_pts:
                    trade_pnl = -stop_pts * lots * mult - rtc; total += 1; break
            else:
                lp = fp[-1]; chg = lp - entry if pred == 1 else entry - lp
                trade_pnl = chg * lots * mult - rtc
                wins += (1 if trade_pnl > 0 else 0); total += 1
            
            pnl_rmb.append(trade_pnl)
            current_capital += trade_pnl
            
            # Track peak and DD
            if current_capital > peak_capital:
                peak_capital = current_capital
            dd_pct = (peak_capital - current_capital) / peak_capital * 100
            if dd_pct > max_dd:
                max_dd = dd_pct
            
            # Ruin check
            if current_capital <= 30000:  # 10% left = ruin
                break
    
    if total == 0: return None
    wr = wins / total * 100; tp = current_capital - START_CAPITAL
    test_days = n - ts; yrs = test_days / 252
    ar = ((current_capital / START_CAPITAL) ** (1/yrs) - 1) * 100 if yrs > 0 and current_capital > 0 else 0
    
    return {'trades': total, 'wr': round(wr, 1), 'pf': 0, 'pnl': round(tp, 0),
            'ar': round(ar, 1), 'dd': round(max_dd, 1), 'sm': sm, 'rr': rr,
            'final_capital': round(current_capital, 0),
            'ruin': current_capital <= 100000}

data = {}
for sk, (sc, _, _, _, _, _, _, _, _) in SYMBOLS.items():
    data[sk] = fetch(sc)
    print(f'{sk}: {len(data[sk])} rows')

total = len(SM_LIST) * len(RR_LIST) * 3
print(f'\nGrid: {len(SM_LIST)} stops × {len(RR_LIST)} RRs × 3 = {total} combos')
print(f'Dynamic capital: lots scale with PnL, floor 30%, half-size below 10万')
print(f'{"Sym":<4} {"Stop":<6} {"RR":<5} {"Trd":<7} {"WR":<6} {"PnL":<9} {"Ann":<8} {"DD":<8} {"FinalCap":<10}')
print('-' * 75)

all_results = {}
for sk, (sc, cost, rev, ne, d, lr, win, mult, base) in SYMBOLS.items():
    df = data[sk]
    for sm in SM_LIST:
        for rr in RR_LIST:
            t0 = time.time()
            r = wf(df, ne, d, lr, win, cost, rev, mult, base, sm, rr)
            elapsed = time.time() - t0
            if r:
                k = f'{sk}_sm{sm}_rr{rr}'
                all_results[k] = r
                ruin = '💀' if r['ruin'] else ''
                print(f'{sk:<4} {sm}xcost {rr:<5.1f} {r["trades"]:<7} {r["wr"]:<5.1f}% {r["pnl"]/10000:<+8.1f}万 {r["ar"]:<+7.1f}% {r["dd"]:<7.1f}% {r["final_capital"]/10000:<9.1f}万 {ruin}')

print('\n=== PER SYMBOL BEST (highest final capital) ===')
for sk in ['LH', 'JM', 'RM']:
    sym_results = [(k, v) for k, v in all_results.items() if k.startswith(sk)]
    sym_results.sort(key=lambda x: -x[1]['final_capital'])
    print(f'\n{sk}:')
    for k, v in sym_results[:3]:
        ruin = ' 💀RUIN' if v['ruin'] else ''
        print(f'  Stop={v["sm"]}xCost RR=1:{v["rr"]} → {v["trades"]}t WR={v["wr"]}% PnL={v["pnl"]/10000:+.1f}万 Ann={v["ar"]:+.1f}%/yr DD={v["dd"]}% Final={v["final_capital"]/10000:.1f}万{ruin}')

with open('/home/a/prophet_futures/prophet_futures/final_grid_v2.json', 'w') as f:
    json.dump({k: v for k, v in all_results.items()}, f, indent=2, default=str)
print('\nSaved: final_grid_v2.json')
