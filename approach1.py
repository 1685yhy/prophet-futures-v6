#!/usr/bin/env python3
"""LH Approach 1: Daily bars with wide ATR stops, intraday OHLC simulation, 500 WF"""
import numpy as np, pandas as pd, json, time
from datetime import datetime, timedelta
import akshare as ak, xgboost as xgb

daily = ak.futures_main_sina(symbol='LH0', start_date='20210101', end_date='20260620')
daily.columns = ['date','open','high','low','close','volume','oi','settle']
for x in ['open','high','low','close','volume','oi']:
    daily[x] = pd.to_numeric(daily[x], errors='coerce')
daily = daily.dropna(subset=['close']).reset_index(drop=True)

N_WF = 500
CAP = 300000
win = 60; ne = 200; d = 5; lr = 0.05; cost = 0.0006; mult = 16; base = 3
SM = [1.0, 1.5, 2.0, 2.5]
RR = [3.0, 4.0, 5.0]
CF = [0.50, 0.52]

def build_f(df_local, idx):
    if idx < win + 5: return None
    w = df_local.iloc[idx-win:idx+1]; c = w['close'].values; o = w['open'].values
    h = w['high'].values; l = w['low'].values; v = w['volume'].values; oi = w['oi'].values
    f = []
    if idx >= 1: f.append((o[-1]-c[-2])/c[-2]); f.append(abs(f[-1]))
    else: f.extend([0, 0])
    for lag in [1,3,5,10,20]:
        f.append((c[-1]-c[-lag-1])/c[-lag-1] if len(c) > lag else 0)
    for p in [5,10,20,60]:
        ma = np.mean(c[-min(p,len(c)):]); f.append((c[-1]-ma)/ma)
    f.append(np.std(c[-20:])/np.mean(c[-20:])); f.append((h[-1]-l[-1])/c[-1])
    vma = np.mean(v[-20:]) if np.mean(v[-20:]) > 0 else 1; f.append(v[-1]/vma)
    f.append(oi[-1]/np.mean(oi[-20:]) if len(oi) >= 20 and np.mean(oi[-20:]) > 0 else 1)
    ema12 = c[-1]; ema26 = c[-1]
    for j in range(len(c)-2, -1, -1):
        ema12 = (2/13)*c[j] + (11/13)*ema12; ema26 = (2/27)*c[j] + (25/27)*ema26
    f.append((ema12-ema26)/c[-1])
    dd_ = np.diff(c[-15:]); g = dd_[dd_>0].sum() if len(dd_[dd_>0]) > 0 else 0
    lo = abs(dd_[dd_<0].sum()) if len(dd_[dd_<0]) > 0 else 1e-10
    f.append(100 - 100/(1+g/lo) if lo > 0 else 50)
    bb = np.std(c[-20:]); ma20 = np.mean(c[-20:]); f.append((c[-1]-ma20)/(2*bb+1e-10))
    try:
        m = int(str(df_local.iloc[idx]['date'])[5:7])
        f.append(np.sin(2*np.pi*m/12)); f.append(np.cos(2*np.pi*m/12))
    except:
        f.extend([0, 0])
    f.append(c[-1]/1000.0)
    return np.array(f, dtype=np.float32)

def run_wf(sm, rr_val, conf):
    n = len(daily); ts = int(n * 0.6)
    if ts < 200: return None
    
    pnl_rmb = []; wins = 0; total = 0
    for run in range(min(N_WF, (n - ts) // 10)):
        sp = ts + run * 10
        if sp + 10 > n: break
        td = daily.iloc[:sp]; t1 = sp; t2 = min(sp + 10, n)
        
        # Train
        X, y = [], []
        for i in range(win, len(td) - 1):
            f = build_f(td, i)
            if f is None: continue
            X.append(f); y.append(1 if td.iloc[i+1]['close'] > td.iloc[i]['close'] else 0)
        if len(X) < 100: continue
        ya = np.array(y)
        if len(np.unique(ya)) < 2: continue
        model = xgb.XGBClassifier(n_estimators=ne, max_depth=d, learning_rate=lr,
                                   use_label_encoder=False, eval_metric='logloss',
                                   verbosity=0, random_state=42)
        model.fit(np.array(X), ya)
        
        # Test — simulate intraday from OHLC
        pos = 0; direction = None; entry_px = 0; lots = 0
        for j in range(t1, t2):
            bar = daily.iloc[j]
            o = bar['open']; h = bar['high']; l = bar['low']; c = bar['close']
            
            av = [abs(daily.iloc[k]['high'] - daily.iloc[k]['low']) for k in range(max(0, j-20), j+1)]
            atr = np.mean(av) if av else o * 0.02; ap = atr / o
            if ap < 0.01: lev = 3.0
            elif ap < 0.02: lev = 2.0
            elif ap < 0.03: lev = 1.5
            elif ap < 0.05: lev = 0.5
            else: lev = 0
            
            cap_ratio = max(0.3, (CAP + sum(pnl_rmb)) / CAP)
            max_lots = max(1, int(lev * base * cap_ratio)) if lev > 0 else 0
            
            # Simulate intraday: high hits target first, or low hits stop first?
            if pos > 0:
                sd = sm * atr; td_val = rr_val * sd
                # Use OHLC to determine which hits first
                hit = False; trade_pnl = 0
                if direction == 'LONG':
                    # Which is further from open: high or low?
                    # If both hit, check which one was reached first (approximate by distance from open)
                    tgt_hit = h >= entry_px + td_val
                    stop_hit = l <= entry_px - sd
                    if tgt_hit and not stop_hit:
                        trade_pnl = td_val * mult * lots - 2 * cost * entry_px * mult * lots; hit = True
                    elif stop_hit and not tgt_hit:
                        trade_pnl = -sd * mult * lots - 2 * cost * entry_px * mult * lots; hit = True
                    elif tgt_hit and stop_hit:
                        # Both hit — which first? Distance from open
                        tgt_dist = entry_px + td_val - o
                        stop_dist = o - (entry_px - sd)
                        if tgt_dist <= stop_dist:
                            trade_pnl = td_val * mult * lots - 2 * cost * entry_px * mult * lots
                        else:
                            trade_pnl = -sd * mult * lots - 2 * cost * entry_px * mult * lots
                        hit = True
                    else:
                        # Neither hit — hold to close
                        trade_pnl = (c - entry_px) * mult * lots - 2 * cost * entry_px * mult * lots
                        hit = True
                else:  # SHORT
                    tgt_hit = l <= entry_px - td_val
                    stop_hit = h >= entry_px + sd
                    if tgt_hit and not stop_hit:
                        trade_pnl = td_val * mult * lots - 2 * cost * entry_px * mult * lots; hit = True
                    elif stop_hit and not tgt_hit:
                        trade_pnl = -sd * mult * lots - 2 * cost * entry_px * mult * lots; hit = True
                    elif tgt_hit and stop_hit:
                        tgt_dist = o - (entry_px - td_val)
                        stop_dist = entry_px + sd - o
                        if tgt_dist <= stop_dist:
                            trade_pnl = td_val * mult * lots - 2 * cost * entry_px * mult * lots
                        else:
                            trade_pnl = -sd * mult * lots - 2 * cost * entry_px * mult * lots
                        hit = True
                    else:
                        trade_pnl = (entry_px - c) * mult * lots - 2 * cost * entry_px * mult * lots
                        hit = True
                
                if hit:
                    pnl_rmb.append(trade_pnl)
                    wins += (1 if trade_pnl > 0 else 0); total += 1
                    pos = 0
            
            # New entry (1 per day)
            if pos == 0 and max_lots > 0:
                f = build_f(daily, j)
                if f is not None:
                    prob = model.predict_proba(f.reshape(1, -1))[0]
                    c2 = prob[1] if prob[1] > 0.5 else 1 - prob[1]
                    if c2 >= conf:
                        entry_px = o; direction = 'LONG' if prob[1] > 0.5 else 'SHORT'
                        lots = max_lots; pos = lots
    
    if total == 0: return None
    tp = sum(pnl_rmb)
    wr = wins / total * 100
    cum = np.cumsum(pnl_rmb); pk = np.maximum.accumulate(cum)
    dd = np.max((pk - cum) / (pk + 1e-10)) * 100
    gp = sum(p for p in pnl_rmb if p > 0); gl = abs(sum(p for p in pnl_rmb if p < 0))
    pf = gp / gl if gl > 0 else 999
    test_days = n - ts; yrs = test_days / 252
    ar = ((1 + tp / CAP) ** (1 / yrs) - 1) * 100 if yrs > 0 and tp > -CAP else tp / CAP / yrs * 100
    return {'trades': total, 'wr': round(wr, 1), 'pf': round(pf, 2), 'pnl': round(tp, 0),
            'ar': round(ar, 1), 'dd': round(dd, 1), 'sm': sm, 'rr': rr_val, 'conf': conf}

results = []
total_combos = len(SM) * len(RR) * len(CF)
print(f'Approach 1: Daily + Intraday OHLC, 500 WF, {total_combos} combos')
print(f'{"Stop":<7} {"RR":<5} {"Conf":<6} {"Trd":<6} {"WR":<6} {"PF":<6} {"PnL":<9} {"Ann":<8} {"DD":<7}')

for sm in SM:
    for rr_val in RR:
        for conf in CF:
            t0 = time.time()
            r = run_wf(sm, rr_val, conf)
            if r:
                results.append(r)
                elapsed = time.time() - t0
                label = '  <--' if r['ar'] > 10 else ''
                print(f'{sm:<7.1f} {rr_val:<5.1f} {conf:<6.2f} {r["trades"]:<6} {r["wr"]:<5.1f}% {r["pf"]:<6.2f} {r["pnl"]/10000:<+8.1f}万 {r["ar"]:<+7.1f}% {r["dd"]:<6.1f}%{label}  {elapsed:.0f}s')

results.sort(key=lambda x: -x['pnl'])
with open('/home/a/prophet_futures/prophet_futures/approach1_daily.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)

print('\n=== TOP 5 ===')
for r in results[:5]:
    print(f'  stop={r["sm"]:.1f}xATR RR={r["rr"]:.1f} conf={r["conf"]:.2f} '
          f'{r["trades"]}t WR={r["wr"]}% PnL={r["pnl"]/10000:+.1f}万 Ann={r["ar"]:+.1f}%/yr DD={r["dd"]}% PF={r["pf"]}')
print('Saved: approach1_daily.json')
