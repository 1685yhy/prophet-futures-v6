#!/usr/bin/env python3
"""Approach 1 v3: Daily + OHLC, 500 WF, 1 entry/day, correct DD"""
import numpy as np, pandas as pd, json, time
import akshare as ak, xgboost as xgb

daily = ak.futures_main_sina(symbol='LH0', start_date='20210101', end_date='20260620')
daily.columns = ['date','open','high','low','close','volume','oi','settle']
for x in ['open','high','low','close','volume','oi']:
    daily[x] = pd.to_numeric(daily[x], errors='coerce')
daily = daily.dropna(subset=['close']).reset_index(drop=True)

N_WF = 500; CAP = 300000; win = 60; ne = 200; d = 5; lr = 0.05
cost = 0.0006; mult = 16; base = 3
SM = [1.5, 2.0, 2.5]; RR = [3, 4, 5]; CF = [0.50, 0.52]

def build_f(df2, idx):
    if idx < win + 5: return None
    w = df2.iloc[idx-win:idx+1]; c = w['close'].values; o = w['open'].values
    h = w['high'].values; l = w['low'].values; v = w['volume'].values; oi = w['oi'].values
    f = []
    if idx >= 1: f.append((o[-1]-c[-2])/c[-2]); f.append(abs(f[-1]))
    else: f.extend([0, 0])
    for lag in [1,3,5,10,20]: f.append((c[-1]-c[-lag-1])/c[-lag-1] if len(c) > lag else 0)
    for p in [5,10,20,60]: ma = np.mean(c[-min(p,len(c)):]); f.append((c[-1]-ma)/ma)
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
    try: m = int(str(df2.iloc[idx]['date'])[5:7]); f.append(np.sin(2*np.pi*m/12)); f.append(np.cos(2*np.pi*m/12))
    except: f.extend([0, 0])
    f.append(c[-1]/1000.0)
    return np.array(f, dtype=np.float32)

total_combos = len(SM)*len(RR)*len(CF)
print(f'A1 v3: Daily+OHLC 500WF {total_combos} combos, 1 trade/day, correct DD')
print(f'{"S":<5} {"RR":<4} {"Conf":<5} {"Trd":<6} {"WR":<6} {"PF":<6} {"PnL":<8} {"Ann":<8} {"DD":<6} {"Time"}')

results = []
for sm in SM:
    for rr_val in RR:
        for conf in CF:
            t0 = time.time()
            n = len(daily); ts = int(n * 0.6)
            pnl_seq = []; wins = 0; total = 0
            
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
                
                # Test with OHLC simulation, max 1 trade per day
                for j in range(t1, t2 - 1):
                    bar = daily.iloc[j]; o = bar['open']; h = bar['high']; l = bar['low']
                    nxt = daily.iloc[j+1]; nh = nxt['high']; nl = nxt['low']; nc = nxt['close']
                    
                    av = [abs(daily.iloc[k]['high'] - daily.iloc[k]['low']) for k in range(max(0, j-20), j+1)]
                    atr = np.mean(av) if av else o * 0.02; ap = atr / o
                    if ap < 0.01: lev = 3.0
                    elif ap < 0.02: lev = 2.0
                    elif ap < 0.03: lev = 1.5
                    elif ap < 0.05: lev = 0.5
                    else: continue
                    
                    f = build_f(daily, j)
                    if f is None: continue
                    prob = model.predict_proba(f.reshape(1, -1))[0]
                    c2 = prob[1] if prob[1] > 0.5 else 1 - prob[1]
                    if c2 < conf: continue
                    
                    dr = 'LONG' if prob[1] > 0.5 else 'SHORT'
                    lots = max(1, int(lev * base))
                    sd = sm * atr; td_val = rr_val * sd
                    
                    # Simulate: next bar OHLC — which hits first?
                    trade_pnl = 0
                    if dr == 'LONG':
                        tgt_px = o + td_val; stp_px = o - sd
                        if nh >= tgt_px and nl > stp_px:
                            trade_pnl = td_val * mult * lots - 2 * cost * o * mult * lots  # Only target
                        elif nl <= stp_px and nh < tgt_px:
                            trade_pnl = -sd * mult * lots - 2 * cost * o * mult * lots  # Only stop
                        elif nh >= tgt_px and nl <= stp_px:
                            # Both hit: distance from open determines first
                            if (tgt_px - o) <= (o - stp_px):
                                trade_pnl = td_val * mult * lots - 2 * cost * o * mult * lots
                            else:
                                trade_pnl = -sd * mult * lots - 2 * cost * o * mult * lots
                        else:
                            chg = nc - o
                            trade_pnl = chg * mult * lots - 2 * cost * o * mult * lots
                    else:  # SHORT
                        tgt_px = o - td_val; stp_px = o + sd
                        if nl <= tgt_px and nh < stp_px:
                            trade_pnl = td_val * mult * lots - 2 * cost * o * mult * lots
                        elif nh >= stp_px and nl > tgt_px:
                            trade_pnl = -sd * mult * lots - 2 * cost * o * mult * lots
                        elif nl <= tgt_px and nh >= stp_px:
                            if (o - tgt_px) <= (stp_px - o):
                                trade_pnl = td_val * mult * lots - 2 * cost * o * mult * lots
                            else:
                                trade_pnl = -sd * mult * lots - 2 * cost * o * mult * lots
                        else:
                            chg = o - nc
                            trade_pnl = chg * mult * lots - 2 * cost * o * mult * lots
                    
                    pnl_seq.append(trade_pnl)
                    wins += (1 if trade_pnl > 0 else 0); total += 1
            
            if total == 0: continue
            tp = sum(pnl_seq); wr = wins / total * 100
            cum = np.cumsum(pnl_seq); pk = np.maximum.accumulate(cum)
            dd_val = np.max((pk - cum) / (pk + 1e-10)) * 100 if pk[-1] > 0 else 0
            gp = sum(p for p in pnl_seq if p > 0); gl = abs(sum(p for p in pnl_seq if p < 0))
            pf = gp / gl if gl > 0 else 999
            yrs = (n - ts) / 252
            if tp > -CAP and yrs > 0:
                ar = ((CAP + tp) / CAP) ** (1 / yrs) - 1
            else: ar = tp / CAP / yrs if yrs > 0 else -1
            
            results.append({'sm': sm, 'rr': rr_val, 'conf': conf, 'trades': total,
                          'wr': round(wr, 1), 'pf': round(pf, 2), 'pnl': round(tp, 0),
                          'ar': round(ar*100, 1), 'dd': round(dd_val, 1)})
            label = ' <-- +' if tp > 0 else ''
            print(f'{sm:<5.1f} {rr_val:<4} {conf:<5.2f} {total:<6} {wr:<5.1f}% {pf:<6.2f} {tp/10000:<+7.1f}万 {ar*100:<+7.1f}% {dd_val:<5.1f}%{label}  {time.time()-t0:.0f}s')

results.sort(key=lambda x: -x['pnl'])
with open('/home/a/prophet_futures/prophet_futures/approach1_v3.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)

print('\n=== TOP 3 ===')
for r in results[:3]:
    print(f'  stop={r["sm"]:.1f}xATR RR={r["rr"]} conf={r["conf"]:.2f} {r["trades"]}t WR={r["wr"]}% PnL={r["pnl"]/10000:+.1f}万 DD={r["dd"]}%')
