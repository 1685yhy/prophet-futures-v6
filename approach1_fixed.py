#!/usr/bin/env python3
"""Approach 1 FIXED: 500 WF + OHLC validation + correct cost/DD"""
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

results = []
print(f'Approach 1 FIXED: 500 WF + OHLC, {len(SM)*len(RR)*len(CF)} combos')
print(f'{"Stop":<7} {"RR":<5} {"Conf":<6} {"Trd":<6} {"WR":<6} {"PF":<6} {"PnL":<9} {"Ann":<8} {"DD":<7}')

for sm in SM:
    for rr_val in RR:
        for conf in CF:
            t0 = time.time()
            n = len(daily); ts = int(n * 0.6)
            if ts < 200: continue
            pnl_rmb = []; wins = 0; total = 0
            current_capital = CAP
            
            for run in range(min(N_WF, (n - ts) // 10)):
                sp = ts + run * 10
                if sp + 10 > n: break
                td = daily.iloc[:sp]; t1 = sp; t2 = min(sp + 10, n)
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
                
                for j in range(t1, t2 - 1):
                    bar = daily.iloc[j]; o = bar['open']; h = bar['high']; l = bar['low']
                    av = [abs(daily.iloc[k]['high'] - daily.iloc[k]['low']) for k in range(max(0, j-20), j+1)]
                    atr = np.mean(av) if av else o * 0.02; ap = atr / o
                    if ap < 0.01: lev = 3.0
                    elif ap < 0.02: lev = 2.0
                    elif ap < 0.03: lev = 1.5
                    elif ap < 0.05: lev = 0.5
                    else: continue
                    
                    cap_ratio = max(0.3, current_capital / CAP)
                    lots = max(1, int(lev * base * cap_ratio))
                    
                    f = build_f(daily, j)
                    if f is not None:
                        prob = model.predict_proba(f.reshape(1, -1))[0]
                        c2 = prob[1] if prob[1] > 0.5 else 1 - prob[1]
                        if c2 >= conf:
                            entry_px = o
                            dr = 'LONG' if prob[1] > 0.5 else 'SHORT'
                            sd = sm * atr; td_val = rr_val * sd
                            
                            # Simulate: check next bar's OHLC
                            nxt = daily.iloc[j+1]
                            nh = nxt['high']; nl = nxt['low']; nc = nxt['close']
                            
                            # Check target/stop — which hits first?
                            win_pnl = 0; loss_pnl = 0; hit = False
                            if dr == 'LONG':
                                tgt_px = entry_px + td_val; stp_px = entry_px - sd
                                if nh >= tgt_px and nl > stp_px:
                                    # Only target hit
                                    win_pnl = td_val * mult * lots - 2*cost*entry_px*mult*lots; hit = True
                                elif nl <= stp_px and nh < tgt_px:
                                    # Only stop hit
                                    loss_pnl = -sd * mult * lots - 2*cost*entry_px*mult*lots; hit = True
                                elif nh >= tgt_px and nl <= stp_px:
                                    # Both hit — check distance
                                    tgt_dist = tgt_px - o; stp_dist = o - stp_px
                                    if tgt_dist <= stp_dist:
                                        win_pnl = td_val * mult * lots - 2*cost*entry_px*mult*lots
                                    else:
                                        loss_pnl = -sd * mult * lots - 2*cost*entry_px*mult*lots
                                    hit = True
                                else:
                                    # Neither — hold to close
                                    chg = nc - entry_px
                                    trade_pnl = chg * mult * lots - 2*cost*entry_px*mult*lots
                                    pnl_rmb.append(trade_pnl)
                                    wins += (1 if trade_pnl > 0 else 0); total += 1
                                    hit = True
                            else:  # SHORT
                                tgt_px = entry_px - td_val; stp_px = entry_px + sd
                                if nl <= tgt_px and nh < stp_px:
                                    win_pnl = td_val * mult * lots - 2*cost*entry_px*mult*lots; hit = True
                                elif nh >= stp_px and nl > tgt_px:
                                    loss_pnl = -sd * mult * lots - 2*cost*entry_px*mult*lots; hit = True
                                elif nl <= tgt_px and nh >= stp_px:
                                    tgt_dist = o - tgt_px; stp_dist = stp_px - o
                                    if tgt_dist <= stp_dist:
                                        win_pnl = td_val * mult * lots - 2*cost*entry_px*mult*lots
                                    else:
                                        loss_pnl = -sd * mult * lots - 2*cost*entry_px*mult*lots
                                    hit = True
                                else:
                                    chg = entry_px - nc
                                    trade_pnl = chg * mult * lots - 2*cost*entry_px*mult*lots
                                    pnl_rmb.append(trade_pnl)
                                    wins += (1 if trade_pnl > 0 else 0); total += 1
                                    hit = True
                            
                            if hit:
                                trade_pnl = win_pnl if win_pnl != 0 else loss_pnl
                                pnl_rmb.append(trade_pnl)
                                current_capital += trade_pnl
                                wins += (1 if trade_pnl > 0 else 0); total += 1
            
            if total == 0: continue
            tp = sum(pnl_rmb); wr = wins / total * 100
            cum = np.cumsum(pnl_rmb)
            if len(cum) > 0:
                pk = np.maximum.accumulate(cum)
                dd_val = np.max((pk - cum) / (pk + 1e-10)) * 100
            else: dd_val = 0
            gp = sum(p for p in pnl_rmb if p > 0); gl = abs(sum(p for p in pnl_rmb if p < 0))
            pf = gp / gl if gl > 0 else 999
            yrs = (n - ts) / 252
            ar = ((1 + tp / CAP) ** (1 / yrs) - 1) * 100 if yrs > 0 and tp > -CAP else tp / CAP / yrs * 100
            
            results.append({'sm': sm, 'rr': rr_val, 'conf': conf, 'trades': total,
                          'wr': round(wr, 1), 'pf': round(pf, 2), 'pnl': round(tp, 0),
                          'ar': round(ar, 1), 'dd': round(dd_val, 1)})
            label = ' <--' if tp > 0 else ''
            print(f'{sm:<7.1f} {rr_val:<5} {conf:<6.2f} {total:<6} {wr:<5.1f}% {pf:<6.2f} {tp/10000:<+8.1f}万 {ar:<+7.1f}% {dd_val:<6.1f}%{label}  {time.time()-t0:.0f}s')

results.sort(key=lambda x: -x['pnl'])
with open('/home/a/prophet_futures/prophet_futures/approach1_fixed.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)

print(f'\n=== APPROACH 1 FIXED ===')
for r in results[:5]:
    print(f'  stop={r["sm"]:.1f}xATR RR={r["rr"]} conf={r["conf"]:.2f} {r["trades"]}t WR={r["wr"]}% PF={r["pf"]} PnL={r["pnl"]/10000:+.1f}万 Ann={r["ar"]:+.1f}%/yr DD={r["dd"]}%')
