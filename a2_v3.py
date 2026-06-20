#!/usr/bin/env python3
"""Approach 2 v3: 30-min WF, proper sliding windows, full run"""
import numpy as np, pandas as pd, json, time
import akshare as ak, xgboost as xgb

m30 = ak.futures_zh_minute_sina(symbol='LH2609', period='30')
m30['datetime'] = pd.to_datetime(m30['datetime'])
m30 = m30.sort_values('datetime').reset_index(drop=True)

CAP = 300000; mult = 16; base = 3; cost = 0.0006
ne = 200; d = 5; lr = 0.05
N_WF = 30
SM = [1.5, 2.0, 2.5]; RR = [3.0, 4.0, 5.0]; CF = [0.50, 0.52]

def build_f(df_local, idx, lookback=13):
    if idx < lookback + 5: return None
    w = df_local.iloc[idx-lookback:idx+1]
    c = w['close'].values; h = w['high'].values; l = w['low'].values; v = w['volume'].values
    f = []
    for lag in [1, 3, 5, 13]:
        f.append((c[-1] - c[-lag-1]) / c[-lag-1] if len(c) > lag else 0)
    for p in [3, 5, 13]:
        ma = np.mean(c[-min(p,len(c)):]); f.append((c[-1] - ma) / ma)
    f.append(np.std(c[-5:]) / np.mean(c[-5:]) if np.mean(c[-5:]) > 0 else 0)
    f.append((h[-1] - l[-1]) / c[-1])
    vma = np.mean(v[-5:]) if np.mean(v[-5:]) > 0 else 1; f.append(v[-1] / vma)
    trend = np.polyfit(range(len(c[-13:])), c[-13:], 1)[0]; f.append(trend / c[-1])
    dd_ = np.diff(c[-8:]); gain = dd_[dd_>0].sum() if len(dd_[dd_>0]) > 0 else 0
    loss = abs(dd_[dd_<0].sum()) if len(dd_[dd_<0]) > 0 else 1e-10
    f.append(100 - 100/(1+gain/loss))
    bb_std = np.std(c[-13:]); bb_ma = np.mean(c[-13:])
    f.append((c[-1]-bb_ma)/(2*bb_std+1e-10))
    hour = df_local.iloc[idx]['datetime'].hour; minute = df_local.iloc[idx]['datetime'].minute
    tod = (hour * 60 + minute) / (24 * 60)
    f.append(np.sin(2 * np.pi * tod)); f.append(np.cos(2 * np.pi * tod))
    f.append(c[-1] / 1000.0)
    return np.array(f, dtype=np.float32)

total_combos = len(SM) * len(RR) * len(CF)
total_bars = len(m30)
train_pct = 0.7
print(f'A2 v3: 30-min WF, {total_bars} bars, {N_WF} windows, {total_combos} combos')
print(f'Each window: train ~{int(total_bars*train_pct)} bars, test ~{int(total_bars*(1-train_pct)/N_WF)} bars')

results = []
for sm in SM:
    for rr_val in RR:
        for conf in CF:
            t0 = time.time()
            all_pnl = []; all_wins = 0; all_total = 0
            
            train_size = int(total_bars * train_pct)
            step = max(1, (total_bars - train_size) // N_WF)
            
            for wf in range(min(N_WF, (total_bars - train_size) // step)):
                sp = train_size + wf * step
                ep = min(sp + step, total_bars)
                if sp >= total_bars: break
                
                train_df = m30.iloc[:sp]
                test_df = m30.iloc[sp:ep]
                if len(test_df) < 5: continue
                
                # Build training samples
                X, y = [], []
                for i in range(13, len(train_df) - 3):
                    f = build_f(train_df, i)
                    if f is None: continue
                    X.append(f)
                    future = train_df.iloc[min(i+3, len(train_df)-1)]['close']
                    y.append(1 if future > train_df.iloc[i]['close'] else 0)
                if len(X) < 50: continue
                
                ya = np.array(y)
                if len(np.unique(ya)) < 2: continue
                
                model = xgb.XGBClassifier(n_estimators=ne, max_depth=d, learning_rate=lr,
                                           use_label_encoder=False, eval_metric='logloss',
                                           verbosity=0, random_state=42)
                model.fit(np.array(X), ya)
                
                # Test — re-entry allowed (match simnow_live.py)
                capital = CAP; positions = []
                for i in range(len(test_df)):
                    bar = test_df.iloc[i]; o = bar['open']; h = bar['high']; l = bar['low']
                    start_idx = max(0, i - 13)
                    recent = [abs(test_df.iloc[k]['high'] - test_df.iloc[k]['low']) for k in range(start_idx, i+1)]
                    atr = np.mean(recent) if recent else o * 0.005
                    ap = atr / o
                    if ap < 0.005: lev = 3.0
                    elif ap < 0.01: lev = 2.0
                    elif ap < 0.015: lev = 1.5
                    elif ap < 0.03: lev = 0.5
                    else: lev = 0
                    cap_ratio = max(0.3, capital / CAP)
                    max_lots = max(1, int(lev * base * cap_ratio)) if lev > 0 else 0
                    
                    # Exit existing positions
                    new_positions = []
                    for ep2, dr2, lots2 in positions:
                        sd_val = sm * atr; td_val = rr_val * sd_val
                        hit = False; pnl2 = 0
                        if dr2 == 'LONG':
                            if h >= ep2 + td_val: pnl2 = td_val*mult*lots2 - 2*cost*ep2*mult*lots2; hit = True
                            elif l <= ep2 - sd_val: pnl2 = -sd_val*mult*lots2 - 2*cost*ep2*mult*lots2; hit = True
                        else:
                            if l <= ep2 - td_val: pnl2 = td_val*mult*lots2 - 2*cost*ep2*mult*lots2; hit = True
                            elif h >= ep2 + sd_val: pnl2 = -sd_val*mult*lots2 - 2*cost*ep2*mult*lots2; hit = True
                        if hit:
                            capital += pnl2; all_total += 1
                            if pnl2 > 0: all_wins += 1
                        else:
                            new_positions.append((ep2, dr2, lots2))
                    positions = new_positions
                    
                    # New entry (re-entry allowed)
                    if max_lots > 0 and i < len(test_df) - 3:
                        f = build_f(test_df, i)
                        if f is not None:
                            prob = model.predict_proba(f.reshape(1, -1))[0]
                            c2 = prob[1] if prob[1] > 0.5 else 1 - prob[1]
                            if c2 >= conf:
                                dr = 'LONG' if prob[1] > 0.5 else 'SHORT'
                                positions.append((o, dr, max_lots))
                
                all_pnl.append(capital - CAP)
            
            if all_total == 0: continue
            total_pnl = sum(all_pnl)
            wr = all_wins / all_total * 100
            cum = np.cumsum(all_pnl); pk = np.maximum.accumulate(cum)
            dd_val = np.max((pk - cum) / (pk + 1e-10)) * 100 if pk[-1] > 0 else 0
            gp = sum(p for p in all_pnl if p > 0); gl = abs(sum(p for p in all_pnl if p < 0))
            pf = gp / gl if gl > 0 else 999
            
            total_days = (m30['datetime'].max() - m30['datetime'].min()).days / 365.0
            if total_days > 0 and total_pnl > -CAP * 0.9:
                ann = ((CAP + total_pnl) / CAP / N_WF) ** (1/total_days) - 1
            else: ann = -1
            
            results.append({'sm': sm, 'rr': rr_val, 'conf': conf, 'trades': all_total,
                          'wr': round(wr, 1), 'pf': round(pf, 2), 'pnl': round(total_pnl, 0),
                          'ar': round(ann*100, 1), 'dd': round(dd_val, 1)})
            label = ' <-- +' if total_pnl > 0 else ''
            print(f'{sm:.1f}x {rr_val} {conf:.2f} {all_total}t WR={wr:.1f}% PF={pf:.2f} PnL={total_pnl/10000:+.1f}万 DD={dd_val:.1f}%{label}  {time.time()-t0:.0f}s')

results.sort(key=lambda x: -x['pnl'])
with open('/home/a/prophet_futures/prophet_futures/approach2_wf.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)

print('\n=== TOP 3 ===')
for r in results[:3]:
    print(f'  stop={r["sm"]:.1f}xATR RR={r["rr"]} conf={r["conf"]:.2f} {r["trades"]}t WR={r["wr"]}% PnL={r["pnl"]/10000:+.1f}万 DD={r["dd"]}%')
