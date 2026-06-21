#!/usr/bin/env python3
"""Final validation: 2000WF, fine Struct + RR, all 3 symbols, vol sizing"""
import numpy as np, pandas as pd, json, time
import akshare as ak, xgboost as xgb

SYM = {
    'LH': ('LH0', 0.0006, False, 200, 5, 0.05, 60, 16, 3),
    'JM': ('JM0', 0.0011, False, 100, 4, 0.03, 60, 60, 5),
    'RM': ('RM0', 0.0011, True,  100, 5, 0.03, 120, 10, 20),
}
N_WF = 2000; CAP = 300000
SN = [22, 24, 25, 26, 28]
RRs = [3.5, 4.0, 4.5]

def precompute(df, win=60):
    n = len(df); F = np.zeros((n, 20), dtype=np.float32); A = np.zeros(n); V = np.zeros(n, dtype=bool)
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
        bb = np.std(c[-20:]); ma20 = np.mean(c[-20:]); f[17] = (c[-1]-ma20)/(2*bb+1e-10)
        try: m = int(str(df.iloc[idx]['date'])[5:7]); f[18] = np.sin(2*np.pi*m/12); f[19] = np.cos(2*np.pi*m/12)
        except: pass
        V[idx] = True
        A[idx] = np.mean([abs(df.iloc[k]['high']-df.iloc[k]['low']) for k in range(max(0,idx-20),idx+1)])
    L = np.zeros(n, dtype=int)
    for i in range(n-1):
        if V[i]: L[i] = 1 if df.iloc[i+1]['close'] > df.iloc[i]['close'] else 0
    return F, A, V, L

all_results = []
for sk, (sc, cost, rev, ne, d, lr, win, mult, base) in SYM.items():
    print(f'\n===== {sk} =====')
    df = ak.futures_main_sina(symbol=sc, start_date='20210101', end_date='20260620')
    df.columns = ['date','open','high','low','close','volume','oi','settle']
    for x in ['open','high','low','close','volume','oi']: df[x] = pd.to_numeric(df[x], errors='coerce')
    df = df.dropna(subset=['close']).reset_index(drop=True)
    F, A, V, L = precompute(df, win)
    n = len(df); ts = int(n * 0.7)
    
    for sn in SN:
        for rr_val in RRs:
            t0 = time.time(); pnl_seq = []; wins = 0; total = 0
            cap = CAP; peak = CAP; max_dd = 0; noise = 0
            
            for run in range(min(N_WF, (n-ts)//10)):
                sp = ts + run * 10
                if sp + 10 > n: break
                t1 = sp; t2 = min(sp + 10, n)
                tr = [i for i in range(win, t1) if V[i]]
                if len(tr) < 100: continue
                X_tr = F[tr]; y_tr = L[tr]
                if len(np.unique(y_tr)) < 2: continue
                m = xgb.XGBClassifier(n_estimators=ne, max_depth=d, learning_rate=lr, verbosity=0, random_state=42)
                m.fit(X_tr, y_tr)
                
                for j in range(t1, t2-1):
                    if not V[j]: continue
                    p = m.predict_proba(F[j].reshape(1, -1))[0]
                    pred = 1 if p[1] > 0.5 else 0
                    if rev: pred = 1 - pred
                    entry = df.iloc[j]['close']; atr = A[j]; ap = atr / entry
                    if ap < 0.01: lev = 3.0
                    elif ap < 0.02: lev = 2.0
                    elif ap < 0.03: lev = 1.5
                    elif ap < 0.05: lev = 0.5
                    else: continue
                    lots = max(1, int(lev * base))
                    
                    if pred == 1:
                        stp = min([df.iloc[k]['low'] for k in range(max(0, j-sn), j+1)])
                    else:
                        stp = max([df.iloc[k]['high'] for k in range(max(0, j-sn), j+1)])
                    tgt = entry + rr_val * abs(stp - entry) if pred == 1 else entry - rr_val * abs(stp - entry)
                    
                    trade_pnl = 0; stopped = False; noisy = False
                    for k in range(j+1, min(j+16, n)):
                        ph = df.iloc[k]['high']; pl = df.iloc[k]['low']; pc = df.iloc[k]['close']
                        hit = False
                        if pred == 1:
                            if ph >= tgt: trade_pnl = (tgt-entry)*mult*lots - 2*cost*entry*mult*lots; hit = True
                            elif pl <= stp: trade_pnl = (stp-entry)*mult*lots - 2*cost*entry*mult*lots; stopped = True; hit = True
                        else:
                            if pl <= tgt: trade_pnl = (entry-tgt)*mult*lots - 2*cost*entry*mult*lots; hit = True
                            elif ph >= stp: trade_pnl = (entry-stp)*mult*lots - 2*cost*entry*mult*lots; stopped = True; hit = True
                        if hit:
                            if stopped and ((pred == 1 and pc > entry) or (pred == 0 and pc < entry)): noisy = True
                            break
                    else:
                        lc = df.iloc[min(j+15, n-1)]['close']
                        trade_pnl = (lc-entry)*mult*lots - 2*cost*entry*mult*lots if pred == 1 else (entry-lc)*mult*lots - 2*cost*entry*mult*lots
                    
                    pnl_seq.append(trade_pnl); cap += trade_pnl
                    if cap > peak: peak = cap
                    dd_pct = (peak - cap) / peak * 100
                    if dd_pct > max_dd: max_dd = dd_pct
                    wins += (1 if trade_pnl > 0 else 0); total += 1
                    if noisy: noise += 1
            
            if total == 0: continue
            tp = sum(pnl_seq); wr = wins / total * 100
            gp = sum(p for p in pnl_seq if p > 0); gl = abs(sum(p for p in pnl_seq if p < 0))
            pf = gp / gl if gl > 0 else 999
            yrs = (n - ts) / 252
            ar = ((CAP + tp) / CAP) ** (1 / yrs) - 1 if tp > -CAP and yrs > 0 else tp / CAP / yrs
            cm = ar * 100 / max_dd if max_dd > 0 else 0
            npct = noise / total * 100
            elapsed = time.time() - t0
            tag = ''
            if tp > 0: tag = ' +'
            print(f'  N={sn} RR={rr_val} {total}t WR={wr:.1f}% PF={pf:.2f} PnL={tp/10000:+.1f}万 AR={ar*100:.1f}% DD={max_dd:.0f}% CM={cm:.2f} ns={npct:.1f}%{tag} {elapsed:.0f}s')
            all_results.append({'symbol': sk, 'sn': sn, 'rr': rr_val, 'trades': total,
                              'wr': round(wr,1), 'pf': round(pf,2), 'pnl': round(tp,0),
                              'ar': round(ar*100,1), 'dd': round(max_dd,1), 'cm': round(cm,2), 'ns': round(npct,1)})

# Per symbol best
print('\n===== BEST PER SYMBOL (by Calmar) =====')
for sk in ['LH', 'JM', 'RM']:
    sym = [r for r in all_results if r['symbol'] == sk and r['dd'] < 200 and r['ns'] < 10]
    sym.sort(key=lambda x: -x['cm'])
    if sym:
        r = sym[0]
        print(f'{sk}: N={r["sn"]} RR={r["rr"]} {r["trades"]}t WR={r["wr"]}% PnL={r["pnl"]/10000:+.1f}万 AR={r["ar"]}% DD={r["dd"]}% CM={r["cm"]}')

with open('/home/a/prophet_futures/prophet_futures/final_validation.json', 'w') as f:
    json.dump(all_results, f, indent=2, default=str)
print('\nSaved: final_validation.json')
