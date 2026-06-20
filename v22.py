#!/usr/bin/env python3
"""v22: v17 base + ATR dynamic stop/target optimization, 500WF, 5yr data"""
import numpy as np, pandas as pd, json, time
import akshare as ak, xgboost as xgb

# ====== DATA ======
SYMBOLS = {
    'LH': ('LH0', 0.0006, False, 200, 5, 0.05, 60, 16, 3),
    'JM': ('JM0', 0.0011, False, 100, 4, 0.03, 60, 60, 5),
    'RM': ('RM0', 0.0011, True,  100, 5, 0.03, 120, 10, 20),
}
N_WF = 500; CAP = 300000

# Grid: stop = sm * ATR, target = rr * stop
SM_LIST = [0.5, 0.7, 1.0, 1.3, 1.5, 2.0]
RR_LIST = [2.0, 2.5, 3.0, 3.5, 4.0, 5.0]

print(f'V22: v17 base + ATR stop/target, {len(SM_LIST)*len(RR_LIST)} combos per symbol, {N_WF}WF')
print(f'Symbols: LH JM RM  Data: 5yr+  Cost: real')

def fetch(c):
    e = pd.Timestamp.now(); s = e - pd.Timedelta(days=2500)
    df = ak.futures_main_sina(symbol=c, start_date=s.strftime('%Y%m%d'), end_date=e.strftime('%Y%m%d'))
    df.columns = ['date','open','high','low','close','volume','oi','settle']
    for x in ['open','high','low','close','volume','oi']:
        df[x] = pd.to_numeric(df[x], errors='coerce')
    return df.dropna(subset=['close']).reset_index(drop=True)

def precompute(df, win=60):
    """Precompute features once"""
    n = len(df)
    feats = np.zeros((n, 20), dtype=np.float32)
    valid = np.zeros(n, dtype=bool)
    for idx in range(win+5, n):
        w = df.iloc[idx-win:idx+1]
        c = w['close'].values; o = w['open'].values
        h = w['high'].values; l = w['low'].values
        v = w['volume'].values; oi = w['oi'].values
        f = feats[idx]
        if idx >= 1: f[0] = (o[-1]-c[-2])/c[-2]; f[1] = abs(f[0])
        for li, lag in enumerate([1,3,5,10,20], 2):
            f[li] = (c[-1]-c[-lag-1])/c[-lag-1] if len(c) > lag else 0
        for pi, p in enumerate([5,10,20,60], 7):
            ma = np.mean(c[-min(p, len(c)):]); f[pi] = (c[-1]-ma)/ma
        f[11] = np.std(c[-20:])/np.mean(c[-20:])
        f[12] = (h[-1]-l[-1])/c[-1]
        vma = np.mean(v[-20:]) if np.mean(v[-20:]) > 0 else 1; f[13] = v[-1]/vma
        f[14] = oi[-1]/np.mean(oi[-20:]) if len(oi) >= 20 and np.mean(oi[-20:]) > 0 else 1
        ema12 = c[-1]; ema26 = c[-1]
        for j in range(len(c)-2, -1, -1):
            ema12 = (2/13)*c[j] + (11/13)*ema12; ema26 = (2/27)*c[j] + (25/27)*ema26
        f[15] = (ema12-ema26)/c[-1]
        dd_ = np.diff(c[-15:]); g = dd_[dd_>0].sum() if len(dd_[dd_>0]) > 0 else 0
        lo = abs(dd_[dd_<0].sum()) if len(dd_[dd_<0]) > 0 else 1e-10
        f[16] = 100 - 100/(1+g/lo) if lo > 0 else 50
        bb = np.std(c[-20:]); ma20 = np.mean(c[-20:])
        f[17] = (c[-1]-ma20)/(2*bb+1e-10)
        try:
            m = int(str(df.iloc[idx]['date'])[5:7])
            f[18] = np.sin(2*np.pi*m/12); f[19] = np.cos(2*np.pi*m/12)
        except:
            pass
        f[19] = c[-1]/1000.0  # overwrite cos if error, but we have 19 features
        valid[idx] = True
    # Labels
    labels = np.zeros(n, dtype=int)
    for i in range(n-1):
        if valid[i]:
            labels[i] = 1 if df.iloc[i+1]['close'] > df.iloc[i]['close'] else 0
    return feats, valid, labels

all_results = {}
for sk, (sc, cost, rev, ne, d, lr, win, mult, base) in SYMBOLS.items():
    print(f'\n===== {sk} (n={ne} d={d} lr={lr} w={win} base={base}) =====')
    df = fetch(sc)
    if df is None: continue
    feats, valid, labels = precompute(df, win)
    n = len(df)
    print(f'  Data: {n} days  Features: {valid.sum()} valid')
    
    sym_results = []
    for sm in SM_LIST:
        for rr_val in RR_LIST:
            t0 = time.time()
            ts = int(n * 0.6)
            if ts < 200: continue
            
            pnl_rmb = []; wins = 0; total = 0
            capital = CAP
            
            for run_idx in range(min(N_WF, (n - ts) // 10)):
                sp = ts + run_idx * 10
                if sp + 10 > n: break
                t1 = sp; t2 = min(sp + 10, n)
                
                # Train indices
                tr_idx = [i for i in range(win, t1) if valid[i] and i < t1]
                if len(tr_idx) < 100: continue
                X_tr = feats[tr_idx]; y_tr = labels[tr_idx]
                if len(np.unique(y_tr)) < 2: continue
                
                model = xgb.XGBClassifier(n_estimators=ne, max_depth=d, learning_rate=lr,
                                           device='cuda', verbosity=0, random_state=42)
                model.fit(X_tr, y_tr)
                
                # Test with ATR stop/target
                for j in range(t1, t2 - 1):
                    if not valid[j]: continue
                    bar = df.iloc[j]; o = bar['open']; h = bar['high']; l = bar['low']
                    nxt = df.iloc[j+1]; nh = nxt['high']; nl = nxt['low']; nc = nxt['close']
                    
                    # Volatility sizing
                    av = [abs(df.iloc[k]['high'] - df.iloc[k]['low']) for k in range(max(0, j-20), j+1)]
                    atr = np.mean(av) if av else o * 0.02; ap = atr / o
                    if ap < 0.01: lev = 3.0
                    elif ap < 0.02: lev = 2.0
                    elif ap < 0.03: lev = 1.5
                    elif ap < 0.05: lev = 0.5
                    else: continue
                    
                    prob = model.predict_proba(feats[j].reshape(1, -1))[0]
                    pred = 1 if prob[1] > 0.5 else 0
                    if rev: pred = 1 - pred
                    
                    lots = max(1, int(lev * base))
                    sd = sm * atr; td_val = rr_val * sd
                    
                    # Check next bar OHLC
                    if pred == 1:  # LONG
                        tgt = o + td_val; stp = o - sd
                        if nh >= tgt and nl > stp:
                            trade_pnl = td_val * mult * lots - 2 * cost * o * mult * lots
                        elif nl <= stp and nh < tgt:
                            trade_pnl = -sd * mult * lots - 2 * cost * o * mult * lots
                        elif nh >= tgt and nl <= stp:
                            trade_pnl = (td_val * mult * lots - 2 * cost * o * mult * lots) if (tgt - o) <= (o - stp) else (-sd * mult * lots - 2 * cost * o * mult * lots)
                        else:
                            trade_pnl = (nc - o) * mult * lots - 2 * cost * o * mult * lots
                    else:  # SHORT
                        tgt = o - td_val; stp = o + sd
                        if nl <= tgt and nh < stp:
                            trade_pnl = td_val * mult * lots - 2 * cost * o * mult * lots
                        elif nh >= stp and nl > tgt:
                            trade_pnl = -sd * mult * lots - 2 * cost * o * mult * lots
                        elif nl <= tgt and nh >= stp:
                            trade_pnl = (td_val * mult * lots - 2 * cost * o * mult * lots) if (o - tgt) <= (stp - o) else (-sd * mult * lots - 2 * cost * o * mult * lots)
                        else:
                            trade_pnl = (o - nc) * mult * lots - 2 * cost * o * mult * lots
                    
                    pnl_rmb.append(trade_pnl)
                    capital += trade_pnl
                    wins += (1 if trade_pnl > 0 else 0); total += 1
            
            if total == 0: continue
            tp = sum(pnl_rmb); wr = wins / total * 100
            cum = np.cumsum(pnl_rmb); pk = np.maximum.accumulate(cum)
            dd_val = np.max((pk - cum) / (pk + 1e-10)) * 100 if pk[-1] > 0 else 0
            gp = sum(p for p in pnl_rmb if p > 0); gl = abs(sum(p for p in pnl_rmb if p < 0))
            pf = gp / gl if gl > 0 else 999
            yrs = (n - ts) / 252
            
            if tp > -CAP and yrs > 0:
                ar = ((CAP + tp) / CAP) ** (1 / yrs) - 1
            else:
                ar = tp / CAP / yrs if yrs > 0 else -1
            
            key = f'sm{sm}_rr{rr_val}'
            entry = {'sm': sm, 'rr': rr_val, 'trades': total, 'wr': round(wr, 1),
                     'pf': round(pf, 2), 'pnl': round(tp, 0), 'ar': round(ar * 100, 1),
                     'dd': round(dd_val, 1)}
            sym_results.append(entry)
            all_results[f'{sk}_{key}'] = entry
            
            label = '+  ' if tp > 0 else ''
            elapsed = time.time() - t0
            print(f'  sm={sm:.1f} rr={rr_val:.1f} {total:>4}t WR={wr:>5.1f}% PF={pf:.2f} PnL={tp/10000:>+7.1f}万 AR={ar*100:>+6.1f}% DD={dd_val:>5.1f}%{label}  {elapsed:.0f}s')
    
    # Best per symbol
    sym_results.sort(key=lambda x: -x['pnl'])
    best = sym_results[0]
    print(f'  ★ BEST: sm={best["sm"]} rr={best["rr"]} {best["trades"]}t WR={best["wr"]}% PnL={best["pnl"]/10000:+.1f}万 AR={best["ar"]:+.1f}%')

# Save all
with open('/home/a/prophet_futures/prophet_futures/v22_results.json', 'w') as f:
    json.dump(all_results, f, indent=2, default=str)

print(f'\n===== V22 FINAL =====')
for sk in ['LH', 'JM', 'RM']:
    sym = [v for k, v in all_results.items() if k.startswith(sk)]
    sym.sort(key=lambda x: -x['pnl'])
    best = sym[0]
    print(f'{sk}: stop={best["sm"]}xATR target={best["rr"]}xStop {best["trades"]}t WR={best["wr"]}% PnL={best["pnl"]/10000:+.1f}万 AR={best["ar"]:+.1f}%/yr DD={best["dd"]}% PF={best["pf"]}')
print('Saved: v22_results.json')
