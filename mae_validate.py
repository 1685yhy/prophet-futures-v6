#!/usr/bin/env python3
"""Compare MAE-optimized ATR stops vs v17 cost stops, 1000 WF"""
import numpy as np, pandas as pd, json, time
import akshare as ak, xgboost as xgb

SYM = {
    'LH': ('LH0', 0.0006, False, 200, 5, 0.05, 60, 16, 3, 1.0),  # + atr_stop
    'JM': ('JM0', 0.0011, False, 100, 4, 0.03, 60, 60, 5, 1.5),
    'RM': ('RM0', 0.0011, True,  100, 5, 0.03, 120, 10, 20, 1.0),
}
N_WF = 1000; CAP = 300000; RR = 3.0

def fetch(c):
    e = pd.Timestamp.now(); s = e - pd.Timedelta(days=2500)
    df = ak.futures_main_sina(symbol=c, start_date=s.strftime('%Y%m%d'), end_date=e.strftime('%Y%m%d'))
    df.columns = ['date','open','high','low','close','volume','oi','settle']
    for x in ['open','high','low','close','volume','oi']:
        df[x] = pd.to_numeric(df[x], errors='coerce')
    return df.dropna(subset=['close']).reset_index(drop=True)

def feats(df2, i, L=60):
    if i < L + 5: return None
    w = df2.iloc[i-L:i+1]; c = w['close'].values; o = w['open'].values
    h = w['high'].values; l = w['low'].values; v = w['volume'].values; oi = w['oi'].values
    f = []
    if i >= 1: f.append((o[-1]-c[-2])/c[-2]); f.append(abs(f[-1]))
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
    bb = np.std(c[-20:]); ma20 = np.mean(c[-20:])
    f.append((c[-1]-ma20)/(2*bb+1e-10))
    f.append(c[-1]/1000.0)
    return np.array(f, dtype=np.float32)

def run_wf(df, ne, d, lr, win, cost, rev, mult, base, atr_stop):
    """500 WF with ATR-based stop + OHLC simulation"""
    n = len(df); ts = int(n * 0.6)
    if ts < 200: return None
    pnl_seq = []; wins = 0; total = 0; capital = CAP; peak_cap = CAP; max_dd = 0
    
    for run in range(min(N_WF, (n - ts) // 10)):
        sp = ts + run * 10
        if sp + 10 > n: break
        td = df.iloc[:sp]; t1 = sp; t2 = min(sp + 10, n)
        X, y = [], []
        for i in range(win, len(td) - 1):
            f = feats(td, i, win)
            if f is None: continue
            X.append(f); y.append(1 if td.iloc[i+1]['close'] > td.iloc[i]['close'] else 0)
        if len(X) < 100: continue
        ya = np.array(y)
        if len(np.unique(ya)) < 2: continue
        m = xgb.XGBClassifier(n_estimators=ne, max_depth=d, learning_rate=lr,
                               verbosity=0, random_state=42)
        m.fit(np.array(X), ya)
        
        for j in range(t1, t2 - 1):
            f = feats(df, j, win)
            if f is None: continue
            p = m.predict_proba(f.reshape(1, -1))[0]
            pred = 1 if p[1] > 0.5 else 0
            if rev: pred = 1 - pred
            
            bar = df.iloc[j]; o = bar['open']; h = bar['high']; l = bar['low']
            nxt = df.iloc[j+1]; nh = nxt['high']; nl = nxt['low']; nc = nxt['close']
            
            av = [abs(df.iloc[k]['high'] - df.iloc[k]['low']) for k in range(max(0, j-20), j+1)]
            atr = np.mean(av) if av else o * 0.02
            ap = atr / o
            if ap < 0.01: lev = 3.0
            elif ap < 0.02: lev = 2.0
            elif ap < 0.03: lev = 1.5
            elif ap < 0.05: lev = 0.5
            else: continue
            
            lots = max(1, int(lev * base))
            sd = atr_stop * atr; td_val = RR * sd
            
            # OHLC simulation
            trade_pnl = 0
            if pred == 1:  # LONG
                tgt = o + td_val; stp = o - sd
                if nh >= tgt and nl > stp:
                    trade_pnl = td_val * mult * lots - 2 * cost * o * mult * lots
                elif nl <= stp and nh < tgt:
                    trade_pnl = -sd * mult * lots - 2 * cost * o * mult * lots
                elif nh >= tgt and nl <= stp:
                    trade_pnl = (td_val*mult*lots-2*cost*o*mult*lots) if (tgt-o)<=(o-stp) else (-sd*mult*lots-2*cost*o*mult*lots)
                else:
                    trade_pnl = (nc - o) * mult * lots - 2 * cost * o * mult * lots
            else:  # SHORT
                tgt = o - td_val; stp = o + sd
                if nl <= tgt and nh < stp:
                    trade_pnl = td_val * mult * lots - 2 * cost * o * mult * lots
                elif nh >= stp and nl > tgt:
                    trade_pnl = -sd * mult * lots - 2 * cost * o * mult * lots
                elif nl <= tgt and nh >= stp:
                    trade_pnl = (td_val*mult*lots-2*cost*o*mult*lots) if (o-tgt)<=(stp-o) else (-sd*mult*lots-2*cost*o*mult*lots)
                else:
                    trade_pnl = (o - nc) * mult * lots - 2 * cost * o * mult * lots
            
            pnl_seq.append(trade_pnl)
            capital += trade_pnl
            if capital > peak_cap: peak_cap = capital
            dd_pct = (peak_cap - capital) / peak_cap * 100
            if dd_pct > max_dd: max_dd = dd_pct
            wins += (1 if trade_pnl > 0 else 0); total += 1
    
    if total == 0: return None
    tp = sum(pnl_seq); wr = wins / total * 100
    gp = sum(p for p in pnl_seq if p > 0); gl = abs(sum(p for p in pnl_seq if p < 0))
    pf = gp / gl if gl > 0 else 999
    yrs = (n - ts) / 252
    ar = ((CAP + tp) / CAP) ** (1 / yrs) - 1 if tp > -CAP and yrs > 0 else tp / CAP / yrs
    return {'trades': total, 'wr': round(wr, 1), 'pf': round(pf, 2), 'pnl': round(tp, 0),
            'ar': round(ar * 100, 1), 'dd': round(max_dd, 1), 'atr_stop': atr_stop}

print('Comparing MAE-optimized ATR stops (OHLC sim) at RR=3')
print(f'{"Sym":<4} {"Stop":<8} {"Trd":<7} {"WR":<6} {"PF":<6} {"PnL":<9} {"Ann":<8} {"DD":<7}')

for sk, (sc, cost, rev, ne, d, lr, win, mult, base, atr_stop) in SYM.items():
    print(f'{sk}...', end='', flush=True)
    df = fetch(sc)
    r = run_wf(df, ne, d, lr, win, cost, rev, mult, base, atr_stop)
    if r:
        print(f' {r["trades"]:<7} {r["wr"]:<5.1f}% {r["pf"]:<6.2f} {r["pnl"]/10000:<+8.1f}万 {r["ar"]:<+7.1f}% {r["dd"]:<6.1f}%')
    else:
        print(' FAIL')
