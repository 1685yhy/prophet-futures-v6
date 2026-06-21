#!/usr/bin/env python3
"""Fast compare: MAE-ATR vs v17 close-to-close, 1000WF, precomputed"""
import numpy as np, pandas as pd, json, time
import akshare as ak, xgboost as xgb

S = {
    'LH': ('LH0', 0.0006, False, 200, 5, 0.05, 60, 16, 3, 1.0),
    'JM': ('JM0', 0.0011, False, 100, 4, 0.03, 60, 60, 5, 1.5),
    'RM': ('RM0', 0.0011, True,  100, 5, 0.03, 120, 10, 20, 1.0),
}
N_WF = 500; CAP = 300000; RR = 3.0

def precompute(df, win=60):
    n = len(df); F = np.zeros((n, 20), dtype=np.float32); V = np.zeros(n, dtype=bool)
    for idx in range(win+5, n):
        w = df.iloc[idx-win:idx+1]; c = w['close'].values; o = w['open'].values
        h = w['high'].values; l = w['low'].values; v = w['volume'].values; oi = w['oi'].values
        f = F[idx]
        if idx >= 1: f[0] = (o[-1]-c[-2])/c[-2]; f[1] = abs(f[0])
        for li, lag in enumerate([1,3,5,10,20], 2):
            f[li] = (c[-1]-c[-lag-1])/c[-lag-1] if len(c) > lag else 0
        for pi, p in enumerate([5,10,20,60], 7):
            ma = np.mean(c[-min(p,len(c)):]); f[pi] = (c[-1]-ma)/ma
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
        V[idx] = True
    L = np.zeros(n, dtype=int)
    for i in range(n-1):
        if V[i]: L[i] = 1 if df.iloc[i+1]['close'] > df.iloc[i]['close'] else 0
    return F, V, L

def run_ohlc(df, F, V, L, ne, d, lr, win, cost, rev, mult, base, atr_stop):
    n = len(df); ts = int(n * 0.6)
    pnl_seq = []; wins = 0; total = 0; cap = CAP; peak = CAP; max_dd = 0
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
            bar = df.iloc[j]; o = bar['open']; h = bar['high']; l = bar['low']
            nxt = df.iloc[j+1]; nh = nxt['high']; nl = nxt['low']; nc = nxt['close']
            av = [abs(df.iloc[k]['high'] - df.iloc[k]['low']) for k in range(max(0, j-20), j+1)]
            atr = np.mean(av) if av else o * 0.02; ap = atr / o
            if ap < 0.01: lev = 3.0
            elif ap < 0.02: lev = 2.0
            elif ap < 0.03: lev = 1.5
            elif ap < 0.05: lev = 0.5
            else: continue
            lots = max(1, int(lev * base))
            sd = atr_stop * atr; td_val = RR * sd
            if pred == 1:
                tgt = o + td_val; stp = o - sd
                if nh >= tgt and nl > stp: trade_pnl = td_val*mult*lots - 2*cost*o*mult*lots
                elif nl <= stp and nh < tgt: trade_pnl = -sd*mult*lots - 2*cost*o*mult*lots
                elif nh >= tgt and nl <= stp: trade_pnl = (td_val*mult*lots-2*cost*o*mult*lots) if (tgt-o)<=(o-stp) else (-sd*mult*lots-2*cost*o*mult*lots)
                else: trade_pnl = (nc-o)*mult*lots - 2*cost*o*mult*lots
            else:
                tgt = o - td_val; stp = o + sd
                if nl <= tgt and nh < stp: trade_pnl = td_val*mult*lots - 2*cost*o*mult*lots
                elif nh >= stp and nl > tgt: trade_pnl = -sd*mult*lots - 2*cost*o*mult*lots
                elif nl <= tgt and nh >= stp: trade_pnl = (td_val*mult*lots-2*cost*o*mult*lots) if (o-tgt)<=(stp-o) else (-sd*mult*lots-2*cost*o*mult*lots)
                else: trade_pnl = (o-nc)*mult*lots - 2*cost*o*mult*lots
            pnl_seq.append(trade_pnl); cap += trade_pnl
            if cap > peak: peak = cap
            dd_pct = (peak - cap) / peak * 100
            if dd_pct > max_dd: max_dd = dd_pct
            wins += (1 if trade_pnl > 0 else 0); total += 1
    if total == 0: return None
    tp = sum(pnl_seq); wr = wins / total * 100
    gp = sum(p for p in pnl_seq if p > 0); gl = abs(sum(p for p in pnl_seq if p < 0))
    pf = gp / gl if gl > 0 else 999
    yrs = (n - ts) / 252
    ar = ((CAP + tp) / CAP) ** (1 / yrs) - 1 if tp > -CAP and yrs > 0 else tp / CAP / yrs
    return {'trades': total, 'wr': round(wr, 1), 'pf': round(pf, 2), 'pnl': round(tp, 0),
            'ar': round(ar*100, 1), 'dd': round(max_dd, 1)}

def run_c2c(df, F, V, L, ne, d, lr, win, cost, rev, mult, base):
    n = len(df); ts = int(n * 0.6)
    pnl_seq = []; wins = 0; total = 0; cap = CAP; peak = CAP; max_dd = 0
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
            o = df.iloc[j]['close']; nc = df.iloc[j+1]['close']
            chg = nc - o if pred == 1 else o - nc
            trade_pnl = chg * mult * base - cost * o * mult * base
            pnl_seq.append(trade_pnl); cap += trade_pnl
            if cap > peak: peak = cap
            dd_pct = (peak - cap) / peak * 100
            if dd_pct > max_dd: max_dd = dd_pct
            wins += (1 if trade_pnl > 0 else 0); total += 1
    if total == 0: return None
    tp = sum(pnl_seq); wr = wins / total * 100
    gp = sum(p for p in pnl_seq if p > 0); gl = abs(sum(p for p in pnl_seq if p < 0))
    pf = gp / gl if gl > 0 else 999
    yrs = (n - ts) / 252
    ar = ((CAP + tp) / CAP) ** (1 / yrs) - 1 if tp > -CAP and yrs > 0 else tp / CAP / yrs
    return {'trades': total, 'wr': round(wr, 1), 'pf': round(pf, 2), 'pnl': round(tp, 0),
            'ar': round(ar*100, 1), 'dd': round(max_dd, 1)}

# Run both modes
print('Mode          Sym  Trades  WR      PF     PnL       Ann      DD')
print('-' * 72)

for mode_name, fn in [('MAE-ATR-stops', run_ohlc), ('v17-close2close', run_c2c)]:
    for sk, (sc, cost, rev, ne, d, lr, win, mult, base, atr_stop) in S.items():
        df = pd.DataFrame()
        for attempt in range(3):
            try:
                e = pd.Timestamp.now(); s = e - pd.Timedelta(days=2500)
                df = ak.futures_main_sina(symbol=sc, start_date=s.strftime('%Y%m%d'), end_date=e.strftime('%Y%m%d'))
                if len(df) > 100: break
            except: time.sleep(1)
        df.columns = ['date','open','high','low','close','volume','oi','settle']
        for x in ['open','high','low','close','volume','oi']:
            df[x] = pd.to_numeric(df[x], errors='coerce')
        df = df.dropna(subset=['close']).reset_index(drop=True)
        
        F, V, L = precompute(df, win)
        if fn == run_ohlc:
            r = fn(df, F, V, L, ne, d, lr, win, cost, rev, mult, base, atr_stop)
        else:
            r = fn(df, F, V, L, ne, d, lr, win, cost, rev, mult, base)
        
        if r:
            print(f'{mode_name:<15} {sk:<4} {r["trades"]:<7} {r["wr"]:<5.1f}% {r["pf"]:<6.2f} {r["pnl"]/10000:<+8.1f}万 {r["ar"]:<+7.1f}% {r["dd"]:<6.1f}%')
        else:
            print(f'{mode_name:<15} {sk:<4} FAIL')
