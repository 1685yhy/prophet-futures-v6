#!/usr/bin/env python3
"""
v9 — 大规模统计验证
50次随机Walk-Forward × 多时间段 × 置信区间
"""

import sys; sys.path.insert(0, ".")
import numpy as np, xgboost as xgb
from datetime import datetime
from run import *

def cal_feat(ds):
    try:
        d=datetime.strptime(str(ds)[:10],"%Y-%m-%d")
        return [d.weekday()/6, d.month/12, (d.month-1)//3/3,
                1 if d.day>25 else 0, 1 if d.month in[1,2,9,10]else 0]
    except: return [0]*5

def single_wf(df, sym, conf, risk, stop, seed, use_cal=False):
    """Single WF run with random train/test split."""
    np.random.seed(seed)
    nsp = np.random.randint(3,6)  # 3-5 random splits
    splits = sorted(np.random.choice(range(60, len(df)-60), nsp, replace=False))
    pnls = []
    for sp in range(len(splits)-1):
        te = splits[sp]; ve = splits[sp+1]
        if ve >= len(df): break
        td = df.iloc[:te]; vd = df.iloc[te:ve].reset_index(drop=True)
        fx=[]; ly=[]; W=60
        for i in range(W, len(td)-1):
            w = td.iloc[i-W:i+1]; ind = calc_indicators(w)
            sg = rule_signal(w, ind, 7)
            if sg is None: continue
            f = build_features(td, i)
            if f is None: continue
            if use_cal: f = np.concatenate([f, cal_feat(td.iloc[i]['date'])])
            nc = float(td.iloc[i+1]['close']); c = float(td.iloc[i]['close'])
            ly.append(1 if (sg=='LONG' and nc>c) or (sg=='SHORT' and nc<c) else 0)
            fx.append(f)
        if len(fx) < 50: continue
        m = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.03,
                              random_state=seed, verbosity=0)
        m.fit(np.array(fx), np.array(ly))
        for i in range(W, len(vd)-1):
            w = vd.iloc[i-W:i+1]; ind = calc_indicators(w)
            sg = rule_signal(w, ind, 7)
            if sg is None: continue
            f = build_features(vd, i)
            if f is None: continue
            if use_cal: f = np.concatenate([f, cal_feat(vd.iloc[i]['date'])])
            pr = m.predict_proba(f.reshape(1,-1))[0,1]
            if pr < conf: continue
            c = float(vd.iloc[i]['close']); atr = ind['atr14']
            e = c + 0.0002*c*(1 if sg=='LONG' else -1)
            sd = max(atr*0.3, atr*stop); td = atr*3.0
            st = e-sd if sg=='LONG' else e+sd; tg = e+td if sg=='LONG' else e-td
            l = LOT.get(sym,10); q = max(1.0, min(20.0, 1_000_000*risk/(sd*l)))
            nc = float(vd.iloc[i+1]['close']); nh = float(vd.iloc[i+1]['high']); nl = float(vd.iloc[i+1]['low'])
            ep = st if (sg=='LONG' and nl<=st) or (sg=='SHORT' and nh>=st) else (tg if (sg=='LONG' and nh>=tg) or (sg=='SHORT' and nl<=tg) else nc)
            pnl = (ep-e)*l*q*(1 if sg=='LONG' else -1) - abs(ep*l*q*0.0001)*2
            pnls.append(pnl)
    return pnls

def stats(pnls):
    if len(pnls) < 5: return None
    w = [p for p in pnls if p>0]; n = len(pnls)
    wr = len(w)/n; tp = sum(pnls); ret = tp/1_000_000*100
    cum = np.cumsum(pnls); rm = np.maximum.accumulate(cum)
    dd = abs(float(np.min((cum-rm)/1_000_000*100)))
    return {"n":n, "wr":wr, "pnl":tp, "ret":ret, "dd":dd}

print("="*60)
print("  v9 — 大规模统计验证 (50次随机WF)")
print(f"  {datetime.now():%Y-%m-%d %H:%M}")
print("="*60)

N_TRIALS = 50

for sym in ["jm", "lh"]:
    df = fetch(sym, 2500)
    if df is None: continue
    print(f"\n{'─'*60}")
    print(f"  {sym.upper()} ({len(df)}天) — {N_TRIALS}次随机WF")

    for cal_name, use_cal in [("BASE", False), ("+CAL", True)]:
        results = []
        for seed in range(N_TRIALS):
            pnls = single_wf(df, sym, 0.50, 0.02, 1.0, seed*7+1, use_cal)
            s = stats(pnls)
            if s and s["n"] >= 5: results.append(s)

        if not results: continue
        wr = [r["wr"] for r in results]
        ret = [r["ret"] for r in results]
        dd = [r["dd"] for r in results]
        n_ = [r["n"] for r in results]
        total_trades = sum(n_)
        total_pnl = sum(r["pnl"] for r in results)

        print(f"  {cal_name}: {len(results)}次有效")
        print(f"    交易: {np.mean(n_):.0f}±{np.std(n_):.0f}/次  合计{total_trades}笔")
        print(f"    胜率: {np.mean(wr):.1%} ±{np.std(wr):.1%}  (范围 {np.min(wr):.0%}-{np.max(wr):.0%})")
        print(f"    收益: {np.mean(ret):.1f}% ±{np.std(ret):.1f}%")
        print(f"    回撤: {np.mean(dd):.1f}% ±{np.std(dd):.1f}%")
        print(f"    累计PnL: {total_pnl:+,.0f}")

        # 95% confidence interval for win rate
        ci = 1.96 * np.std(wr) / np.sqrt(len(wr))
        print(f"    胜率95%CI: [{np.mean(wr)-ci:.1%}, {np.mean(wr)+ci:.1%}]")
