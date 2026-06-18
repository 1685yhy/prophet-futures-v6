#!/usr/bin/env python3
"""v8 — 深度XGBoost + 日历特征 + 集成投票"""

import sys; sys.path.insert(0, ".")
import numpy as np, xgboost as xgb
from datetime import datetime
from run import *

def cal_feat(ds):
    """Calendar features: day_of_week, month, quarter, month_end proximity."""
    try:
        d=datetime.strptime(str(ds)[:10],"%Y-%m-%d")
        return [d.weekday()/6, d.month/12, (d.month-1)//3/3,
                1 if d.day>25 else 0, 1 if d.month in[1,2,9,10]else 0]
    except: return [0]*5

def wf_with_calendar(df,sym,conf,risk,stop,n_est=200,depth=7,lr=0.03):
    nsp=4;sps=np.linspace(60,len(df),nsp+1).astype(int);pnls=[]
    for sp in range(nsp-1):
        te=sps[sp+1];ve=sps[sp+2]
        if ve>=len(df):break
        td=df.iloc[:te];vd=df.iloc[te:ve].reset_index(drop=True)
        fx=[];ly=[];W=60
        for i in range(W,len(td)-1):
            w=td.iloc[i-W:i+1];ind=calc_indicators(w)
            sg=rule_signal(w,ind,7)
            if sg is None:continue
            f=build_features(td,i)
            if f is None:continue
            cf=cal_feat(td.iloc[i]['date'])
            nc=float(td.iloc[i+1]['close']);c=float(td.iloc[i]['close'])
            ly.append(1 if(sg=='LONG' and nc>c)or(sg=='SHORT' and nc<c)else 0)
            fx.append(np.concatenate([f,cf]))
        if len(fx)<50:continue
        m=xgb.XGBClassifier(n_estimators=n_est,max_depth=depth,learning_rate=lr,random_state=42,verbosity=0)
        m.fit(np.array(fx),np.array(ly))
        for i in range(W,len(vd)-1):
            w=vd.iloc[i-W:i+1];ind=calc_indicators(w)
            sg=rule_signal(w,ind,7)
            if sg is None:continue
            f=build_features(vd,i)
            if f is None:continue
            cf=cal_feat(vd.iloc[i]['date'])
            pr=m.predict_proba(np.concatenate([f,cf]).reshape(1,-1))[0,1]
            if pr<conf:continue
            c=float(vd.iloc[i]['close']);atr=ind['atr14']
            e=c+0.0002*c*(1 if sg=='LONG'else-1)
            sd=max(atr*0.3,atr*stop);td=atr*3.0
            st=e-sd if sg=='LONG'else e+sd;tg=e+td if sg=='LONG'else e-td
            l=LOT.get(sym,10);q=max(1.0,min(20.0,1_000_000*risk/(sd*l)))
            nc=float(vd.iloc[i+1]['close']);nh=float(vd.iloc[i+1]['high']);nl=float(vd.iloc[i+1]['low'])
            ep=st if(sg=='LONG'and nl<=st)or(sg=='SHORT'and nh>=st)else(tg if(sg=='LONG'and nh>=tg)or(sg=='SHORT'and nl<=tg)else nc)
            pnl=(ep-e)*l*q*(1 if sg=='LONG'else-1)-abs(ep*l*q*0.0001)*2
            pnls.append(pnl)
    return pnls

print("="*60)
print("  v8 — 深度XGBoost + 日历特征")
print("="*60)

for sym in["jm","lh"]:
    df=fetch(sym,2500)
    print(f"\n  {sym.upper()}:")
    for n_est in[150,200,250]:
        for depth in[6,7,8]:
            for lr in[0.03,0.02]:
                for cal in[False,True]:
                    pnls=wf_with_calendar(df,sym,0.5,0.02,1.0,n_est,depth,lr) if cal else None
                    # Without calendar — use original WF
                    if not cal:
                        from v7_tune import wf_backtest, make_model
                        m=make_model("XGBoost",{"n_estimators":n_est,"max_depth":depth,"learning_rate":lr})
                        pnls=wf_backtest(df,sym,m,0.5,0.02,1.0)
                    if len(pnls)<15:continue
                    w=[p for p in pnls if p>0];n=len(pnls)
                    wr=len(w)/n;tp=sum(pnls);ret=tp/1_000_000*100
                    cum=np.cumsum(pnls);rm=np.maximum.accumulate(cum)
                    dd=abs(float(np.min((cum-rm)/1_000_000*100)))
                    cal_tag="+CAL" if cal else "BASE"
                    m=f"🔥" if wr>=0.62 else("⭐" if wr>=0.58 else"")
                    print(f"  {m} {cal_tag} est={n_est} d={depth} lr={lr}: {n}t {wr:.0%}wr {ret:+.1f}% DD{dd:.1f}%")
