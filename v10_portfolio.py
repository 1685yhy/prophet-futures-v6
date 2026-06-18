#!/usr/bin/env python3
"""v10 — 组合优化: LH+JM双品种 + 动态仓位 + 集成投票 + 增强特征"""

import sys; sys.path.insert(0, ".")
import numpy as np, xgboost as xgb, lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime
from run import *

def cal_feat(ds):
    try:
        d=datetime.strptime(str(ds)[:10],"%Y-%m-%d")
        return [d.weekday()/6,d.month/12,(d.month-1)//3/3,
                1 if d.day>25 else 0,1 if d.month in[1,2,9,10]else 0]
    except:return[0]*5

def enhanced_features(df,i):
    """Add Bollinger, KDJ, CCI, momentum divergence."""
    base=build_features(df,i)
    if base is None:return None
    c=df["close"].values.astype(float);h=df["high"].values.astype(float)
    l=df["low"].values.astype(float);cl=c[i]
    # Bollinger position
    ma20=np.mean(c[max(0,i-20):i+1]);std20=np.std(c[max(0,i-20):i+1])
    bb_pos=(cl-ma20)/(std20+1e-8)
    # KDJ
    if i>=9:
        h9=max(h[i-9:i+1]);l9=min(l[i-9:i+1])
        rsv=(cl-l9)/(h9-l9+1e-8)*100
    else:rsv=50
    # CCI
    tp=(h[i]+l[i]+cl)/3;ma_tp=np.mean([(h[j]+l[j]+c[j])/3 for j in range(max(0,i-20),i+1)])
    cci=(tp-ma_tp)/(0.015*np.std([(h[j]+l[j]+c[j])/3 for j in range(max(0,i-20),i+1)])+1e-8)
    return np.concatenate([base,[bb_pos,rsv/100,cci/100]])

def train_model(df,use_cal=False,use_enhanced=False,model_type="xgb"):
    fx=[];ly=[];W=60
    for i in range(W,len(df)-1):
        w=df.iloc[i-W:i+1];ind=calc_indicators(w)
        sg=rule_signal(w,ind,7)
        if sg is None:continue
        f=enhanced_features(df,i) if use_enhanced else build_features(df,i)
        if f is None:continue
        if use_cal:f=np.concatenate([f,cal_feat(df.iloc[i]['date'])])
        nc=float(df.iloc[i+1]['close']);c=float(df.iloc[i]['close'])
        ly.append(1 if(sg=='LONG' and nc>c)or(sg=='SHORT' and nc<c)else 0);fx.append(f)
    if len(fx)<50:return None
    X=np.array(fx);y=np.array(ly)
    if model_type=="xgb":m=xgb.XGBClassifier(n_estimators=200,max_depth=6,learning_rate=0.03,random_state=42,verbosity=0)
    elif model_type=="lgb":m=lgb.LGBMClassifier(n_estimators=100,max_depth=5,learning_rate=0.05,random_state=42,verbose=-1)
    else:m=RandomForestClassifier(n_estimators=100,max_depth=6,random_state=42)
    m.fit(X,y);return m

print("="*60)
print("  v10 — 双品种组合 + 增强特征 + 集成投票")
print("="*60)

CONFIGS={
    "jm":{"cal":True,"enhanced":False,"model":"xgb"},
    "lh":{"cal":False,"enhanced":False,"model":"xgb"},
}

results=[]
for seed in range(30):
    np.random.seed(seed)
    nsp=np.random.randint(3,6)
    lh_df=fetch("lh",2500);jm_df=fetch("jm",2500)
    min_len=min(len(lh_df),len(jm_df))
    splits=sorted(np.random.choice(range(60,min_len-60),nsp,replace=False))
    pnls=[]
    for sp in range(len(splits)-1):
        te=splits[sp];ve=splits[sp+1]
        for sym,df in[("lh",lh_df),("jm",jm_df)]:
            cfg=CONFIGS[sym];td=df.iloc[:te];vd=df.iloc[te:ve].reset_index(drop=True)
            m=train_model(td,cfg["cal"],cfg["enhanced"],cfg["model"])
            if m is None:continue
            for i in range(60,len(vd)-1):
                w=vd.iloc[i-60:i+1];ind=calc_indicators(w)
                sg=rule_signal(w,ind,7)
                if sg is None:continue
                f=enhanced_features(vd,i) if cfg["enhanced"] else build_features(vd,i)
                if f is None:continue
                if cfg["cal"]:f=np.concatenate([f,cal_feat(vd.iloc[i]['date'])])
                pr=m.predict_proba(f.reshape(1,-1))[0,1]
                if pr<0.5:continue
                c=float(vd.iloc[i]['close']);atr=ind["atr14"]
                e=c+0.0002*c*(1 if sg=='LONG'else-1)
                sd=max(atr*0.3,atr*1.0);td=atr*3.0
                st=e-sd if sg=='LONG'else e+sd;tg=e+td if sg=='LONG'else e-td
                l=LOT.get(sym,10);q=max(1.0,min(20.0,1_000_000*0.015/(sd*l)))
                nc=float(vd.iloc[i+1]['close']);nh=float(vd.iloc[i+1]['high']);nl=float(vd.iloc[i+1]['low'])
                ep=st if(sg=='LONG'and nl<=st)or(sg=='SHORT'and nh>=st)else(tg if(sg=='LONG'and nh>=tg)or(sg=='SHORT'and nl<=tg)else nc)
                pnl=(ep-e)*l*q*(1 if sg=='LONG'else-1)-abs(ep*l*q*0.0001)*2
                pnls.append(pnl)
    if pnls:
        w=[p for p in pnls if p>0];n=len(pnls);wr=len(w)/n;tp=sum(pnls)
        ret=tp/1_000_000*100;cum=np.cumsum(pnls);rm=np.maximum.accumulate(cum)
        dd=abs(float(np.min((cum-rm)/1_000_000*100)))
        results.append({'n':n,'wr':wr,'ret':ret,'dd':dd})

if results:
    wr=[r['wr'] for r in results];ret=[r['ret'] for r in results]
    dd=[r['dd'] for r in results];n_=[r['n'] for r in results]
    print(f"\n  组合(LH+JM): {len(results)}次")
    print(f"  交易: {np.mean(n_):.0f}±{np.std(n_):.0f}/次")
    print(f"  胜率: {np.mean(wr):.1%}±{np.std(wr):.1%}")
    ci=1.96*np.std(wr)/np.sqrt(len(wr))
    print(f"  胜率95%CI: [{np.mean(wr)-ci:.1%},{np.mean(wr)+ci:.1%}]")
    print(f"  收益: {np.mean(ret):.1f}%±{np.std(ret):.1f}%")
    print(f"  回撤: {np.mean(dd):.1f}%±{np.std(dd):.1f}%")
    print(f"  累计PnL: {sum(r['pnl'] if'pnl'in r else r['ret']*10000 for r in results):+,.0f}")
