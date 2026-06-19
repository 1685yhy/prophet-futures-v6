#!/usr/bin/env python3
"""v14 — 多模型集成 + 限价单优化 (Renaissance思路)"""

import sys; sys.path.insert(0,".")
import numpy as np, xgboost as xgb
from datetime import datetime
from tools.indicators import calc_indicators,_calc_macd
from tools.cycle_detector import detect_cycle,detect_rollover_noise

def fetch(sym,days=2500):
    import akshare as ak
    e=datetime.now();from datetime import timedelta; s=e-timedelta(days=days+200)
    df=ak.futures_main_sina(sym.upper()+"0",s.strftime("%Y%m%d"),e.strftime("%Y%m%d"))
    df.columns=["date","open","high","low","close","volume","oi","settle"]
    for c in["open","high","low","close","volume","oi"]:df[c]=df[c].astype(float)
    return df.reset_index(drop=True)

def feat(df,i,L=60):
    if i<L:return None
    c=df["close"].values.astype(float);o=df["open"].values.astype(float)
    h=df["high"].values.astype(float);l=df["low"].values.astype(float)
    v=df["volume"].values.astype(float)
    oi=df["oi"].values.astype(float) if"oi"in df.columns else np.zeros(len(c))
    wc=c[i-L:i+1];wv=v[i-L:i+1];wi=oi[i-L:i+1];wh=h[i-L:i+1];wl=l[i-L:i+1];cl=c[i];op=o[i]
    f=[]
    for j in[1,3,5,10,21]:
        if len(wc)>j:f.append((cl-wc[-j-1])/(wc[-j-1]+1e-8))
        else:f.append(0)
    f.append((cl-np.mean(wc[-20:]))/(np.std(wc[-20:])+1e-8))
    f.append((cl-np.mean(wc[-5:]))/(np.std(wc[-5:])+1e-8))
    for Lv in[3,5,8,10,13,20,30,min(60,len(wc))]:
        ma=np.mean(wc[-Lv:]);f.append((ma-cl)/(cl+1e-8))
    ma5=np.mean(wc[-5:]);ma8=np.mean(wc[-8:]);ma20=np.mean(wc[-20:]);ma60=np.mean(wc[-min(60,len(wc)):])
    f.extend([(ma5-ma8)/(ma8+1e-8),(ma5-ma20)/(ma20+1e-8),(ma20-ma60)/(ma60+1e-8)])
    tr=[max(wh[j]-wl[j],abs(wh[j]-wc[j-1]),abs(wl[j]-wc[j-1])) for j in range(-14,0)]
    f.extend([np.mean(tr)/(cl+1e-8),np.std(wc[-20:])/(np.mean(wc[-20:])+1e-8)])
    g=[max(0,wc[j]-wc[j-1]) for j in range(-14,0)];ls=[max(0,wc[j-1]-wc[j]) for j in range(-14,0)]
    f.extend([np.mean(g)/(np.mean(ls)+1e-8),(cl-min(wc[-14:]))/(max(wc[-14:])-min(wc[-14:])+1e-8)])
    vm3=np.mean(wv[-3:]);vm5=np.mean(wv[-5:]);vm10=np.mean(wv[-10:]);vm20=np.mean(wv[-20:])
    f.extend([(vm3-vm10)/(vm10+1e-8),(vm5-vm20)/(vm20+1e-8),(wv[-1]-vm20)/(vm20+1e-8)])
    om3=np.mean(wi[-3:]);om5=np.mean(wi[-5:]);om20=np.mean(wi[-20:])
    f.extend([(om3-om5)/(om5+1e-8),(om5-om20)/(om20+1e-8)])
    f.extend([(cl-op)/(op+1e-8),1 if cl>op else-1,(h[i]-l[i])/(cl+1e-8)])
    f.extend([1 if cl>wc[-2] else-1,1 if cl>wc[-3] else-1,1 if ma5>ma8 else-1,1 if ma5>ma20 else-1])
    return np.array(f,dtype=np.float64)

def sig(df_w,ind,mc=7):
    c=df_w["close"].values.astype(float)
    cy=detect_cycle(df_w);ns=detect_rollover_noise(df_w)
    if cy["cycle"]not in("BULL","BEAR"):return None
    if ns["is_noise"]:return None
    _,_,h0=_calc_macd(c);_,_,h1=_calc_macd(c[:-1]) if len(c)>1 else(0,0,0);_,_,h2=_calc_macd(c[:-2]) if len(c)>2 else(0,0,0)
    mi=bool(h0<0 and h1<0 and abs(h0)<abs(h1)<abs(h2))
    adx=ind.get("adx14",0);rsi=ind.get("rsi14",50);m5=ind.get("ma5",0);m20=ind.get("ma20",0);m60=ind.get("ma60",0);mh=ind.get("macd_hist",0)
    oic="oi"if"oi"in df_w.columns else None;oi_=df_w[oic].values.astype(float)if oic else np.zeros(10)
    o3=float(oi_[-1]-oi_[-4])if len(oi_)>=4 else 0;o5=float(oi_[-1]-oi_[-6])if len(oi_)>=6 else o3
    ot="ACCUMULATING"if(o3>0 and o5>0)else("REDUCING"if(o3<0 and o5<0)else"FLAT")
    mb=m5>m20>m60;mbe=m5<m20<m60
    sc=sum([cy["cycle"]=="BEAR",mbe,mh<0 and not mi,ot in("REDUCING","FLAT"),adx>20,32<rsi<72,True,True])
    lc=sum([cy["cycle"]=="BULL",mb,mh>0,ot=="ACCUMULATING",adx>22,30<rsi<65,True,True])
    if sc>=mc:return"SHORT"
    if lc>=mc:return"LONG"
    return None

LOT={"jm":60,"lh":16,"rm":10}; COST=0.0004
N_ENSEMBLE=21  # 21个小模型

def train_ensemble(td):
    """Train 21 models on bootstrapped data."""
    fx,ly=[],[]
    for i in range(60,len(td)-1):
        w=td.iloc[i-60:i+1];ind=calc_indicators(w)
        sg_=sig(w,ind,7)
        if sg_ is None:continue
        f_=feat(td,i)
        if f_ is None:continue
        nc=float(td.iloc[i+1]["close"]);c=float(td.iloc[i]["close"])
        ly.append(1 if(sg_=="LONG"and nc>c)or(sg_=="SHORT"and nc<c)else 0);fx.append(f_)
    if len(fx)<80:return None
    X=np.array(fx);y=np.array(ly)
    models=[]
    for k in range(N_ENSEMBLE):
        idx=np.random.choice(len(X),size=int(len(X)*0.8),replace=True)
        m=xgb.XGBClassifier(n_estimators=30,max_depth=3,learning_rate=0.05,
                            subsample=0.7,colsample_bytree=0.7,random_state=k+42,verbosity=0)
        m.fit(X[idx],y[idx]);models.append(m)
    return models

def ensemble_predict(models, f):
    """21个模型投票，返回通过比例和平均置信度."""
    votes=0;confs=[]
    for m in models:
        prob=m.predict_proba(f.reshape(1,-1))[0,1]
        confs.append(prob)
        if prob>0.5:votes+=1
    return votes/N_ENSEMBLE, np.mean(confs)

print("="*60)
print("  v14 — 21模型集成 + 限价单优化")
print(f"  {datetime.now():%Y-%m-%d %H:%M}")
print("="*60)

for sym in["jm","lh","rm"]:
    df=fetch(sym,2500)
    print(f"\n{sym.upper()} 50次WF...")
    results=[]
    for seed in range(50):
        np.random.seed(seed)
        nsp=np.random.randint(3,6)
        splits=sorted(np.random.choice(range(300,len(df)-60),nsp,replace=False))
        pnls=[]
        for sp in range(len(splits)-1):
            te,ve=splits[sp],splits[sp+1]
            td=df.iloc[:te];vd=df.iloc[te:ve].reset_index(drop=True)
            models=train_ensemble(td)
            if models is None:continue
            for i in range(60,len(vd)-1):
                w=vd.iloc[i-60:i+1];ind=calc_indicators(w)
                sg_=sig(w,ind,7)
                if sg_ is None:continue
                f_=feat(vd,i)
                if f_ is None:continue
                vote_ratio,avg_conf=ensemble_predict(models,f_)
                if vote_ratio<0.67:continue  # 至少14/21同意
                c=float(vd.iloc[i]["close"]);atr=ind["atr14"]
                # 限价单: 挂低0.1%等成交
                limit_price=c-0.001*c if sg_=="LONG" else c+0.001*c
                e=limit_price
                sd=max(atr*0.3,atr*1.0);td=atr*3.0
                st=e-sd if sg_=="LONG"else e+sd;tg=e+td if sg_=="LONG"else e-td
                l=LOT.get(sym,10);q=max(1.0,min(20.0,500000*0.02/(sd*l)))
                nc=float(vd.iloc[i+1]["close"]);nh=float(vd.iloc[i+1]["high"]);nl=float(vd.iloc[i+1]["low"])
                # 检查限价单是否成交
                if sg_=="LONG" and nl>limit_price:continue  # 没碰到
                if sg_=="SHORT" and nh<limit_price:continue
                ep=st if(sg_=="LONG"and nl<=st)or(sg_=="SHORT"and nh>=st)else(tg if(sg_=="LONG"and nh>=tg)or(sg_=="SHORT"and nl<=tg)else nc)
                pnl=(ep-e)*l*q*(1 if sg_=="LONG"else-1)-abs(e*l*q*COST)
                pnls.append(pnl)
        if len(pnls)>=5:
            w=[p for p in pnls if p>0];wr=len(w)/len(pnls);tp=sum(pnls)
            eq=500000;me=500000;mdd=0
            for p in pnls:eq+=p;me=max(me,eq);mdd=min(mdd,(eq-me)/me*100)
            results.append({"wr":wr,"pnl":tp,"dd":abs(mdd),"n":len(pnls)})
    if results:
        wrs=[r["wr"] for r in results];rets=[r["pnl"]/500000*100 for r in results]
        dds=[r["dd"] for r in results];ns=[r["n"] for r in results]
        print(f"  交易{np.mean(ns):.0f} 胜率{np.mean(wrs):.1%}±{np.std(wrs):.1%}")
        print(f"  收益{np.mean(rets):.1f}±{np.std(rets):.1f}% DD{np.mean(dds):.1f}%")
        print(f"  盈利{sum(1 for r in rets if r>0)/len(rets):.0%}")
