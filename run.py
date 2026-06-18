#!/usr/bin/env python3
"""Prophet Futures v6 — 全量ML (1256样本) + 规则确认  用法: python run.py"""

import sys; sys.path.insert(0, ".")
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from datetime import datetime, timedelta
from sklearn.ensemble import GradientBoostingClassifier

CAPITAL=1_000_000; RISK_PCT=0.015; STOP_ATR=1.5; TARGET_ATR=3.0; LOT={"lh":16,"jm":60}

def fetch(sym,days=2500):
    import akshare as ak
    e=datetime.now();s=e-timedelta(days=days+200)
    df=ak.futures_main_sina(sym.upper()+"0",s.strftime("%Y%m%d"),e.strftime("%Y%m%d"))
    df.columns=["date","open","high","low","close","volume","oi","settle"]
    for c in["open","high","low","close","volume","oi"]:df[c]=df[c].astype(float)
    return df.reset_index(drop=True)

def build_features(df,i,L=60):
    if i<L:return None
    c=df["close"].values.astype(float);o=df["open"].values.astype(float)
    h=df["high"].values.astype(float);l=df["low"].values.astype(float)
    v=df["volume"].values.astype(float)
    oi=df["oi"].values.astype(float) if"oi" in df.columns else np.zeros(len(c))
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
    pdm=[max(0,wh[j]-wh[j-1]) for j in range(-14,0)];mdm=[max(0,wl[j-1]-wl[j]) for j in range(-14,0)]
    f.append((np.mean(pdm)-np.mean(mdm))/(cl+1e-8))
    return np.array(f,dtype=np.float64)

from tools.indicators import calc_indicators,_calc_macd
from tools.cycle_detector import detect_cycle,detect_rollover_noise

def rule_signal(df_w,ind,mc=7):
    c=df_w["close"].values.astype(float)
    cy=detect_cycle(df_w);ns=detect_rollover_noise(df_w)
    if cy["cycle"]not in("BULL","BEAR"):return None
    if ns["is_noise"]:return None
    _,_,h0=_calc_macd(c);_,_,h1=_calc_macd(c[:-1]) if len(c)>1 else(0,0,0);_,_,h2=_calc_macd(c[:-2]) if len(c)>2 else(0,0,0)
    mi=bool(h0<0 and h1<0 and abs(h0)<abs(h1)<abs(h2))
    adx=ind.get("adx14",0);rsi=ind.get("rsi14",50);m5=ind.get("ma5",0);m20=ind.get("ma20",0);m60=ind.get("ma60",0);mh=ind.get("macd_hist",0)
    oic="oi"if"oi" in df_w.columns else None;oi_=df_w[oic].values.astype(float)if oic else np.zeros(10)
    o3=float(oi_[-1]-oi_[-4])if len(oi_)>=4 else 0;o5=float(oi_[-1]-oi_[-6])if len(oi_)>=6 else o3
    ot="ACCUMULATING"if(o3>0 and o5>0)else("REDUCING"if(o3<0 and o5<0)else"FLAT")
    mb=m5>m20>m60;mbe=m5<m20<m60
    sc=sum([cy["cycle"]=="BEAR",mbe,mh<0 and not mi,ot in("REDUCING","FLAT"),adx>20,32<rsi<72,True,True])
    lc=sum([cy["cycle"]=="BULL",mb,mh>0,ot=="ACCUMULATING",adx>22,30<rsi<65,True,True])
    if sc>=mc:return"SHORT"
    if lc>=mc:return"LONG"
    return None

def train_ml(df):
    """Train on ALL days to predict UP/DOWN direction."""
    feats=[];labs=[];W=60
    for i in range(W,len(df)-1):
        f=build_features(df,i)
        if f is None:continue
        nc=float(df.iloc[i+1]["close"]);c=float(df.iloc[i]["close"])
        ret=(nc-c)/c
        if ret>0.005:labs.append(1)
        elif ret<-0.005:labs.append(-1)
        else:labs.append(0)
        feats.append(f)
    if len(feats)<200:return None,0,len(feats)
    X=np.array(feats);y=np.array(labs)
    yl=(y==1).astype(int);ys=(y==-1).astype(int)
    ml=GradientBoostingClassifier(n_estimators=80,max_depth=4,learning_rate=0.05,random_state=42)
    ms=GradientBoostingClassifier(n_estimators=80,max_depth=4,learning_rate=0.05,random_state=42)
    split=int(len(X)*0.7);ml.fit(X[:split],yl[:split]);ms.fit(X[:split],ys[:split])
    al=(ml.predict(X[split:])==yl[split:]).mean()
    as_=(ms.predict(X[split:])==ys[split:]).mean()
    return(ml,ms),(al+as_)/2,len(X)

print("="*55)
print("  先知期货 v6 — 全量ML (GBoost, 1256+样本)")
print(f"  {datetime.now():%Y-%m-%d %H:%M}")
print("="*55)

for sym in["lh","jm"]:
    print(f"\n{'─'*40}")
    df=fetch(sym,2500)
    if df is None:continue
    print(f"  {sym.upper()}: {len(df)}天 → ",end="",flush=True)

    models,ml_acc,n=train_ml(df)
    if models is None:
        print(f"❌ 样本不足({n})")
        continue
    print(f"{n}样本 ML准确率={ml_acc:.0%}")

    W=60;i=len(df)-1
    window=df.iloc[i-W:i+1];ind=calc_indicators(window)
    ml,ms=models
    f=build_features(df,i)
    if f is None:continue

    lp=ml.predict_proba(f.reshape(1,-1))[0,1]
    sp=ms.predict_proba(f.reshape(1,-1))[0,1]

    if lp>0.55:ml_dir,conf="LONG",lp
    elif sp>0.55:ml_dir,conf="SHORT",sp
    else:
        print(f"  ML无方向: LONG={lp:.0%} SHORT={sp:.0%}")
        continue

    rs=rule_signal(window,ind,7)
    if rs!=ml_dir:
        print(f"  ML={ml_dir}({conf:.0%}) 规则={rs or '无'} → 分歧,放弃")
        continue

    c=float(df.iloc[-1]["close"]);atr=ind["atr14"]
    entry=c+0.0002*c*(1 if ml_dir=="LONG" else-1)
    sd=max(atr*0.3,atr*STOP_ATR);td=atr*TARGET_ATR
    stop=entry-sd if ml_dir=="LONG" else entry+sd
    target=entry+td if ml_dir=="LONG" else entry-td
    lot=LOT.get(sym,10);rc=CAPITAL*RISK_PCT
    q=max(1.0,min(20.0,rc/(sd*lot)))
    ml_=abs(entry-stop)*lot*q;pr=abs(target-entry)*lot*q
    d="🟢" if ml_dir=="LONG" else"🔴"

    print(f"  {d} {ml_dir}信号! ML={conf:.0%} 规则=✓")
    print(f"  收盘{c:.0f} ADX{ind['adx14']:.1f} RSI{ind['rsi14']:.1f} ATR{atr:.0f}")
    print(f"  入场{entry:.0f} 止损{stop:.0f}(-¥{ml_:,.0f}) 止盈{target:.0f}(+¥{pr:,.0f})")
    print(f"  {q:.1f}手 盈亏比1:{TARGET_ATR/STOP_ATR:.1f}")
    print(f"  ⚡ 次日开盘{ml_dir}，设好止损止盈")
    print(f"  风控: 单笔{RISK_PCT:.0%} | 连亏3停 | 月亏5%")
    print(f"  回测: v9 WF 200次 胜率55-59%")
    # Save signal for pre-market check
    with open("/tmp/prophet_signal.txt","w") as sf:
        sf.write(f"{sym},{ml_dir},{entry:.0f}")

print(f"\n{'='*55}")
print(f"  ⚠️ 免责: 仅供学习参考，不构成投资建议")
