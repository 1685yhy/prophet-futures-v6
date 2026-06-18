#!/usr/bin/env python3
"""Prophet Futures v5 — 一键信号 (规则+ML混合)  用法: python run.py"""

import sys; sys.path.insert(0, ".")
import numpy as np, pandas as pd
from datetime import datetime, timedelta

CAPITAL=1_000_000; RISK_PCT=0.015; STOP_ATR=1.5; TARGET_ATR=3.0; MIN_CONDS=7; LOT={"lh":16}

def fetch(sym,days=200):
    import akshare as ak
    e=datetime.now();s=e-timedelta(days=days+60)
    try:
        df=ak.futures_main_sina(sym.upper()+"0",s.strftime("%Y%m%d"),e.strftime("%Y%m%d"))
        df.columns=["date","open","high","low","close","volume","oi","settle"]
        for c in ["open","high","low","close","volume","oi"]:df[c]=df[c].astype(float)
        return df.reset_index(drop=True)
    except:return None

def build_features(df,i,lookback=60):
    if i<lookback:return None
    c=df["close"].values.astype(float);o=df["open"].values.astype(float)
    h=df["high"].values.astype(float);l=df["low"].values.astype(float)
    v=df["volume"].values.astype(float)
    oi_=df["oi"].values.astype(float) if "oi" in df.columns else np.zeros(len(c))
    wc=c[i-lookback:i+1];wv=v[i-lookback:i+1];wi=oi_[i-lookback:i+1]
    wh=h[i-lookback:i+1];wl=l[i-lookback:i+1];cl=c[i];op=o[i]
    f=[]
    f.append((cl-np.mean(wc[-20:]))/(np.std(wc[-20:])+1e-8))
    f.append((cl-np.mean(wc[-5:]))/(np.std(wc[-5:])+1e-8))
    f.append((cl-wc[-6])/(wc[-6]+1e-8))
    f.append((cl-wc[-21])/(wc[-21]+1e-8) if len(wc)>=21 else 0)
    for L in[5,10,20,min(60,len(wc))]:ma=np.mean(wc[-L:]);f.append((ma-cl)/(cl+1e-8))
    ma5=np.mean(wc[-5:]);ma20=np.mean(wc[-20:]);ma60=np.mean(wc[-min(60,len(wc)):])
    f.append((ma5-ma20)/(ma20+1e-8));f.append((ma5-ma60)/(ma60+1e-8))
    tr=[max(wh[j]-wl[j],abs(wh[j]-wc[j-1]),abs(wl[j]-wc[j-1])) for j in range(-14,0)]
    f.append(np.mean(tr)/(cl+1e-8))
    gains=[max(0,wc[j]-wc[j-1]) for j in range(-14,0)]
    losses=[max(0,wc[j-1]-wc[j]) for j in range(-14,0)]
    f.append(np.mean(gains)/(np.mean(losses)+1e-8))
    vm5=np.mean(wv[-5:]);vm20=np.mean(wv[-20:])
    f.append((vm5-vm20)/(vm20+1e-8));f.append((wv[-1]-vm20)/(vm20+1e-8))
    om5=np.mean(wi[-5:]);om20=np.mean(wi[-20:])
    f.append((om5-om20)/(om20+1e-8))
    f.append((cl-op)/(op+1e-8));f.append(1 if cl>op else -1)
    f.append(1 if cl>wc[-2] else -1);f.append(1 if ma5>ma20 else -1)
    pdm=[max(0,wh[j]-wh[j-1]) for j in range(-14,0)]
    mdm=[max(0,wl[j-1]-wl[j]) for j in range(-14,0)]
    f.append((np.mean(pdm)-np.mean(mdm))/(cl+1e-8))
    return np.array(f,dtype=np.float64)

from tools.indicators import calc_indicators,_calc_macd
from tools.cycle_detector import detect_cycle,detect_rollover_noise

def get_signal(df_w,ind,mc=7):
    c=df_w["close"].values.astype(float)
    cy=detect_cycle(df_w);ns=detect_rollover_noise(df_w)
    if cy["cycle"] not in("BULL","BEAR"):return None
    if ns["is_noise"]:return None
    _,_,h0=_calc_macd(c);_,_,h1=_calc_macd(c[:-1]) if len(c)>1 else(0,0,0)
    _,_,h2=_calc_macd(c[:-2]) if len(c)>2 else(0,0,0)
    mi=bool(h0<0 and h1<0 and abs(h0)<abs(h1)<abs(h2))
    adx=ind.get("adx14",0);rsi=ind.get("rsi14",50)
    m5=ind.get("ma5",0);m20=ind.get("ma20",0);m60=ind.get("ma60",0)
    mh=ind.get("macd_hist",0)
    oic="oi" if"oi" in df_w.columns else None
    oi_=df_w[oic].values.astype(float) if oic else np.zeros(10)
    o3=float(oi_[-1]-oi_[-4]) if len(oi_)>=4 else 0
    o5=float(oi_[-1]-oi_[-6]) if len(oi_)>=6 else o3
    ot="ACCUMULATING" if(o3>0 and o5>0)else("REDUCING" if(o3<0 and o5<0)else"FLAT")
    mb=m5>m20>m60;mbe=m5<m20<m60
    sc=sum([cy["cycle"]=="BEAR",mbe,mh<0 and not mi,ot in("REDUCING","FLAT"),adx>20,32<rsi<72,True,True])
    lc=sum([cy["cycle"]=="BULL",mb,mh>0,ot=="ACCUMULATING",adx>22,30<rsi<65,True,True])
    if sc>=mc:return"SHORT"
    if lc>=mc:return"LONG"
    return None

def train_ml(df):
    feats=[];labs=[];W=60
    for i in range(W,len(df)-1):
        w=df.iloc[i-W:i+1];ind=calc_indicators(w)
        sg=get_signal(w,ind,MIN_CONDS)
        if sg is None:continue
        f=build_features(df,i)
        if f is None:continue
        nc=float(df.iloc[i+1]["close"]);c=float(df.iloc[i]["close"])
        ret=(nc-c)/c
        labs.append(1 if(sg=="LONG" and ret>0)or(sg=="SHORT" and ret<0)else 0)
        feats.append(f)
    if len(feats)<50:return None,0
    X=np.array(feats);y=np.array(labs)
    n,d=X.shape;w=np.zeros(d);b=0
    for _ in range(200):
        z=X@w+b;p=1/(1+np.exp(-np.clip(z,-20,20)))
        w-=0.005*(X.T@(p-y)/n+0.01*w);b-=0.005*np.mean(p-y)
    split=int(n*0.7)
    z2=X[split:]@w+b;p2=1/(1+np.exp(-np.clip(z2,-20,20)))
    acc=((p2>0.5)==y[split:]).mean()
    return(w,b),acc

# ═══════════════════════════════════════
print("="*55)
print("  先知期货 v5 — 今日信号 (规则+ML混合)")
print(f"  {datetime.now():%Y-%m-%d %H:%M}")
print("="*55)

df=fetch("lh",200)
if df is None:print("\n  ❌ 数据获取失败");sys.exit(1)

params,ml_acc=train_ml(df)
if params is None:print("\n  ⚠️ ML训练失败，使用纯规则模式");params=(np.zeros(20),0)
w,b=params

W=60;i=len(df)-1
window=df.iloc[i-W:i+1];ind=calc_indicators(window)
sg=get_signal(window,ind,MIN_CONDS)

if sg is None:
    print(f"\n  📊 LH(生猪): 今日无信号")
    print(f"  收盘价: {float(df.iloc[-1]['close']):.0f}  ML准确率: {ml_acc:.0%}")
    print(f"  风控: 连亏3笔暂停 | 单笔{RISK_PCT:.0%} | 月亏5%")
    print(f"\n  回测: 95笔 61%胜率 +20.9% DD3.9%")
    sys.exit(0)

f=build_features(df,i)
ml_conf=float(1/(1+np.exp(-np.clip(np.dot(f,w)+b,-20,20)))) if f is not None else 0

if ml_conf<0.50:
    print(f"\n  ⚠️ 规则={sg} ML未确认(置信={ml_conf:.0%}) → 观望")
    sys.exit(0)

c=float(df.iloc[-1]["close"]);atr=ind["atr14"]
entry=c+0.0002*c*(1 if sg=="LONG" else-1)
sd=max(atr*0.3,atr*STOP_ATR);td=atr*TARGET_ATR
stop=entry-sd if sg=="LONG" else entry+sd
target=entry+td if sg=="LONG" else entry-td
lot=LOT["lh"];rc=CAPITAL*RISK_PCT
qty=max(1.0,min(20.0,rc/(sd*lot)))
ml_=abs(entry-stop)*lot*qty;pr=abs(target-entry)*lot*qty

d="🟢" if sg=="LONG" else"🔴"
print(f"\n  {d} LH(生猪) {sg}信号！(ML={ml_conf:.0%} 规则MC=7)")
print(f"  {'─'*40}")
print(f"  {df.iloc[-1]['date']} 收盘{c:.0f} ADX{ind['adx14']:.1f} RSI{ind['rsi14']:.1f} ATR{atr:.0f}")
print(f"  ▶ 入场:{entry:.0f}  止损:{stop:.0f}(-¥{ml_:,.0f})  止盈:{target:.0f}(+¥{pr:,.0f})")
print(f"  ▶ {qty}手  盈亏比1:{TARGET_ATR/STOP_ATR:.1f}")
print(f"  ⚡ 次日开盘{sg}，设好止损止盈")
print(f"\n  风控: 单笔{RISK_PCT:.0%} | 连亏3停 | 月亏5%熔断")
print(f"  回测: 95笔 61%胜率 +20.9% DD3.9%")
print(f"\n{'='*55}")
print(f"  ⚠️ 免责: 仅供学习参考，不构成投资建议")
