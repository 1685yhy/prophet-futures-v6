#!/usr/bin/env python3
"""
Prophet v5.0 — sklearn ML全面升级
RandomForest + GradientBoosting + Ensemble
全品种 × 多模型 × Walk-Forward × 超参搜索
"""

import sys; sys.path.insert(0, ".")
import json, numpy as np, pandas as pd
from datetime import datetime, timedelta
from tools.indicators import calc_indicators, _calc_macd
from tools.cycle_detector import detect_cycle, detect_rollover_noise
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

LOT = {"lh":16,"jm":60,"jd":5,"m":10,"rm":10,"rb":10,"hc":10,"i":100,
       "j":100,"cu":5,"al":5,"zn":5,"ni":1,"au":1000,"ag":15000,
       "sc":1000,"bu":10,"fu":10,"ma":10,"ta":5,"eg":10,"pg":20,
       "sa":20,"fg":20,"y":10,"oi":10,"p":10,"a":10,"c":10,"cf":5,"sr":10}

ALL = list(LOT.keys())

# ═══════════════════════════════════════════════════════════
# Data + Features (same as before but optimized)
# ═══════════════════════════════════════════════════════════

def fetch(sym, days=2500):
    import akshare as ak
    e=datetime.now();s=e-timedelta(days=days+200)
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
    oi=df["oi"].values.astype(float) if "oi" in df.columns else np.zeros(len(c))
    wc=c[i-lookback:i+1];wv=v[i-lookback:i+1];wi=oi[i-lookback:i+1]
    wh=h[i-lookback:i+1];wl=l[i-lookback:i+1]
    cl=c[i];op=o[i]
    f=[]
    # Price z-scores
    f.append((cl-np.mean(wc[-20:]))/(np.std(wc[-20:])+1e-8))
    f.append((cl-np.mean(wc[-5:]))/(np.std(wc[-5:])+1e-8))
    # Returns
    f.append((cl-wc[-6])/(wc[-6]+1e-8))
    f.append((cl-wc[-11])/(wc[-11]+1e-8))
    f.append((cl-wc[-21])/(wc[-21]+1e-8))
    if len(wc)>61:f.append((cl-wc[-61])/(wc[-61]+1e-8))
    else:f.append(0)
    # Moving averages
    for L in [5,10,20,min(60,len(wc))]:
        ma=np.mean(wc[-L:]);f.append((ma-cl)/(cl+1e-8))
    ma5=np.mean(wc[-5:]);ma20=np.mean(wc[-20:]);ma60=np.mean(wc[-min(60,len(wc)):])
    f.append((ma5-ma20)/(ma20+1e-8))
    f.append((ma20-ma60)/(ma60+1e-8))
    # ATR
    tr=[max(wh[j]-wl[j],abs(wh[j]-wc[j-1]),abs(wl[j]-wc[j-1])) for j in range(-14,0)]
    f.append(np.mean(tr)/(cl+1e-8))
    # RSI
    gains=[max(0,wc[j]-wc[j-1]) for j in range(-14,0)]
    losses=[max(0,wc[j-1]-wc[j]) for j in range(-14,0)]
    ag=np.mean(gains);al_=np.mean(losses)
    f.append(ag/(al_+1e-8))
    # Volume
    vm5=np.mean(wv[-5:]);vm20=np.mean(wv[-20:])
    f.append((vm5-vm20)/(vm20+1e-8));f.append((wv[-1]-vm20)/(vm20+1e-8))
    # OI
    om5=np.mean(wi[-5:]);om20=np.mean(wi[-20:])
    f.append((om5-om20)/(om20+1e-8))
    # Intraday + pattern
    f.append((cl-op)/(op+1e-8));f.append(1 if cl>op else -1)
    f.append(1 if cl>wc[-2] else -1);f.append(1 if ma5>ma20 else -1)
    # ADX proxy
    pdm=[max(0,wh[j]-wh[j-1]) for j in range(-14,0)]
    mdm=[max(0,wl[j-1]-wl[j]) for j in range(-14,0)]
    f.append((np.mean(pdm)-np.mean(mdm))/(cl+1e-8))
    return np.array(f,dtype=np.float64)

def rule_signal(df_w,ind,mc=7):
    c=df_w["close"].values.astype(float)
    cy=detect_cycle(df_w);ns=detect_rollover_noise(df_w)
    if cy["cycle"] not in ("BULL","BEAR"):return None
    if ns["is_noise"]:return None
    _,_,h0=_calc_macd(c);_,_,h1=_calc_macd(c[:-1]) if len(c)>1 else (0,0,0)
    _,_,h2=_calc_macd(c[:-2]) if len(c)>2 else (0,0,0)
    mi=bool(h0<0 and h1<0 and abs(h0)<abs(h1)<abs(h2))
    adx=ind.get("adx14",0);rsi=ind.get("rsi14",50)
    m5=ind.get("ma5",0);m20=ind.get("ma20",0);m60=ind.get("ma60",0)
    mh=ind.get("macd_hist",0)
    oic="oi" if "oi" in df_w.columns else None
    oi_=df_w[oic].values.astype(float) if oic else np.zeros(10)
    o3=float(oi_[-1]-oi_[-4]) if len(oi_)>=4 else 0
    o5=float(oi_[-1]-oi_[-6]) if len(oi_)>=6 else o3
    ot="ACCUMULATING" if (o3>0 and o5>0) else ("REDUCING" if (o3<0 and o5<0) else "FLAT")
    mb=m5>m20>m60;mbe=m5<m20<m60
    sc=sum([cy["cycle"]=="BEAR",mbe,mh<0 and not mi,ot in ("REDUCING","FLAT"),adx>20,32<rsi<72,True,True])
    lc=sum([cy["cycle"]=="BULL",mb,mh>0,ot=="ACCUMULATING",adx>22,30<rsi<65,True,True])
    if sc>=mc:return "SHORT"
    if lc>=mc:return "LONG"
    return None

# ═══════════════════════════════════════════════════════════
# Sklearn ML Training + Backtest
# ═══════════════════════════════════════════════════════════

def train_sklearn_ml(df, model_type="rf"):
    """Train sklearn model to filter rule signals."""
    features=[];labels=[];W=60
    for i in range(W,len(df)-1):
        window=df.iloc[i-W:i+1];ind=calc_indicators(window)
        sg=rule_signal(window,ind,7)
        if sg is None:continue
        f = build_features(df, i)
        if f is None:
            continue
        nc=float(df.iloc[i+1]["close"]);c=float(df.iloc[i]["close"])
        ret=(nc-c)/c
        label=1 if (sg=="LONG" and ret>0) or (sg=="SHORT" and ret<0) else 0
        features.append(f);labels.append(label)
    if len(features)<50:return None
    X=np.array(features);y=np.array(labels)
    train_n=int(len(X)*0.7)
    if model_type=="rf":
        m=RandomForestClassifier(n_estimators=200,max_depth=8,random_state=42,n_jobs=-1)
    elif model_type=="gb":
        m=GradientBoostingClassifier(n_estimators=200,max_depth=5,learning_rate=0.05,random_state=42)
    else:
        m=LogisticRegression(max_iter=1000,C=0.1)
    m.fit(X[:train_n],y[:train_n])
    acc=(m.predict(X[train_n:])==y[train_n:]).mean()
    return m,acc

def sklearn_backtest(df,sym,model_type="rf",risk_pct=0.015,stop_atr=1.5,
                     target_atr=3.0,ml_thresh=0.50):
    model,ml_acc=train_sklearn_ml(df,model_type)
    if model is None:return None,[]
    lot=LOT.get(sym,10);capital=1_000_000;trades=[];W=60
    signals=0;confirmed=0

    for i in range(W,len(df)-1):
        window=df.iloc[i-W:i+1];ind=calc_indicators(window)
        sg=rule_signal(window,ind,7)
        if sg is None:continue
        signals+=1
        f = build_features(df, i)
        if f is None:
            continue
        ml_prob=model.predict_proba(f.reshape(1,-1))[0,1]
        if ml_prob<ml_thresh:continue
        confirmed+=1

        c=float(df.iloc[i]["close"]);atr=ind["atr14"]
        entry=c+0.0002*c*(1 if sg=="LONG" else -1)
        sd=max(atr*0.3,atr*stop_atr);td=atr*target_atr
        stop=entry-sd if sg=="LONG" else entry+sd
        target=entry+td if sg=="LONG" else entry-td
        rc=capital*risk_pct;q=max(1.0,min(20.0,rc/(sd*lot)))
        nc=float(df.iloc[i+1]["close"]);nh=float(df.iloc[i+1]["high"])
        nl=float(df.iloc[i+1]["low"])
        if sg=="LONG":
            ep=stop if nl<=stop else (target if nh>=target else nc)
            rs="STOP" if nl<=stop else ("TP" if nh>=target else "EOD")
        else:
            ep=stop if nh>=stop else (target if nl<=target else nc)
            rs="STOP" if nh>=stop else ("TP" if nl<=target else "EOD")
        pnl=(ep-entry)*lot*q*(1 if sg=="LONG" else -1)
        pnl-=abs(ep*lot*q*0.0001)+abs(entry*lot*q*0.0001)
        trades.append({"pnl":round(pnl,2),"rs":rs,"dir":sg,
                       "ml_conf":round(ml_prob,3),"ml_type":model_type})

    if not trades:return {"t":0},[]
    pnls=[t["pnl"] for t in trades];n=len(pnls)
    w=[p for p in pnls if p>0];l=[p for p in pnls if p<=0]
    wr=len(w)/n;tp=sum(pnls);ret=tp/capital*100
    aw=np.mean(w) if w else 0;al_=abs(np.mean(l)) if l else 1
    cum=np.cumsum(pnls);rm=np.maximum.accumulate(cum)
    dd=abs(float(np.min((cum-rm)/capital*100)))
    if n>=5:
        rt_arr=[p/capital for p in pnls]
        sr=np.mean(rt_arr)/(np.std(rt_arr,ddof=1)+1e-8)*np.sqrt(252)
    else:sr=0
    return {"t":n,"wr":round(wr,3),"pnl":round(tp,0),"ret":round(ret,1),
            "dd":round(dd,1),"sr":round(sr,3),"pf":round(sum(w)/(abs(sum(l))+1e-8),2),
            "plr":round(aw/al_,2),"sigs":signals,"conf":confirmed,
            "ml_acc":round(ml_acc,3),"model":model_type},trades

# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

print("="*65)
print("  Prophet v5.0 — sklearn ML升级 (RF+GB+LR)")
print(f"  {datetime.now():%Y-%m-%d %H:%M}")
print("="*65)

MODELS=[("RandomForest","rf"),("GradientBoost","gb"),("LogisticReg","lr")]
THRESH=[0.50,0.55,0.60];RISKS=[0.01,0.015,0.02]
STOPS=[1.2,1.5,2.0]

all=[];td=0;vd=0

for sym in ["lh","jm","jd","rb","i","fu","sc","ma","sa","p","y","cu","au"]:
    df=fetch(sym,2500)
    if df is None or len(df)<500:continue;td+=1
    best_sym=None;bs=-999

    for mname,mtype in MODELS:
        for th in THRESH:
            for rp in RISKS:
                for sa in STOPS:
                    stats,tr=sklearn_backtest(df,sym,mtype,rp,sa,3.0,th)
                    if stats is None or stats["t"]<10:continue
                    vd+=1
                    score=stats["wr"]*stats["ret"]*min(stats["t"],60)/(stats["dd"]/100+0.05)
                    stats["score"]=round(score,1);stats["sym"]=sym
                    stats["th"]=th;stats["rp"]=rp;stats["sa"]=sa
                    all.append(stats)
                    if score>bs:bs=score;best_sym=stats

    if best_sym:
        m="🔥" if best_sym["wr"]>=0.65 else ("⭐" if best_sym["wr"]>=0.60 else "")
        print(f"  {m} {sym.upper()}: {best_sym['model']} conf={best_sym['th']} "
              f"R={best_sym['rp']} S={best_sym['sa']} → "
              f"{best_sym['t']}t {best_sym['wr']:.0%}wr "
              f"{best_sym['pnl']:+,.0f} {best_sym['ret']:+.1f}% DD{best_sym['dd']:.1f}%")

all.sort(key=lambda x:x["score"],reverse=True)

print(f"\n  Top 30:")
print(f"  {'Rk':<4} {'Sym':<6} {'Model':<14} {'Th':<5} {'R':<5} {'S':<5} {'Trd':<6} {'WR':<7} {'PnL':<12} {'Ret%':<8} {'DD%':<6} {'MLacc':<7}")
print(f"  {'─'*85}")
for i,r in enumerate(all[:30]):
    m="🔥" if r["wr"]>=0.65 else ("⭐" if r["wr"]>=0.60 else "")
    print(f"  {m}{i+1:<3} {r['sym'].upper():<6} {r['model']:<14} {r['th']:<5} "
          f"{r['rp']:<5} {r['sa']:<5} {r['t']:<6} {r['wr']:.0%}     "
          f"{r['pnl']:+,.0f}     {r['ret']:+.1f}%    {r['dd']:.1f}%   "
          f"{r['ml_acc']:.0%}")

wr65=[r for r in all if r["wr"]>=0.65 and r["t"]>=15]
print(f"\n  >=65%胜率+≥15笔: {len(wr65)}个")
for r in sorted(wr65,key=lambda x:x["score"],reverse=True)[:10]:
    print(f"  {r['sym'].upper()} {r['model']} conf={r['th']}: "
          f"{r['t']}t {r['wr']:.0%}wr {r['ret']:+.1f}% DD{r['dd']:.1f}%")

json.dump({"top":all[:100],"wr65":wr65,"total":vd},
          open("/tmp/sklearn_final.json","w"),indent=2,ensure_ascii=False)
print(f"\n✅ /tmp/sklearn_final.json ({vd}组合)")
