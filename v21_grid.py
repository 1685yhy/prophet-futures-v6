#!/usr/bin/env python3
"""Prophet v21 — Pure XGBoost parameter grid search (no feature tricks)"""
import sys, json, time
import numpy as np, pandas as pd
from datetime import datetime, timedelta
import akshare as ak
import xgboost as xgb

SYMBOLS = {
    'LH': {'code': 'LH0', 'cost': 0.0006, 'rev': False, 'name': '生猪'},
    'JM': {'code': 'JM0', 'cost': 0.0011, 'rev': False, 'name': '焦煤'},
    'RM': {'code': 'RM0', 'cost': 0.0011, 'rev': True,  'name': '菜粕'},
}
N_WF = 500; RR = 3.0; POS_CAP = 2.0; CONF = 0.55

# Expanded param grid
GRID = [
    (100, 4, 0.03, 60), (100, 5, 0.03, 60), (100, 5, 0.05, 60),
    (150, 4, 0.03, 60), (150, 5, 0.03, 60), (150, 6, 0.03, 90),
    (200, 4, 0.03, 60), (200, 4, 0.05, 60), (200, 5, 0.03, 60),
    (200, 5, 0.05, 60), (200, 6, 0.03, 90), (200, 4, 0.03, 90),
    (250, 4, 0.03, 60), (250, 5, 0.03, 60), (250, 4, 0.03, 120),
    (300, 4, 0.03, 60), (300, 5, 0.03, 120), (100, 5, 0.03, 120),
    (100, 5, 0.01, 120), (150, 5, 0.03, 120), (200, 5, 0.01, 120),
]

def fetch(sym_code):
    e=datetime.now();s=e-timedelta(days=2500)
    df=ak.futures_main_sina(symbol=sym_code,start_date=s.strftime('%Y%m%d'),end_date=e.strftime('%Y%m%d'))
    df.columns=['date','open','high','low','close','volume','oi','settle']
    for c in ['open','high','low','close','volume','oi']:
        df[c]=pd.to_numeric(df[c],errors='coerce')
    return df.dropna(subset=['close']).reset_index(drop=True)

def build_feats(df, idx, L=60):
    if idx<L+5: return None
    w=df.iloc[idx-L:idx+1]
    c=w['close'].values;o=w['open'].values;h=w['high'].values;l=w['low'].values
    v=w['volume'].values;oi=w['oi'].values
    f=[]
    if idx>=1: f.append((o[-1]-c[-2])/c[-2]); f.append(abs((o[-1]-c[-2])/c[-2]))
    else: f.extend([0,0])
    for lag in [1,3,5,10,20]:
        f.append((c[-1]-c[-lag-1])/c[-lag-1] if len(c)>lag else 0)
    for p in [5,10,20,60]:
        ma=np.mean(c[-min(p,len(c)):]); f.append((c[-1]-ma)/ma)
    f.append(np.std(c[-20:])/np.mean(c[-20:]))
    f.append((h[-1]-l[-1])/c[-1])
    vma=np.mean(v[-20:]) if np.mean(v[-20:])>0 else 1
    f.append(v[-1]/vma)
    f.append(oi[-1]/np.mean(oi[-20:]) if len(oi)>=20 and np.mean(oi[-20:])>0 else 1)
    f.append((oi[-1]-oi[-6])/np.mean(oi[-20:]) if len(oi)>=20 and np.mean(oi[-20:])>0 else 0)
    ema12=c[-1];ema26=c[-1]
    for i in range(len(c)-2,-1,-1):
        ema12=(2/13)*c[i]+(11/13)*ema12;ema26=(2/27)*c[i]+(25/27)*ema26
    f.append((ema12-ema26)/c[-1])
    d=np.diff(c[-15:]);g=d[d>0].sum() if len(d[d>0])>0 else 0
    lo=abs(d[d<0].sum()) if len(d[d<0])>0 else 1e-10
    f.append(100-100/(1+g/lo) if lo>0 else 50)
    bb=np.std(c[-20:]);ma20=np.mean(c[-20:])
    f.append((c[-1]-ma20)/(2*bb+1e-10))
    ds=str(df.iloc[idx]['date'])
    try: m=int(ds[5:7]); f.append(np.sin(2*np.pi*m/12)); f.append(np.cos(2*np.pi*m/12))
    except: f.extend([0,0])
    f.append(c[-1]/1000.0)
    return np.array(f,dtype=np.float32)

def wf_run(df, n_est, depth, lr, win, cost, rev=False):
    n=len(df);train_size=int(n*0.6)
    if train_size<200: return None
    results=[];pnls=[]
    for run in range(min(N_WF,(n-train_size)//10)):
        split=train_size+run*10
        if split+10>n: break
        train_df=df.iloc[:split];test_start=split;test_end=min(split+10,n)
        X,y=[],[]
        for i in range(win,len(train_df)-1):
            f=build_feats(train_df,i,win)
            if f is None: continue
            y.append(1 if train_df.iloc[i+1]['close']>train_df.iloc[i]['close'] else 0)
            X.append(f)
        if len(X)<100: continue
        y_arr=np.array(y)
        if len(np.unique(y_arr))<2: continue
        model=xgb.XGBClassifier(n_estimators=n_est,max_depth=depth,learning_rate=lr,
                                 use_label_encoder=False,eval_metric='logloss',
                                 verbosity=0,random_state=42)
        model.fit(np.array(X),y_arr)
        for j in range(test_start,test_end-1):
            f=build_feats(df,j,win)
            if f is None: continue
            prob=model.predict_proba(f.reshape(1,-1))[0]
            conf=prob[1] if prob[1]>0.5 else 1-prob[1]
            if conf<CONF: continue
            pred=1 if prob[1]>0.5 else 0
            if rev: pred=1-pred
            entry=df.iloc[j]['close']
            fp=df.iloc[j+1:min(j+15,len(df))]['close'].values
            if len(fp)==0: continue
            atr_vals=[abs(df.iloc[k]['high']-df.iloc[k]['low']) for k in range(max(0,j-20),j+1)]
            atr=np.mean(atr_vals) if atr_vals else entry*0.02
            atr_pct=atr/entry
            if atr_pct<0.01: pos=POS_CAP
            elif atr_pct<0.02: pos=1.5
            elif atr_pct<0.03: pos=1.0
            elif atr_pct<0.05: pos=0.5
            else: continue
            for px in fp:
                if pred==1: pnl=(px-entry)/entry
                else: pnl=(entry-px)/entry
                if pnl>=RR*cost: results.append(1);pnls.append((RR*cost-cost)*pos);break
                elif pnl<=-cost: results.append(0);pnls.append(-cost*pos);break
            else:
                lp=fp[-1]
                if pred==1: pnl=(lp-entry)/entry-cost
                else: pnl=(entry-lp)/entry-cost
                results.append(1 if pnl>0 else 0);pnls.append(pnl*pos)
    if not results: return None
    total=len(results);nw=sum(results);wr=nw/total*100
    tp=sum(pnls)
    cum=np.cumsum(pnls);peak=np.maximum.accumulate(cum)
    dd=np.max((peak-cum)/(peak+1e-10))*100
    gp=sum(p for p,w in zip(pnls,results) if w==1)
    gl=abs(sum(p for p,w in zip(pnls,results) if w==0))
    pf=gp/gl if gl>0 else 999
    test_days=n-train_size;years=test_days/252
    ann_ret=((1+tp)**(1/years)-1)*100 if years>0 and tp>-1 else tp/years*100
    score=wr*np.log1p(total)*pf/max(dd,1)
    return {'trades':total,'wr':round(wr,1),'ann':round(ann_ret,1),'dd':round(dd,1),
            'pf':round(pf,2),'score':round(score,1),'p':(n_est,depth,lr,win)}

print("=== Prophet v21 — Pure XGBoost Grid Search ===")
print(f"  Grid: {len(GRID)} combos  WF: {N_WF}  Conf: >{CONF}  PosCap: {POS_CAP}x")
print()

for sym_key,cfg in SYMBOLS.items():
    print(f"\n{'='*60}\n  {sym_key} {cfg['name']}\n{'='*60}")
    df=fetch(cfg['code'])
    if df is None: continue
    print(f"  Data: {len(df)} rows  Cost: {cfg['cost']*100:.2f}%  Rev: {cfg['rev']}")
    
    best_score=0;best_r=None;all_r=[]
    for i,(n_est,depth,lr,win) in enumerate(GRID):
        start=time.time()
        r=wf_run(df,n_est,depth,lr,win,cfg['cost'],cfg['rev'])
        elapsed=time.time()-start
        if r is None: continue
        all_r.append(r)
        mark=""
        if r['score']>best_score:
            best_score=r['score'];best_r=r;mark=" ★"
        if i%5==0 or mark:
            print(f"  [{i+1}/{len(GRID)}] n={n_est} d={depth} lr={lr:.2f} w={win} → "
                  f"{r['trades']}t {r['wr']}% Ann={r['ann']:+.1f}% DD={r['dd']}% "
                  f"PF={r['pf']} s={r['score']:.0f} {elapsed:.0f}s{mark}")
    
    top3=sorted(all_r,key=lambda x:-x['score'])[:3]
    print(f"\n  TOP 3:")
    for i,r in enumerate(top3):
        n_est,depth,lr,win=r['p']
        print(f"  {i+1}. n={n_est} d={depth} lr={lr:.2f} w={win} → "
              f"{r['trades']}t {r['wr']}% Ann={r['ann']:+.1f}%/yr DD={r['dd']}% PF={r['pf']}")

    print(f"  ★ BEST: n={best_r['p'][0]} d={best_r['p'][1]} lr={best_r['p'][2]:.2f} w={best_r['p'][3]} → "
          f"{best_r['trades']}t {best_r['wr']}% Ann={best_r['ann']:+.1f}%/yr DD={best_r['dd']}% PF={best_r['pf']}")
