#!/usr/bin/env python3
"""Prophet v20 — Multi-TF features + 3D label + Feature pruning + 500 WF"""
import sys, json, time, joblib
import numpy as np, pandas as pd
from datetime import datetime, timedelta
import akshare as ak
import xgboost as xgb

SYMBOLS = {
    'LH': {'code': 'LH0', 'name': '生猪', 'cost': 0.0006, 'rev': False, 'has_night': False},
    'JM': {'code': 'JM0', 'name': '焦煤', 'cost': 0.0011, 'rev': False, 'has_night': True},
    'RM': {'code': 'RM0', 'name': '菜粕', 'cost': 0.0011, 'rev': True,  'has_night': True},
}
N_WF = 500
RR = 3.0
POS_CAP = 2.0
LABEL_FWD = 3  # 3-day forward label (was 1)
LABEL_THR = 0.003  # 0.3% threshold for meaningful move
CONF_THRESH = 0.55
TOP_FEATURES = 18  # Keep top N features by importance

def fetch(sym_code):
    e=datetime.now();s=e-timedelta(days=2500)
    df=ak.futures_main_sina(symbol=sym_code,start_date=s.strftime('%Y%m%d'),end_date=e.strftime('%Y%m%d'))
    df.columns=['date','open','high','low','close','volume','oi','settle']
    for c in ['open','high','low','close','volume','oi']:
        df[c]=pd.to_numeric(df[c],errors='coerce')
    return df.dropna(subset=['close']).reset_index(drop=True)

def build_features_v20(df, idx, cfg, L=60):
    """Multi-timeframe features: daily + weekly + monthly"""
    if idx < max(L, 60) + 5: return None
    w = df.iloc[idx-L:idx+1]
    c=w['close'].values;o=w['open'].values;h=w['high'].values;l=w['low'].values
    v=w['volume'].values;oi_vals=w['oi'].values
    
    f = []
    
    # === Session (3) ===
    f.append((o[-1]-c[-2])/c[-2] if idx>=1 else 0)
    f.append(abs((o[-1]-c[-2])/c[-2]) if idx>=1 else 0)
    f.append(1.0 if cfg['has_night'] else 0.0)
    
    # === Daily returns (5) ===
    for lag in [1,3,5,10,20]:
        f.append((c[-1]-c[-lag-1])/c[-lag-1] if len(c)>lag else 0)
    
    # === Weekly returns (3) ===
    for lag in [5,10,20]:
        f.append((c[-1]-c[-lag-1])/c[-lag-1] if len(c)>lag else 0)
    
    # === MA divergences: daily (4) + weekly (2) ===
    for p in [5,10,20,60]:
        ma=np.mean(c[-min(p,len(c)):])
        f.append((c[-1]-ma)/ma)
    for p in [20,60]:
        ma_w=np.mean(c[-min(p,len(c)):])
        f.append((c[-1]-ma_w)/ma_w)
    
    # === MA crosses (2) ===
    ma5=np.mean(c[-5:]);ma20=np.mean(c[-20:])
    f.append((ma5-ma20)/ma20)
    ma10=np.mean(c[-10:]);ma60=np.mean(c[-min(60,len(c)):])
    f.append((ma10-ma60)/ma60)
    
    # === Volatility: daily (2) + weekly (1) ===
    f.append(np.std(c[-20:])/np.mean(c[-20:]))
    f.append((h[-1]-l[-1])/c[-1])
    f.append(np.std(c[-min(60,len(c)):])/np.mean(c[-min(60,len(c)):]))
    
    # === Volume (3) ===
    vma=np.mean(v[-20:]) if np.mean(v[-20:])>0 else 1
    f.append(v[-1]/vma)
    vma5=np.mean(v[-5:]) if np.mean(v[-5:])>0 else 1
    f.append(vma5/vma)
    f.append(oi_vals[-1]/np.mean(oi_vals[-20:]) if len(oi_vals)>=20 and np.mean(oi_vals[-20:])>0 else 1)
    
    # === OI changes (2) ===
    oima20=np.mean(oi_vals[-20:]) if len(oi_vals)>=20 else 1
    f.append((oi_vals[-1]-oi_vals[-6])/oima20 if oima20>0 else 0)
    f.append((oi_vals[-1]-oi_vals[-21])/oima20 if oima20>0 else 0)
    
    # === MACD + RSI (2) ===
    ema12=c[-1];ema26=c[-1]
    for i in range(len(c)-2,-1,-1):
        ema12=(2/13)*c[i]+(11/13)*ema12;ema26=(2/27)*c[i]+(25/27)*ema26
    f.append((ema12-ema26)/c[-1])
    d=np.diff(c[-15:]);g=d[d>0].sum() if len(d[d>0])>0 else 0
    lo=abs(d[d<0].sum()) if len(d[d<0])>0 else 1e-10
    f.append(100-100/(1+g/lo) if lo>0 else 50)
    
    # === Bollinger (2) ===
    bb=np.std(c[-20:]);ma20=np.mean(c[-20:])
    f.append((c[-1]-ma20)/(2*bb+1e-10))
    f.append(bb/ma20)  # BB width
    
    # === Seasonality (3) ===
    ds=str(df.iloc[idx]['date'])
    try:
        m=int(ds[5:7])
        f.append(np.sin(2*np.pi*m/12))
        f.append(np.cos(2*np.pi*m/12))
        f.append(np.sin(4*np.pi*m/12))
    except:
        f.extend([0,0,0])
    
    # === Price level (2) ===
    f.append(c[-1]/1000.0)
    f.append(c[-1]/c[-min(60,len(c))] if len(c)>=60 else 1.0)
    
    return np.array(f, dtype=np.float32)

def get_label(df, idx, cfg):
    """3-day trend label with threshold"""
    if idx+LABEL_FWD >= len(df): return None
    ret = (df.iloc[idx+LABEL_FWD]['close'] - df.iloc[idx]['close']) / df.iloc[idx]['close']
    if ret > LABEL_THR: return 1
    if ret < -LABEL_THR: return 0
    return None  # Skip flat days

def train_and_prune(X, y, n_est=200, depth=5, lr=0.05):
    """Train XGBoost and keep only top features"""
    model = xgb.XGBClassifier(n_estimators=n_est, max_depth=depth, learning_rate=lr,
                               use_label_encoder=False, eval_metric='logloss',
                               verbosity=0, random_state=42)
    model.fit(np.array(X), np.array(y))
    
    # Get feature importance
    imp = model.feature_importances_
    top_idx = np.argsort(imp)[-TOP_FEATURES:]
    
    # Retrain with top features
    X_top = np.array(X)[:, top_idx]
    model2 = xgb.XGBClassifier(n_estimators=n_est, max_depth=depth, learning_rate=lr,
                                use_label_encoder=False, eval_metric='logloss',
                                verbosity=0, random_state=42)
    model2.fit(X_top, np.array(y))
    
    return model2, top_idx

def wf_backtest(df, cfg):
    """500 WF with feature-pruned XGBoost + 3D label"""
    cost=cfg['cost'];rev=cfg['rev']
    n=len(df);train_size=int(n*0.6)
    if train_size<200: return None
    
    results=[];pnls=[]
    win=60  # Fixed window for stability
    
    for run in range(min(N_WF,(n-train_size)//10)):
        split=train_size+run*10
        if split+10>n: break
        train_df=df.iloc[:split];test_start=split;test_end=min(split+10,n)
        
        X,y=[],[]
        for i in range(win,min(len(train_df)-LABEL_FWD,test_start)):
            f=build_features_v20(train_df,i,cfg,win)
            if f is None: continue
            lab=get_label(train_df,i,cfg)
            if lab is None: continue
            X.append(f);y.append(lab)
        
        if len(X)<100: continue
        y_arr=np.array(y)
        if len(np.unique(y_arr))<2: continue
        
        # Train with feature pruning
        model,top_idx=train_and_prune(X,y)
        
        for j in range(test_start,test_end-LABEL_FWD-1):
            f_full=build_features_v20(df,j,cfg,win)
            if f_full is None: continue
            f=f_full[top_idx]
            prob=model.predict_proba(f.reshape(1,-1))[0]
            conf=prob[1] if prob[1]>0.5 else 1-prob[1]
            if conf<CONF_THRESH: continue
            
            pred=1 if prob[1]>0.5 else 0
            if rev: pred=1-pred
            
            entry=df.iloc[j]['close']
            fp=df.iloc[j+1:min(j+15,len(df))]['close'].values
            if len(fp)==0: continue
            
            # Position sizing
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
    dd=np.max((peak-cum)/(peak+1e-10))*100 if len(cum)>0 else 0
    gp=sum(p for p,w in zip(pnls,results) if w==1)
    gl=abs(sum(p for p,w in zip(pnls,results) if w==0))
    pf=gp/gl if gl>0 else 999
    test_days=n-train_size;years=test_days/252
    ann_ret=((1+tp)**(1/years)-1)*100 if years>0 and tp>-1 else tp/years*100
    return {'trades':total,'wr':round(wr,1),'cum_ret':round(tp*100,1),
            'ann_ret':round(ann_ret,1),'dd':round(dd,1),'pf':round(pf,2)}

print(f"=== Prophet v20 — Multi-TF + 3D Label + Feature Pruning ===")
print(f"  Features: 38-dim (daily+weekly+monthly) → pruned to {TOP_FEATURES}")
print(f"  Label: {LABEL_FWD}d forward >{LABEL_THR*100:.1f}%")
print(f"  WF: {N_WF}  Conf: >{CONF_THRESH}  PosCap: {POS_CAP}x")
print()

final={}
for sym_key,cfg in SYMBOLS.items():
    print(f"\n{'='*60}\n  {sym_key} {cfg['name']} (Rev={cfg['rev']})\n{'='*60}")
    df=fetch(cfg['code'])
    if df is None or len(df)<300: continue
    print(f"  Data: {len(df)} rows  Cost: {cfg['cost']*100:.2f}%")
    
    start=time.time()
    r=wf_backtest(df,cfg)
    elapsed=time.time()-start
    
    if r:
        final[sym_key]=r
        print(f"  Result: {r['trades']}t {r['wr']}% Cum={r['cum_ret']:+.1f}% "
              f"Ann={r['ann_ret']:+.1f}%/yr DD={r['dd']}% PF={r['pf']} ({elapsed:.0f}s)")

print(f"\n{'='*60}\n  V20 vs V17 COMPARISON\n{'='*60}")
print(f"  {'Sym':<4} {'Trades':<6} {'WR':<7} {'Ann%/yr':<9} {'DD%':<7} {'PF':<6} {'v17 Ann':<9} {'v17 DD'}")
print(f"  {'-'*70}")
v17={'LH':(416,49.5,24.4,97.7,2.01),'JM':(528,48.5,63.4,100,1.88),'RM':(264,46.6,33.9,155.6,1.74)}
for sym,r in final.items():
    v=v17.get(sym,('','','','',''))
    print(f"  {sym:<4} {r['trades']:<6} {r['wr']:<6.1f}% {r['ann_ret']:<+8.1f}% {r['dd']:<6.1f}% {r['pf']:<5.2f} "
          f"{v[2]:<+8.1f}% {v[3]:<5.1f}%")
