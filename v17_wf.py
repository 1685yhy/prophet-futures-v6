#!/usr/bin/env python3
"""v17 base, close-to-close, 1000WF, real costs"""
import numpy as np, pandas as pd, json, time
import akshare as ak, xgboost as xgb

SYM={'LH':('LH0',0.0006,False,200,5,0.05,60,16,3),
     'JM':('JM0',0.0011,False,100,4,0.03,60,60,5),
     'RM':('RM0',0.0011,True,100,5,0.03,120,10,20)}
N_WF=1000;CAP=300000

def fetch(c):
    e=pd.Timestamp.now();s=e-pd.Timedelta(days=2500)
    df=ak.futures_main_sina(symbol=c,start_date=s.strftime('%Y%m%d'),end_date=e.strftime('%Y%m%d'))
    df.columns=['date','open','high','low','close','volume','oi','settle']
    for x in ['open','high','low','close','volume','oi']:df[x]=pd.to_numeric(df[x],errors='coerce')
    return df.dropna(subset=['close']).reset_index(drop=True)

def feats(df,i,L):
    if i<L+5:return None
    w=df.iloc[i-L:i+1];c=w['close'].values;o=w['open'].values
    h=w['high'].values;l=w['low'].values;v=w['volume'].values;oi=w['oi'].values
    f=[]
    if i>=1:f.append((o[-1]-c[-2])/c[-2]);f.append(abs(f[-1]))
    else:f.extend([0,0])
    for lag in[1,3,5,10,20]:f.append((c[-1]-c[-lag-1])/c[-lag-1]if len(c)>lag else 0)
    for p in[5,10,20,60]:ma=np.mean(c[-min(p,len(c)):]);f.append((c[-1]-ma)/ma)
    f.append(np.std(c[-20:])/np.mean(c[-20:]));f.append((h[-1]-l[-1])/c[-1])
    vma=np.mean(v[-20:])if np.mean(v[-20:])>0 else 1;f.append(v[-1]/vma)
    f.append(oi[-1]/np.mean(oi[-20:])if len(oi)>=20 and np.mean(oi[-20:])>0 else 1)
    ema12=c[-1];ema26=c[-1]
    for j in range(len(c)-2,-1,-1):ema12=(2/13)*c[j]+(11/13)*ema12;ema26=(2/27)*c[j]+(25/27)*ema26
    f.append((ema12-ema26)/c[-1])
    dd_=np.diff(c[-15:]);g=dd_[dd_>0].sum()if len(dd_[dd_>0])>0 else 0
    lo=abs(dd_[dd_<0].sum())if len(dd_[dd_<0])>0 else 1e-10;f.append(100-100/(1+g/lo)if lo>0 else 50)
    bb=np.std(c[-20:]);ma20=np.mean(c[-20:]);f.append((c[-1]-ma20)/(2*bb+1e-10))
    f.append(c[-1]/1000.0)
    return np.array(f,dtype=np.float32)

tot_t=0;tot_w=0;tot_pnl=0
for sk,(sc,cost,rev,ne,d,lr,win,mult,base) in SYM.items():
    print(f'{sk}...',end='',flush=True)
    df=fetch(sc);n=len(df);ts=int(n*0.6)
    pnl=[];wins=0;total=0
    for run in range(min(N_WF,(n-ts)//10)):
        sp=ts+run*10
        if sp+10>n:break
        td=df.iloc[:sp];t1=sp;t2=min(sp+10,n)
        X,y=[],[]
        for i in range(win,len(td)-1):
            f=feats(td,i,win)
            if f is None:continue
            X.append(f);y.append(1 if td.iloc[i+1]['close']>td.iloc[i]['close'] else 0)
        if len(X)<100:continue
        ya=np.array(y)
        if len(np.unique(ya))<2:continue
        m=xgb.XGBClassifier(n_estimators=ne,max_depth=d,learning_rate=lr,verbosity=0,random_state=42)
        m.fit(np.array(X),ya)
        for j in range(t1,t2-1):
            f=feats(df,j,win)
            if f is None:continue
            p=m.predict_proba(f.reshape(1,-1))[0]
            pred=1 if p[1]>0.5 else 0
            if rev:pred=1-pred
            o=df.iloc[j]['close'];nc=df.iloc[j+1]['close']
            chg=nc-o if pred==1 else o-nc
            trade_pnl=chg*mult*base-cost*o*mult*base
            pnl.append(trade_pnl);wins+=(1 if trade_pnl>0 else 0);total+=1
    if total==0:print('FAIL');continue
    tp=sum(pnl);wr=wins/total*100
    cum=np.cumsum(pnl);pk=np.maximum.accumulate(cum)
    dd_val=np.max((pk-cum)/(pk+1e-10))*100 if pk[-1]>0 else 0
    gp=sum(p for p in pnl if p>0);gl=abs(sum(p for p in pnl if p<0))
    pf=gp/gl if gl>0 else 999
    yrs=(n-ts)/252;ar=((CAP+tp)/CAP)**(1/yrs)-1 if tp>-CAP and yrs>0 else tp/CAP/yrs
    tot_t+=total;tot_w+=total*wr/100;tot_pnl+=tp
    print(f' {total}t WR={wr:.1f}% PnL={tp/10000:+.1f}万 AR={ar*100:+.1f}%/yr DD={dd_val:.0f}% PF={pf:.2f}')

avg_wr=tot_w/tot_t*100 if tot_t>0 else 0
print(f'\nTotal: {tot_t}t WR={avg_wr:.1f}% PnL={tot_pnl/10000:+.1f}万')
