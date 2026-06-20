#!/usr/bin/env python3
"""Fast A1: precompute features, full WF, OHLC simulation, save JSON"""
import numpy as np, pandas as pd, json, time
import akshare as ak, xgboost as xgb

daily = ak.futures_main_sina(symbol='LH0', start_date='20210101', end_date='20260620')
daily.columns = ['date','open','high','low','close','volume','oi','settle']
for x in ['open','high','low','close','volume','oi']:
    daily[x] = pd.to_numeric(daily[x], errors='coerce')
daily = daily.dropna(subset=['close']).reset_index(drop=True)

win=60;t0=time.time();n=len(daily)
features=np.zeros((n,20),dtype=np.float32);valid=np.zeros(n,dtype=bool)

for idx in range(win+5,n):
    w=daily.iloc[idx-win:idx+1];c=w['close'].values;o=w['open'].values
    h=w['high'].values;l=w['low'].values;v=w['volume'].values;oi=w['oi'].values
    f=features[idx]
    f[0]=(o[-1]-c[-2])/c[-2] if idx>=1 else 0;f[1]=abs(f[0])
    for li,lag in enumerate([1,3,5,10,20],2):f[li]=(c[-1]-c[-lag-1])/c[-lag-1] if len(c)>lag else 0
    for pi,p in enumerate([5,10,20,60],7):ma=np.mean(c[-min(p,len(c)):]);f[pi]=(c[-1]-ma)/ma
    f[11]=np.std(c[-20:])/np.mean(c[-20:]);f[12]=(h[-1]-l[-1])/c[-1]
    vma=np.mean(v[-20:]) if np.mean(v[-20:])>0 else 1;f[13]=v[-1]/vma
    f[14]=oi[-1]/np.mean(oi[-20:]) if len(oi)>=20 and np.mean(oi[-20:])>0 else 1
    ema12=c[-1];ema26=c[-1]
    for j in range(len(c)-2,-1,-1):ema12=(2/13)*c[j]+(11/13)*ema12;ema26=(2/27)*c[j]+(25/27)*ema26
    f[15]=(ema12-ema26)/c[-1]
    dd_=np.diff(c[-15:]);g=dd_[dd_>0].sum() if len(dd_[dd_>0])>0 else 0
    lo=abs(dd_[dd_<0].sum()) if len(dd_[dd_<0])>0 else 1e-10
    f[16]=100-100/(1+g/lo) if lo>0 else 50
    bb=np.std(c[-20:]);ma20=np.mean(c[-20:]);f[17]=(c[-1]-ma20)/(2*bb+1e-10)
    try:m=int(str(daily.iloc[idx]['date'])[5:7]);f[18]=np.sin(2*np.pi*m/12);f[19]=np.cos(2*np.pi*m/12)
    except:f[18]=f[19]=0
    valid[idx]=True

labels=np.zeros(n,dtype=int)
for i in range(n-1):
    if valid[i]:labels[i]=1 if daily.iloc[i+1]['close']>daily.iloc[i]['close'] else 0

print(f'Precomputed {valid.sum()} features ({len(daily)} days) in {time.time()-t0:.1f}s')

N_WF=500;CAP=300000;cost=0.0006;mult=16;base=3;ne=200;d=5;lr=0.05
SM=[1.5,2.0,2.5];RR=[3,4,5];CF=[0.50,0.52]

print(f'\nA1 FAST: {len(SM)*len(RR)*len(CF)} combos x {N_WF}WF, OHLC simulation')
print(f'{"Stop":<6} {"RR":<4} {"Conf":<5} {"Trd":<6} {"WR":<6} {"PF":<6} {"PnL":<9} {"Ann":<8} {"DD":<6} {"Time"}')
print('-'*72)

results=[]
for sm in SM:
    for rr_val in RR:
        for conf in CF:
            t0=time.time();ts=int(n*0.6)
            pnl_seq=[];wins=0;total=0
            
            for run in range(min(N_WF,(n-ts)//10)):
                sp=ts+run*10
                if sp+10>n:break
                t1=sp;t2=min(sp+10,n)
                tr_idx=[i for i in range(win,t1) if valid[i]]
                if len(tr_idx)<100:continue
                X_tr=features[tr_idx];y_tr=labels[tr_idx]
                if len(np.unique(y_tr))<2:continue
                m=xgb.XGBClassifier(n_estimators=ne,max_depth=d,learning_rate=lr,
                    device='cuda',verbosity=0,random_state=42)
                m.fit(X_tr,y_tr)
                
                for j in range(t1,t2-1):
                    if not valid[j]:continue
                    bar=daily.iloc[j];o=bar['open'];h=bar['high'];l=bar['low']
                    nxt=daily.iloc[j+1];nh=nxt['high'];nl=nxt['low'];nc=nxt['close']
                    
                    av=[abs(daily.iloc[k]['high']-daily.iloc[k]['low']) for k in range(max(0,j-20),j+1)]
                    atr=np.mean(av)if av else o*0.02;ap=atr/o
                    if ap<0.01:lev=3.0
                    elif ap<0.02:lev=2.0
                    elif ap<0.03:lev=1.5
                    elif ap<0.05:lev=0.5
                    else:continue
                    
                    prob=m.predict_proba(features[j].reshape(1,-1))[0]
                    c2=prob[1]if prob[1]>0.5 else 1-prob[1]
                    if c2<conf:continue
                    
                    dr='LONG'if prob[1]>0.5 else'SHORT'
                    lots=max(1,int(lev*base))
                    sd=sm*atr;td_val=rr_val*sd
                    
                    if dr=='LONG':
                        tgt=o+td_val;stp=o-sd
                        if nh>=tgt and nl>stp:pnl=td_val*mult*lots-2*cost*o*mult*lots
                        elif nl<=stp and nh<tgt:pnl=-sd*mult*lots-2*cost*o*mult*lots
                        elif nh>=tgt and nl<=stp:pnl=(td_val*mult*lots-2*cost*o*mult*lots)if(tgt-o)<=(o-stp)else(-sd*mult*lots-2*cost*o*mult*lots)
                        else:pnl=(nc-o)*mult*lots-2*cost*o*mult*lots
                    else:
                        tgt=o-td_val;stp=o+sd
                        if nl<=tgt and nh<stp:pnl=td_val*mult*lots-2*cost*o*mult*lots
                        elif nh>=stp and nl>tgt:pnl=-sd*mult*lots-2*cost*o*mult*lots
                        elif nl<=tgt and nh>=stp:pnl=(td_val*mult*lots-2*cost*o*mult*lots)if(o-tgt)<=(stp-o)else(-sd*mult*lots-2*cost*o*mult*lots)
                        else:pnl=(o-nc)*mult*lots-2*cost*o*mult*lots
                    
                    pnl_seq.append(pnl)
                    wins+=(1 if pnl>0 else 0);total+=1
            
            if total==0:continue
            tp=sum(pnl_seq);wr=wins/total*100
            cum=np.cumsum(pnl_seq);pk=np.maximum.accumulate(cum)
            dd_val=np.max((pk-cum)/(pk+1e-10))*100 if pk[-1]>0 else 0
            gp=sum(p for p in pnl_seq if p>0);gl=abs(sum(p for p in pnl_seq if p<0))
            pf=gp/gl if gl>0 else 999
            yrs=(n-ts)/252
            ar=((CAP+tp)/CAP)**(1/yrs)-1 if tp>-CAP and yrs>0 else tp/CAP/yrs
            
            results.append({'sm':sm,'rr':rr_val,'conf':conf,'trades':total,'wr':round(wr,1),
                'pf':round(pf,2),'pnl':round(tp,0),'ar':round(ar*100,1),'dd':round(dd_val,1)})
            label='+ 'if tp>0 else''
            print(f'{sm:<6.1f} {rr_val:<4} {conf:<5.2f} {total:<6} {wr:<5.1f}% {pf:<6.2f} {tp/10000:<+8.1f}万 {ar*100:<+7.1f}% {dd_val:<5.1f}% {label} {time.time()-t0:.0f}s')

results.sort(key=lambda x:-x['pnl'])
with open('/home/a/prophet_futures/prophet_futures/a1_fast_results.json','w')as f:json.dump(results,f,indent=2,default=str)

print(f'\n===== A1 FASTFULL RESULTS ({len(results)} combos) =====')
print(f'{"Rank":<5} {"Stop":<6} {"RR":<4} {"Conf":<5} {"Trd":<6} {"WR":<6} {"PF":<6} {"PnL":<9} {"Ann":<8} {"DD":<6}')
for i,r in enumerate(results):
    m='+ 'if r['pnl']>0 else'  '
    print(f'{i+1:<5} {r["sm"]:<6.1f} {r["rr"]:<4} {r["conf"]:<5.2f} {r["trades"]:<6} {r["wr"]:<5.1f}% {r["pf"]:<6.2f} {r["pnl"]/10000:<+8.1f}万 {r["ar"]:<+7.1f}% {r["dd"]:<5.1f}% {m}')
print('\nSaved: a1_fast_results.json')
