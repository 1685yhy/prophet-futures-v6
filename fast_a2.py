#!/usr/bin/env python3
"""Fast A2: precompute intraday features, WF with 30-min data, full save"""
import numpy as np, pandas as pd, json, time
import akshare as ak, xgboost as xgb

m30 = ak.futures_zh_minute_sina(symbol='LH2609', period='30')
m30['datetime'] = pd.to_datetime(m30['datetime'])
m30 = m30.sort_values('datetime').reset_index(drop=True)

CAP=300000;mult=16;base=3;cost=0.0006;ne=200;d=5;lr=0.05
N_WF=30;lookback=13
SM=[1.5,2.0,2.5];RR=[3,4,5];CF=[0.50,0.52]

n_bars=len(m30)
# Precompute features for all bars
t0=time.time()
features=np.zeros((n_bars,15),dtype=np.float32);valid=np.zeros(n_bars,dtype=bool)

for idx in range(lookback+5,n_bars):
    w=m30.iloc[idx-lookback:idx+1];c=w['close'].values;h=w['high'].values;l=w['low'].values
    v=w['volume'].values;f=features[idx]
    for li,lag in enumerate([1,3,5,13]):f[li]=(c[-1]-c[-lag-1])/c[-lag-1]if len(c)>lag else 0
    for pi,p in enumerate([3,5,13],4):ma=np.mean(c[-min(p,len(c)):]);f[pi]=(c[-1]-ma)/ma
    f[7]=np.std(c[-5:])/np.mean(c[-5:])if np.mean(c[-5:])>0 else 0
    f[8]=(h[-1]-l[-1])/c[-1]
    vma=np.mean(v[-5:])if np.mean(v[-5:])>0 else 1;f[9]=v[-1]/vma
    trend=np.polyfit(range(len(c[-13:])),c[-13:],1)[0];f[10]=trend/c[-1]
    dd_=np.diff(c[-8:]);g=dd_[dd_>0].sum()if len(dd_[dd_>0])>0 else 0
    lo=abs(dd_[dd_<0].sum())if len(dd_[dd_<0])>0 else 1e-10
    f[11]=100-100/(1+g/lo)if lo>0 else 50
    bb_std=np.std(c[-13:]);bb_ma=np.mean(c[-13:]);f[12]=(c[-1]-bb_ma)/(2*bb_std+1e-10)
    hour=m30.iloc[idx]['datetime'].hour;minute=m30.iloc[idx]['datetime'].minute
    tod=(hour*60+minute)/(24*60);f[13]=np.sin(2*np.pi*tod);f[14]=np.cos(2*np.pi*tod)
    valid[idx]=True

# Labels: next 3 bars up or down
labels=np.zeros(n_bars,dtype=int)
for i in range(n_bars-3):
    if valid[i]:labels[i]=1 if m30.iloc[i+3]['close']>m30.iloc[i]['close'] else 0

print(f'Precomputed {valid.sum()} intraday features in {time.time()-t0:.1f}s')

total=len(SM)*len(RR)*len(CF)
print(f'\nA2 FAST: {total} combos x {N_WF}WF, 30-min data')
print(f'{"Stop":<6} {"RR":<4} {"Conf":<5} {"Trd":<6} {"WR":<6} {"PF":<6} {"PnL":<9} {"Ann":<8} {"DD":<6} {"Time"}')
print('-'*72)

results=[]
train_pct=0.7;train_size=int(n_bars*train_pct)

for sm in SM:
    for rr_val in RR:
        for conf in CF:
            t0=time.time()
            all_pnl=[];all_wins=0;all_total=0
            step=max(1,(n_bars-train_size)//N_WF)
            
            for wf in range(min(N_WF,(n_bars-train_size)//step)):
                sp=train_size+wf*step;ep=min(sp+step,n_bars)
                if sp>=n_bars or ep-sp<5:continue
                
                tr_idx=[i for i in range(lookback,sp)if valid[i]and i<ep-3]
                if len(tr_idx)<50:continue
                X_tr=features[tr_idx];y_tr=labels[tr_idx]
                if len(np.unique(y_tr))<2:continue
                m=xgb.XGBClassifier(n_estimators=ne,max_depth=d,learning_rate=lr,
                    device='cuda',verbosity=0,random_state=42)
                m.fit(X_tr,y_tr)
                
                capital=CAP;positions=[]
                for i in range(sp,ep):
                    bar=m30.iloc[i];o=bar['open'];h=bar['high'];l=bar['low']
                    if not valid[i]:continue
                    
                    start_idx=max(0,i-13)
                    recent=[abs(m30.iloc[k]['high']-m30.iloc[k]['low'])for k in range(start_idx,i+1)]
                    atr=np.mean(recent)if recent else o*0.005;ap=atr/o
                    if ap<0.005:lev=3.0
                    elif ap<0.01:lev=2.0
                    elif ap<0.015:lev=1.5
                    elif ap<0.03:lev=0.5
                    else:lev=0
                    cap_ratio=max(0.3,capital/CAP)
                    max_lots=max(1,int(lev*base*cap_ratio))if lev>0 else 0
                    
                    new_positions=[]
                    for ep2,dr2,lots2 in positions:
                        sd_val=sm*atr;td_val=rr_val*sd_val
                        hit=False;pnl2=0
                        if dr2=='LONG':
                            if h>=ep2+td_val:pnl2=td_val*mult*lots2-2*cost*ep2*mult*lots2;hit=True
                            elif l<=ep2-sd_val:pnl2=-sd_val*mult*lots2-2*cost*ep2*mult*lots2;hit=True
                        else:
                            if l<=ep2-td_val:pnl2=td_val*mult*lots2-2*cost*ep2*mult*lots2;hit=True
                            elif h>=ep2+sd_val:pnl2=-sd_val*mult*lots2-2*cost*ep2*mult*lots2;hit=True
                        if hit:capital+=pnl2;all_total+=1;all_wins+=(1 if pnl2>0 else 0)
                        else:new_positions.append((ep2,dr2,lots2))
                    positions=new_positions
                    
                    if max_lots>0 and i<ep-3:
                        prob=m.predict_proba(features[i].reshape(1,-1))[0]
                        c2=prob[1]if prob[1]>0.5 else 1-prob[1]
                        if c2>=conf:positions.append((o,'LONG'if prob[1]>0.5 else'SHORT',max_lots))
                
                all_pnl.append(capital-CAP)
            
            if all_total==0:continue
            tp=sum(all_pnl);wr=all_wins/all_total*100
            cum=np.cumsum(all_pnl);pk=np.maximum.accumulate(cum)
            dd_val=np.max((pk-cum)/(pk+1e-10))*100 if pk[-1]>0 else 0
            gp=sum(p for p in all_pnl if p>0);gl=abs(sum(p for p in all_pnl if p<0))
            pf=gp/gl if gl>0 else 999
            total_days=(m30['datetime'].max()-m30['datetime'].min()).days/365.0
            ar=((CAP+tp)/CAP/N_WF)**(1/total_days)-1 if total_days>0 and tp>-CAP*.9 else-1
            
            results.append({'sm':sm,'rr':rr_val,'conf':conf,'trades':all_total,'wr':round(wr,1),
                'pf':round(pf,2),'pnl':round(tp,0),'ar':round(ar*100,1),'dd':round(dd_val,1)})
            label='+ 'if tp>0 else''
            print(f'{sm:<6.1f} {rr_val:<4} {conf:<5.2f} {all_total:<6} {wr:<5.1f}% {pf:<6.2f} {tp/10000:<+8.1f}万 {ar*100:<+7.1f}% {dd_val:<5.1f}% {label} {time.time()-t0:.0f}s')

results.sort(key=lambda x:-x['pnl'])
with open('/home/a/prophet_futures/prophet_futures/a2_fast_results.json','w')as f:json.dump(results,f,indent=2,default=str)

print(f'\n===== A2 FASTFULL RESULTS ({len(results)} combos) =====')
print(f'{"Rank":<5} {"Stop":<6} {"RR":<4} {"Conf":<5} {"Trd":<6} {"WR":<6} {"PF":<6} {"PnL":<9} {"Ann":<8} {"DD":<6}')
for i,r in enumerate(results):
    m='+ 'if r['pnl']>0 else'  '
    print(f'{i+1:<5} {r["sm"]:<6.1f} {r["rr"]:<4} {r["conf"]:<5.2f} {r["trades"]:<6} {r["wr"]:<5.1f}% {r["pf"]:<6.2f} {r["pnl"]/10000:<+8.1f}万 {r["ar"]:<+7.1f}% {r["dd"]:<5.1f}% {m}')
print('\nSaved: a2_fast_results.json')
