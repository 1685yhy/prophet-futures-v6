#!/usr/bin/env python3
"""
Prophet Futures v28 — 实盘连续Walk-Forward回测
资本连续滚动 | 仓位随资金变化 | V25 vs V28
"""
import numpy as np, pandas as pd
from datetime import datetime
import akshare as ak
import xgboost as xgb

# ===== CONFIG =====
SYMBOLS = {
    'lh2609': {
        'code': 'LH0', 'name': 'LH', 'multiplier': 16, 'cost': 0.0006,
        'max_pos': 6, 'max_total': 12,
        'atr_stop': 1.5, 'rr': 4.0,
        'add_conf': 0.65, 'add_atr': 2.0,
        'reduce_conf': 0.55, 'reverse_conf': 0.35,
        'trail_atr': 2.0, 'be_atr': 1.0, 'min_hold': 3,
        'init_cap': 300000, 'margin': 0.15,
    },
    'jm2609': {
        'code': 'JM0', 'name': 'JM', 'multiplier': 60, 'cost': 0.0011,
        'max_pos': 4, 'max_total': 8,
        'atr_stop': 2.0, 'rr': 3.5,
        'add_conf': 0.65, 'add_atr': 2.5,
        'reduce_conf': 0.55, 'reverse_conf': 0.30,
        'trail_atr': 3.0, 'be_atr': 2.0, 'min_hold': 5,
        'init_cap': 300000, 'margin': 0.15,
    },
}

def bf(df, idx, w=60):
    if idx < w+5: return None
    s = df.iloc[idx-w:idx+1]
    c=s['close'].values.astype(float);o=s['open'].values.astype(float)
    h=s['high'].values.astype(float);l=s['low'].values.astype(float)
    v=s['volume'].values.astype(float);oi=s['oi'].values.astype(float)
    oc_ret = float((o[-1]-c[-2])/c[-2]) if idx>=1 else 0.0
    f=[oc_ret, abs(oc_ret)]
    for lag in[1,3,5,10,20]: f.append(float((c[-1]-c[-lag-1])/c[-lag-1] if len(c)>lag else 0))
    for p in[5,10,20,60]: ma=np.mean(c[-min(p,len(c)):]);f.append(float((c[-1]-ma)/ma))
    f.append(float(np.std(c[-20:])/np.mean(c[-20:])))
    f.append(float((h[-1]-l[-1])/c[-1]))
    vm=np.mean(v[-20:])if np.mean(v[-20:])>0 else 1;f.append(float(v[-1]/vm))
    f.append(float(oi[-1]/np.mean(oi[-20:]))if len(oi)>=20 and np.mean(oi[-20:])>0 else 1)
    e12=c[-1];e26=c[-1]
    for j in range(len(c)-2,-1,-1):e12=(2/13)*c[j]+(11/13)*e12;e26=(2/27)*c[j]+(25/27)*e26
    f.append(float((e12-e26)/c[-1]))
    dd=np.diff(c[-15:]);g=float(dd[dd>0].sum())if len(dd[dd>0])>0 else 0
    lo=float(abs(dd[dd<0].sum()))if len(dd[dd<0])>0 else 1e-10
    f.append(float(100-100/(1+g/lo)if lo>0 else 50))
    bb=np.std(c[-20:]);m20=np.mean(c[-20:]);f.append(float((c[-1]-m20)/(2*bb+1e-10)))
    f.append(float(c[-1]/1000.0))
    return np.array(f,dtype=np.float32)

def atr(df,i,p=20):
    if i<p:return None
    return np.mean([abs(float(df.iloc[j]['high'])-float(df.iloc[j]['low']))for j in range(i-p+1,i+1)])

def pos_size(cap,price,atr,cfg):
    ap=atr/price
    if ap<0.01:lev=3.0
    elif ap<0.02:lev=2.0
    elif ap<0.03:lev=1.5
    else:lev=0.5
    bl=max(1,int(lev*(cfg['max_pos']//2)))if lev>0 else 0
    cr=cap/cfg['init_cap']
    lots=max(0,int(bl*cr))
    if cap<100000:lots=max(1,lots//2)if lots>0 else 0
    return min(lots,cfg['max_pos'])

def run_v25(df,model,cfg,cap):
    """V25: single position, fixed ATR stop + RR TP"""
    pos=None;trades=[];eq=[]
    for i in range(70,len(df)):
        ft=bf(df,i,60)
        if ft is None:eq.append(cap);continue
        try:prob=float(model.predict_proba(ft.reshape(1,-1))[0][1])
        except:eq.append(cap);continue
        pr=float(df.iloc[i]['close']);hi=float(df.iloc[i]['high']);lo=float(df.iloc[i]['low'])
        a=atr(df,i,20)
        if a is None or pr<=0:eq.append(cap);continue
        
        if pos:
            d,en,st,tp,eb,lt,mg=pos
            if d=='LONG':
                if lo<=st:
                    pnl=mg*(((st-en)/en)/cfg['margin']-cfg['cost']*2);cap+=mg+pnl
                    trades.append({'t':'S','d':d,'en':en,'ex':st,'lt':lt,'pnl':pnl,'b':i-eb});pos=None
                elif hi>=tp:
                    pnl=mg*(((tp-en)/en)/cfg['margin']-cfg['cost']*2);cap+=mg+pnl
                    trades.append({'t':'T','d':d,'en':en,'ex':tp,'lt':lt,'pnl':pnl,'b':i-eb});pos=None
            else:
                if hi>=st:
                    pnl=mg*(((en-st)/en)/cfg['margin']-cfg['cost']*2);cap+=mg+pnl
                    trades.append({'t':'S','d':d,'en':en,'ex':st,'lt':lt,'pnl':pnl,'b':i-eb});pos=None
                elif lo<=tp:
                    pnl=mg*(((en-tp)/en)/cfg['margin']-cfg['cost']*2);cap+=mg+pnl
                    trades.append({'t':'T','d':d,'en':en,'ex':tp,'lt':lt,'pnl':pnl,'b':i-eb});pos=None
        
        if pos is None:
            ps=pos_size(cap,pr,a,cfg)
            if ps>0:
                sd='LONG'if prob>0.5 else'SHORT';sd2=a*cfg['atr_stop']
                mpl=pr*cfg['multiplier']*cfg['margin'];m=ps*mpl
                if m<=cap*0.8:
                    if sd=='LONG':
                        sv=pr-sd2;tv=pr+sd2*cfg['rr']
                        if lo>sv:cap-=m;pos=(sd,pr,sv,tv,i,ps,m)
                    else:
                        sv=pr+sd2;tv=pr-sd2*cfg['rr']
                        if hi<sv:cap-=m;pos=(sd,pr,sv,tv,i,ps,m)
        
        eqv=cap
        if pos:
            d,en,st,tp,eb,lt,mg=pos
            up=lt*(pr-en)*cfg['multiplier']*cfg['margin']if d=='LONG'else lt*(en-pr)*cfg['multiplier']*cfg['margin']
            eqv+=mg+up
        eq.append(eqv)
    
    lp=float(df.iloc[-1]['close'])
    if pos:
        d,en,st,tp,eb,lt,mg=pos
        pnl=mg*(((lp-en)/en)/cfg['margin']-cfg['cost']*2)if d=='LONG'else mg*(((en-lp)/en)/cfg['margin']-cfg['cost']*2)
        cap+=mg+pnl
        trades.append({'t':'E','d':d,'en':en,'ex':lp,'lt':lt,'pnl':pnl,'b':len(df)-1-eb})
    return trades,eq,cap

def run_v28(df,model,cfg,cap):
    """V28: dynamic add/reduce/reverse"""
    positions=[];trades=[];eq=[];tl=0;rv=0
    for i in range(70,len(df)):
        ft=bf(df,i,60)
        if ft is None:eq.append(cap);continue
        try:prob=float(model.predict_proba(ft.reshape(1,-1))[0][1])
        except:eq.append(cap);continue
        pr=float(df.iloc[i]['close']);hi=float(df.iloc[i]['high']);lo=float(df.iloc[i]['low'])
        a=atr(df,i,20)
        if a is None or pr<=0:eq.append(cap);continue
        
        cd='LONG'if prob>0.5 else'SHORT';cf=prob if prob>0.5 else 1-prob
        
        sv=[]
        for pos in positions:
            d,en,tr,eb,lt,mg=pos;bh=i-eb
            pp=(pr-en)/en if d=='LONG'else(en-pr)/en;pa=pp*en/a if a>0 else 0
            
            if d=='LONG':
                hs=pr-a*cfg['atr_stop']
                if pa>cfg['trail_atr']:tr=max(tr,pr-a*(cfg['atr_stop']-0.3))
                if pa>cfg['be_atr']:tr=max(tr,en)
                es=max(hs,tr)
                srd=(cd=='LONG'and cf<cfg['reduce_conf']and bh>=cfg['min_hold'])
                srv=(prob<cfg['reverse_conf']and bh>=cfg['min_hold'])
                if lo<=es:
                    ep=es;pnl=mg*(((ep-en)/en)/cfg['margin']-cfg['cost']*2);cap+=mg+pnl
                    trades.append({'t':'S','d':d,'en':en,'ex':ep,'lt':lt,'pnl':pnl,'b':bh});tl-=lt
                    rv+=1 if prob<0.5 else 0
                elif srv and rv>=2:
                    pnl=mg*(pp/cfg['margin']-cfg['cost']*2);cap+=mg+pnl
                    trades.append({'t':'R','d':d,'en':en,'ex':pr,'lt':lt,'pnl':pnl,'b':bh});tl-=lt;rv+=1
                elif srd and lt>1:
                    rl=lt//2;rm=mg*(rl/lt);pnl=rm*(pp/cfg['margin']-cfg['cost']);cap+=rm+pnl
                    trades.append({'t':'D','d':d,'en':en,'ex':pr,'lt':rl,'pnl':pnl,'b':bh});tl-=rl
                    sv.append((d,en,tr,eb,lt-rl,mg-rm));rv=0
                else:
                    sv.append((d,en,tr,eb,lt,mg));rv=0 if cd=='LONG'else rv
            else:
                hs=pr+a*cfg['atr_stop']
                if -pa>cfg['trail_atr']:tr=min(tr,pr+a*(cfg['atr_stop']-0.3))
                if -pa>cfg['be_atr']:tr=min(tr,en)
                es=min(hs,tr)
                srd=(cd=='SHORT'and cf<cfg['reduce_conf']and bh>=cfg['min_hold'])
                srv=(prob>1-cfg['reverse_conf']and bh>=cfg['min_hold'])
                if hi>=es:
                    ep=es;pnl=mg*(((en-ep)/en)/cfg['margin']-cfg['cost']*2);cap+=mg+pnl
                    trades.append({'t':'S','d':d,'en':en,'ex':ep,'lt':lt,'pnl':pnl,'b':bh});tl-=lt
                    rv+=1 if prob>0.5 else 0
                elif srv and rv>=2:
                    pnl=mg*(pp/cfg['margin']-cfg['cost']*2);cap+=mg+pnl
                    trades.append({'t':'R','d':d,'en':en,'ex':pr,'lt':lt,'pnl':pnl,'b':bh});tl-=lt;rv+=1
                elif srd and lt>1:
                    rl=lt//2;rm=mg*(rl/lt);pnl=rm*(pp/cfg['margin']-cfg['cost']);cap+=rm+pnl
                    trades.append({'t':'D','d':d,'en':en,'ex':pr,'lt':rl,'pnl':pnl,'b':bh});tl-=rl
                    sv.append((d,en,tr,eb,lt-rl,mg-rm));rv=0
                else:
                    sv.append((d,en,tr,eb,lt,mg));rv=0 if cd=='SHORT'else rv
        positions=sv
        
        ps=pos_size(cap,pr,a,cfg)
        mx=max(1,min(int(cfg['max_total']*cap/cfg['init_cap']),cfg['max_total']))
        if ps>0 and tl+ps<=mx:
            sd='LONG'if prob>0.5 else'SHORT';sd2=a*cfg['atr_stop']
            mpl=pr*cfg['multiplier']*cfg['margin'];m=ps*mpl
            if not positions:
                if sd=='LONG':
                    sv_=pr-sd2
                    if lo>sv_ and m<=cap*0.8:cap-=m;positions.append((sd,pr,sv_,i,ps,m));tl+=ps
                else:
                    sv_=pr+sd2
                    if hi<sv_ and m<=cap*0.8:cap-=m;positions.append((sd,pr,sv_,i,ps,m));tl+=ps
            else:
                ed=positions[0][0]
                if sd==ed:
                    ae=np.mean([p[1]for p in positions])
                    pa=(pr-ae)/a if sd=='LONG'else(ae-pr)/a
                    if cf>cfg['add_conf']and pa>cfg['add_atr']:
                        if sd=='LONG':
                            sv_=pr-sd2
                            if lo>sv_ and m<=cap*0.8:cap-=m;positions.append((sd,pr,sv_,i,ps,m));tl+=ps
                        else:
                            sv_=pr+sd2
                            if hi<sv_ and m<=cap*0.8:cap-=m;positions.append((sd,pr,sv_,i,ps,m));tl+=ps
        
        eqv=cap
        for pos in positions:
            d,en,tr,eb,lt,mg=pos
            up=lt*(pr-en)*cfg['multiplier']*cfg['margin']if d=='LONG'else lt*(en-pr)*cfg['multiplier']*cfg['margin']
            eqv+=mg+up
        eq.append(eqv)
    
    lp=float(df.iloc[-1]['close'])
    for pos in positions:
        d,en,tr,eb,lt,mg=pos
        pnl=mg*(((lp-en)/en)/cfg['margin']-cfg['cost']*2)if d=='LONG'else mg*(((en-lp)/en)/cfg['margin']-cfg['cost']*2)
        cap+=mg+pnl
        trades.append({'t':'E','d':d,'en':en,'ex':lp,'lt':lt,'pnl':pnl,'b':len(df)-1-eb})
    return trades,eq,cap

def report(name,trades,eq_arr,final_cap,start_cap):
    n=len(trades)
    if n==0:return None,{'n':0,'ret':0,'mdd':0,'wr':0,'pf':0}
    wins=[t for t in trades if t['pnl']>0];losses=[t for t in trades if t['pnl']<=0]
    wr=len(wins)/n;tp=sum(t['pnl']for t in trades)
    gw=sum(t['pnl']for t in wins);gl=abs(sum(t['pnl']for t in losses));pf=gw/gl if gl>0 else 99
    aw=np.mean([t['pnl']for t in wins])if wins else 0
    al=np.mean([t['pnl']for t in losses])if losses else 0
    ret=(final_cap-start_cap)/start_cap*100
    arr=np.array(eq_arr)
    if len(arr)>0:
        pk=np.maximum.accumulate(arr);mdd=np.min((arr-pk)/pk)*100
    else:mdd=0
    tb=sum(t['b']for t in trades);yrs=tb/240 if tb>0 else 1
    ar=((final_cap/start_cap)**(1/yrs)-1)*100 if yrs>0 else 0
    ts={}
    for t in trades:ts[t['t']]=ts.get(t['t'],0)+1
    d={'n':n,'wr':wr,'pnl':tp,'ret':ret,'mdd':mdd,'pf':pf,'ar':ar,'fc':final_cap,'types':ts}
    s=f"{name}: {n}笔 WR={wr:.0%} 总盈亏{tp:+,.0f} 收益{ret:+.1f}% 年化{ar:+.1f}% MDD{mdd:.1f}% PF={pf:.2f}"
    print(f"  {s}")
    return d,s

# ===== MAIN =====
print("="*60)
print("  Prophet v28 — 实盘连续WF回测")
print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
print("="*60)

for sk,cfg in SYMBOLS.items():
    print(f"\n{'='*60}")
    print(f"  {sk} ({cfg['name']}) 起始 ¥{cfg['init_cap']:,}")
    print(f"{'='*60}")
    
    print(f"  📡 取数...")
    df=ak.futures_main_sina(symbol=cfg['code'])
    df.columns=['date','open','high','low','close','volume','oi','settle']
    for c in['open','high','low','close','volume','oi']:df[c]=pd.to_numeric(df[c],errors='coerce')
    df=df.dropna(subset=['close']).reset_index(drop=True)
    print(f"  ✅ {len(df)}行 {df.iloc[0]['date']}~{df.iloc[-1]['date']}")
    
    print(f"  🔄 特征...")
    Xa,ya,fi=[],[],[]
    for i in range(70,len(df)-1):
        ft=bf(df,i,60)
        if ft is not None:Xa.append(ft);ya.append(1 if float(df.iloc[i+1]['close'])>float(df.iloc[i]['close'])else 0);fi.append(i)
    Xa=np.array(Xa,dtype=np.float32);ya=np.array(ya)
    print(f"  ✅ {len(Xa)}条")
    
    NT=500;NTS=30;ST=15
    
    # V25
    cap25=cfg['init_cap'];t25=[];eq25=[]
    for ts in range(NT,len(Xa)-NTS,ST):
        trs=max(0,ts-NT)
        m=xgb.XGBClassifier(n_estimators=100,max_depth=4,learning_rate=0.05,
                             subsample=0.8,colsample_bytree=0.8,random_state=42,n_jobs=1,verbosity=0)
        m.fit(Xa[trs:ts][-1000:],ya[trs:ts][-1000:])
        # Include warmup rows before test period (70 bars needed)
        t0=fi[max(0,ts-70)]; t1=fi[min(ts+NTS-1,len(fi)-1)]+1
        tdf=df.iloc[t0:min(len(df),t1+1)].copy().reset_index(drop=True)
        if len(tdf)<72:continue
        cu=min(cap25,cfg['init_cap']*3)
        tt,te,cap25=run_v25(tdf,m,cfg,cu)
        t25.extend(tt);eq25.extend(te)
    
    # V28
    cap28=cfg['init_cap'];t28=[];eq28=[]
    for ts in range(NT,len(Xa)-NTS,ST):
        trs=max(0,ts-NT)
        m=xgb.XGBClassifier(n_estimators=100,max_depth=4,learning_rate=0.05,
                             subsample=0.8,colsample_bytree=0.8,random_state=42,n_jobs=1,verbosity=0)
        m.fit(Xa[trs:ts][-1000:],ya[trs:ts][-1000:])
        t0=fi[max(0,ts-70)]; t1=fi[min(ts+NTS-1,len(fi)-1)]+1
        tdf=df.iloc[t0:min(len(df),t1+1)].copy().reset_index(drop=True)
        if len(tdf)<72:continue
        cu=min(cap28,cfg['init_cap']*3)
        tt,te,cap28=run_v28(tdf,m,cfg,cu)
        t28.extend(tt);eq28.extend(te)
    
    print(f"\n  {'─'*50}")
    d25,_=report('V25',t25,eq25,cap25,cfg['init_cap'])
    d28,_=report('V28',t28,eq28,cap28,cfg['init_cap'])
    print(f"  {'─'*50}")
    
    if d25 and d28 and d25['n']>0 and d28['n']>0:
        w='V28'if d28['ret']>d25['ret']else'V25'
        print(f"  {w} 🏆 | 收益差{d28['ret']-d25['ret']:+.1f}% | 回撤差{d28['mdd']-d25['mdd']:+.1f}%")
        print(f"  V25终值 ¥{d25['fc']:,.0f} | V28终值 ¥{d28['fc']:,.0f}")
        print(f"  V25退出: {d25['types']} | V28退出: {d28['types']}")

print(f"\n{'='*60}")
print(f"  完成")
print(f"{'='*60}")
