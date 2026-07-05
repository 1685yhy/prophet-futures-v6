#!/usr/bin/env python3
"""
Prophet Futures v28 — Full WF Backtest
全量数据 | 30天测试窗口 | 滚动训练 | 真实成本
"""
import numpy as np, pandas as pd, pickle, os, time
from datetime import datetime, timedelta
import akshare as ak
import xgboost as xgb

SYMBOLS = {
    'lh2609': {
        'code': 'LH0', 'name': 'LH', 'multiplier': 16, 'cost': 0.0006,
        'max_pos': 6, 'max_total': 12,
        'atr_stop': 1.5, 'rr': 4.0,
        'add_conf': 0.65, 'add_atr': 2.0,
        'reduce_conf': 0.55, 'reverse_conf': 0.35,
        'trail_atr': 2.0, 'be_atr': 1.0, 'min_hold': 3,
    },
    'jm2609': {
        'code': 'JM0', 'name': 'JM', 'multiplier': 60, 'cost': 0.0011,
        'max_pos': 4, 'max_total': 8,
        'atr_stop': 2.0, 'rr': 3.5,
        'add_conf': 0.65, 'add_atr': 2.5,
        'reduce_conf': 0.55, 'reverse_conf': 0.30,
        'trail_atr': 3.0, 'be_atr': 2.0, 'min_hold': 5,
    },
}

def build_features(df, idx, window=60):
    if idx < window + 5: return None
    w = df.iloc[idx-window:idx+1]
    c = w['close'].values.astype(float); o = w['open'].values.astype(float)
    h = w['high'].values.astype(float); l = w['low'].values.astype(float)
    v = w['volume'].values.astype(float); oi = w['oi'].values.astype(float)
    f = []
    f.append(float((o[-1]-c[-2])/c[-2]) if idx>=1 else 0.0)
    f.append(abs(f[-1]))
    for lag in [1,3,5,10,20]:
        f.append(float((c[-1]-c[-lag-1])/c[-lag-1] if len(c)>lag else 0))
    for p in [5,10,20,60]:
        ma = np.mean(c[-min(p,len(c)):]); f.append(float((c[-1]-ma)/ma))
    f.append(float(np.std(c[-20:])/np.mean(c[-20:])))
    f.append(float((h[-1]-l[-1])/c[-1]))
    vma = np.mean(v[-20:]) if np.mean(v[-20:])>0 else 1; f.append(float(v[-1]/vma))
    f.append(float(oi[-1]/np.mean(oi[-20:])) if len(oi)>=20 and np.mean(oi[-20:])>0 else 1)
    ema12=c[-1];ema26=c[-1]
    for j in range(len(c)-2,-1,-1): ema12=(2/13)*c[j]+(11/13)*ema12; ema26=(2/27)*c[j]+(25/27)*ema26
    f.append(float((ema12-ema26)/c[-1]))
    dd_=np.diff(c[-15:]); g=float(dd_[dd_>0].sum())if len(dd_[dd_>0])>0 else 0
    lo=float(abs(dd_[dd_<0].sum()))if len(dd_[dd_<0])>0 else 1e-10
    f.append(float(100-100/(1+g/lo)if lo>0 else 50))
    bb=np.std(c[-20:]);ma20=np.mean(c[-20:]);f.append(float((c[-1]-ma20)/(2*bb+1e-10)))
    f.append(float(c[-1]/1000.0))
    return np.array(f,dtype=np.float32)

def calc_atr(df,idx,period=20):
    if idx<period: return None
    return np.mean([abs(float(df.iloc[i]['high'])-float(df.iloc[i]['low'])) for i in range(idx-period+1,idx+1)])

def run_v25(df,model,cfg):
    trades=[];pos=None
    for i in range(70,len(df)):
        feats=build_features(df,i,60)
        if feats is None: continue
        try: prob=float(model.predict_proba(feats.reshape(1,-1))[0][1])
        except: continue
        price=float(df.iloc[i]['close']);high=float(df.iloc[i]['high']);low=float(df.iloc[i]['low'])
        atr=calc_atr(df,i,20)
        if atr is None or price<=0: continue
        if pos:
            d,entry,stop,tp,entry_i,vol=pos
            if d=='LONG':
                if low<=stop:
                    pnl=((stop-entry)/entry)-cfg['cost']*2
                    trades.append({'pnl':pnl,'bars':i-entry_i,'type':'STOP'});pos=None
                elif high>=tp:
                    pnl=((tp-entry)/entry)-cfg['cost']*2
                    trades.append({'pnl':pnl,'bars':i-entry_i,'type':'TP'});pos=None
            else:
                if high>=stop:
                    pnl=((entry-stop)/entry)-cfg['cost']*2
                    trades.append({'pnl':pnl,'bars':i-entry_i,'type':'STOP'});pos=None
                elif low<=tp:
                    pnl=((entry-tp)/entry)-cfg['cost']*2
                    trades.append({'pnl':pnl,'bars':i-entry_i,'type':'TP'});pos=None
        if pos is None:
            atr_pct=atr/price
            if atr_pct<0.01:lev=3.0
            elif atr_pct<0.02:lev=2.0
            elif atr_pct<0.03:lev=1.5
            else:lev=0.5
            ps=max(1,int(lev*(cfg['max_pos']//2)))if lev>0 else 0
            if ps>0:
                sd='LONG'if prob>0.5 else'SHORT';sd2=atr*cfg['atr_stop']
                if sd=='LONG':
                    sv=price-sd2;tv=price+sd2*cfg['rr']
                    if low>sv:pos=(sd,price,sv,tv,i,ps)
                else:
                    sv=price+sd2;tv=price-sd2*cfg['rr']
                    if high<sv:pos=(sd,price,sv,tv,i,ps)
    if pos:
        d,entry,stop,tp,entry_i,vol=pos;lp=float(df.iloc[-1]['close'])
        pnl=((lp-entry)/entry if d=='LONG'else(entry-lp)/entry)-cfg['cost']*2
        trades.append({'pnl':pnl,'bars':len(df)-1-entry_i,'type':'EOD'})
    return trades

def run_v28(df,model,cfg):
    trades=[];positions=[];total_lots=0;rev_bars=0
    for i in range(70,len(df)):
        feats=build_features(df,i,60)
        if feats is None: continue
        try: prob=float(model.predict_proba(feats.reshape(1,-1))[0][1])
        except: continue
        price=float(df.iloc[i]['close']);high=float(df.iloc[i]['high']);low=float(df.iloc[i]['low'])
        atr=calc_atr(df,i,20)
        if atr is None or price<=0: continue
        cur_dir='LONG'if prob>0.5 else'SHORT';conf=prob if prob>0.5 else 1-prob
        
        surviving=[]
        for pos in positions:
            d,entry,trail,entry_i,vol=pos;bars=i-entry_i
            pnl_pct=(price-entry)/entry if d=='LONG'else(entry-price)/entry
            pnl_atr=pnl_pct*entry/atr if atr>0 else 0
            
            if d=='LONG':
                hs=price-atr*cfg['atr_stop']
                if pnl_atr>cfg['trail_atr']:trail=max(trail,price-atr*(cfg['atr_stop']-0.3))
                if pnl_atr>cfg['be_atr']:trail=max(trail,entry)
                es=max(hs,trail)
                should_reduce=(cur_dir=='LONG'and conf<cfg['reduce_conf']and bars>=cfg['min_hold'])
                should_reverse=(prob<cfg['reverse_conf']and bars>=cfg['min_hold'])
                if low<=es:
                    ep=es;pnl=((ep-entry)/entry)-cfg['cost']*2
                    trades.append({'pnl':pnl,'bars':bars,'type':'STOP','vol':vol});total_lots-=vol
                    rev_bars+=1 if prob<0.5 else 0
                elif should_reverse and rev_bars>=2:
                    pnl=pnl_pct-cfg['cost']*2
                    trades.append({'pnl':pnl,'bars':bars,'type':'REVERSE','vol':vol});total_lots-=vol;rev_bars+=1
                elif should_reduce and vol>1:
                    rv=vol//2;pnl=pnl_pct-cfg['cost']
                    trades.append({'pnl':pnl*0.5,'bars':bars,'type':'REDUCE','vol':rv});total_lots-=rv
                    surviving.append((d,entry,trail,entry_i,vol-rv));rev_bars=0
                else:
                    surviving.append((d,entry,trail,entry_i,vol));rev_bars=0 if cur_dir=='LONG'else rev_bars
            else:
                hs=price+atr*cfg['atr_stop']
                if -pnl_atr>cfg['trail_atr']:trail=min(trail,price+atr*(cfg['atr_stop']-0.3))
                if -pnl_atr>cfg['be_atr']:trail=min(trail,entry)
                es=min(hs,trail)
                should_reduce=(cur_dir=='SHORT'and conf<cfg['reduce_conf']and bars>=cfg['min_hold'])
                should_reverse=(prob>1-cfg['reverse_conf']and bars>=cfg['min_hold'])
                if high>=es:
                    ep=es;pnl=((entry-ep)/entry)-cfg['cost']*2
                    trades.append({'pnl':pnl,'bars':bars,'type':'STOP','vol':vol});total_lots-=vol
                    rev_bars+=1 if prob>0.5 else 0
                elif should_reverse and rev_bars>=2:
                    pnl=pnl_pct-cfg['cost']*2
                    trades.append({'pnl':pnl,'bars':bars,'type':'REVERSE','vol':vol});total_lots-=vol;rev_bars+=1
                elif should_reduce and vol>1:
                    rv=vol//2;pnl=pnl_pct-cfg['cost']
                    trades.append({'pnl':pnl*0.5,'bars':bars,'type':'REDUCE','vol':rv});total_lots-=rv
                    surviving.append((d,entry,trail,entry_i,vol-rv));rev_bars=0
                else:
                    surviving.append((d,entry,trail,entry_i,vol));rev_bars=0 if cur_dir=='SHORT'else rev_bars
        positions=surviving
        
        atr_pct=atr/price
        if atr_pct<0.01:lev=3.0
        elif atr_pct<0.02:lev=2.0
        elif atr_pct<0.03:lev=1.5
        else:lev=0.5
        ps=max(1,int(lev*(cfg['max_pos']//2)))if lev>0 else 0
        
        if ps>0 and total_lots+ps<=cfg['max_total']:
            sd='LONG'if prob>0.5 else'SHORT';sd2=atr*cfg['atr_stop']
            if not positions:
                if sd=='LONG':
                    sv=price-sd2
                    if low>sv:positions.append((sd,price,sv,i,ps));total_lots+=ps
                else:
                    sv=price+sd2
                    if high<sv:positions.append((sd,price,sv,i,ps));total_lots+=ps
            else:
                ed=positions[0][0]
                if sd==ed:
                    avg_entry=np.mean([p[1]for p in positions])
                    pa=(price-avg_entry)/atr if sd=='LONG'else(avg_entry-price)/atr
                    if conf>cfg['add_conf']and pa>cfg['add_atr']:
                        if sd=='LONG':
                            sv=price-sd2
                            if low>sv:positions.append((sd,price,sv,i,ps));total_lots+=ps
                        else:
                            sv=price+sd2
                            if high<sv:positions.append((sd,price,sv,i,ps));total_lots+=ps
    
    lp=float(df.iloc[-1]['close'])
    for pos in positions:
        d,entry,trail,entry_i,vol=pos
        pnl=((lp-entry)/entry if d=='LONG'else(entry-lp)/entry)-cfg['cost']*2
        trades.append({'pnl':pnl,'bars':len(df)-1-entry_i,'type':'EOD','vol':vol})
    return trades

def compound_stats(trades):
    if not trades: return {'wr':0,'eq':1.0,'mdd':0,'n':0,'types':{}}
    wins=[t for t in trades if t['pnl']>0];losses=[t for t in trades if t['pnl']<=0]
    wr=len(wins)/len(trades)
    eq=1.0;peak=1.0;mdd=0;eq_curve=[]
    for t in trades:
        eq*=(1+t['pnl']);peak=max(peak,eq);mdd=min(mdd,(eq-peak)/peak);eq_curve.append(eq)
    gw=sum(t['pnl']for t in wins);gl=abs(sum(t['pnl']for t in losses))
    pf=gw/gl if gl>0 else 99
    avg_win=np.mean([t['pnl']for t in wins])if wins else 0
    avg_loss=np.mean([t['pnl']for t in losses])if losses else 0
    types={}
    for t in trades:types[t['type']]=types.get(t['type'],0)+1
    return {'wr':wr,'eq':eq,'mdd':mdd,'n':len(trades),'pf':pf,
            'avg_win':avg_win,'avg_loss':avg_loss,'types':types,'pnl_sum':eq-1}

# ===== MAIN =====
print("="*60)
print("  Prophet v28 — Full WF Backtest")
print(f"  30天窗口 | 滚动训练 | {datetime.now().strftime('%Y-%m-%d %H:%M')}")
print("="*60)

for sym_key, cfg in SYMBOLS.items():
    print(f"\n{'='*60}")
    print(f"  {sym_key} ({cfg['name']})")
    print(f"{'='*60}")
    
    print(f"  📡 取全量数据...")
    try:
        df = ak.futures_main_sina(symbol=cfg['code'])
        df.columns = ['date','open','high','low','close','volume','oi','settle']
        for c in ['open','high','low','close','volume','oi']:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        df = df.dropna(subset=['close']).reset_index(drop=True)
        print(f"  ✅ {len(df)}行 {df.iloc[0]['date']} ~ {df.iloc[-1]['date']}")
    except Exception as e:
        print(f"  ❌ {e}"); continue
    
    # Precompute
    print(f"  🔄 预计算特征...")
    X_all, y_all = [], []
    for i in range(70, len(df)-1):
        feats = build_features(df, i, 60)
        if feats is not None:
            X_all.append(feats)
            y_all.append(1 if float(df.iloc[i+1]['close']) > float(df.iloc[i]['close']) else 0)
    X_all = np.array(X_all, dtype=np.float32); y_all = np.array(y_all)
    n_feat = len(X_all)
    print(f"  ✅ {n_feat}条特征 (df {len(df)}行 → {n_feat}特征)")
    
    # WF params
    n_train = 500  # 训练窗口
    n_test = 30    # 测试窗口 (月度)
    step = 15      # 步进
    
    v25_compound = []
    v28_compound = []
    
    for test_start in range(n_train, n_feat - n_test, step):
        test_end = test_start + n_test
        train_start = max(0, test_start - n_train)
        
        X_tr = X_all[train_start:test_start]; y_tr = y_all[train_start:test_start]
        if len(y_tr) < 100: continue
        
        model = xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05,
                                   subsample=0.8, colsample_bytree=0.8, random_state=42,
                                   n_jobs=1, verbosity=0)
        model.fit(X_tr[-1000:], y_tr[-1000:])
        
        tdf = df.iloc[test_start:min(len(df), test_end+71)].copy().reset_index(drop=True)
        if len(tdf) < 72: continue
        
        t25 = run_v25(tdf, model, cfg)
        s25 = compound_stats(t25)
        if s25['n'] > 0: v25_compound.append(s25['eq'])
        
        t28 = run_v28(tdf, model, cfg)
        s28 = compound_stats(t28)
        if s28['n'] > 0: v28_compound.append(s28['eq'])
    
    n_wf = len(v25_compound)
    print(f"  ✅ {n_wf}次WF完成")
    
    if n_wf == 0: continue
    
    def wf_report(name, eqs):
        arr = np.array(eqs)
        cum = np.prod(arr)
        pos_rate = (arr > 1.0).sum() / len(arr)
        mean_ret = (np.mean(arr) - 1) * 100
        # Compound MDD
        curve = np.cumprod(arr); peak = np.maximum.accumulate(curve)
        cmdd = np.min((curve - peak) / peak) * 100
        return f"{name}: {len(arr)}窗 正{pos_rate:.0%} 均值{mean_ret:+.1f}% 累计{(cum-1)*100:+.1f}% MDD{cmdd:.1f}%"
    
    print(f"  {wf_report('V25', v25_compound)}")
    print(f"  {wf_report('V28', v28_compound)}")
    
    c25 = np.prod(v25_compound); c28 = np.prod(v28_compound)
    better = sum(1 for a,b in zip(v28_compound,v25_compound) if a>b)/n_wf
    winner = 'V28' if c28 > c25 else 'V25'
    print(f"  {'─'*50}")
    print(f"  {winner} 🏆 | 累计差{(c28-c25)*100:+.1f}% | V28优于V25: {better:.0%}窗")

print(f"\n{'='*60}")
print(f"  V28 Full WF 完成")
print(f"{'='*60}")
