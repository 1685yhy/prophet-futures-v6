#!/usr/bin/env python3
"""
Prophet Futures — WF 完整对比回测
改前原版 vs 当前版本 + 滚动训练基线
WF: 500训练/30测试/15步进, 无前视偏差
"""
import sys, os, pickle, time
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from realtime_data import build_features

CAPITAL = 300000
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
PRE_DIR = os.path.join(MODEL_DIR, 'backups', 'pre_calibration_20260705_0922')

V25_CFG = {
    'lh2609': {'code':'LH0','name':'LH','multiplier':16,'cost':0.0006,'max_pos':6,
               'hard_atr':0.8,'trail_atr':2.0,'be_atr':1.0,'trail_dist':1.5,'rr':4.0},
    'jm2609': {'code':'JM0','name':'JM','multiplier':60,'cost':0.0011,'max_pos':4,
               'hard_atr':1.8,'trail_atr':3.0,'be_atr':2.0,'trail_dist':2.5,'rr':3.5},
}

V28_CFG = {
    'lh2609': {'code':'LH0','name':'LH','multiplier':16,'cost':0.0006,
               'max_pos':6,'max_total':12,'atr_stop':1.5,'rr':4.0,
               'add_conf':0.65,'add_atr':2.0,'reduce_conf':0.55,'reverse_conf':0.35,
               'trail_atr':2.0,'be_atr':1.0,'min_hold':3},
    'jm2609': {'code':'JM0','name':'JM','multiplier':60,'cost':0.0011,
               'max_pos':4,'max_total':8,'atr_stop':2.0,'rr':3.5,
               'add_conf':0.65,'add_atr':2.5,'reduce_conf':0.55,'reverse_conf':0.30,
               'trail_atr':3.0,'be_atr':2.0,'min_hold':5},
}


def fetch_data(code, days=1500):
    import akshare as ak, pandas as pd
    from datetime import timedelta
    end = datetime.now(); start = end - timedelta(days=days)
    df = ak.futures_main_sina(symbol=code, start_date=start.strftime('%Y%m%d'), end_date=end.strftime('%Y%m%d'))
    df.columns = ['date','open','high','low','close','volume','oi','settle']
    for c in ['open','high','low','close','volume','oi']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    return df.dropna(subset=['close']).reset_index(drop=True)


def calc_atr(df, idx, period=20):
    if idx < period: return None
    tr = [max(float(df.iloc[i]['high'])-float(df.iloc[i]['low']),
              abs(float(df.iloc[i]['high'])-float(df.iloc[i-1]['close'])),
              abs(float(df.iloc[i]['low'])-float(df.iloc[i-1]['close'])))
          for i in range(idx-period+1, idx+1)]
    return float(np.mean(tr))


# ── 交易逻辑 (和 v28_full_wf.py 100%一致) ──

def run_v25(df, model, cfg):
    trades = []; pos = None
    for i in range(70, len(df)):
        feats = build_features(df, i, 60)
        if feats is None: continue
        try: prob = float(model.predict_proba(feats.reshape(1,-1))[0][1])
        except: continue
        price = float(df.iloc[i]['close']); high = float(df.iloc[i]['high']); low = float(df.iloc[i]['low'])
        atr = calc_atr(df, i, 20)
        if atr is None or price <= 0: continue

        if pos:
            d, entry, stop, tp, entry_i, vol = pos
            if d == 'LONG':
                if low <= stop:
                    trades.append({'pnl':((stop-entry)/entry)-cfg['cost']*2,'bars':i-entry_i,'type':'STOP'});pos=None
                elif high >= tp:
                    trades.append({'pnl':((tp-entry)/entry)-cfg['cost']*2,'bars':i-entry_i,'type':'TP'});pos=None
            else:
                if high >= stop:
                    trades.append({'pnl':((entry-stop)/entry)-cfg['cost']*2,'bars':i-entry_i,'type':'STOP'});pos=None
                elif low <= tp:
                    trades.append({'pnl':((entry-tp)/entry)-cfg['cost']*2,'bars':i-entry_i,'type':'TP'});pos=None

        if pos is None:
            sd2=atr*cfg['hard_atr'];sd='LONG' if prob>0.5 else'SHORT'
            if sd=='LONG':
                sv=price-sd2
                if low>sv:pos=(sd,price,sv,price+sd2*cfg['rr'],i,1)
            else:
                sv=price+sd2
                if high<sv:pos=(sd,price,sv,price-sd2*cfg['rr'],i,1)

    if pos:
        d,entry,stop,tp,entry_i,vol=pos;lp=float(df.iloc[-1]['close'])
        trades.append({'pnl':((lp-entry)/entry if d=='LONG' else(entry-lp)/entry)-cfg['cost']*2,'bars':len(df)-1-entry_i,'type':'EOD'})
    return trades


def run_v28(df, model, cfg):
    trades=[];positions=[];total_lots=0;rev_bars=0
    for i in range(70,len(df)):
        feats=build_features(df,i,60)
        if feats is None:continue
        try:prob=float(model.predict_proba(feats.reshape(1,-1))[0][1])
        except:continue
        price=float(df.iloc[i]['close']);high=float(df.iloc[i]['high']);low=float(df.iloc[i]['low'])
        atr=calc_atr(df,i,20)
        if atr is None or price<=0:continue
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
                sr=(cur_dir=='LONG'and conf<cfg['reduce_conf']and bars>=cfg['min_hold'])
                sv=(prob<cfg['reverse_conf']and bars>=cfg['min_hold'])
                if low<=es:
                    trades.append({'pnl':((es-entry)/entry)-cfg['cost']*2,'bars':bars,'type':'STOP','vol':vol});total_lots-=vol
                    rev_bars+=1 if prob<0.5 else 0
                elif sv and rev_bars>=2:
                    trades.append({'pnl':pnl_pct-cfg['cost']*2,'bars':bars,'type':'REVERSE','vol':vol});total_lots-=vol;rev_bars+=1
                elif sr and vol>1:
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
                sr=(cur_dir=='SHORT'and conf<cfg['reduce_conf']and bars>=cfg['min_hold'])
                sv=(prob>1-cfg['reverse_conf']and bars>=cfg['min_hold'])
                if high>=es:
                    trades.append({'pnl':((entry-es)/entry)-cfg['cost']*2,'bars':bars,'type':'STOP','vol':vol});total_lots-=vol
                    rev_bars+=1 if prob>0.5 else 0
                elif sv and rev_bars>=2:
                    trades.append({'pnl':pnl_pct-cfg['cost']*2,'bars':bars,'type':'REVERSE','vol':vol});total_lots-=vol;rev_bars+=1
                elif sr and vol>1:
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
        trades.append({'pnl':((lp-entry)/entry if d=='LONG'else(entry-lp)/entry)-cfg['cost']*2,'bars':len(df)-1-entry_i,'type':'EOD','vol':vol})
    return trades


def compound_stats(trades):
    if not trades:return{'wr':0,'eq':1.0,'mdd':0,'n':0,'pf':0,'avg_win':0,'avg_loss':0}
    wins=[t for t in trades if t['pnl']>0];losses=[t for t in trades if t['pnl']<=0]
    wr=len(wins)/len(trades)
    eq=1.0;peak=1.0;mdd=0
    for t in trades:eq*=(1+t['pnl']);peak=max(peak,eq);mdd=min(mdd,(eq-peak)/peak)
    gw=sum(t['pnl']for t in wins);gl=abs(sum(t['pnl']for t in losses))
    pf=gw/gl if gl>0 else 99
    avg_win=np.mean([t['pnl']for t in wins])if wins else 0
    avg_loss=np.mean([t['pnl']for t in losses])if losses else 0
    return{'wr':wr,'eq':eq,'mdd':mdd,'n':len(trades),'pf':pf,'avg_win':avg_win,'avg_loss':avg_loss}


def wf_fixed_model(df, run_fn, cfg, model_path):
    """WF回测使用固定模型文件 (当前部署版本的真实表现)"""
    model = pickle.load(open(model_path,'rb'))
    X_all,y_all=[],[]
    for i in range(70,len(df)-1):
        feats=build_features(df,i,60)
        if feats is not None:X_all.append(feats);y_all.append(1 if float(df.iloc[i+1]['close'])>float(df.iloc[i]['close'])else 0)
    X_all=np.array(X_all,dtype=np.float32);y_all=np.array(y_all)
    n_feat=len(X_all);n_train,n_test,step=500,30,15
    windows=[]
    for test_start in range(n_train,n_feat-n_test,step):
        test_end=test_start+n_test
        tdf=df.iloc[test_start:min(len(df),test_end+71)].copy().reset_index(drop=True)
        if len(tdf)<72:continue
        trades=run_fn(tdf,model,cfg)
        stats=compound_stats(trades)
        if stats['n']>0:windows.append(stats['eq'])
    if not windows:return None
    arr=np.array(windows);cum=np.prod(arr)
    pos_rate=(arr>1.0).sum()/len(arr)
    curve=np.cumprod(arr);peak=np.maximum.accumulate(curve);cmdd=np.min((curve-peak)/peak)*100
    return{'windows':len(arr),'positive':f'{pos_rate:.0%}','cum_ret':f'{(cum-1)*100:+.1f}%','mdd':f'{cmdd:.1f}%','eq_mult':cum}


def wf_rolling_train(df, run_fn, cfg):
    """WF回测使用滚动训练 (方法论评估, 无前视偏差)"""
    import xgboost as xgb
    X_all,y_all=[],[]
    for i in range(70,len(df)-1):
        feats=build_features(df,i,60)
        if feats is not None:X_all.append(feats);y_all.append(1 if float(df.iloc[i+1]['close'])>float(df.iloc[i]['close'])else 0)
    X_all=np.array(X_all,dtype=np.float32);y_all=np.array(y_all)
    n_feat=len(X_all);n_train,n_test,step=500,30,15
    windows=[]
    for test_start in range(n_train,n_feat-n_test,step):
        test_end=test_start+n_test;train_start=max(0,test_start-n_train)
        X_tr=X_all[train_start:test_start];y_tr=y_all[train_start:test_start]
        if len(y_tr)<100:continue
        model=xgb.XGBClassifier(n_estimators=100,max_depth=4,learning_rate=0.05,subsample=0.8,colsample_bytree=0.8,random_state=42,n_jobs=1,verbosity=0)
        model.fit(X_tr[-1000:],y_tr[-1000:])
        tdf=df.iloc[test_start:min(len(df),test_end+71)].copy().reset_index(drop=True)
        if len(tdf)<72:continue
        trades=run_fn(tdf,model,cfg)
        stats=compound_stats(trades)
        if stats['n']>0:windows.append(stats['eq'])
    if not windows:return None
    arr=np.array(windows);cum=np.prod(arr)
    pos_rate=(arr>1.0).sum()/len(arr)
    curve=np.cumprod(arr);peak=np.maximum.accumulate(curve);cmdd=np.min((curve-peak)/peak)*100
    return{'windows':len(arr),'positive':f'{pos_rate:.0%}','cum_ret':f'{(cum-1)*100:+.1f}%','mdd':f'{cmdd:.1f}%','eq_mult':cum}


def main():
    import pandas as pd
    pd.set_option('display.width',200);pd.set_option('display.max_colwidth',30)
    print("="*75)
    print("  Prophet Futures — WF 完整对比: 改前原版 vs 当前版本")
    print(f"  资金: ¥{CAPITAL:,} | WF: 500train/30test/15step")
    print("="*75)

    for sym_name,sym,code in[('LH 生猪','lh2609','LH0'),('JM 焦煤','jm2609','JM0')]:
        print(f"\n{'─'*75}")
        print(f"  {sym_name} ({sym})")
        print(f"{'─'*75}")
        print(f"  获取数据...",end=' ',flush=True)
        df=fetch_data(code,days=1500)
        print(f"{len(df)}条日线")

        # 定义要跑的版本: (label, run_fn, cfg, model_source)
        tests=[
            # 改前原版 (pre_calibration)
            ('改前V25',run_v25,V25_CFG,f'{PRE_DIR}/{sym}_xgb.pkl'),
            ('改前V28',run_v28,V28_CFG,f'{PRE_DIR}/{sym}_xgb.pkl'),
            ('改前V29',run_v28,V28_CFG,f'{PRE_DIR}/{sym}_xgb_new.pkl'),
            # 当前版本
            ('当前V25(校)',run_v25,V25_CFG,f'{MODEL_DIR}/{sym}_xgb_calibrated.pkl'),
            ('当前V28(旧)',run_v28,V28_CFG,f'{MODEL_DIR}/{sym}_xgb.pkl'),
            ('当前V30(校)',run_v28,V28_CFG,f'{MODEL_DIR}/{sym}_xgb_calibrated.pkl'),
        ]

        results={}
        # 固定模型版本
        for label,run_fn,cfg_dict,mpath in tests:
            if not os.path.exists(mpath):
                print(f"  {label}: 模型不存在,跳过")
                continue
            t0=time.time()
            r=wf_fixed_model(df,run_fn,cfg_dict[sym],mpath)
            if r:
                results[label]=r
                print(f"  {label:<16} {r['windows']:>3}窗 正{r['positive']:>5} 累计{r['cum_ret']:>9} MDD{r['mdd']:>6} ({time.time()-t0:.1f}s)")
            else:
                print(f"  {label:<16} 无有效窗口")

        # 滚动训练基线
        t0=time.time()
        r=wf_rolling_train(df,run_v28,V28_CFG[sym])
        if r:
            results['WF训练V28']=r
            print(f"  {'WF训练V28':<16} {r['windows']:>3}窗 正{r['positive']:>5} 累计{r['cum_ret']:>9} MDD{r['mdd']:>6} ({time.time()-t0:.1f}s)")

        # 汇总表
        if results:
            print(f"\n  {'指标':<12}",end='')
            for v in results:print(f"{v:>16}",end='')
            print(f"\n  {'─'*12}─"+"─"*16*len(results))
            for m in['windows','cum_ret','mdd','eq_mult']:
                print(f"  {m:<12}",end='')
                for v in results:
                    val=results[v][m]
                    if isinstance(val,float):
                        print(f"{val:>16.2f}"if m=='eq_mult'else f"{val:>16}",end='')
                    else:
                        print(f"{val:>16}",end='')
                print()

    print(f"\n{'='*75}")
    print(f"  改前原版 = pre_calibration_20260705_0922 (校准前)")
    print(f"  当前版本 = 含 Platt Scaling 校准模型")
    print(f"  WF训练V28 = 每窗重训 (纯方法论, 无前视偏差)")
    print(f"{'='*75}")


if __name__=='__main__':
    main()
