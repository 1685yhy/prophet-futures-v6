#!/usr/bin/env python3
"""
Prophet Futures V31 — 降回撤版 (Walk-Forward 滚动训练)
V28 vs V31 同框对比

V31 改进 (5项):
  1. 置信度门槛: 入场需 conf > 0.55 (V28: >0.50)
  2. 趋势过滤: 做多需 price > MA20, 做空需 price < MA20
  3. 更严反手: LH 0.35→0.25, JM 0.30→0.20
  4. 收紧止损: LH 1.5→1.2 ATR, JM 2.0→1.5 ATR
  5. 分批建仓: 初始 max_pos/3 (V28: max_pos/2)

Walk-Forward: 250天滚动训练 → 5天测试 → 滑动 → 重复
"""
import numpy as np, pandas as pd, os, time
from datetime import datetime, timedelta
import akshare as ak
import xgboost as xgb

# ===== CONFIG =====
SYMBOLS = {
    'lh2609': {
        'code': 'LH0', 'name': 'LH', 'multiplier': 16, 'cost': 0.0006,
        'max_pos': 6, 'max_total': 12,
        'v28': {
            'atr_stop_mult': 1.5, 'rr': 4.0,
            'add_conf': 0.65, 'add_atr': 2.0, 'reduce_conf': 0.55,
            'reverse_conf': 0.35, 'trail_atr': 2.0, 'be_atr': 1.0, 'min_hold': 3,
            'entry_conf': 0.50, 'init_frac': 2, 'trend_filter': False,
        },
        'v31': {
            'atr_stop_mult': 1.2, 'rr': 4.0,
            'add_conf': 0.68, 'add_atr': 2.0, 'reduce_conf': 0.55,
            'reverse_conf': 0.25, 'trail_atr': 2.0, 'be_atr': 1.0, 'min_hold': 3,
            'entry_conf': 0.55, 'init_frac': 3, 'trend_filter': True,
        },
    },
    'jm2609': {
        'code': 'JM0', 'name': 'JM', 'multiplier': 60, 'cost': 0.0011,
        'max_pos': 4, 'max_total': 8,
        'v28': {
            'atr_stop_mult': 2.0, 'rr': 3.5,
            'add_conf': 0.65, 'add_atr': 2.5, 'reduce_conf': 0.55,
            'reverse_conf': 0.30, 'trail_atr': 3.0, 'be_atr': 2.0, 'min_hold': 5,
            'entry_conf': 0.50, 'init_frac': 2, 'trend_filter': False,
        },
        'v31': {
            'atr_stop_mult': 1.5, 'rr': 3.5,
            'add_conf': 0.68, 'add_atr': 2.5, 'reduce_conf': 0.55,
            'reverse_conf': 0.20, 'trail_atr': 3.0, 'be_atr': 2.0, 'min_hold': 5,
            'entry_conf': 0.55, 'init_frac': 3, 'trend_filter': True,
        },
    },
}
CAPITAL = 300000


def build_features(df, idx, window=60):
    if idx < window + 5: return None
    w = df.iloc[idx - window:idx + 1]
    c = w['close'].values.astype(float); o = w['open'].values.astype(float)
    h = w['high'].values.astype(float); l = w['low'].values.astype(float)
    v = w['volume'].values.astype(float); oi_v = w['oi'].values.astype(float)
    f = []
    if idx >= 1:
        f.append(float((o[-1]-c[-2])/c[-2])); f.append(abs(f[-1]))
    else: f.extend([0.0,0.0])
    for lag in [1,3,5,10,20]:
        f.append(float((c[-1]-c[-lag-1])/c[-lag-1] if len(c)>lag else 0))
    for p in [5,10,20,60]:
        ma=np.mean(c[-min(p,len(c)):]); f.append(float((c[-1]-ma)/ma))
    f.append(float(np.std(c[-20:])/np.mean(c[-20:])))
    f.append(float((h[-1]-l[-1])/c[-1]))
    vma=np.mean(v[-20:]) if np.mean(v[-20:])>0 else 1; f.append(float(v[-1]/vma))
    f.append(float(oi_v[-1]/np.mean(oi_v[-20:])) if len(oi_v)>=20 and np.mean(oi_v[-20:])>0 else 1)
    e12=c[-1];e26=c[-1]
    for j in range(len(c)-2,-1,-1): e12=(2/13)*c[j]+(11/13)*e12; e26=(2/27)*c[j]+(25/27)*e26
    f.append(float((e12-e26)/c[-1]))
    dd=np.diff(c[-15:])
    g=float(dd[dd>0].sum()) if len(dd[dd>0])>0 else 0
    lo=float(abs(dd[dd<0].sum())) if len(dd[dd<0])>0 else 1e-10
    f.append(float(100-100/(1+g/lo) if lo>0 else 50))
    bb=np.std(c[-20:]);m20=np.mean(c[-20:])
    f.append(float((c[-1]-m20)/(2*bb+1e-10)))
    f.append(float(c[-1]/1000.0))
    return np.array(f, dtype=np.float32)

def calc_atr(df, idx, period=20):
    if idx<period: return None
    vals=[abs(float(df.iloc[i]['high'])-float(df.iloc[i]['low'])) for i in range(idx-period+1,idx+1)]
    return np.mean(vals)

def calc_pos_size(atr_pct, cfg, init_frac):
    if atr_pct<0.01: lev=3.0
    elif atr_pct<0.02: lev=2.0
    elif atr_pct<0.03: lev=1.5
    elif atr_pct<0.05: lev=0.5
    else: lev=0
    return max(1,int(lev*(cfg['max_pos']//init_frac))) if lev>0 else 0

def run_strategy(df, model, sym_key, test_start, params):
    """运行策略 (V28 或 V31 参数)"""
    cfg = SYMBOLS[sym_key]
    p = params
    init_frac = p['init_frac']
    trades = []; positions = []; total_lots = 0
    equity_curve = [CAPITAL]; cash = CAPITAL

    for i in range(test_start, len(df)):
        feats = build_features(df, i, 60)
        if feats is None: continue
        prob = float(model.predict_proba(feats.reshape(1,-1))[0][1])

        price=float(df.iloc[i]['close']); high=float(df.iloc[i]['high']); low=float(df.iloc[i]['low'])
        atr=calc_atr(df,i,20)
        if atr is None or price<=0: continue

        cur_dir='LONG' if prob>0.5 else 'SHORT'
        conf=prob if prob>0.5 else 1-prob

        # 趋势过滤 (V31 only)
        trend_long=trend_short=True
        if p.get('trend_filter'):
            ma20=float(np.mean([float(df.iloc[j]['close']) for j in range(max(0,i-19),i+1)]))
            trend_long=price>ma20; trend_short=price<ma20

        surviving=[]
        for pos in positions:
            d,entry,trail,entry_i,vol=pos
            bars=i-entry_i
            pnl_pct=(price-entry)/entry if d=='LONG' else (entry-price)/entry
            pnl_atr=pnl_pct*entry/atr if atr>0 else 0

            if d=='LONG':
                hard_stop=price-atr*p['atr_stop_mult']
                if pnl_atr>p['trail_atr']: trail=max(trail,price-atr*(p['atr_stop_mult']-0.3))
                if pnl_atr>p['be_atr']: trail=max(trail,entry)
                eff_stop=max(hard_stop,trail)
                should_reduce=(cur_dir=='LONG' and conf<p['reduce_conf'] and bars>=p['min_hold'])
                should_reverse=(prob<p['reverse_conf'] and bars>=p['min_hold'])

                if low<=eff_stop:
                    ep=eff_stop
                    pnl=vol*cfg['multiplier']*(ep-entry)
                    cash+=pnl+vol*entry*cfg['multiplier']*0.15
                    trades.append({'pnl':pnl,'bars':bars,'type':'STOP'}); total_lots-=vol
                elif should_reverse:
                    pnl=vol*cfg['multiplier']*(price-entry)
                    cash+=pnl+vol*entry*cfg['multiplier']*0.15
                    trades.append({'pnl':pnl,'bars':bars,'type':'REVERSE'}); total_lots-=vol
                elif should_reduce and vol>1:
                    cut=vol//2
                    pnl=cut*cfg['multiplier']*(price-entry)
                    cash+=pnl+cut*entry*cfg['multiplier']*0.15
                    trades.append({'pnl':pnl,'bars':bars,'type':'REDUCE'}); total_lots-=cut
                    surviving.append((d,entry,trail,entry_i,vol-cut))
                else:
                    surviving.append((d,entry,trail,entry_i,vol))
            else:
                hard_stop=price+atr*p['atr_stop_mult']
                if pnl_atr>p['trail_atr']: trail=min(trail,price+atr*(p['atr_stop_mult']-0.3))
                if pnl_atr>p['be_atr']: trail=min(trail,entry)
                eff_stop=min(hard_stop,trail)
                should_reduce=(cur_dir=='SHORT' and conf<p['reduce_conf'] and bars>=p['min_hold'])
                should_reverse=(prob>1-p['reverse_conf'] and bars>=p['min_hold'])

                if high>=eff_stop:
                    ep=eff_stop
                    pnl=vol*cfg['multiplier']*(entry-ep)
                    cash+=pnl+vol*entry*cfg['multiplier']*0.15
                    trades.append({'pnl':pnl,'bars':bars,'type':'STOP'}); total_lots-=vol
                elif should_reverse:
                    pnl=vol*cfg['multiplier']*(entry-price)
                    cash+=pnl+vol*entry*cfg['multiplier']*0.15
                    trades.append({'pnl':pnl,'bars':bars,'type':'REVERSE'}); total_lots-=vol
                elif should_reduce and vol>1:
                    cut=vol//2
                    pnl=cut*cfg['multiplier']*(entry-price)
                    cash+=pnl+cut*entry*cfg['multiplier']*0.15
                    trades.append({'pnl':pnl,'bars':bars,'type':'REDUCE'}); total_lots-=cut
                    surviving.append((d,entry,trail,entry_i,vol-cut))
                else:
                    surviving.append((d,entry,trail,entry_i,vol))
        positions=surviving

        atr_pct=atr/price; ps=calc_pos_size(atr_pct,cfg,init_frac)
        if ps>0 and total_lots+ps<=cfg['max_total']:
            sd='LONG' if prob>0.5 else 'SHORT'; sd2=atr*p['atr_stop_mult']
            margin=ps*price*cfg['multiplier']*0.15
            if margin>cash: continue

            if not positions:
                # 入场过滤
                entry_ok=conf>p['entry_conf']
                trend_ok=True
                if p.get('trend_filter'):
                    trend_ok=(sd=='LONG' and trend_long) or (sd=='SHORT' and trend_short)
                if entry_ok and trend_ok:
                    if sd=='LONG':
                        s_val=price-sd2
                        if low>s_val: cash-=margin; positions.append((sd,price,s_val,i,ps)); total_lots+=ps
                    else:
                        s_val=price+sd2
                        if high<s_val: cash-=margin; positions.append((sd,price,s_val,i,ps)); total_lots+=ps
            else:
                existing_dir=positions[0][0]
                if sd==existing_dir:
                    avg_entry=np.mean([p[1] for p in positions])
                    pa=(price-avg_entry)/atr if sd=='LONG' else (avg_entry-price)/atr
                    if conf>p['add_conf'] and pa>p['add_atr']:
                        if sd=='LONG':
                            s_val=price-sd2
                            if low>s_val and margin<=cash: cash-=margin; positions.append((sd,price,s_val,i,ps)); total_lots+=ps
                        else:
                            s_val=price+sd2
                            if high<s_val and margin<=cash: cash-=margin; positions.append((sd,price,s_val,i,ps)); total_lots+=ps

        floating=0
        for pos in positions:
            d,entry,_,_,vol=pos
            floating+=vol*cfg['multiplier']*((price-entry) if d=='LONG' else (entry-price))
        equity_curve.append(cash+floating)

    lp=float(df.iloc[-1]['close'])
    for pos in positions:
        d,entry,_,_,vol=pos
        pnl=vol*cfg['multiplier']*((lp-entry) if d=='LONG' else (entry-lp))
        cash+=pnl; trades.append({'pnl':pnl,'bars':len(df)-1-pos[3],'type':'EOD'})
    return trades,equity_curve

def analyze(name,trades,equity):
    if not trades: return {'trades':0,'wr':0,'ret':0,'pf':0,'mdd':0,'sharpe':0,'total_pnl':0}
    n=len(trades)
    wins=[t for t in trades if t['pnl']>0]
    losses=[t for t in trades if t['pnl']<=0]
    wr=len(wins)/n
    ret=(equity[-1]-CAPITAL)/CAPITAL
    gw=sum(t['pnl'] for t in wins); gl=abs(sum(t['pnl'] for t in losses))
    pf=gw/gl if gl>0 else 999
    eq=np.array(equity); peak=np.maximum.accumulate(eq)
    dd=(eq-peak)/(peak+1); mdd=abs(dd.min())
    rets=np.diff(eq)/(eq[:-1]+1)
    sharpe=np.mean(rets)/np.std(rets)*np.sqrt(252) if np.std(rets)>0 else 0
    types={}
    for t in trades: types[t['type']]=types.get(t['type'],0)+1
    return {'trades':n,'wr':wr,'ret':ret,'pf':pf,'mdd':mdd,'sharpe':sharpe,
            'total_pnl':sum(t['pnl'] for t in trades),'types':types}

def print_result(name,r):
    t=r['trades']
    if t==0: print(f"  {name:12s}: 无交易"); return
    print(f"  {name:12s}: {t:>4d}笔  胜率{r['wr']:>5.1%}  收益{r['ret']:>+7.1%}  "
          f"回撤{r['mdd']:>6.1%}  Sharpe{r['sharpe']:>+5.2f}  PF{r['pf']:>5.2f}  "
          f"盈亏¥{r['total_pnl']:>+.0f}")


# ===== MAIN =====
print("="*72)
print("  Prophet V31 — Walk-Forward 降回撤对比")
print(f"  模型: XGBoost滚动训练(250天) → 测试(5天) → 滑动")
print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
print("="*72)

for sym_key,cfg in SYMBOLS.items():
    print(f"\n{'='*72}")
    print(f"  {sym_key} ({cfg['name']}) — V28 vs V31 Walk-Forward")
    print(f"{'='*72}")

    # 取数据
    print(f"  📡 取数据...")
    try:
        end=datetime.now(); start=end-timedelta(days=1500)
        df=ak.futures_main_sina(symbol=cfg['code'],start_date=start.strftime('%Y%m%d'),
                                 end_date=end.strftime('%Y%m%d'))
        df.columns=['date','open','high','low','close','volume','oi','settle']
        for c in ['open','high','low','close','volume','oi']:
            df[c]=pd.to_numeric(df[c],errors='coerce')
        df=df.dropna(subset=['close']).reset_index(drop=True)
    except Exception as e:
        print(f"  ❌ {e}"); continue

    n_total=len(df); n_train=250; n_test=7; step=5
    print(f"  ✅ {n_total}行  {df.iloc[0]['date']} → {df.iloc[-1]['date']}")
    print(f"  训练窗:{n_train}天 | 测试窗:{n_test}天 | 步长:{step}天")

    # 预计算全部特征和标签
    X_all,y_all=[],[]
    for i in range(70,n_total-1):
        feats=build_features(df,i,60)
        if feats is not None:
            X_all.append(feats)
            y_all.append(1 if float(df.iloc[i+1]['close'])>float(df.iloc[i]['close']) else 0)
    X_all=np.array(X_all,dtype=np.float32); y_all=np.array(y_all)

    all_v28=[]; all_v31=[]
    v28_trades_all=[]; v31_trades_all=[]
    wf_count=0; min_train=100

    for test_start in range(n_train,n_total-n_test,step):
        if test_start+n_test>n_total: break

        train_end=test_start
        train_start=max(0,train_end-n_train)

        X_train=X_all[train_start:train_end]; y_train=y_all[train_start:train_end]
        if len(y_train)<min_train: continue

        model=xgb.XGBClassifier(n_estimators=100,max_depth=4,learning_rate=0.05,
                                subsample=0.8,colsample_bytree=0.8,random_state=42,
                                n_jobs=1,verbosity=0)
        model.fit(X_train[-min_train:],y_train[-min_train:])

        test_df_start=test_start; test_df_end=min(n_total,test_start+n_test+71)
        test_df=df.iloc[test_df_start:test_df_end].copy().reset_index(drop=True)

        if len(test_df)<2: continue

        # V28
        t28,eq28=run_strategy(test_df,model,sym_key,0,cfg['v28'])
        if t28:
            v28_trades_all.extend(t28)
            eq_final=eq28[-1]; ret=(eq_final-CAPITAL)/CAPITAL
            all_v28.append(ret)

        # V31
        t31,eq31=run_strategy(test_df,model,sym_key,0,cfg['v31'])
        if t31:
            v31_trades_all.extend(t31)
            eq_final=eq31[-1]; ret=(eq_final-CAPITAL)/CAPITAL
            all_v31.append(ret)

        wf_count+=1
        if wf_count%100==0: print(f"    WF {wf_count}...",flush=True)

    print(f"  ✅ {wf_count}次 Walk-Forward 完成")

    # 每个WF窗口的统计
    def wf_stats(arr,name):
        if not arr: return
        a=np.array(arr)
        pos=(a>0).sum()/len(a)
        mean=np.mean(a)*100; med=np.median(a)*100
        total=(np.prod(1+a)-1)*100
        cum=np.cumprod(1+a); peak=np.maximum.accumulate(cum)
        mdd=np.min((cum-peak)/peak)*100
        print(f"  {name} (WF窗口): {len(arr)}窗  正收益{pos:.0%}  "
              f"均值{mean:+.1f}%  累计{total:+.1f}%  MDD{mdd:.1f}%")

    wf_stats(all_v28,'V28')
    wf_stats(all_v31,'V31')

    # 汇总全部交易
    r28=analyze('V28',v28_trades_all,[CAPITAL]+[CAPITAL*(1+sum(t['pnl'] for t in v28_trades_all[:j+1])/CAPITAL) for j in range(len(v28_trades_all))])
    r31=analyze('V31',v31_trades_all,[CAPITAL]+[CAPITAL*(1+sum(t['pnl'] for t in v31_trades_all[:j+1])/CAPITAL) for j in range(len(v31_trades_all))])

    print(f"\n  ── 全部交易汇总 ──")
    print_result('V28 (原版)',r28)
    print_result('V31 (改进)',r31)

    if r28['trades']>0 and r31['trades']>0:
        print(f"\n  {'─'*68}")
        ret_diff=(r31['ret']-r28['ret'])*100
        mdd_diff=(r31['mdd']-r28['mdd'])*100
        pnl_diff=r31['total_pnl']-r28['total_pnl']
        print(f"  📊 收益差: {ret_diff:+.1f}pp | 回撤差: {mdd_diff:+.1f}pp | "
              f"交易差: {r31['trades']-r28['trades']:+d}笔 | 盈亏差: ¥{pnl_diff:+,.0f}")
        print(f"  📋 V28: {r28['types']}")
        print(f"  📋 V31: {r31['types']}")

print(f"\n{'='*72}")
print("  ✅ V31 Walk-Forward 对比完成")
print("="*72)

# ===== V31b: 仅降回撤,不杀收益 =====
# 保留 V28 入场逻辑, 加 3 道防线:
#   1. 日亏损上限 3%
#   2. 总仓位上限 60% 资金
#   3. 收紧移动止损触发点

SYMBOLS['lh2609']['v31b'] = {
    'atr_stop_mult': 1.3, 'rr': 4.0,
    'add_conf': 0.65, 'add_atr': 2.0, 'reduce_conf': 0.55,
    'reverse_conf': 0.35, 'trail_atr': 1.5, 'be_atr': 0.8, 'min_hold': 3,
    'entry_conf': 0.50, 'init_frac': 2, 'trend_filter': False,
    'daily_loss_cap': 0.03,  # 日亏损上限 3%
    'max_heat': 0.60,         # 总仓位不超 60% 资金
}
SYMBOLS['jm2609']['v31b'] = {
    'atr_stop_mult': 1.5, 'rr': 3.5,
    'add_conf': 0.65, 'add_atr': 2.5, 'reduce_conf': 0.55,
    'reverse_conf': 0.30, 'trail_atr': 2.0, 'be_atr': 1.5, 'min_hold': 5,
    'entry_conf': 0.50, 'init_frac': 2, 'trend_filter': False,
    'daily_loss_cap': 0.03,
    'max_heat': 0.60,
}

CAPITAL=300000

def run_v31b(df, model, sym_key, test_start, params):
    cfg = SYMBOLS[sym_key]; p = params
    init_frac = p['init_frac']
    trades = []; positions = []; total_lots = 0
    equity_curve = [CAPITAL]; cash = CAPITAL
    daily_pnl = 0; last_day = None

    for i in range(test_start, len(df)):
        feats = build_features(df, i, 60)
        if feats is None: continue
        prob = float(model.predict_proba(feats.reshape(1,-1))[0][1])

        price=float(df.iloc[i]['close']); high=float(df.iloc[i]['high']); low=float(df.iloc[i]['low'])
        atr=calc_atr(df,i,20)
        if atr is None or price<=0: continue

        cur_day = str(df.iloc[i]['date'])
        if last_day and cur_day != last_day:
            if daily_pnl < -CAPITAL * p.get('daily_loss_cap', 0.03):
                pass  # 日亏损超限, 继续跟踪但不再开新仓
            daily_pnl = 0
        last_day = cur_day

        cur_dir='LONG' if prob>0.5 else 'SHORT'
        conf=prob if prob>0.5 else 1-prob

        surviving=[]
        for pos in positions:
            d,entry,trail,entry_i,vol=pos
            bars=i-entry_i
            pnl_pct=(price-entry)/entry if d=='LONG' else (entry-price)/entry
            pnl_atr=pnl_pct*entry/atr if atr>0 else 0

            if d=='LONG':
                hard_stop=price-atr*p['atr_stop_mult']
                if pnl_atr>p['trail_atr']: trail=max(trail,price-atr*(p['atr_stop_mult']-0.3))
                if pnl_atr>p['be_atr']: trail=max(trail,entry)
                eff_stop=max(hard_stop,trail)
                should_reduce=(cur_dir=='LONG' and conf<p['reduce_conf'] and bars>=p['min_hold'])
                should_reverse=(prob<p['reverse_conf'] and bars>=p['min_hold'])

                if low<=eff_stop:
                    ep=eff_stop
                    pnl=vol*cfg['multiplier']*(ep-entry)
                    cash+=pnl+vol*entry*cfg['multiplier']*0.15; daily_pnl+=pnl
                    trades.append({'pnl':pnl,'bars':bars,'type':'STOP'}); total_lots-=vol
                elif should_reverse:
                    pnl=vol*cfg['multiplier']*(price-entry)
                    cash+=pnl+vol*entry*cfg['multiplier']*0.15; daily_pnl+=pnl
                    trades.append({'pnl':pnl,'bars':bars,'type':'REVERSE'}); total_lots-=vol
                elif should_reduce and vol>1:
                    cut=vol//2
                    pnl=cut*cfg['multiplier']*(price-entry)
                    cash+=pnl+cut*entry*cfg['multiplier']*0.15; daily_pnl+=pnl
                    trades.append({'pnl':pnl,'bars':bars,'type':'REDUCE'}); total_lots-=cut
                    surviving.append((d,entry,trail,entry_i,vol-cut))
                else:
                    surviving.append((d,entry,trail,entry_i,vol))
            else:
                hard_stop=price+atr*p['atr_stop_mult']
                if pnl_atr>p['trail_atr']: trail=min(trail,price+atr*(p['atr_stop_mult']-0.3))
                if pnl_atr>p['be_atr']: trail=min(trail,entry)
                eff_stop=min(hard_stop,trail)
                should_reduce=(cur_dir=='SHORT' and conf<p['reduce_conf'] and bars>=p['min_hold'])
                should_reverse=(prob>1-p['reverse_conf'] and bars>=p['min_hold'])

                if high>=eff_stop:
                    ep=eff_stop
                    pnl=vol*cfg['multiplier']*(entry-ep)
                    cash+=pnl+vol*entry*cfg['multiplier']*0.15; daily_pnl+=pnl
                    trades.append({'pnl':pnl,'bars':bars,'type':'STOP'}); total_lots-=vol
                elif should_reverse:
                    pnl=vol*cfg['multiplier']*(entry-price)
                    cash+=pnl+vol*entry*cfg['multiplier']*0.15; daily_pnl+=pnl
                    trades.append({'pnl':pnl,'bars':bars,'type':'REVERSE'}); total_lots-=vol
                elif should_reduce and vol>1:
                    cut=vol//2
                    pnl=cut*cfg['multiplier']*(entry-price)
                    cash+=pnl+cut*entry*cfg['multiplier']*0.15; daily_pnl+=pnl
                    trades.append({'pnl':pnl,'bars':bars,'type':'REDUCE'}); total_lots-=cut
                    surviving.append((d,entry,trail,entry_i,vol-cut))
                else:
                    surviving.append((d,entry,trail,entry_i,vol))
        positions=surviving

        atr_pct=atr/price; ps=calc_pos_size(atr_pct,cfg,init_frac)
        max_heat=p.get('max_heat',0.6)

        if ps>0 and total_lots+ps<=cfg['max_total']:
            sd='LONG' if prob>0.5 else 'SHORT'; sd2=atr*p['atr_stop_mult']
            margin=ps*price*cfg['multiplier']*0.15

            # 仓位热度检查
            total_heat=sum(pos[4]*pos[1]*cfg['multiplier']*0.15 for pos in positions)
            if (total_heat+margin)/CAPITAL > max_heat: continue
            if margin>cash: continue

            # 日亏损检查
            if daily_pnl < -CAPITAL * p.get('daily_loss_cap', 0.03): continue

            if not positions:
                entry_ok=conf>p['entry_conf']
                if entry_ok:
                    if sd=='LONG':
                        s_val=price-sd2
                        if low>s_val: cash-=margin; positions.append((sd,price,s_val,i,ps)); total_lots+=ps
                    else:
                        s_val=price+sd2
                        if high<s_val: cash-=margin; positions.append((sd,price,s_val,i,ps)); total_lots+=ps
            else:
                existing_dir=positions[0][0]
                if sd==existing_dir:
                    avg_entry=np.mean([p[1] for p in positions])
                    pa=(price-avg_entry)/atr if sd=='LONG' else (avg_entry-price)/atr
                    if conf>p['add_conf'] and pa>p['add_atr']:
                        if sd=='LONG':
                            s_val=price-sd2
                            if low>s_val and margin<=cash: cash-=margin; positions.append((sd,price,s_val,i,ps)); total_lots+=ps
                        else:
                            s_val=price+sd2
                            if high<s_val and margin<=cash: cash-=margin; positions.append((sd,price,s_val,i,ps)); total_lots+=ps

        floating=0
        for pos in positions:
            d,entry,_,_,vol=pos
            floating+=vol*cfg['multiplier']*((price-entry) if d=='LONG' else (entry-price))
        equity_curve.append(cash+floating)

    lp=float(df.iloc[-1]['close'])
    for pos in positions:
        d,entry,_,_,vol=pos
        pnl=vol*cfg['multiplier']*((lp-entry) if d=='LONG' else (entry-lp))
        cash+=pnl; trades.append({'pnl':pnl,'bars':len(df)-1-pos[3],'type':'EOD'})
    return trades,equity_curve

# Re-run comparison with V31b added
print("\n" + "="*72)
print("  🔄 追加 V31b (仅降回撤,不杀收益)")
print("="*72)

for sym_key,cfg in SYMBOLS.items():
    print(f"\n  {sym_key} ({cfg['name']}) — V31b 追测")
    try:
        end=datetime.now(); start=end-timedelta(days=1500)
        df=ak.futures_main_sina(symbol=cfg['code'],start_date=start.strftime('%Y%m%d'),
                                 end_date=end.strftime('%Y%m%d'))
        df.columns=['date','open','high','low','close','volume','oi','settle']
        for c in ['open','high','low','close','volume','oi']:
            df[c]=pd.to_numeric(df[c],errors='coerce')
        df=df.dropna(subset=['close']).reset_index(drop=True)
    except Exception as e:
        print(f"  ❌ {e}"); continue

    n_total=len(df); n_train=250; n_test=7; step=5

    X_all,y_all=[],[]
    for i in range(70,n_total-1):
        feats=build_features(df,i,60)
        if feats is not None:
            X_all.append(feats)
            y_all.append(1 if float(df.iloc[i+1]['close'])>float(df.iloc[i]['close']) else 0)
    X_all=np.array(X_all,dtype=np.float32); y_all=np.array(y_all)

    v31b_trades_all=[]
    wf_count=0; min_train=100

    for test_start in range(n_train,n_total-n_test,step):
        if test_start+n_test>n_total: break
        train_end=test_start; train_start=max(0,train_end-n_train)
        X_train=X_all[train_start:train_end]; y_train=y_all[train_start:train_end]
        if len(y_train)<min_train: continue

        model=xgb.XGBClassifier(n_estimators=100,max_depth=4,learning_rate=0.05,
                                subsample=0.8,colsample_bytree=0.8,random_state=42,
                                n_jobs=1,verbosity=0)
        model.fit(X_train[-min_train:],y_train[-min_train:])

        test_df_start=test_start; test_df_end=min(n_total,test_start+n_test+71)
        test_df=df.iloc[test_df_start:test_df_end].copy().reset_index(drop=True)
        if len(test_df)<2: continue

        t31b,eq31b=run_v31b(test_df,model,sym_key,0,cfg['v31b'])
        if t31b: v31b_trades_all.extend(t31b)
        wf_count+=1
        if wf_count%100==0: print(f"    V31b WF {wf_count}...",flush=True)

    r31b=analyze('V31b',v31b_trades_all,[CAPITAL]+[CAPITAL*(1+sum(t['pnl'] for t in v31b_trades_all[:j+1])/CAPITAL) for j in range(len(v31b_trades_all))])
    print_result('V31b(降回撤)',r31b)
    print(f"  📋 V31b: {r31b['types']}")

print(f"\n{'='*72}")
print("  ✅ 三版本对比完成: V28 vs V31 vs V31b")
print("="*72)
