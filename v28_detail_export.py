#!/usr/bin/env python3
"""
V28 Walk-Forward 完整回测明细
用滚动训练(250天窗口) → 每周重训 → 覆盖全部回测期
输出: v28_trade_detail.xlsx
每笔交易含: 日期/方向/开平仓价/手续费/净盈亏/当天OHLC/涨跌幅
"""
import numpy as np, pandas as pd, os, time
from datetime import datetime, timedelta
import akshare as ak
import xgboost as xgb

CAPITAL = 300000

SYM_CFG = {
    'lh2609': {'code': 'LH0', 'name': '生猪LH', 'multiplier': 16, 'cost': 0.0006,
               'max_pos': 6, 'max_total': 12,
               'atr_stop_mult': 1.5, 'rr': 4.0,
               'add_conf': 0.65, 'add_atr': 2.0, 'reduce_conf': 0.55,
               'reverse_conf': 0.35, 'trail_atr': 2.0, 'be_atr': 1.0, 'min_hold': 3,
               'entry_conf': 0.50, 'init_frac': 2},
    'jm2609': {'code': 'JM0', 'name': '焦煤JM', 'multiplier': 60, 'cost': 0.0011,
               'max_pos': 4, 'max_total': 8,
               'atr_stop_mult': 2.0, 'rr': 3.5,
               'add_conf': 0.65, 'add_atr': 2.5, 'reduce_conf': 0.55,
               'reverse_conf': 0.30, 'trail_atr': 3.0, 'be_atr': 2.0, 'min_hold': 5,
               'entry_conf': 0.50, 'init_frac': 2},
}

def build_features(df, idx, window=60):
    if idx < window + 5: return None
    w = df.iloc[idx - window:idx + 1]
    c = w['close'].values.astype(float); o = w['open'].values.astype(float)
    h = w['high'].values.astype(float); l = w['low'].values.astype(float)
    v = w['volume'].values.astype(float); oi_v = w['oi'].values.astype(float)
    f = []
    if idx >= 1: f.append(float((o[-1]-c[-2])/c[-2])); f.append(abs(f[-1]))
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

def run_v28_on_testdf(test_df, model, sym_key, running_pnl_start):
    """在测试切片上运行V28,返回交易记录"""
    cfg = SYM_CFG[sym_key]; p = cfg
    trades_detail = []
    positions = []
    total_lots = 0
    cash = CAPITAL + running_pnl_start
    running_pnl = running_pnl_start

    for i in range(0, len(test_df)):
        feats = build_features(test_df, i, 60)
        if feats is None: continue
        prob = float(model.predict_proba(feats.reshape(1,-1))[0][1])

        price=float(test_df.iloc[i]['close'])
        high=float(test_df.iloc[i]['high'])
        low=float(test_df.iloc[i]['low'])
        atr=calc_atr(test_df,i,20)
        if atr is None or price<=0: continue

        cur_date=str(test_df.iloc[i]['date'])
        cur_dir='LONG' if prob>0.5 else 'SHORT'
        conf=prob if prob>0.5 else 1-prob

        surviving=[]
        for pos in positions:
            d,entry,trail,entry_i,vol,entry_date=pos
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
                    gross=vol*cfg['multiplier']*(ep-entry)
                    fee=vol*cfg['multiplier']*(entry+ep)*cfg['cost']
                    net=gross-fee; cash+=net+vol*entry*cfg['multiplier']*0.15; running_pnl+=net
                    trades_detail.append({
                        '品种':cfg['name'],'开仓日':entry_date,'平仓日':cur_date,
                        '方向':'多','开仓价':round(entry,0),'平仓价':round(ep,0),
                        '手数':vol,'持天数':bars,
                        '当日高':round(high,0),'当日低':round(low,0),'当日收':round(price,0),
                        '毛盈亏(¥)':round(gross,0),'手续费(¥)':round(fee,0),
                        '净盈亏(¥)':round(net,0),'累计(¥)':round(running_pnl,0),'类型':'止损'})
                    total_lots-=vol
                elif should_reverse:
                    gross=vol*cfg['multiplier']*(price-entry)
                    fee=vol*cfg['multiplier']*(entry+price)*cfg['cost']
                    net=gross-fee; cash+=net+vol*entry*cfg['multiplier']*0.15; running_pnl+=net
                    trades_detail.append({
                        '品种':cfg['name'],'开仓日':entry_date,'平仓日':cur_date,
                        '方向':'多','开仓价':round(entry,0),'平仓价':round(price,0),
                        '手数':vol,'持天数':bars,
                        '当日高':round(high,0),'当日低':round(low,0),'当日收':round(price,0),
                        '毛盈亏(¥)':round(gross,0),'手续费(¥)':round(fee,0),
                        '净盈亏(¥)':round(net,0),'累计(¥)':round(running_pnl,0),'类型':'反手'})
                    total_lots-=vol
                elif should_reduce and vol>1:
                    cut=vol//2
                    gross=cut*cfg['multiplier']*(price-entry)
                    fee=cut*cfg['multiplier']*(entry+price)*cfg['cost']
                    net=gross-fee; cash+=net+cut*entry*cfg['multiplier']*0.15; running_pnl+=net
                    trades_detail.append({
                        '品种':cfg['name'],'开仓日':entry_date,'平仓日':cur_date,
                        '方向':'多','开仓价':round(entry,0),'平仓价':round(price,0),
                        '手数':cut,'持天数':bars,
                        '当日高':round(high,0),'当日低':round(low,0),'当日收':round(price,0),
                        '毛盈亏(¥)':round(gross,0),'手续费(¥)':round(fee,0),
                        '净盈亏(¥)':round(net,0),'累计(¥)':round(running_pnl,0),'类型':'减仓'})
                    total_lots-=cut
                    surviving.append((d,entry,trail,entry_i,vol-cut,entry_date))
                else:
                    surviving.append((d,entry,trail,entry_i,vol,entry_date))
            else:
                hard_stop=price+atr*p['atr_stop_mult']
                if pnl_atr>p['trail_atr']: trail=min(trail,price+atr*(p['atr_stop_mult']-0.3))
                if pnl_atr>p['be_atr']: trail=min(trail,entry)
                eff_stop=min(hard_stop,trail)
                should_reduce=(cur_dir=='SHORT' and conf<p['reduce_conf'] and bars>=p['min_hold'])
                should_reverse=(prob>1-p['reverse_conf'] and bars>=p['min_hold'])

                if high>=eff_stop:
                    ep=eff_stop
                    gross=vol*cfg['multiplier']*(entry-ep)
                    fee=vol*cfg['multiplier']*(entry+ep)*cfg['cost']
                    net=gross-fee; cash+=net+vol*entry*cfg['multiplier']*0.15; running_pnl+=net
                    trades_detail.append({
                        '品种':cfg['name'],'开仓日':entry_date,'平仓日':cur_date,
                        '方向':'空','开仓价':round(entry,0),'平仓价':round(ep,0),
                        '手数':vol,'持天数':bars,
                        '当日高':round(high,0),'当日低':round(low,0),'当日收':round(price,0),
                        '毛盈亏(¥)':round(gross,0),'手续费(¥)':round(fee,0),
                        '净盈亏(¥)':round(net,0),'累计(¥)':round(running_pnl,0),'类型':'止损'})
                    total_lots-=vol
                elif should_reverse:
                    gross=vol*cfg['multiplier']*(entry-price)
                    fee=vol*cfg['multiplier']*(entry+price)*cfg['cost']
                    net=gross-fee; cash+=net+vol*entry*cfg['multiplier']*0.15; running_pnl+=net
                    trades_detail.append({
                        '品种':cfg['name'],'开仓日':entry_date,'平仓日':cur_date,
                        '方向':'空','开仓价':round(entry,0),'平仓价':round(price,0),
                        '手数':vol,'持天数':bars,
                        '当日高':round(high,0),'当日低':round(low,0),'当日收':round(price,0),
                        '毛盈亏(¥)':round(gross,0),'手续费(¥)':round(fee,0),
                        '净盈亏(¥)':round(net,0),'累计(¥)':round(running_pnl,0),'类型':'反手'})
                    total_lots-=vol
                elif should_reduce and vol>1:
                    cut=vol//2
                    gross=cut*cfg['multiplier']*(entry-price)
                    fee=cut*cfg['multiplier']*(entry+price)*cfg['cost']
                    net=gross-fee; cash+=net+cut*entry*cfg['multiplier']*0.15; running_pnl+=net
                    trades_detail.append({
                        '品种':cfg['name'],'开仓日':entry_date,'平仓日':cur_date,
                        '方向':'空','开仓价':round(entry,0),'平仓价':round(price,0),
                        '手数':cut,'持天数':bars,
                        '当日高':round(high,0),'当日低':round(low,0),'当日收':round(price,0),
                        '毛盈亏(¥)':round(gross,0),'手续费(¥)':round(fee,0),
                        '净盈亏(¥)':round(net,0),'累计(¥)':round(running_pnl,0),'类型':'减仓'})
                    total_lots-=cut
                    surviving.append((d,entry,trail,entry_i,vol-cut,entry_date))
                else:
                    surviving.append((d,entry,trail,entry_i,vol,entry_date))
        positions=surviving

        atr_pct=atr/price; ps=calc_pos_size(atr_pct,cfg,p['init_frac'])
        if ps>0 and total_lots+ps<=cfg['max_total']:
            sd='LONG' if prob>0.5 else 'SHORT'; sd2=atr*p['atr_stop_mult']
            margin=ps*price*cfg['multiplier']*0.15
            if margin>cash: continue
            if not positions:
                if conf>p['entry_conf']:
                    if sd=='LONG':
                        s_val=price-sd2
                        if low>s_val:
                            cash-=margin; positions.append((sd,price,s_val,i,ps,cur_date)); total_lots+=ps
                    else:
                        s_val=price+sd2
                        if high<s_val:
                            cash-=margin; positions.append((sd,price,s_val,i,ps,cur_date)); total_lots+=ps
            else:
                existing_dir=positions[0][0]
                if sd==existing_dir:
                    avg_entry=np.mean([p[1] for p in positions])
                    pa=(price-avg_entry)/atr if sd=='LONG' else (avg_entry-price)/atr
                    if conf>p['add_conf'] and pa>p['add_atr']:
                        if sd=='LONG':
                            s_val=price-sd2
                            if low>s_val and margin<=cash: cash-=margin; positions.append((sd,price,s_val,i,ps,cur_date)); total_lots+=ps
                        else:
                            s_val=price+sd2
                            if high<s_val and margin<=cash: cash-=margin; positions.append((sd,price,s_val,i,ps,cur_date)); total_lots+=ps

    return trades_detail, running_pnl


# ===== MAIN =====
print("="*60)
print("  V28 Walk-Forward 全量回测明细")
print(f"  滚动训练(250天) → 测试(7天) → 滑动(5天)")
print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
print("="*60)

all_sheets = {}

for sym_key, cfg in SYM_CFG.items():
    print(f"\n{'='*60}")
    print(f"  {cfg['name']}")
    print(f"{'='*60}")

    print("  📡 取数据...")
    end=datetime.now(); start=end-timedelta(days=1500)
    df=ak.futures_main_sina(symbol=cfg['code'],
                             start_date=start.strftime('%Y%m%d'),
                             end_date=end.strftime('%Y%m%d'))
    df.columns=['date','open','high','low','close','volume','oi','settle']
    for c in ['open','high','low','close','volume','oi']:
        df[c]=pd.to_numeric(df[c],errors='coerce')
    df=df.dropna(subset=['close']).reset_index(drop=True)
    print(f"  ✅ {len(df)}行  {df.iloc[0]['date']} → {df.iloc[-1]['date']}")

    n_total=len(df); n_train=250; n_test=7; step=5

    # 预计算特征
    X_all,y_all=[],[]
    for i in range(70,n_total-1):
        feats=build_features(df,i,60)
        if feats is not None:
            X_all.append(feats)
            y_all.append(1 if float(df.iloc[i+1]['close'])>float(df.iloc[i]['close']) else 0)
    X_all=np.array(X_all,dtype=np.float32); y_all=np.array(y_all)

    all_trades=[]; running_total=0; wf_count=0

    for test_start in range(n_train,n_total-n_test,step):
        if test_start+n_test>n_total: break
        train_end=test_start; train_start=max(0,train_end-n_train)
        X_train=X_all[train_start:train_end]; y_train=y_all[train_start:train_end]
        if len(y_train)<100: continue

        model=xgb.XGBClassifier(n_estimators=100,max_depth=4,learning_rate=0.05,
                                subsample=0.8,colsample_bytree=0.8,random_state=42,
                                n_jobs=1,verbosity=0)
        model.fit(X_train[-100:],y_train[-100:])

        test_df_start=test_start; test_df_end=min(n_total,test_start+n_test+71)
        test_df=df.iloc[test_df_start:test_df_end].copy().reset_index(drop=True)
        if len(test_df)<2: continue

        trades,new_total=run_v28_on_testdf(test_df,model,sym_key,running_total)
        all_trades.extend(trades); running_total=new_total
        wf_count+=1
        if wf_count%50==0: print(f"    WF {wf_count}... 累计盈亏¥{running_total:,.0f}  {len(all_trades)}笔",flush=True)

    print(f"  ✅ {wf_count}次WF | {len(all_trades)}笔交易")

    if all_trades:
        df_t=pd.DataFrame(all_trades)
        total_pnl=df_t['净盈亏(¥)'].sum()
        total_fee=df_t['手续费(¥)'].sum()
        wins=df_t[df_t['净盈亏(¥)']>0]
        losses=df_t[df_t['净盈亏(¥)']<=0]
        print(f"  总盈亏: ¥{total_pnl:,.0f} | 手续费: ¥{total_fee:,.0f}")
        print(f"  胜率: {len(wins)/len(all_trades):.1%} | "
              f"均盈利¥{wins['净盈亏(¥)'].mean():,.0f} | 均亏损¥{losses['净盈亏(¥)'].mean():,.0f}")
        print(f"  类型: {df_t['类型'].value_counts().to_dict()}")
        all_sheets[cfg['name']]=df_t

    # 日线表 (全部数据,含涨跌幅)
    df_day=df.copy()
    df_day['date']=df_day['date'].astype(str)
    pc=df_day['close'].shift(1)
    df_day['涨跌幅%']=((df_day['close']-pc)/pc*100).round(2)
    df_day['累计涨跌%']=((df_day['close']/df_day['close'].iloc[0]-1)*100).round(2)
    pdf=df_day[['date','open','high','low','close','涨跌幅%','累计涨跌%']].copy()
    pdf.columns=['日期','开盘','最高','最低','收盘','涨跌幅%','累计涨跌%']
    all_sheets[f'{cfg["name"]}_日线']=pdf

# 导出
output_path='/home/a/prophet_futures/prophet_futures/v28_trade_detail.xlsx'
with pd.ExcelWriter(output_path,engine='openpyxl') as writer:
    for sname,sdf in all_sheets.items():
        safe=sname[:31]; sdf.to_excel(writer,sheet_name=safe,index=False)
        ws=writer.sheets[safe]
        for ci,cc in enumerate(ws.columns,1):
            ml=max((len(str(c.value or '')) for c in cc),default=0)
            ws.column_dimensions[ws.cell(1,ci).column_letter].width=min(ml+3,30)

print(f"\n{'='*60}")
print(f"  ✅ 已导出: {output_path}")
print(f"  工作表: {list(all_sheets.keys())}")
print("  Walk-Forward 滚动训练 → 全部 out-of-sample 交易记录")
print(f"{'='*60}")
