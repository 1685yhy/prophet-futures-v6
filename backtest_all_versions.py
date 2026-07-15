#!/usr/bin/env python3
"""
四版本真实连续回测: V25 / V28 / V29
• 统一方法论: 前60%训练 → 后40%回测, 每30天重训
• 本金¥300,000一直滚, 不断链
• 全部足量数据(1500天)
"""
import numpy as np, pandas as pd, os, pickle, time
from datetime import datetime, timedelta
import akshare as ak
import xgboost as xgb

CAPITAL = 300000
RETRAIN_EVERY = 30
TRAIN_WINDOW = 250

CFG = {'code': 'LH0', 'name': '生猪LH', 'multiplier': 16, 'cost': 0.0006,
       'max_pos': 6, 'max_total': 12,
       'add_conf': 0.65, 'add_atr': 2.0, 'reduce_conf': 0.55,
       'trail_atr': 2.0, 'be_atr': 1.0, 'min_hold': 3}

VERSIONS = {
    'V25': {
        'model_file': 'lh2609_xgb.pkl',
        'desc': '固定止损+模型退出',
        'atr_stop_mult': 1.5, 'rr': 4.0,
        'reverse_conf': 0.35, 'entry_conf': 0.50,
        'trail_atr': 2.0, 'be_atr': 1.0, 'min_hold': 3,
        'dynamic': False,
    },
    'V28': {
        'model_file': 'lh2609_xgb.pkl',
        'desc': '动态加仓/减仓/反手',
        'atr_stop_mult': 1.5, 'rr': 4.0,
        'reverse_conf': 0.35, 'entry_conf': 0.50,
        'trail_atr': 2.0, 'be_atr': 1.0, 'min_hold': 3,
        'add_conf': 0.65, 'add_atr': 2.0, 'reduce_conf': 0.55,
        'dynamic': True,
    },
    'V29': {
        'model_file': 'lh2609_xgb_new.pkl',
        'desc': '校准模型+动态策略',
        'atr_stop_mult': 1.5, 'rr': 4.0,
        'reverse_conf': 0.35, 'entry_conf': 0.50,
        'trail_atr': 2.0, 'be_atr': 1.0, 'min_hold': 3,
        'add_conf': 0.65, 'add_atr': 2.0, 'reduce_conf': 0.55,
        'dynamic': True,
    },
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

def train_model(df, up_to_idx, train_window=TRAIN_WINDOW):
    Xt,yt=[],[]
    end=up_to_idx; start=max(70,end-train_window)
    for i in range(start,end-1):
        feats=build_features(df,i,60)
        if feats is not None:
            Xt.append(feats)
            yt.append(1 if float(df.iloc[i+1]['close'])>float(df.iloc[i]['close']) else 0)
    if len(yt)<100: return None
    Xt=np.array(Xt,dtype=np.float32); yt=np.array(yt)
    model=xgb.XGBClassifier(n_estimators=100,max_depth=3,learning_rate=0.03,
                             subsample=0.7,colsample_bytree=0.7,
                             reg_alpha=1.0,reg_lambda=1.0,
                             random_state=42,n_jobs=1,verbosity=0)
    model.fit(Xt,yt)
    return model

def calc_pos_size_v25(price, atr, cash):
    atr_pct=atr/price
    if atr_pct<0.01: lev=3.0
    elif atr_pct<0.02: lev=2.0
    elif atr_pct<0.03: lev=1.5
    elif atr_pct<0.05: lev=0.5
    else: lev=0
    ratio=cash/CAPITAL
    lots=max(1,int(lev*3*ratio)) if lev>0 else 0
    if cash<100000: lots=max(1,lots//2) if lots>0 else 0
    return min(lots,CFG['max_pos'])

def calc_pos_size(price, atr, cash):
    atr_pct=atr/price
    if atr_pct<0.01: lev=3.0
    elif atr_pct<0.02: lev=2.0
    elif atr_pct<0.03: lev=1.5
    elif atr_pct<0.05: lev=0.5
    else: lev=0
    ratio=cash/CAPITAL
    lots=max(1,int(lev*(CFG['max_pos']//2)*ratio)) if lev>0 else 0
    if cash<100000: lots=max(1,lots//2) if lots>0 else 0
    return min(lots,CFG['max_pos'])


def run_v25(df, model, test_start, n_total, ver_cfg):
    """V25: 单一持仓, ATR止损+止盈, 模型退出"""
    p=ver_cfg; positions=[]; total_lots=0
    cash=CAPITAL; running_pnl=0; trade_count=0; rows=[]

    for day_i,gi in enumerate(range(test_start,n_total)):
        if day_i>0 and day_i%RETRAIN_EVERY==0:
            model=train_model(df,gi)
        feats=build_features(df,gi,60)
        if feats is None or model is None: continue
        prob=float(model.predict_proba(feats.reshape(1,-1))[0][1])
        price=float(df.iloc[gi]['close'])
        high=float(df.iloc[gi]['high']); low=float(df.iloc[gi]['low'])
        atr=calc_atr(df,gi,20)
        if atr is None or price<=0: continue
        cur_date=str(df.iloc[gi]['date'])
        prev_close=float(df.iloc[gi-1]['close']) if gi>0 else price
        change_pct=round((price/prev_close-1)*100,2)
        today_trades=[]

        # 管理持仓 (V25: 单一持仓)
        surviving=[]
        for pos in positions:
            d,entry,trail,entry_i,vol,entry_date=pos
            bars=gi-entry_i
            pnl_pct=(price-entry)/entry if d=='LONG' else (entry-price)/entry
            pnl_atr=pnl_pct*entry/atr if atr>0 else 0

            if d=='LONG':
                hard_stop=price-atr*p['atr_stop_mult']
                tp=entry+atr*p['atr_stop_mult']*p['rr']
                if pnl_atr>p['trail_atr']: trail=max(trail,price-atr*(p['atr_stop_mult']-0.3))
                if pnl_atr>p['be_atr']: trail=max(trail,entry)
                eff_stop=max(hard_stop,trail)
                model_exit=(prob<p['reverse_conf'] and bars>=p['min_hold'])
                tp_hit=(high>=tp)

                if low<=eff_stop:
                    ep=eff_stop
                    gross=vol*CFG['multiplier']*(ep-entry)
                    fee=vol*CFG['multiplier']*(entry+ep)*CFG['cost']
                    net=gross-fee; cash+=net+vol*entry*CFG['multiplier']*0.15
                    running_pnl+=net; trade_count+=1
                    ret_pct=round((ep/entry-1)*100,2)
                    today_trades.append(f"止损平多{vol}手 @{entry:.0f}→{ep:.0f} | {ret_pct:+.1f}% ¥{net:+,.0f} | 费¥{fee:.0f}")
                    total_lots-=vol
                elif tp_hit:
                    gross=vol*CFG['multiplier']*(tp-entry)
                    fee=vol*CFG['multiplier']*(entry+tp)*CFG['cost']
                    net=gross-fee; cash+=net+vol*entry*CFG['multiplier']*0.15
                    running_pnl+=net; trade_count+=1
                    today_trades.append(f"止盈平多{vol}手 @{entry:.0f}→{tp:.0f} | +{p['rr']*100:.0f}%R ¥{net:+,.0f}")
                    total_lots-=vol
                elif model_exit:
                    gross=vol*CFG['multiplier']*(price-entry)
                    fee=vol*CFG['multiplier']*(entry+price)*CFG['cost']
                    net=gross-fee; cash+=net+vol*entry*CFG['multiplier']*0.15
                    running_pnl+=net; trade_count+=1
                    ret_pct=round((price/entry-1)*100,2)
                    today_trades.append(f"模型平多{vol}手 @{entry:.0f}→{price:.0f} | {ret_pct:+.1f}% ¥{net:+,.0f}")
                    total_lots-=vol
                else:
                    surviving.append((d,entry,trail,entry_i,vol,entry_date))
            else:
                hard_stop=price+atr*p['atr_stop_mult']
                tp=entry-atr*p['atr_stop_mult']*p['rr']
                if pnl_atr>p['trail_atr']: trail=min(trail,price+atr*(p['atr_stop_mult']-0.3))
                if pnl_atr>p['be_atr']: trail=min(trail,entry)
                eff_stop=min(hard_stop,trail)
                model_exit=(prob>1-p['reverse_conf'] and bars>=p['min_hold'])
                tp_hit=(low<=tp)

                if high>=eff_stop:
                    ep=eff_stop
                    gross=vol*CFG['multiplier']*(entry-ep)
                    fee=vol*CFG['multiplier']*(entry+ep)*CFG['cost']
                    net=gross-fee; cash+=net+vol*entry*CFG['multiplier']*0.15
                    running_pnl+=net; trade_count+=1
                    ret_pct=round((entry/ep-1)*100,2)
                    today_trades.append(f"止损平空{vol}手 @{entry:.0f}→{ep:.0f} | {ret_pct:+.1f}% ¥{net:+,.0f}")
                    total_lots-=vol
                elif tp_hit:
                    gross=vol*CFG['multiplier']*(entry-tp)
                    fee=vol*CFG['multiplier']*(entry+tp)*CFG['cost']
                    net=gross-fee; cash+=net+vol*entry*CFG['multiplier']*0.15
                    running_pnl+=net; trade_count+=1
                    today_trades.append(f"止盈平空{vol}手 @{entry:.0f}→{tp:.0f} | +{p['rr']*100:.0f}%R ¥{net:+,.0f}")
                    total_lots-=vol
                elif model_exit:
                    gross=vol*CFG['multiplier']*(entry-price)
                    fee=vol*CFG['multiplier']*(entry+price)*CFG['cost']
                    net=gross-fee; cash+=net+vol*entry*CFG['multiplier']*0.15
                    running_pnl+=net; trade_count+=1
                    ret_pct=round((entry/price-1)*100,2)
                    today_trades.append(f"模型平空{vol}手 @{entry:.0f}→{price:.0f} | {ret_pct:+.1f}% ¥{net:+,.0f}")
                    total_lots-=vol
                else:
                    surviving.append((d,entry,trail,entry_i,vol,entry_date))
        positions=surviving

        # 开仓 (V25: 无持仓时开)
        if not positions:
            atr_pct=atr/price; ps=calc_pos_size_v25(price,atr,cash)
            if ps>0 and ps<=CFG['max_total']:
                sd='LONG' if prob>0.5 else 'SHORT'; sd2=atr*p['atr_stop_mult']
                margin=ps*price*CFG['multiplier']*0.15
                if margin<=cash:
                    if sd=='LONG' and low>price-sd2 and (prob>0.55 or prob<0.45):
                        cash-=margin; positions.append((sd,price,price-sd2,gi,ps,cur_date)); total_lots+=ps
                        today_trades.append(f"开多{ps}手 @{price:.0f} 止{price-sd2:.0f} 盈{price+sd2*p['rr']:.0f}")
                    elif sd=='SHORT' and high<price+sd2 and (prob>0.55 or prob<0.45):
                        cash-=margin; positions.append((sd,price,price+sd2,gi,ps,cur_date)); total_lots+=ps
                        today_trades.append(f"开空{ps}手 @{price:.0f} 止{price+sd2:.0f} 盈{price-sd2*p['rr']:.0f}")

        floating=0; margin_locked=0
        for pos in positions:
            d,entry,_,_,vol,_=pos
            floating+=vol*CFG['multiplier']*((price-entry) if d=='LONG' else (entry-price))
            margin_locked+=vol*entry*CFG['multiplier']*0.15
        net_asset=cash+margin_locked+floating
        pos_desc='空仓'
        if positions:
            pos_desc=f"{'多' if positions[0][0]=='LONG' else '空'}{total_lots}手"
        avg_entry=round(np.mean([p[1] for p in positions]),0) if positions else 0

        rows.append({
            '日期':cur_date,'开盘':round(float(df.iloc[gi]['open']),0),
            '最高':round(high,0),'最低':round(low,0),'收盘':round(price,0),
            '涨跌幅%':change_pct,'持仓':pos_desc,'持仓均价':avg_entry if avg_entry>0 else '',
            '浮盈¥':round(floating,0),'保证金¥':round(margin_locked,0),
            '累计实盈¥':round(running_pnl,0),'净资产¥':round(net_asset,0),
            '收益率%':round((net_asset-CAPITAL)/CAPITAL*100,1),
            '操作记录':'\n'.join(today_trades) if today_trades else '无操作',
        })

    # EOD
    lp=float(df.iloc[-1]['close'])
    for pos in positions:
        d,entry,_,_,vol,_=pos
        gross=vol*CFG['multiplier']*((lp-entry) if d=='LONG' else (entry-lp))
        fee=vol*CFG['multiplier']*(entry+lp)*CFG['cost']
        cash+=gross-fee; running_pnl+=gross-fee
    final_net=cash
    return rows,final_net,trade_count


def run_dynamic(df, model, test_start, n_total, ver_cfg):
    """V28/V29: 动态加仓/减仓/反手"""
    p=ver_cfg; positions=[]; total_lots=0
    cash=CAPITAL; running_pnl=0; trade_count=0; rows=[]

    for day_i,gi in enumerate(range(test_start,n_total)):
        if day_i>0 and day_i%RETRAIN_EVERY==0:
            model=train_model(df,gi)
        feats=build_features(df,gi,60)
        if feats is None or model is None: continue
        prob=float(model.predict_proba(feats.reshape(1,-1))[0][1])
        price=float(df.iloc[gi]['close'])
        high=float(df.iloc[gi]['high']); low=float(df.iloc[gi]['low'])
        atr=calc_atr(df,gi,20)
        if atr is None or price<=0: continue
        cur_date=str(df.iloc[gi]['date'])
        prev_close=float(df.iloc[gi-1]['close']) if gi>0 else price
        change_pct=round((price/prev_close-1)*100,2)
        cur_dir='LONG' if prob>0.5 else 'SHORT'
        conf=prob if prob>0.5 else 1-prob
        today_trades=[]

        surviving=[]
        for pos in positions:
            d,entry,trail,entry_i,vol,entry_date=pos
            bars=gi-entry_i
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
                    gross=vol*CFG['multiplier']*(ep-entry)
                    fee=vol*CFG['multiplier']*(entry+ep)*CFG['cost']
                    net=gross-fee; cash+=net+vol*entry*CFG['multiplier']*0.15
                    running_pnl+=net; trade_count+=1
                    ret_pct=round((ep/entry-1)*100,2)
                    today_trades.append(f"止损多{vol}手 @{entry:.0f}→{ep:.0f} | {ret_pct:+.1f}% ¥{net:+,.0f} | 费¥{fee:.0f}")
                    total_lots-=vol
                elif should_reverse:
                    gross=vol*CFG['multiplier']*(price-entry)
                    fee=vol*CFG['multiplier']*(entry+price)*CFG['cost']
                    net=gross-fee; cash+=net+vol*entry*CFG['multiplier']*0.15
                    running_pnl+=net; trade_count+=1
                    ret_pct=round((price/entry-1)*100,2)
                    today_trades.append(f"反手多{vol}手 @{entry:.0f}→{price:.0f} | {ret_pct:+.1f}% ¥{net:+,.0f}")
                    total_lots-=vol
                elif should_reduce and vol>1:
                    cut=vol//2
                    gross=cut*CFG['multiplier']*(price-entry)
                    fee=cut*CFG['multiplier']*(entry+price)*CFG['cost']
                    net=gross-fee; cash+=net+cut*entry*CFG['multiplier']*0.15
                    running_pnl+=net; trade_count+=1
                    ret_pct=round((price/entry-1)*100,2)
                    today_trades.append(f"减多{cut}/{vol}手 @{entry:.0f}→{price:.0f} | {ret_pct:+.1f}% ¥{net:+,.0f}")
                    total_lots-=cut
                    surviving.append((d,entry,trail,entry_i,vol-cut,entry_date))
                else: surviving.append((d,entry,trail,entry_i,vol,entry_date))
            else:
                hard_stop=price+atr*p['atr_stop_mult']
                if pnl_atr>p['trail_atr']: trail=min(trail,price+atr*(p['atr_stop_mult']-0.3))
                if pnl_atr>p['be_atr']: trail=min(trail,entry)
                eff_stop=min(hard_stop,trail)
                should_reduce=(cur_dir=='SHORT' and conf<p['reduce_conf'] and bars>=p['min_hold'])
                should_reverse=(prob>1-p['reverse_conf'] and bars>=p['min_hold'])
                if high>=eff_stop:
                    ep=eff_stop
                    gross=vol*CFG['multiplier']*(entry-ep)
                    fee=vol*CFG['multiplier']*(entry+ep)*CFG['cost']
                    net=gross-fee; cash+=net+vol*entry*CFG['multiplier']*0.15
                    running_pnl+=net; trade_count+=1
                    ret_pct=round((entry/ep-1)*100,2)
                    today_trades.append(f"止损空{vol}手 @{entry:.0f}→{ep:.0f} | {ret_pct:+.1f}% ¥{net:+,.0f}")
                    total_lots-=vol
                elif should_reverse:
                    gross=vol*CFG['multiplier']*(entry-price)
                    fee=vol*CFG['multiplier']*(entry+price)*CFG['cost']
                    net=gross-fee; cash+=net+vol*entry*CFG['multiplier']*0.15
                    running_pnl+=net; trade_count+=1
                    ret_pct=round((entry/price-1)*100,2)
                    today_trades.append(f"反手空{vol}手 @{entry:.0f}→{price:.0f} | {ret_pct:+.1f}% ¥{net:+,.0f}")
                    total_lots-=vol
                elif should_reduce and vol>1:
                    cut=vol//2
                    gross=cut*CFG['multiplier']*(entry-price)
                    fee=cut*CFG['multiplier']*(entry+price)*CFG['cost']
                    net=gross-fee; cash+=net+cut*entry*CFG['multiplier']*0.15
                    running_pnl+=net; trade_count+=1
                    ret_pct=round((entry/price-1)*100,2)
                    today_trades.append(f"减空{cut}/{vol}手 @{entry:.0f}→{price:.0f} | {ret_pct:+.1f}% ¥{net:+,.0f}")
                    total_lots-=cut
                    surviving.append((d,entry,trail,entry_i,vol-cut,entry_date))
                else: surviving.append((d,entry,trail,entry_i,vol,entry_date))
        positions=surviving

        # 开仓/加仓
        atr_pct=atr/price; ps=calc_pos_size(price,atr,cash)
        if ps>0 and total_lots+ps<=CFG['max_total']:
            sd='LONG' if prob>0.5 else 'SHORT'; sd2=atr*p['atr_stop_mult']
            margin=ps*price*CFG['multiplier']*0.15
            if margin<=cash:
                if not positions and conf>p['entry_conf']:
                    if sd=='LONG' and low>price-sd2:
                        cash-=margin; positions.append((sd,price,price-sd2,gi,ps,cur_date)); total_lots+=ps
                        today_trades.append(f"开多{ps}手 @{price:.0f} 止{price-sd2:.0f} | 保¥{margin:,.0f}")
                    elif sd=='SHORT' and high<price+sd2:
                        cash-=margin; positions.append((sd,price,price+sd2,gi,ps,cur_date)); total_lots+=ps
                        today_trades.append(f"开空{ps}手 @{price:.0f} 止{price+sd2:.0f} | 保¥{margin:,.0f}")
                elif positions and positions[0][0]==sd:
                    avg_entry=np.mean([p[1] for p in positions])
                    pa=(price-avg_entry)/atr if sd=='LONG' else (avg_entry-price)/atr
                    if conf>p['add_conf'] and pa>p['add_atr']:
                        if sd=='LONG' and low>price-sd2 and margin<=cash:
                            cash-=margin; positions.append((sd,price,price-sd2,gi,ps,cur_date)); total_lots+=ps
                            today_trades.append(f"加多{ps}手 @{price:.0f} 共{total_lots}手")
                        elif sd=='SHORT' and high<price+sd2 and margin<=cash:
                            cash-=margin; positions.append((sd,price,price+sd2,gi,ps,cur_date)); total_lots+=ps
                            today_trades.append(f"加空{ps}手 @{price:.0f} 共{total_lots}手")

        floating=0; margin_locked=0
        for pos in positions:
            d,entry,_,_,vol,_=pos
            floating+=vol*CFG['multiplier']*((price-entry) if d=='LONG' else (entry-price))
            margin_locked+=vol*entry*CFG['multiplier']*0.15
        net_asset=cash+margin_locked+floating
        pos_desc='空仓'
        if positions:
            pos_desc=f"{'多' if positions[0][0]=='LONG' else '空'}{total_lots}手"
        avg_entry=round(np.mean([p[1] for p in positions]),0) if positions else 0

        rows.append({
            '日期':cur_date,'开盘':round(float(df.iloc[gi]['open']),0),
            '最高':round(high,0),'最低':round(low,0),'收盘':round(price,0),
            '涨跌幅%':change_pct,'持仓':pos_desc,'持仓均价':avg_entry if avg_entry>0 else '',
            '浮盈¥':round(floating,0),'保证金¥':round(margin_locked,0),
            '累计实盈¥':round(running_pnl,0),'净资产¥':round(net_asset,0),
            '收益率%':round((net_asset-CAPITAL)/CAPITAL*100,1),
            '操作记录':'\n'.join(today_trades) if today_trades else '无操作',
        })

    lp=float(df.iloc[-1]['close'])
    for pos in positions:
        d,entry,_,_,vol,_=pos
        gross=vol*CFG['multiplier']*((lp-entry) if d=='LONG' else (entry-lp))
        fee=vol*CFG['multiplier']*(entry+lp)*CFG['cost']
        cash+=gross-fee; running_pnl+=gross-fee
    final_net=cash
    return rows,final_net,trade_count


# ===== MAIN =====
print("="*60)
print("  四版本真实连续回测")
print(f"  前60%训练 → 后40%回测 | 每{RETRAIN_EVERY}天重训")
print("="*60)

# 取数据
print("📡 取数据(1500天)...")
end=datetime.now(); start=end-timedelta(days=1500)
df=ak.futures_main_sina(symbol=CFG['code'],
                         start_date=start.strftime('%Y%m%d'),
                         end_date=end.strftime('%Y%m%d'))
df.columns=['date','open','high','low','close','volume','oi','settle']
for c in ['open','high','low','close','volume','oi']:
    df[c]=pd.to_numeric(df[c],errors='coerce')
df=df.dropna(subset=['close']).reset_index(drop=True)
n_total=len(df); test_start=int(n_total*0.6)
print(f"✅ {n_total}行 | 训练:{test_start}天 | 回测:{n_total-test_start}天")

# 初始模型
model_initial=train_model(df,test_start)

all_results={}; all_dfs={}

for ver_name, ver_cfg in VERSIONS.items():
    print(f"\n{'='*60}")
    print(f"  {ver_name}: {ver_cfg['desc']}")
    print(f"{'='*60}")

    t0=time.time()
    model=model_initial  # 同一初始模型
    if ver_cfg['dynamic']:
        rows,final_net,trades=run_dynamic(df,model,test_start,n_total,ver_cfg)
    else:
        rows,final_net,trades=run_v25(df,model,test_start,n_total,ver_cfg)

    ret=(final_net-CAPITAL)/CAPITAL*100
    df_out=pd.DataFrame(rows)
    peak=df_out['净资产¥'].cummax()
    mdd=(df_out['净资产¥']-peak)/peak*100
    elapsed=time.time()-t0

    print(f"  最终净资产: ¥{final_net:,.0f} | 收益率: {ret:+.1f}%")
    print(f"  最高净资产: ¥{df_out['净资产¥'].max():,.0f} | 最大回撤: {mdd.min():.1f}%")
    print(f"  交易笔数: {trades} | 耗时: {elapsed:.1f}s")

    all_results[ver_name]={
        'final':final_net,'ret':ret,'peak':df_out['净资产¥'].max(),
        'mdd':mdd.min(),'trades':trades,
    }
    all_dfs[ver_name]=df_out

# 导出
output_path='/home/a/prophet_futures/prophet_futures/backtest_all_versions.xlsx'
with pd.ExcelWriter(output_path,engine='openpyxl') as writer:
    # 汇总表
    summary_rows=[]
    for vn,r in all_results.items():
        summary_rows.append({
            '版本':vn,'策略':VERSIONS[vn]['desc'],
            '最终净资产¥':r['final'],'总收益率%':round(r['ret'],1),
            '最高净资产¥':r['peak'],'最大回撤%':round(r['mdd'],1),
            '交易笔数':r['trades'],
        })
    pd.DataFrame(summary_rows).to_excel(writer,sheet_name='汇总对比',index=False)

    # 各版本明细
    for vn,df_out in all_dfs.items():
        df_out.to_excel(writer,sheet_name=f'{vn}_每日日志',index=False)

print(f"\n{'='*60}")
print(f"  ✅ 已导出: {output_path}")
print(f"  工作表: 汇总对比 + {' + '.join(f'{v}_每日日志' for v in all_dfs)}")
print(f"\n{'='*60}")
print(f"  汇总对比:")
print(f"  {'版本':<6} {'最终¥':>12} {'收益率':>8} {'最大回撤':>8} {'交易笔数':>6}")
for vn,r in all_results.items():
    print(f"  {vn:<6} ¥{r['final']:>10,.0f} {r['ret']:>+7.1f}% {r['mdd']:>+7.1f}% {r['trades']:>5}笔")
print(f"{'='*60}")
