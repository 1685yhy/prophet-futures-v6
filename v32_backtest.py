#!/usr/bin/env python3
"""
V32 — 三步改进对比 V28
1. 置信度过滤: 只做高置信度交易 (>0.60 开仓, <0.45 反手)
2. 三模型投票: XGBoost + LightGBM + CatBoost 一致才入场
3. 波动率仓位: 高波动时减半仓位
连续回测: ¥30万一直滚, 后40% out-of-sample
"""
import numpy as np, pandas as pd, os, pickle, time
from datetime import datetime, timedelta
import akshare as ak
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

CAPITAL = 300000
RETRAIN_EVERY = 30; TRAIN_WINDOW = 250

CFG = {'code': 'LH0', 'name': '生猪LH', 'multiplier': 16, 'cost': 0.0006,
       'max_pos': 6, 'max_total': 12}

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

def train_ensemble(df, up_to_idx):
    Xt,yt=[],[]
    end=up_to_idx; start=max(70,end-TRAIN_WINDOW)
    for i in range(start,end-1):
        feats=build_features(df,i,60)
        if feats is not None:
            Xt.append(feats)
            yt.append(1 if float(df.iloc[i+1]['close'])>float(df.iloc[i]['close']) else 0)
    if len(yt)<100: return None
    Xt=np.array(Xt,dtype=np.float32); yt=np.array(yt)
    models={}
    models['xgb']=xgb.XGBClassifier(n_estimators=100,max_depth=3,learning_rate=0.03,
        subsample=0.7,colsample_bytree=0.7,reg_alpha=1.0,reg_lambda=1.0,
        random_state=42,verbosity=0)
    models['xgb'].fit(Xt,yt)
    models['lgb']=lgb.LGBMClassifier(n_estimators=100,max_depth=3,learning_rate=0.03,
        subsample=0.7,colsample_bytree=0.7,reg_alpha=1.0,reg_lambda=1.0,
        random_state=42,verbose=-1)
    models['lgb'].fit(Xt,yt)
    if len(yt)<500:
        models['cb']=CatBoostClassifier(iterations=100,depth=3,learning_rate=0.03,
            random_seed=42,verbose=0)
        models['cb'].fit(Xt,yt)
    return models

def ensemble_predict(models, feats):
    """返回(概率, 一致否)"""
    if models is None: return 0.5, False
    probs=[]
    for name,m in models.items():
        try:
            p=float(m.predict_proba(feats.reshape(1,-1))[0][1])
            probs.append(p)
        except: probs.append(0.5)
    avg=np.mean(probs)
    # 三模型是否一致(都看多或都看空)
    all_long=all(p>0.5 for p in probs)
    all_short=all(p<0.5 for p in probs)
    unanimous=all_long or all_short
    return avg, unanimous

def calc_pos_size_v32(price, atr, cash):
    atr_pct=atr/price
    if atr_pct<0.01: lev=3.0
    elif atr_pct<0.02: lev=2.0
    elif atr_pct<0.03: lev=1.5
    elif atr_pct<0.05: lev=0.5
    else: lev=0
    ratio=cash/CAPITAL
    lots=max(1,int(lev*(CFG['max_pos']//2)*ratio)) if lev>0 else 0
    # 波动率调整: 高波动减半
    if atr_pct>0.03: lots=max(1,lots//2)
    if cash<100000: lots=max(1,lots//2) if lots>0 else 0
    return min(lots,CFG['max_pos'])


def run_backtest(df, models_initial, test_start, n_total, version_name, use_ensemble=False, use_conf_filter=False):
    """通用回测引擎"""
    p = {
        'atr_stop_mult': 1.5, 'add_conf': 0.65, 'add_atr': 2.0,
        'reduce_conf': 0.55, 'reverse_conf': 0.35,
        'trail_atr': 2.0, 'be_atr': 1.0, 'min_hold': 3,
        'entry_conf': 0.60 if use_conf_filter else 0.50,
    }
    models = models_initial
    positions = []; total_lots = 0
    cash = CAPITAL; running_pnl = 0; trade_count = 0; rows = []
    wins = 0; losses = 0

    for day_i, gi in enumerate(range(test_start, n_total)):
        if day_i > 0 and day_i % RETRAIN_EVERY == 0:
            if use_ensemble:
                models = train_ensemble(df, gi)
            else:
                m = train_ensemble(df, gi)
                models = m

        feats = build_features(df, gi, 60)
        if feats is None: continue
        
        if use_ensemble and models is not None:
            prob, unanimous = ensemble_predict(models, feats)
        else:
            if models is None: continue
            prob = float(models['xgb'].predict_proba(feats.reshape(1,-1))[0][1])
            unanimous = True

        price = float(df.iloc[gi]['close'])
        high = float(df.iloc[gi]['high']); low = float(df.iloc[gi]['low'])
        atr = calc_atr(df, gi, 20)
        if atr is None or price <= 0: continue

        cur_date = str(df.iloc[gi]['date'])
        cur_dir = 'LONG' if prob > 0.5 else 'SHORT'
        conf = prob if prob > 0.5 else 1 - prob
        today_trades = []

        # 管理持仓
        surviving = []
        for pos in positions:
            d, entry, trail, entry_i, vol, entry_date = pos
            bars = gi - entry_i
            pnl_pct = (price-entry)/entry if d=='LONG' else (entry-price)/entry
            pnl_atr = pnl_pct*entry/atr if atr > 0 else 0
            if d == 'LONG':
                hard_stop = price - atr*p['atr_stop_mult']
                if pnl_atr > p['trail_atr']: trail = max(trail, price-atr*(p['atr_stop_mult']-0.3))
                if pnl_atr > p['be_atr']: trail = max(trail, entry)
                eff_stop = max(hard_stop, trail)
                should_reverse = (prob < p['reverse_conf'] and bars >= p['min_hold'])
                should_reduce = (cur_dir=='LONG' and conf < p['reduce_conf'] and bars >= p['min_hold'])
                if low <= eff_stop:
                    ep = eff_stop
                    gross = vol*CFG['multiplier']*(ep-entry)
                    fee = vol*CFG['multiplier']*(entry+ep)*CFG['cost']
                    net = gross-fee
                    cash += net + vol*entry*CFG['multiplier']*0.15
                    running_pnl += net; trade_count += 1
                    if net > 0: wins += 1
                    else: losses += 1
                    today_trades.append(f"止损多{vol}手 @{entry:.0f}→{ep:.0f} ¥{net:+,.0f}")
                    total_lots -= vol
                elif should_reverse:
                    gross = vol*CFG['multiplier']*(price-entry)
                    fee = vol*CFG['multiplier']*(entry+price)*CFG['cost']
                    net = gross-fee
                    cash += net + vol*entry*CFG['multiplier']*0.15
                    running_pnl += net; trade_count += 1
                    if net > 0: wins += 1
                    else: losses += 1
                    today_trades.append(f"反手多{vol}手 @{entry:.0f}→{price:.0f} ¥{net:+,.0f}")
                    total_lots -= vol
                elif should_reduce and vol > 1:
                    cut = vol//2
                    gross = cut*CFG['multiplier']*(price-entry)
                    fee = cut*CFG['multiplier']*(entry+price)*CFG['cost']
                    net = gross-fee
                    cash += net + cut*entry*CFG['multiplier']*0.15
                    running_pnl += net; trade_count += 1
                    today_trades.append(f"减多{cut}/{vol}手 ¥{net:+,.0f}")
                    total_lots -= cut
                    surviving.append((d,entry,trail,entry_i,vol-cut,entry_date))
                else: surviving.append((d,entry,trail,entry_i,vol,entry_date))
            else:
                hard_stop = price + atr*p['atr_stop_mult']
                if pnl_atr > p['trail_atr']: trail = min(trail, price+atr*(p['atr_stop_mult']-0.3))
                if pnl_atr > p['be_atr']: trail = min(trail, entry)
                eff_stop = min(hard_stop, trail)
                should_reverse = (prob > 1-p['reverse_conf'] and bars >= p['min_hold'])
                should_reduce = (cur_dir=='SHORT' and conf < p['reduce_conf'] and bars >= p['min_hold'])
                if high >= eff_stop:
                    ep = eff_stop
                    gross = vol*CFG['multiplier']*(entry-ep)
                    fee = vol*CFG['multiplier']*(entry+ep)*CFG['cost']
                    net = gross-fee
                    cash += net + vol*entry*CFG['multiplier']*0.15
                    running_pnl += net; trade_count += 1
                    if net > 0: wins += 1
                    else: losses += 1
                    today_trades.append(f"止损空{vol}手 @{entry:.0f}→{ep:.0f} ¥{net:+,.0f}")
                    total_lots -= vol
                elif should_reverse:
                    gross = vol*CFG['multiplier']*(entry-price)
                    fee = vol*CFG['multiplier']*(entry+price)*CFG['cost']
                    net = gross-fee
                    cash += net + vol*entry*CFG['multiplier']*0.15
                    running_pnl += net; trade_count += 1
                    if net > 0: wins += 1
                    else: losses += 1
                    today_trades.append(f"反手空{vol}手 @{entry:.0f}→{price:.0f} ¥{net:+,.0f}")
                    total_lots -= vol
                elif should_reduce and vol > 1:
                    cut = vol//2
                    gross = cut*CFG['multiplier']*(entry-price)
                    fee = cut*CFG['multiplier']*(entry+price)*CFG['cost']
                    net = gross-fee
                    cash += net + cut*entry*CFG['multiplier']*0.15
                    running_pnl += net; trade_count += 1
                    today_trades.append(f"减空{cut}/{vol}手 ¥{net:+,.0f}")
                    total_lots -= cut
                    surviving.append((d,entry,trail,entry_i,vol-cut,entry_date))
                else: surviving.append((d,entry,trail,entry_i,vol,entry_date))
        positions = surviving

        # 开仓/加仓
        atr_pct = atr/price
        ps = calc_pos_size_v32(price, atr, cash)
        if ps > 0 and total_lots+ps <= CFG['max_total']:
            sd = 'LONG' if prob > 0.5 else 'SHORT'
            sd2 = atr*p['atr_stop_mult']
            margin = ps*price*CFG['multiplier']*0.15
            if margin <= cash:
                # 入场过滤
                entry_ok = conf > p['entry_conf']
                if use_conf_filter and use_ensemble:
                    entry_ok = entry_ok and unanimous  # 三模型一致+高置信度
                elif use_ensemble:
                    entry_ok = unanimous  # 仅三模型一致
                elif use_conf_filter:
                    entry_ok = conf > p['entry_conf']  # 仅高置信度

                if not positions and entry_ok:
                    if sd == 'LONG' and low > price-sd2:
                        cash -= margin
                        positions.append((sd, price, price-sd2, gi, ps, cur_date))
                        total_lots += ps
                        today_trades.append(f"开多{ps}手 @{price:.0f}止{price-sd2:.0f}")
                    elif sd == 'SHORT' and high < price+sd2:
                        cash -= margin
                        positions.append((sd, price, price+sd2, gi, ps, cur_date))
                        total_lots += ps
                        today_trades.append(f"开空{ps}手 @{price:.0f}止{price+sd2:.0f}")
                elif positions and positions[0][0] == sd:
                    avg_entry = np.mean([p[1] for p in positions])
                    pa = (price-avg_entry)/atr if sd=='LONG' else (avg_entry-price)/atr
                    if conf > p['add_conf'] and pa > p['add_atr']:
                        if sd=='LONG' and low>price-sd2 and margin<=cash:
                            cash -= margin
                            positions.append((sd,price,price-sd2,gi,ps,cur_date))
                            total_lots += ps
                            today_trades.append(f"加多{ps}手 共{total_lots}手")
                        elif sd=='SHORT' and high<price+sd2 and margin<=cash:
                            cash -= margin
                            positions.append((sd,price,price+sd2,gi,ps,cur_date))
                            total_lots += ps
                            today_trades.append(f"加空{ps}手 共{total_lots}手")

        floating = 0; margin_locked = 0
        for pos in positions:
            d, entry, _, _, vol, _ = pos
            floating += vol*CFG['multiplier']*((price-entry) if d=='LONG' else (entry-price))
            margin_locked += vol*entry*CFG['multiplier']*0.15
        net_asset = cash + margin_locked + floating
        pos_desc = '空仓'
        if positions:
            pos_desc = f"{'多' if positions[0][0]=='LONG' else '空'}{total_lots}手"
        avg_entry = round(np.mean([p[1] for p in positions]), 0) if positions else 0

        rows.append({
            '日期': cur_date, '收盘': round(price, 0), '涨跌幅%': round((price/float(df.iloc[gi-1]['close'])-1)*100, 2) if gi>0 else 0,
            '持仓': pos_desc, '持仓均价': avg_entry if avg_entry > 0 else '',
            '浮盈¥': round(floating, 0), '净资产¥': round(net_asset, 0),
            '收益率%': round((net_asset-CAPITAL)/CAPITAL*100, 1),
            '操作记录': '\n'.join(today_trades) if today_trades else '无操作',
        })

        if day_i % 100 == 0:
            print(f"    {version_name} 第{day_i}天: 净资产¥{net_asset:,.0f} ({(net_asset-CAPITAL)/CAPITAL*100:+.1f}%)  {pos_desc}")

    # EOD
    lp = float(df.iloc[-1]['close'])
    for pos in positions:
        d, entry, _, _, vol, _ = pos
        gross = vol*CFG['multiplier']*((lp-entry) if d=='LONG' else (entry-lp))
        fee = vol*CFG['multiplier']*(entry+lp)*CFG['cost']
        cash += gross-fee; running_pnl += gross-fee
    final_net = cash
    return rows, final_net, trade_count, wins, losses


# ===== MAIN =====
print("="*60)
print("  V32 改进对比 V28 — 三个版本")
print("  V28: 原始 | V32a: 置信度过滤 | V32b: 三模型投票 | V32c: 全部")
print("="*60)

print("📡 取数据...")
end=datetime.now(); start=end-timedelta(days=1500)
df=ak.futures_main_sina(symbol=CFG['code'],
    start_date=start.strftime('%Y%m%d'), end_date=end.strftime('%Y%m%d'))
df.columns=['date','open','high','low','close','volume','oi','settle']
for c in ['open','high','low','close','volume','oi']:
    df[c]=pd.to_numeric(df[c],errors='coerce')
df=df.dropna(subset=['close']).reset_index(drop=True)
n_total=len(df); test_start=int(n_total*0.6)
print(f"✅ {n_total}行 回测:{n_total-test_start}天")

# 训练初始模型
print("🔧 训练初始模型...")
models_init=train_ensemble(df,test_start)

# 三个版本
configs=[
    ('V28_原版', False, False),
    ('V32a_置信度过滤', True, False),
    ('V32b_三模型投票', False, True),
    ('V32c_置信度+投票', True, True),
]

results={}

for name, use_conf, use_ens in configs:
    print(f"\n{'─'*50}")
    print(f"  {name}")
    t0=time.time()
    rows,fn,tc,w,l=run_backtest(df,models_init,test_start,n_total,name,use_ens,use_conf)
    ret=(fn-CAPITAL)/CAPITAL*100
    df_out=pd.DataFrame(rows)
    peak=df_out['净资产¥'].cummax()
    mdd=(df_out['净资产¥']-peak)/peak*100
    elapsed=time.time()-t0
    wr=w/(w+l)*100 if w+l>0 else 0
    print(f"  → 净资产¥{fn:,.0f} | 收益{ret:+.1f}% | 回撤{mdd.min():.1f}% | {tc}笔 | 胜率{wr:.0f}% | {elapsed:.1f}s")
    results[name]={'final':fn,'ret':ret,'mdd':mdd.min(),'trades':tc,'wr':wr,'rows':df_out}

# 导出
output='/home/a/prophet_futures/prophet_futures/v32_compare.xlsx'
with pd.ExcelWriter(output,engine='openpyxl') as writer:
    sr=[]
    for n,r in results.items():
        sr.append({'版本':n,'最终¥':r['final'],'收益率%':round(r['ret'],1),
                   '最大回撤%':round(r['mdd'],1),'交易':r['trades'],'胜率%':round(r['wr'],0)})
    pd.DataFrame(sr).to_excel(writer,sheet_name='汇总',index=False)
    for n,r in results.items():
        r['rows'].to_excel(writer,sheet_name=n[:31],index=False)

print(f"\n{'='*60}")
print(f"  ✅ {output}")
for n,r in results.items():
    print(f"  {n:<18s}: ¥{r['final']:>10,.0f}  {r['ret']:>+6.1f}%  DD{r['mdd']:>+6.1f}%  {r['trades']:>3}笔")
