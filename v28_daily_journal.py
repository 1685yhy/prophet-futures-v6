#!/usr/bin/env python3
"""
V28 生猪LH — 真实连续回测
• 初始本金 ¥300,000，之后一直滚
• 每30天用最近250天数据重训模型（防过拟合）
• 回测期: 后40%数据 (真正的out-of-sample)
• 训练数据量: 前60% = ~600天
"""
import numpy as np, pandas as pd, os, time
from datetime import datetime, timedelta
import akshare as ak
import xgboost as xgb

CAPITAL = 300000
CFG = {'code': 'LH0', 'name': '生猪LH', 'multiplier': 16, 'cost': 0.0006,
       'max_pos': 6, 'max_total': 12,
       'atr_stop_mult': 1.5, 'add_conf': 0.65, 'add_atr': 2.0,
       'reduce_conf': 0.55, 'reverse_conf': 0.35,
       'trail_atr': 2.0, 'be_atr': 1.0, 'min_hold': 3,
       'entry_conf': 0.50, 'init_frac': 2}

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

def calc_pos_size(price, atr, cash, init_frac):
    """仓位大小用当前资金比例"""
    atr_pct = atr/price
    if atr_pct<0.01: lev=3.0
    elif atr_pct<0.02: lev=2.0
    elif atr_pct<0.03: lev=1.5
    elif atr_pct<0.05: lev=0.5
    else: lev=0
    if lev==0: return 0
    # 用当前资金比例
    ratio = cash / CAPITAL
    base = max(1, int(lev * (CFG['max_pos']//init_frac)))
    lots = max(0, int(base * ratio))
    if cash < 100000: lots = max(1, lots//2) if lots>0 else 0
    return min(lots, CFG['max_pos'])


# ===== MAIN =====
print("="*60)
print("  V28 生猪LH — 真实连续回测（本金滚动）")
print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
print("="*60)

# 取全部数据
print("📡 取数据(1500天)...")
end=datetime.now(); start=end-timedelta(days=1500)
df=ak.futures_main_sina(symbol=CFG['code'],
                         start_date=start.strftime('%Y%m%d'),
                         end_date=end.strftime('%Y%m%d'))
df.columns=['date','open','high','low','close','volume','oi','settle']
for c in ['open','high','low','close','volume','oi']:
    df[c]=pd.to_numeric(df[c],errors='coerce')
df=df.dropna(subset=['close']).reset_index(drop=True)
n_total=len(df)
print(f"✅ {n_total}行  {df.iloc[0]['date']} → {df.iloc[-1]['date']}")

# 切分
train_end = int(n_total * 0.6)  # 前60%用于训练
test_start = train_end          # 后40%用于回测
print(f"训练数据: 0→{train_end} ({train_end}天)")
print(f"回测数据: {test_start}→{n_total} ({n_total-test_start}天)")

# 回测区间日期
test_dates = [str(df.iloc[i]['date']) for i in range(test_start, n_total)]

# 初始化模型（用前60%数据）
X_train, y_train = [], []
for i in range(70, train_end-1):
    feats = build_features(df, i, 60)
    if feats is not None:
        X_train.append(feats)
        y_train.append(1 if float(df.iloc[i+1]['close'])>float(df.iloc[i]['close']) else 0)
X_train = np.array(X_train, dtype=np.float32); y_train = np.array(y_train)
print(f"初始训练样本: {len(X_train)}条")

# 训练模型（抗过拟合参数）
model = xgb.XGBClassifier(
    n_estimators=100, max_depth=3, learning_rate=0.03,
    subsample=0.7, colsample_bytree=0.7,
    reg_alpha=1.0, reg_lambda=1.0,
    random_state=42, n_jobs=1, verbosity=0
)
model.fit(X_train, y_train)
train_acc = (model.predict(X_train)==y_train).mean()
print(f"✅ 初始模型训练完成 | 训练准确率: {train_acc:.1%}")

# ===== 连续回测 =====
p = CFG; init_frac = p['init_frac']
positions = []     # [(dir, entry, trail, entry_idx, vol, entry_date)]
total_lots = 0
cash = CAPITAL     # 当前现金（已扣保证金）
running_pnl = 0    # 累计已实现盈亏
retrain_every = 30 # 每30天重训

rows = []
trade_count = 0

for day_i, global_idx in enumerate(range(test_start, n_total)):
    # 定期重训
    if day_i > 0 and day_i % retrain_every == 0:
        retrain_end = global_idx
        retrain_start = max(70, retrain_end - 250)
        Xr, yr = [], []
        for j in range(retrain_start, retrain_end-1):
            feats = build_features(df, j, 60)
            if feats is not None:
                Xr.append(feats)
                yr.append(1 if float(df.iloc[j+1]['close'])>float(df.iloc[j]['close']) else 0)
        if len(yr) >= 100:
            Xr = np.array(Xr, dtype=np.float32); yr = np.array(yr)
            model.fit(Xr, yr)
            acc = (model.predict(Xr)==yr).mean()
            print(f"  🔄 第{day_i}天重训 | 样本{len(yr)} | 准确率{acc:.1%} | "
                  f"净资产¥{cash:,.0f}", flush=True)

    feats = build_features(df, global_idx, 60)
    if feats is None: continue

    prob = float(model.predict_proba(feats.reshape(1,-1))[0][1])
    price = float(df.iloc[global_idx]['close'])
    high = float(df.iloc[global_idx]['high'])
    low = float(df.iloc[global_idx]['low'])
    atr = calc_atr(df, global_idx, 20)
    if atr is None or price <= 0: continue

    cur_date = str(df.iloc[global_idx]['date'])
    cur_dir = 'LONG' if prob > 0.5 else 'SHORT'
    conf = prob if prob > 0.5 else 1-prob
    prev_close = float(df.iloc[global_idx-1]['close']) if global_idx>0 else price
    change_pct = round((price/prev_close-1)*100, 2)
    today_trades = []

    # ===== 管理现有持仓 =====
    surviving = []
    for pos in positions:
        d,entry,trail,entry_i,vol,entry_date = pos
        bars = global_idx - entry_i
        pnl_pct = (price-entry)/entry if d=='LONG' else (entry-price)/entry
        pnl_atr = pnl_pct*entry/atr if atr>0 else 0

        if d == 'LONG':
            hard_stop = price - atr*p['atr_stop_mult']
            if pnl_atr > p['trail_atr']: trail = max(trail, price-atr*(p['atr_stop_mult']-0.3))
            if pnl_atr > p['be_atr']: trail = max(trail, entry)
            eff_stop = max(hard_stop, trail)
            should_reduce = (cur_dir=='LONG' and conf<p['reduce_conf'] and bars>=p['min_hold'])
            should_reverse = (prob < p['reverse_conf'] and bars>=p['min_hold'])

            if low <= eff_stop:
                ep = eff_stop
                gross = vol*CFG['multiplier']*(ep-entry)
                fee = vol*CFG['multiplier']*(entry+ep)*CFG['cost']
                net = gross-fee
                cash += net + vol*entry*CFG['multiplier']*0.15
                running_pnl += net; trade_count += 1
                ret_pct = round((ep/entry-1)*100, 2)
                today_trades.append(f"止损平多{vol}手 @{entry:.0f}→{ep:.0f} | 收益率{ret_pct:+.1f}% | 盈亏¥{net:+,.0f} | 费用¥{fee:.0f}")
                total_lots -= vol
            elif should_reverse:
                gross = vol*CFG['multiplier']*(price-entry)
                fee = vol*CFG['multiplier']*(entry+price)*CFG['cost']
                net = gross-fee
                cash += net + vol*entry*CFG['multiplier']*0.15
                running_pnl += net; trade_count += 1
                ret_pct = round((price/entry-1)*100, 2)
                today_trades.append(f"反手平多{vol}手 @{entry:.0f}→{price:.0f} | 收益率{ret_pct:+.1f}% | 盈亏¥{net:+,.0f} | 费用¥{fee:.0f}")
                total_lots -= vol
            elif should_reduce and vol > 1:
                cut = vol//2
                gross = cut*CFG['multiplier']*(price-entry)
                fee = cut*CFG['multiplier']*(entry+price)*CFG['cost']
                net = gross-fee
                cash += net + cut*entry*CFG['multiplier']*0.15
                running_pnl += net; trade_count += 1
                ret_pct = round((price/entry-1)*100, 2)
                today_trades.append(f"减多{cut}/{vol}手 @{entry:.0f}→{price:.0f} | 收益率{ret_pct:+.1f}% | 盈亏¥{net:+,.0f} | 费用¥{fee:.0f}")
                total_lots -= cut
                surviving.append((d,entry,trail,entry_i,vol-cut,entry_date))
            else:
                surviving.append((d,entry,trail,entry_i,vol,entry_date))
        else:  # SHORT
            hard_stop = price + atr*p['atr_stop_mult']
            if pnl_atr > p['trail_atr']: trail = min(trail, price+atr*(p['atr_stop_mult']-0.3))
            if pnl_atr > p['be_atr']: trail = min(trail, entry)
            eff_stop = min(hard_stop, trail)
            should_reduce = (cur_dir=='SHORT' and conf<p['reduce_conf'] and bars>=p['min_hold'])
            should_reverse = (prob > 1-p['reverse_conf'] and bars>=p['min_hold'])

            if high >= eff_stop:
                ep = eff_stop
                gross = vol*CFG['multiplier']*(entry-ep)
                fee = vol*CFG['multiplier']*(entry+ep)*CFG['cost']
                net = gross-fee
                cash += net + vol*entry*CFG['multiplier']*0.15
                running_pnl += net; trade_count += 1
                ret_pct = round((entry/ep-1)*100, 2)
                today_trades.append(f"止损平空{vol}手 @{entry:.0f}→{ep:.0f} | 收益率{ret_pct:+.1f}% | 盈亏¥{net:+,.0f} | 费用¥{fee:.0f}")
                total_lots -= vol
            elif should_reverse:
                gross = vol*CFG['multiplier']*(entry-price)
                fee = vol*CFG['multiplier']*(entry+price)*CFG['cost']
                net = gross-fee
                cash += net + vol*entry*CFG['multiplier']*0.15
                running_pnl += net; trade_count += 1
                ret_pct = round((entry/price-1)*100, 2)
                today_trades.append(f"反手平空{vol}手 @{entry:.0f}→{price:.0f} | 收益率{ret_pct:+.1f}% | 盈亏¥{net:+,.0f} | 费用¥{fee:.0f}")
                total_lots -= vol
            elif should_reduce and vol > 1:
                cut = vol//2
                gross = cut*CFG['multiplier']*(entry-price)
                fee = cut*CFG['multiplier']*(entry+price)*CFG['cost']
                net = gross-fee
                cash += net + cut*entry*CFG['multiplier']*0.15
                running_pnl += net; trade_count += 1
                ret_pct = round((entry/price-1)*100, 2)
                today_trades.append(f"减空{cut}/{vol}手 @{entry:.0f}→{price:.0f} | 收益率{ret_pct:+.1f}% | 盈亏¥{net:+,.0f} | 费用¥{fee:.0f}")
                total_lots -= cut
                surviving.append((d,entry,trail,entry_i,vol-cut,entry_date))
            else:
                surviving.append((d,entry,trail,entry_i,vol,entry_date))
    positions = surviving

    # ===== 开仓/加仓 =====
    atr_pct = atr/price
    ps = calc_pos_size(price, atr, cash, init_frac)
    if ps > 0 and total_lots+ps <= CFG['max_total']:
        sd = 'LONG' if prob>0.5 else 'SHORT'
        sd2 = atr*p['atr_stop_mult']
        margin = ps*price*CFG['multiplier']*0.15
        if margin <= cash:
            if not positions and conf > p['entry_conf']:
                if sd=='LONG' and low > price-sd2:
                    cash -= margin
                    positions.append((sd, price, price-sd2, global_idx, ps, cur_date))
                    total_lots += ps
                    today_trades.append(f"开多{ps}手 @{price:.0f} | 止损{price-sd2:.0f} | 保证金¥{margin:,.0f}")
                elif sd=='SHORT' and high < price+sd2:
                    cash -= margin
                    positions.append((sd, price, price+sd2, global_idx, ps, cur_date))
                    total_lots += ps
                    today_trades.append(f"开空{ps}手 @{price:.0f} | 止损{price+sd2:.0f} | 保证金¥{margin:,.0f}")
            elif positions and positions[0][0]==sd:
                avg_entry = np.mean([p[1] for p in positions])
                pa = (price-avg_entry)/atr if sd=='LONG' else (avg_entry-price)/atr
                if conf>p['add_conf'] and pa>p['add_atr']:
                    if sd=='LONG' and low>price-sd2 and margin<=cash:
                        cash -= margin
                        positions.append((sd,price,price-sd2,global_idx,ps,cur_date))
                        total_lots += ps
                        today_trades.append(f"加多{ps}手 @{price:.0f} | 共{total_lots}手")
                    elif sd=='SHORT' and high<price+sd2 and margin<=cash:
                        cash -= margin
                        positions.append((sd,price,price+sd2,global_idx,ps,cur_date))
                        total_lots += ps
                        today_trades.append(f"加空{ps}手 @{price:.0f} | 共{total_lots}手")

    # 净资产
    floating = 0; margin_locked = 0
    for pos in positions:
        d,entry,_,_,vol,_ = pos
        floating += vol*CFG['multiplier']*((price-entry) if d=='LONG' else (entry-price))
        margin_locked += vol*entry*CFG['multiplier']*0.15
    net_asset = cash + margin_locked + floating

    pos_desc = '空仓'
    if positions:
        d_name = '多' if positions[0][0]=='LONG' else '空'
        pos_desc = f"{d_name}{total_lots}手"
    avg_entry = round(np.mean([p[1] for p in positions]),0) if positions else 0

    rows.append({
        '日期': cur_date,
        '开盘': round(float(df.iloc[global_idx]['open']),0),
        '最高': round(high,0), '最低': round(low,0), '收盘': round(price,0),
        '涨跌幅%': change_pct,
        '持仓': pos_desc, '持仓均价': avg_entry if avg_entry>0 else '',
        '浮盈¥': round(floating,0), '保证金¥': round(margin_locked,0),
        '累计实盈¥': round(running_pnl,0),
        '净资产¥': round(net_asset,0),
        '收益率%': round((net_asset-CAPITAL)/CAPITAL*100, 1),
        '操作记录': '\n'.join(today_trades) if today_trades else '无操作',
    })

    if day_i % 50 == 0:
        print(f"  第{day_i}天 | {cur_date} | 收盘{price:.0f} | 涨跌{change_pct:+.1f}% | "
              f"{pos_desc} | 净资产¥{net_asset:,.0f}({(net_asset-CAPITAL)/CAPITAL*100:+.1f}%) | "
              f"累计实盈¥{running_pnl:,.0f}", flush=True)

# EOD
lp = float(df.iloc[-1]['close'])
last_date = str(df.iloc[-1]['date'])
for pos in positions:
    d,entry,_,_,vol,_ = pos
    gross = vol*CFG['multiplier']*((lp-entry) if d=='LONG' else (entry-lp))
    fee = vol*CFG['multiplier']*(entry+lp)*CFG['cost']
    net = gross-fee
    cash += net; running_pnl += net
    print(f"  收盘强平: {d} {vol}手 @{entry:.0f}→{lp:.0f} 净¥{net:+,.0f}")

final_net = cash
final_ret = (final_net-CAPITAL)/CAPITAL*100
print(f"\n{'='*60}")
print(f"  最终净资产: ¥{final_net:,.0f}  |  总收益率: {final_ret:+.1f}%")
print(f"  总交易笔数: {trade_count}  |  累计实盈: ¥{running_pnl:,.0f}")
print(f"{'='*60}")

# 导出
df_out = pd.DataFrame(rows)
output_path = '/home/a/prophet_futures/prophet_futures/v28_LH_daily_journal.xlsx'

with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
    df_out.to_excel(writer, sheet_name='每日交易日志', index=False)
    ws = writer.sheets['每日交易日志']
    for ci, cc in enumerate(ws.columns, 1):
        ml = max((len(str(c.value or '')) for c in cc), default=0)
        ws.column_dimensions[ws.cell(1, ci).column_letter].width = min(ml+4, 50)
    op_col = df_out.columns.get_loc('操作记录')
    ws.column_dimensions[ws.cell(1, op_col+1).column_letter].width = 70

    # 汇总
    peak = df_out['净资产¥'].cummax()
    dd = (df_out['净资产¥']-peak)/peak*100
    op_days = df_out[df_out['操作记录']!='无操作']
    pd.DataFrame({
        '指标': ['回测期','总天数','初始资金','最终净资产','总收益率',
                 '最高净资产','最大回撤','总交易笔数','有操作天数',
                 '模型重训间隔','训练窗口','数据总量'],
        '数值': [
            f"{df_out['日期'].iloc[0]} → {df_out['日期'].iloc[-1]}",
            len(df_out), f"¥{CAPITAL:,}",
            f"¥{final_net:,.0f}", f"{final_ret:+.1f}%",
            f"¥{df_out['净资产¥'].max():,.0f}", f"{dd.min():.1f}%",
            trade_count, len(op_days),
            f"{retrain_every}天", "250天", f"{n_total}天",
        ]
    }).to_excel(writer, sheet_name='汇总', index=False)

os.system(f"cp '{output_path}' /mnt/c/Users/{os.listdir('/mnt/c/Users')[0]}/Desktop/v28_LH_daily_journal.xlsx 2>/dev/null")
print(f"✅ 已导出: {output_path}")
print(f"✅ 已复制桌面")
