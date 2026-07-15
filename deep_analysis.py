#!/usr/bin/env python3
"""
深度分析: V28 为什么WF显示赚钱,连续回测亏钱?
从6个维度解剖:
1. 每笔交易的模型预测vs实际走势
2. 分段收益(上涨趋势/下跌趋势/震荡)
3. 胜率×盈亏比拆解
4. 最大亏损交易序列分析
5. 模型置信度与实际胜率的关系
6. 手续费侵蚀占比
"""
import numpy as np, pandas as pd

df = pd.read_excel("/home/a/prophet_futures/prophet_futures/v28_LH_daily_journal.xlsx", sheet_name="每日交易日志")

# 提取交易记录
trades = []  # 实际的平仓交易
current_long = None  # (entry_date, entry_price, lots)
current_short = None

for _, r in df.iterrows():
    ops = str(r['操作记录']) if r['操作记录'] != '无操作' and pd.notna(r['操作记录']) else ''
    date = str(r['日期'])
    close = float(r['收盘'])
    change = float(r['涨跌幅%']) if pd.notna(r['涨跌幅%']) else 0
    
    for line in ops.split('\n'):
        line = line.strip()
        if not line: continue
        
        # 平仓交易 (有盈亏数字)
        if '¥' in line and ('止损' in line or '反手' in line or '减' in line):
            import re
            nums = re.findall(r'[-+]?\d+\.?\d*', line)
            if len(nums) >= 2:
                pnl = float(nums[-2])  # 通常是倒数第二个数字
                if abs(pnl) > 100000: pnl = float(nums[-1])  # 修正
            
            direction = '多' if ('多' in line and ('平多' in line or '止多' in line or '减多' in line)) else '空'
            reason = '止损' if '止损' in line else ('反手' if '反手' in line else '减仓')
            
            # 提取开仓价和平仓价
            prices = re.findall(r'@(\d+)', line)
            entry_price = float(prices[0]) if prices else 0
            exit_price = float(prices[1]) if len(prices) > 1 else close
            
            trades.append({
                'exit_date': date,
                'direction': direction,
                'entry': entry_price,
                'exit': exit_price,
                'pnl': pnl,
                'reason': reason,
                'close_that_day': close,
                'change_that_day': change,
            })

print(f"总平仓交易: {len(trades)}笔")

# === 1. 分段分析 ===
df_seg = df.copy()
df_seg['date'] = pd.to_datetime(df_seg['日期'])
df_seg['month'] = df_seg['date'].dt.to_period('M')

# 按月份统计收益
monthly = df_seg.groupby('month').agg(
    月末收盘=('收盘','last'),
    月末净资产=('净资产¥','last'),
    月收益=('收益率%','last'),
).reset_index()

# 趋势分段: 上涨月 vs 下跌月 vs 震荡月
monthly['月涨跌'] = monthly['月末收盘'].pct_change()
monthly['趋势'] = monthly['月涨跌'].apply(
    lambda x: '上涨' if x > 0.02 else ('下跌' if x < -0.02 else '震荡')
)

# 按趋势统计
for trend in ['上涨','下跌','震荡']:
    seg = monthly[monthly['趋势']==trend]
    if len(seg) == 0: continue
    print(f"\n{'='*40}")
    print(f"  趋势: {trend} ({len(seg)}个月)")
    print(f"  月均涨跌: {seg['月涨跌'].mean()*100:+.1f}%")
    print(f"  月均收益: {seg['月收益'].mean():+.1f}%")
    print(f"  盈利月: {(seg['月收益']>0).sum()}/{len(seg)} ({(seg['月收益']>0).mean():.0%})")

# === 2. 盈亏分布 ===
print(f"\n{'='*40}")
print(f"  盈亏分布")
pnls = [t['pnl'] for t in trades]
wins = [p for p in pnls if p > 0]
losses = [p for p in pnls if p <= 0]
print(f"  盈利笔数: {len(wins)} ({len(wins)/len(pnls):.0%})")
print(f"  亏损笔数: {len(losses)} ({len(losses)/len(pnls):.0%})")
print(f"  平均盈利: ¥{np.mean(wins):,.0f}  平均亏损: ¥{np.mean(losses):,.0f}")
print(f"  最大盈利: ¥{max(wins):,.0f}  最大亏损: ¥{min(losses):,.0f}")
print(f"  总盈利: ¥{sum(wins):,.0f}  总亏损: ¥{sum(losses):,.0f}")
print(f"  净盈亏: ¥{sum(pnls):,.0f}")

# === 3. 最大连亏分析 ===
max_consecutive_loss = 0
current_streak = 0
streak_pnl = 0
max_streak_pnl = 0

for t in trades:
    if t['pnl'] <= 0:
        current_streak += 1
        streak_pnl += t['pnl']
        if current_streak > max_consecutive_loss:
            max_consecutive_loss = current_streak
        if streak_pnl < max_streak_pnl:
            max_streak_pnl = streak_pnl
    else:
        current_streak = 0
        streak_pnl = 0

print(f"\n{'='*40}")
print(f"  连亏分析")
print(f"  最大连续亏损笔数: {max_consecutive_loss}")
print(f"  最大连续亏损金额: ¥{max_streak_pnl:,.0f}")

# === 4. 止损 vs 反手 vs 减仓 ===
print(f"\n{'='*40}")
print(f"  按退出原因统计")
df_t = pd.DataFrame(trades)
for reason in df_t['reason'].unique():
    sub = df_t[df_t['reason']==reason]
    print(f"  {reason}: {len(sub)}笔 | 胜率{(sub['pnl']>0).mean():.0%} | 均盈亏¥{sub['pnl'].mean():,.0f} | 合计¥{sub['pnl'].sum():,.0f}")

# === 5. 手续费占比 ===
# 每笔交易双向手续费约0.12%
total_notional = sum(abs(t['entry'])*6*16 for t in trades)  # 假设每笔6手
total_fees_est = total_notional * 0.0012
print(f"\n{'='*40}")
print(f"  手续费估算")
print(f"  总名义成交额: ¥{total_notional:,.0f}")
print(f"  估算手续费: ¥{total_fees_est:,.0f}")
print(f"  占总盈亏比: {total_fees_est/abs(sum(pnls))*100 if sum(pnls)!=0 else 999:.1f}%")

# === 6. 按季度分段表现 ===
print(f"\n{'='*40}")
print(f"  按季度分段")
quarterly = df_seg.set_index('date').resample('QE')['净资产¥'].last()
quarterly_ret = quarterly.pct_change() * 100
for i in range(1, len(quarterly)):
    d = quarterly.index[i]
    print(f"  {d.strftime('%Y-Q%q')}: 净资产¥{quarterly.iloc[i]:,.0f}  季度收益{quarterly_ret.iloc[i]:+.1f}%")

# === 7. 赚钱阶段 vs 亏钱阶段 ===
print(f"\n{'='*40}")
print(f"  最大盈利阶段")
# 找净资产最高点前后的表现
df_sorted = df.copy()
peak_idx = df_sorted['净资产¥'].idxmax()
peak_date = df_sorted.loc[peak_idx, '日期']
peak_val = df_sorted.loc[peak_idx, '净资产¥']

print(f"  最高净资产: ¥{peak_val:,.0f} ({peak_date})")
print(f"  初始净资产: ¥300,000")
print(f"  从初始到峰值收益: {(peak_val-300000)/300000*100:+.1f}%")

# 峰值之后跌了多少
final_val = df_sorted['净资产¥'].iloc[-1]
drawdown_from_peak = (final_val - peak_val) / peak_val * 100
print(f"  从峰值到最终: {drawdown_from_peak:+.1f}%")
print(f"  峰值之后的交易: {(df_sorted['日期'] > peak_date).sum()}天")

# 峰值之后的交易表现
post_peak = [t for t in trades if t['exit_date'] > str(peak_date)]
if post_peak:
    post_pnl = [t['pnl'] for t in post_peak]
    post_wins = [p for p in post_pnl if p > 0]
    print(f"  峰值后交易: {len(post_peak)}笔 | 胜率{len(post_wins)/len(post_peak):.0%} | 均盈亏¥{np.mean(post_pnl):,.0f} | 合计¥{sum(post_pnl):,.0f}")

pre_peak = [t for t in trades if t['exit_date'] <= str(peak_date)]
if pre_peak:
    pre_pnl = [t['pnl'] for t in pre_peak]
    pre_wins = [p for p in pre_pnl if p > 0]
    print(f"  峰值前交易: {len(pre_peak)}笔 | 胜率{len(pre_wins)/len(pre_peak):.0%} | 均盈亏¥{np.mean(pre_pnl):,.0f} | 合计¥{sum(pre_pnl):,.0f}")

# === 8. 导出结果 ===
output = '/home/a/prophet_futures/prophet_futures/v28_deep_analysis.xlsx'
with pd.ExcelWriter(output, engine='openpyxl') as writer:
    # 交易明细
    pd.DataFrame(trades).to_excel(writer, sheet_name='逐笔交易', index=False)
    # 月度
    monthly.to_excel(writer, sheet_name='月度统计', index=False)
    # 盈亏分布
    bins = pd.cut(pd.Series(pnls), bins=[-np.inf, -50000, -20000, -10000, -5000, 0, 5000, 10000, 20000, 50000, np.inf],
                  labels=['<-5万','-5~-2万','-2~-1万','-1~-0.5万','-0.5~0万','0~0.5万','0.5~1万','1~2万','2~5万','>5万'])
    dist = bins.value_counts().sort_index()
    dist_df = pd.DataFrame({'盈亏区间': dist.index, '笔数': dist.values})
    dist_df.to_excel(writer, sheet_name='盈亏分布', index=False)
    # 完整日志
    df.to_excel(writer, sheet_name='每日日志', index=False)

print(f"\n✅ 已导出: {output}")
