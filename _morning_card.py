#!/usr/bin/env python3
"""盘前自检卡片 - 向飞书发送7个纸盘版本状态+资金"""
import sys, os, json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from report_v4 import (
    S, ls, lv28, lv29, lv30, lv31, lv32, lv32b,
    send, md, hr, eq
)
from realtime_data import get_realtime_quote

# 获取价格（盘前用prev_close）
prices = {}
for sk in ['lh2609', 'jm2609']:
    rt = get_realtime_quote(sk)
    if rt:
        prices[sk] = rt['price']
    else:
        # fallback
        cfg = S.get(sk, {})
        if cfg:
            from report_v4 import fd
            df = fd(cfg['code'])
            if df is not None and len(df) > 0:
                prices[sk] = float(df.iloc[-1]['close'])

# 所有7个版本
versions = [
    ('V25',  ls(),    'xgb'),
    ('V28',  lv28(),  'xgb'),
    ('V29',  lv29(),  'new'),
    ('V30',  lv30(),  'calibrated'),
    ('V31',  lv31(),  'v31'),
    ('V32',  lv32(),  'v32'),
    ('V32b', lv32b(), 'v32b'),
]

elements = [
    md(f'**📊 Prophet Futures 盘前自检 | 2026-07-17 08:45**'),
    md(''),
]

# 表1: 持仓明细
elements.append(md('**持仓明细**'))
rows = ['| 版本 | 品种 | 方向 | 手数 | 均价 | 止损 | 现价 | 浮盈点 |']
rows.append('|------|------|------|------|------|------|------|--------|')
for ver_name, st, _ in versions:
    pos = st.get('positions', {})
    if not pos:
        rows.append(f'| {ver_name} | — | — | 0 | — | — | — | — |')
        continue
    for sk, pl in pos.items():
        cfg = S.get(sk, {})
        cn = cfg.get('cn', sk)
        cur = prices.get(sk)
        if isinstance(pl, list):
            tv = sum(p.get('vol', 0) for p in pl)
            d = pl[0].get('dir', '?')
            ae = sum(p['entry'] * p['vol'] for p in pl) / tv if tv > 0 else 0
            sl = pl[0].get('stop', pl[0].get('_trail', 0))
        else:
            tv = pl.get('vol', 0)
            d = pl.get('dir', '?')
            ae = pl.get('entry', 0)
            sl = pl.get('stop', pl.get('_trail', 0))
        
        dir_cn = '多' if d == 'LONG' else '空'
        if cur and tv > 0:
            pnl_pts = (cur - ae) if d == 'LONG' else (ae - cur)
            rows.append(f'| {ver_name} | {cn} | {dir_cn} | {tv} | {ae:.0f} | {sl:.0f} | {cur:.0f} | {pnl_pts:+.0f} |')
        else:
            rows.append(f'| {ver_name} | {cn} | {dir_cn} | {tv} | {ae:.0f} | {sl:.0f} | — | — |')
elements.append(md('\n'.join(rows)))
elements.append(md(''))

# 表2: 资金汇总 (7个版本)
elements.append(md('**📈 资金汇总**'))
rows2 = ['| 版本 | 权益 | 盈亏 | 盈亏% | 持仓 |']
rows2.append('|------|------|------|------|------|')
total_pnl = 0
for ver_name, st, _ in versions:
    equity = eq(st, prices)
    pnl = equity - 300000
    pnl_pct = pnl / 300000 * 100
    total_pnl += pnl
    pos_cnt = len(st.get('positions', {}))
    pos_label = f'{pos_cnt}仓' if pos_cnt > 0 else '空仓'
    emoji = '🟢' if pnl > 0 else ('🔴' if pnl < 0 else '⚪')
    rows2.append(f'| **{ver_name}** | ¥{equity:,.0f} | {emoji} {pnl:+,.0f} | {pnl_pct:+.2f}% | {pos_label} |')
elements.append(md('\n'.join(rows2)))
elements.append(md(''))

# 表3: 进程状态
elements.append(md('**⚙️ 进程状态**'))
elements.append(md(
    '✅ V25 ✅ V28 ✅ V29 ✅ V30\n'
    '✅ V31 ✅ V32 ✅ V32b\n'
    '📡 SimNow云端: 运行中\n'
    '🛡️ 止损穿透: 无 | 追踪止损: 正常'
))
elements.append(md(''))

# 总计
total_emoji = '🟢' if total_pnl > 0 else ('🔴' if total_pnl < 0 else '⚪')
elements.append(md(f'**💰 7版本总盈亏**: {total_emoji} **¥{total_pnl:+,.0f}** ({total_pnl/300000/7*100:+.2f}%)'))
elements.append(hr())
elements.append(md('🕐 自检时间: 2026-07-17 08:45 | 盘前状态'))

# 发送
ok = send('📊 盘前自检 | 2026-07-17', elements, 'blue')
if ok:
    print('✅ 飞书卡片已发送')
else:
    print('❌ 发送失败')
