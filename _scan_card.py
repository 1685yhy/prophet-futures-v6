#!/usr/bin/env python3
"""5分钟扫描卡 — 多版本持仓+模型预测"""
import json
from datetime import datetime
from feishu_card import send_card, md, hr

now = datetime.now()
time_str = now.strftime('%H:%M')

# ── 实时行情 ──
prices = {'lh2609': 12260.0, 'jm2609': 1294.0}
sym_names = {'lh2609': '生猪', 'jm2609': '焦煤'}
dir_cn = {'LONG': '多', 'SHORT': '空'}
specs = {'lh': 16, 'jm': 60}

# ── 读取4个纸盘状态 ──
versions = {
    'V25': 'paper_state.json',
    'V28': 'paper_state_v28.json',
    'V29': 'paper_state_v29.json',
    'V30': 'paper_state_v30.json',
}

paper_data = {}
for ver, fname in versions.items():
    with open(fname) as f:
        s = json.load(f)
    cash = s['cash']
    pos_raw = s.get('positions', {})

    positions = {}
    floating = {}
    total_floating = 0

    for sym, pdata in pos_raw.items():
        mult = specs.get(sym[:2], 1)
        cur = prices[sym]

        if isinstance(pdata, dict):
            entry = pdata['entry']
            vol = pdata['vol']
            direction = pdata['dir']
        elif isinstance(pdata, list) and len(pdata) > 0:
            entry = pdata[0]['entry']
            vol = sum(p['vol'] for p in pdata)
            direction = pdata[0]['dir']
        else:
            continue

        if direction == 'LONG':
            fl = (cur - entry) * mult * vol
        else:
            fl = (entry - cur) * mult * vol

        positions[sym] = (direction, entry, vol)
        floating[sym] = fl
        total_floating += fl

    init_cap = 500000 if ver in ('V25', 'V28') else 123500
    paper_data[ver] = {
        'positions': positions,
        'floating': floating,
        'total_floating': total_floating,
        'cash': cash,
        'equity': cash + total_floating,
        'initial': init_cap,
    }

# ── 模型预测 ──
import pickle, os, sys
sys.path.insert(0, '.')
from realtime_data import get_daily_history, build_features

model_files = {
    'V25': '_xgb.pkl', 'V28': '_xgb.pkl',
    'V29': '_xgb_new.pkl', 'V30': '_xgb_calibrated.pkl',
}
MODEL_DIR = 'models'

model_preds = {}
for sym in ['lh2609', 'jm2609']:
    df = get_daily_history(sym, 1200)
    if df is None:
        continue
    feats = build_features(df, len(df)-1, 60)
    if feats is None:
        continue
    for ver, suffix in model_files.items():
        mp = os.path.join(MODEL_DIR, f'{sym}{suffix}')
        if not os.path.exists(mp):
            continue
        with open(mp, 'rb') as f:
            model = pickle.load(f)
        prob = float(model.predict_proba(feats.reshape(1, -1))[0][1])
        direction = '多' if prob > 0.5 else '空'
        model_preds.setdefault(ver, {})[sym] = (direction, prob)

# ── 构建卡片 ──
elements = []

# ── 顶部简要概括：只列出有异常或需关注的 ──
alert_lines = []
all_conflicts = []

for ver in ['V25', 'V28', 'V29', 'V30']:
    d = paper_data[ver]
    m = model_preds.get(ver, {})

    for sym in ['lh2609', 'jm2609']:
        if sym not in d['positions'] or sym not in m:
            continue
        direction, entry, vol = d['positions'][sym]
        md_dir, md_conf = m[sym]
        pos_d = dir_cn[direction]
        name = sym_names[sym]
        fl = d['floating'].get(sym, 0)
        fl_wan = fl / 10000

        if pos_d != md_dir:
            gap = abs(md_conf - 0.5)
            all_conflicts.append((ver, name, pos_d, vol, fl_wan, md_dir, gap))

if all_conflicts:
    for ver, name, pos_d, vol, fl_wan, md_dir, gap in all_conflicts:
        sign = '+' if fl_wan >= 0 else ''
        alert_lines.append(
            f'⚠️ {ver} {name}: ⚠️ {name} 做{pos_d}{vol}手 | 浮盈{sign}{fl_wan:.1f}万 | '
            f'持仓做{pos_d} 模型偏看{md_dir}(距反手差{gap:.0%})'
        )
    elements.append(md('\n'.join(alert_lines)))
    elements.append(md('━━━ ⚡ 关注 ━━━'))
else:
    elements.append(md('✅ 所有版本持仓与模型一致，无需特别关注'))

elements.append(md(
    f'**实时行情** | 生猪 {prices["lh2609"]} | 焦煤 {prices["jm2609"]} | {time_str}'
))

# ── 按版本分组 ──
for ver in ['V25', 'V28', 'V29', 'V30']:
    d = paper_data[ver]
    m = model_preds.get(ver, {})

    lines = []

    # 表头
    lines.append('|品种|持仓|入场|手数|浮盈|模型方向|概率|建议|')
    lines.append('|---|---|---|---|---|---|---|---|')

    for sym in ['lh2609', 'jm2609']:
        if sym not in d['positions']:
            continue
        direction, entry, vol = d['positions'][sym]
        fl = d['floating'].get(sym, 0)
        name = sym_names.get(sym, sym)
        fl_sign = '+' if fl >= 0 else ''

        md_info = '-'
        suggestion = '-'
        if sym in m:
            md_dir, md_conf = m[sym]
            md_info = f'{md_dir} {md_conf:.0%}'
            pos_d = dir_cn[direction]
            if pos_d == md_dir:
                suggestion = '✅ 持有'
            else:
                gap = abs(md_conf - 0.5)
                if gap < 0.03:
                    suggestion = f'⚠️ 反手差{gap:.0%}'
                else:
                    suggestion = f'🚨 方向冲突'
        else:
            md_info = '-'

        lines.append(
            f'|{name}|{dir_cn[direction]}|{int(entry)}|{vol}手'
            f'|{fl_sign}{int(fl)}'
            f'|{md_info}'
            f'|{suggestion}|'
        )

    # 总计行
    total_pnl = d['total_floating']
    pnl_emoji = '🟢' if total_pnl >= 0 else '🔴'
    pnl_sign = '+' if total_pnl >= 0 else ''
    pnl_rate = total_pnl / d['initial'] * 100

    lines.append('')
    lines.append(
        f'{pnl_emoji} 总浮盈 {pnl_sign}{int(total_pnl):,} '
        f'({pnl_sign}{pnl_rate:.1f}%) | 权益≈{int(d["equity"]):,} | '
        f'初始¥{d["initial"]:,}'
    )

    elements.append(md(f'━━━ {ver} ━━━'))
    elements.append(md('\n'.join(lines)))

# ── 底部 ──
equities = {ver: int(paper_data[ver]['equity']) for ver in ['V25','V28','V29','V30']}
footer = (
    f'V25 ¥{equities["V25"]:,} | '
    f'V28 ¥{equities["V28"]:,} | '
    f'V29 ¥{equities["V29"]:,} | '
    f'V30 ¥{equities["V30"]:,} | '
    f'{time_str}'
)
elements.append(md(footer))

# ── 发送 ──
success, msg = send_card(f'🔔 扫描 {time_str}', elements, template='blue')
print(f'Send result: success={success}, msg={msg}')
