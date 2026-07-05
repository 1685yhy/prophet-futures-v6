#!/usr/bin/env python3
"""Prophet Futures — 早晚报告生成器 v27
用法:
  python daily_report.py morning          # 早报文本
  python daily_report.py evening           # 晚报文本
  python daily_report.py morning --card    # 早报卡片
  python daily_report.py evening --card   # 晚报卡片

v27: 全面改用扫描风格的表格+精简行，三个版本持仓统一展示
"""
import sys, os, json, numpy as np, pandas as pd, pickle
from datetime import datetime, timedelta
import akshare as ak
from feishu_card import send_card, md, hr
from realtime_data import get_realtime_quote, get_daily_history, build_features, SYMBOL_MAP

SYMBOLS = {
    'lh2609': {'code': 'LH0', 'name': 'LH 生猪', 'cost': 0.0006, 'multiplier': 16,
               'stop_type': 'atr', 'stop_mult': 1.5, 'rr': 4, 'max_pos': 6,
               'trail_atr': 2.0,  'be_atr': 1.0,
               'reduce1_atr': 2.0, 'reduce1_pct': 0.5,
               'reduce2_atr': 4.0, 'reduce2_pct': 0.5},
    'jm2609': {'code': 'JM0', 'name': 'JM 焦煤', 'cost': 0.0011, 'multiplier': 60,
               'stop_type': 'atr', 'stop_mult': 2.0, 'rr': 3.5, 'max_pos': 4,
               'trail_atr': 3.0,  'be_atr': 2.0,
               'reduce1_atr': 3.0, 'reduce1_pct': 0.5,
               'reduce2_atr': 5.0, 'reduce2_pct': 0.5},
}
CAPITAL = 300000
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

STATE_FILES = {
    'V25': 'paper_state.json',
    'V28': 'paper_state_v28.json',
    'V29': 'paper_state_v29.json',
}

# ============================================================
#  数据加载
# ============================================================

def get_model_prediction(sym_key):
    """获取模型预测 — 优先校准模型"""
    for suffix in ['_xgb_calibrated.pkl', '_xgb.pkl']:
        mp = os.path.join(MODEL_DIR, f'{sym_key}{suffix}')
        if os.path.exists(mp):
            break
    if not os.path.exists(mp):
        return None
    try:
        with open(mp, 'rb') as f: model = pickle.load(f)
        daily_df = get_daily_history(sym_key, 1200)
        if daily_df is None or len(daily_df) < 100:
            return None
        feats = build_features(daily_df, len(daily_df)-1, 60)
        if feats is None:
            return None
        prob = float(model.predict_proba(feats.reshape(1, -1))[0][1])
        direction = 'LONG' if prob > 0.5 else 'SHORT'
        confidence = prob if prob > 0.5 else (1 - prob)
        last_date = str(daily_df.iloc[-1]['date'])
        return {
            'direction': direction,
            'prob': prob,
            'confidence': confidence,
            'last_date': last_date,
        }
    except:
        return None

def fetch(sym, days=500):
    raw = sym.upper()
    code = raw[:-1] + '0' if raw.endswith('0') else raw + '0'
    end = datetime.now(); start = end - timedelta(days=days+50)
    try:
        df = ak.futures_main_sina(symbol=code, start_date=start.strftime('%Y%m%d'),
                                   end_date=end.strftime('%Y%m%d'))
        df.columns = ['date','open','high','low','close','volume','oi','settle']
        for c in ['open','high','low','close','volume','oi']:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        return df.dropna(subset=['close']).reset_index(drop=True)
    except: return None

def load_state(path):
    if os.path.exists(path):
        with open(path) as f: return json.load(f)
    return {'cash': CAPITAL, 'positions': {}, 'trades': []}

def save_state(state, path):
    with open(path, 'w') as f: json.dump(state, f, indent=2, default=str)

def adjust_stops():
    """动态调整 V25 持仓的止损 — 只收紧，不放松"""
    state_path = os.path.join(BASE_DIR, 'paper_state.json')
    state = load_state(state_path)
    positions = state.get('positions', {})
    if not positions: return state

    changed = False
    for sym_key, pos in list(positions.items()):
        cfg = SYMBOLS.get(sym_key)
        if not cfg: continue
        df = fetch(cfg['code'], 120)
        if df is None or len(df) < 20: continue
        cur_price = float(df.iloc[-1]['close'])
        old_stop = pos['stop']
        d = pos['dir']

        atr_vals = [abs(float(df.iloc[i]['high']) - float(df.iloc[i]['low']))
                    for i in range(max(0, len(df) - 20), len(df))]
        atr = np.mean(atr_vals) if atr_vals else cur_price * 0.005
        stop_dist = atr * cfg['stop_mult']

        if cfg['stop_type'] == 'atr':
            new_stop = cur_price - stop_dist if d == 'LONG' else cur_price + stop_dist
        else:
            n = cfg['struct_n']
            if d == 'LONG':
                new_stop = min(float(df.iloc[i]['low']) for i in range(max(0, len(df) - n), len(df)))
            else:
                new_stop = max(float(df.iloc[i]['high']) for i in range(max(0, len(df) - n), len(df)))

        if d == 'LONG' and new_stop > old_stop:
            pos['stop'] = round(float(new_stop), 0)
            changed = True
        elif d == 'SHORT' and new_stop < old_stop:
            pos['stop'] = round(float(new_stop), 0)
            changed = True

        old_tp = pos['take_profit']
        entry = pos['entry']
        if d == 'LONG' and pos['stop'] >= old_tp:
            pos['take_profit'] = round(max(entry + stop_dist * cfg['rr'], pos['stop'] + atr * 0.5), 0)
            changed = True
        elif d == 'SHORT' and pos['stop'] <= old_tp:
            pos['take_profit'] = round(min(entry - stop_dist * cfg['rr'], pos['stop'] - atr * 0.5), 0)
            changed = True

    if changed:
        save_state(state, state_path)
    return state

def get_market_data():
    """获取所有品种的行情数据"""
    data = {}
    for sym_key, cfg in SYMBOLS.items():
        df = fetch(cfg['code'], 60)
        if df is None or len(df) < 20: continue

        ma20 = float(np.mean([float(df.iloc[k]['close']) for k in range(max(0, len(df)-20), len(df))]))
        ma60 = float(np.mean([float(df.iloc[k]['close']) for k in range(max(0, len(df)-60), len(df))]))
        atr_vals = [abs(float(df.iloc[k]['high']) - float(df.iloc[k]['low']))
                    for k in range(max(0, len(df)-20), len(df))]
        atr = np.mean(atr_vals) if atr_vals else 0
        prev_close = float(df.iloc[-2]['close']) if len(df) > 1 else 0

        rt = get_realtime_quote(sym_key)
        if rt:
            price = rt['price']
            chg = rt['changepercent']
            o = rt['open']
            h = rt['high']
            l = rt['low']
        else:
            price = float(df.iloc[-1]['close'])
            chg = (price - prev_close) / prev_close if prev_close else 0
            o = float(df.iloc[-1]['open'])
            h = float(df.iloc[-1]['high'])
            l = float(df.iloc[-1]['low'])

        if price > ma20 > ma60: trend = '📈 上涨'
        elif price < ma20 < ma60: trend = '📉 下跌'
        else: trend = '↔️ 震荡'

        data[sym_key] = {
            'price': price, 'prev': prev_close, 'chg': chg, 'atr': atr,
            'trend': trend, 'open': o, 'high': h, 'low': l,
            'atr_pct': atr/price if price>0 else 0,
        }
    return data

def compute_equity(state, market_data=None):
    equity = state['cash']
    for k, p in state.get('positions', {}).items():
        multiplier = SYMBOLS.get(k, {}).get('multiplier', 10)
        cur = p['entry']
        if market_data and k in market_data:
            cur = market_data[k]['price']
        if p['dir'] == 'LONG':
            pnl = (cur - p['entry']) * p['vol'] * multiplier
        else:
            pnl = (p['entry'] - cur) * p['vol'] * multiplier
        equity += pnl
    return equity

# ============================================================
#  多版本持仓收集
# ============================================================

def collect_all_positions(market_data):
    """收集 V25/V28/V29 的所有持仓，返回表格行列表"""
    rows = []
    for ver_label, state_file in STATE_FILES.items():
        path = os.path.join(BASE_DIR, state_file)
        if not os.path.exists(path):
            continue
        state = load_state(path)
        for sym_key, pos_data in state.get('positions', {}).items():
            cfg = SYMBOLS.get(sym_key, {})
            name = cfg.get('name', sym_key)
            m = market_data.get(sym_key, {})
            cur = m.get('price', 0) if m else 0

            # V25 是单仓位 dict，V28/V29 是列表
            pos_list = pos_data if isinstance(pos_data, list) else [pos_data]
            for pos in pos_list:
                entry = pos['entry']
                vol = pos['vol']
                d = pos['dir']
                stop = pos.get('stop', pos.get('_trail', 0))
                tp = pos.get('take_profit', 0)
                multiplier = cfg.get('multiplier', 10)

                if d == 'LONG':
                    pnl_pts = cur - entry
                    dist_stop = cur - stop if stop else 999
                else:
                    pnl_pts = entry - cur
                    dist_stop = stop - cur if stop else 999

                pnl_emoji = '🟢' if pnl_pts >= 0 else '🔴'
                dir_cn = '做多' if d == 'LONG' else '做空'
                risk = f'🚨{dist_stop:.0f}' if dist_stop < 50 else f'⚠️{dist_stop:.0f}' if dist_stop < 100 else f'{dist_stop:.0f}'

                rows.append({
                    'ver': ver_label,
                    'name': name,
                    'dir': dir_cn,
                    'vol': vol,
                    'entry': entry,
                    'cur': cur,
                    'pnl_pts': pnl_pts,
                    'pnl_emoji': pnl_emoji,
                    'stop': stop,
                    'dist_stop': dist_stop,
                    'risk': risk,
                    'tp': tp,
                })
    return rows

# ============================================================
#  卡片构建
# ============================================================

def build_positions_table(positions_rows, market_data):
    """持仓表格"""
    if not positions_rows:
        return None

    lines = [
        "**📌 持仓**\n",
        "| 版本 | 品种 | 方向 | 手 | 成本 | 现价 | 浮盈 | 止损 | 距止损 |",
        "|------|------|------|----|------|------|------|------|--------|",
    ]
    for r in positions_rows:
        lines.append(
            f"| {r['ver']} | {r['name']} | {r['pnl_emoji']} {r['dir']} | {r['vol']} | "
            f"{r['entry']:.0f} | {r['cur']:.0f} | {r['pnl_pts']:+.0f}点 | "
            f"{r['stop']:.0f} | {r['risk']} |"
        )
    return "\n".join(lines)

def build_market_table(market_data):
    """行情表格"""
    if not market_data:
        return None

    lines = [
        "**📊 行情**\n",
        "| 品种 | 现价 | 涨跌 | 开 | 高 | 低 | ATR | 趋势 |",
        "|------|------|------|----|----|----|-----|------|",
    ]
    for sym_key, m in market_data.items():
        name = SYMBOLS[sym_key]['name']
        chg = m['chg']
        emoji = '📈' if chg > 0.001 else ('📉' if chg < -0.001 else '➡️')
        lines.append(
            f"| {name} | {emoji} **{m['price']:.0f}** | {chg:+.2%} | "
            f"{m['open']:.0f} | {m['high']:.0f} | {m['low']:.0f} | "
            f"{m['atr']:.0f} | {m['trend']} |"
        )
    return "\n".join(lines)

def build_model_table(model_preds, positions_rows):
    """模型预测表格"""
    if not model_preds:
        return None

    ref_date = list(model_preds.values())[0]['last_date']
    lines = [
        f"**🧠 模型预测**（{ref_date} 日线）\n",
        "| 品种 | 方向 | 概率 | 置信度 |",
        "|------|------|------|--------|",
    ]
    for sym_key, mp in model_preds.items():
        name = SYMBOLS[sym_key]['name']
        emoji = '🟢' if mp['direction'] == 'LONG' else '🔴'
        dir_cn = '偏多' if mp['direction'] == 'LONG' else '偏空'

        # 检查是否有版本持仓冲突
        conflict_parts = []
        for r in (positions_rows or []):
            if r['name'] == name:
                pos_dir = 'LONG' if r['dir'] == '做多' else 'SHORT'
                if pos_dir != mp['direction']:
                    conflict_parts.append(f"{r['ver']}冲突")
        conflict = f" ⚠️ {'/'.join(conflict_parts)}" if conflict_parts else ""

        lines.append(
            f"| {name} | {emoji} **{dir_cn}** | {mp['prob']:.1%} | {mp['confidence']:.1%}{conflict} |"
        )
    return "\n".join(lines)

def build_actions(positions_rows, market_data):
    """操作建议 — 扫描风格：每行一个明确指令"""
    if not positions_rows:
        # 空仓建议
        lines = []
        for sym_key, m in market_data.items():
            cfg = SYMBOLS[sym_key]
            name = cfg['name']
            atr = m['atr']
            price = m['price']
            atr_pct = m.get('atr_pct', atr/price)
            if atr_pct < 0.01: lev = 3.0
            elif atr_pct < 0.02: lev = 2.0
            elif atr_pct < 0.03: lev = 1.5
            elif atr_pct < 0.05: lev = 0.5
            else: lev = 0
            vol = max(1, int(lev * (cfg['max_pos'] // 2))) if lev > 0 else 0
            if vol > 0:
                sm = cfg['stop_mult']; rr = cfg['rr']
                ls = price - atr * sm; ss = price + atr * sm
                lt = price + (price - ls) * rr; st = price - (ss - price) * rr
                mg = vol * price * cfg['multiplier'] * 0.15
                lines.append(
                    f"⚪ {name} **空仓** | 做多 {price:.0f} 损{ls:.0f} 盈{lt:.0f} | "
                    f"做空 损{ss:.0f} 盈{st:.0f} | {vol}手 ¥{mg/10000:.1f}万"
                )
            else:
                lines.append(f"⚪ {name} **观望** | 波动{atr_pct:.1%}过大")
        return "\n".join(lines) if lines else ""

    lines = []
    for r in positions_rows:
        atr = market_data.get(
            [k for k, v in SYMBOLS.items() if v['name'] == r['name']][0], {}
        ).get('atr', 100) if market_data else 100

        # 风险判断
        if r['dist_stop'] < 50:
            action = f"🚨 **止损在即** — 距止损仅 {r['dist_stop']:.0f} 点，一旦触发立即出场"
        elif r['pnl_pts'] < 0:
            action = f"持有观望 — 浮亏 {abs(r['pnl_pts']):.0f} 点"
        elif r['pnl_pts'] < atr * 1:
            action = f"持有 — 保本触发还需 {atr - r['pnl_pts']:.0f} 点"
        elif r['pnl_pts'] < atr * 2:
            action = f"持有 — 盈利 {r['pnl_pts']:.0f} 点 ({(r['pnl_pts']/atr):.1f}ATR)"
        elif r['pnl_pts'] < atr * 4:
            action = f"⚠️ 建议减仓 — {(r['pnl_pts']/atr):.1f}ATR 大幅盈利"
        else:
            action = f"🔔 锁利 — {(r['pnl_pts']/atr):.1f}ATR 远超目标"

        lines.append(
            f"{r['pnl_emoji']} **[{r['ver']}] {r['name']}** {r['dir']} {r['vol']}手 | "
            f"{r['pnl_pts']:+.0f}点 | → {action}"
        )
    return "\n".join(lines)

def build_equity_table(market_data):
    """权益对比表格"""
    lines = [
        "**💰 权益**\n",
        "| 版本 | 权益 | 盈亏 | 持仓 |",
        "|------|------|------|------|",
    ]
    best_eq = 0
    vers = []
    for ver_label, state_file in STATE_FILES.items():
        path = os.path.join(BASE_DIR, state_file)
        if not os.path.exists(path):
            continue
        state = load_state(path)
        cash = state.get('cash', CAPITAL)
        eq = cash
        for sym_key, pos_data in state.get('positions', {}).items():
            multiplier = SYMBOLS.get(sym_key, {}).get('multiplier', 10)
            cur = market_data.get(sym_key, {}).get('price', 0) if market_data else 0
            p_list = pos_data if isinstance(pos_data, list) else [pos_data]
            for p in p_list:
                if p['dir'] == 'LONG':
                    eq += (cur - p['entry']) * p['vol'] * multiplier
                else:
                    eq += (p['entry'] - cur) * p['vol'] * multiplier
        pnl = eq - CAPITAL
        npos = len(state.get('positions', {}))
        vers.append((ver_label, eq, pnl, npos))
        best_eq = max(best_eq, eq)

    for ver_label, eq, pnl, npos in vers:
        crown = ' 👑' if eq == best_eq and len(vers) > 1 else ''
        emoji = '🟢' if pnl >= 0 else '🔴'
        lines.append(
            f"| {emoji} **{ver_label}** | ¥{eq:,.0f} | {pnl:+,.0f} | {npos}{crown} |"
        )
    return "\n".join(lines)

# ============================================================
#  主生成函数
# ============================================================

def generate_report(mode='morning'):
    state = adjust_stops()
    now = datetime.now()
    today_str = now.strftime('%Y-%m-%d')
    wday = ['周一','周二','周三','周四','周五','周六','周日'][now.weekday()]
    market_data = get_market_data()

    if not market_data:
        return "⚠️ 无法获取行情数据", ("", "", "", [])

    # 模型预测
    model_preds = {}
    for sym_key in SYMBOLS:
        mp = get_model_prediction(sym_key)
        if mp:
            model_preds[sym_key] = mp

    # 所有持仓
    positions_rows = collect_all_positions(market_data)

    # ── 文本输出（兼容）──
    mode_cn = '早报' if mode == 'morning' else '晚报'
    text_lines = [f"══ Prophet Futures {mode_cn} ══", f"{today_str} {wday}"]
    for r in (positions_rows or []):
        text_lines.append(
            f"  {r['pnl_emoji']} [{r['ver']}] {r['name']} {r['dir']} {r['vol']}手 "
            f"@{r['entry']:.0f} → {r['cur']:.0f} ({r['pnl_pts']:+.0f}点)"
        )
    text_report = '\n'.join(text_lines)

    # ── 卡片构建 ──
    title = f"📊 Prophet {mode_cn}"
    subtitle = f"{today_str} {wday}"

    # 风险检测
    has_urgent = any(r['dist_stop'] < 50 for r in (positions_rows or []))
    template = "red" if has_urgent else "blue"

    elements = []

    # 1. 持仓表格
    pos_table = build_positions_table(positions_rows, market_data)
    if pos_table:
        elements.append(md(pos_table))
    else:
        elements.append(md("**📌 持仓**\n\n空仓"))
    elements.append(hr())

    # 2. 行情表格
    mkt_table = build_market_table(market_data)
    elements.append(md(mkt_table))
    elements.append(hr())

    # 3. 模型预测表格
    model_table = build_model_table(model_preds, positions_rows)
    if model_table:
        elements.append(md(model_table))
        elements.append(hr())

    # 4. 操作建议（精简扫描风格）
    actions_text = build_actions(positions_rows, market_data)
    if actions_text:
        elements.append(md(f"**🎯 操作**\n\n{actions_text}"))
        elements.append(hr())

    # 5. 权益对比表格
    equity_table = build_equity_table(market_data)
    elements.append(md(equity_table))

    return text_report, (title, subtitle, template, elements)

if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'evening'
    use_card = '--card' in sys.argv

    text, card_data = generate_report(mode)

    if use_card:
        title, subtitle, template, elements = card_data
        ok, result = send_card(title, elements, template, subtitle)
        if ok:
            print(f"✅ 卡片已发送 ({result})")
        else:
            print(f"❌ 卡片发送失败: {result}")
    else:
        print(text)
