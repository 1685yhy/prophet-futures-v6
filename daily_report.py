#!/usr/bin/env python3
"""Prophet Futures — 早晚报告生成器 v26
用法:
  python daily_report.py morning          # 早报文本
  python daily_report.py evening           # 晚报文本
  python daily_report.py morning --card    # 早报卡片
  python daily_report.py evening --card   # 晚报卡片
"""
import sys, os, json, numpy as np, pandas as pd, pickle
from datetime import datetime, timedelta
import akshare as ak
from feishu_card import send_card, md, hr, build_position_actions, build_market_table
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
STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'paper_state.json')
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')

def get_model_prediction(sym_key):
    """获取模型预测 — 返回 dict 或 None
    
    优先加载校准模型（_calibrated.pkl），找不到则回退到未校准模型
    """
    # 优先校准模型
    for suffix in ['_xgb_calibrated.pkl', '_xgb.pkl']:
        mp = os.path.join(MODEL_DIR, f'{sym_key}{suffix}')
        if os.path.exists(mp):
            break
    else:
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

def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f: return json.load(f)
    return {'cash': CAPITAL, 'positions': {}, 'trades': [], 'equity_history': []}

def save_state(state):
    with open(STATE_FILE, 'w') as f: json.dump(state, f, indent=2, default=str)

def adjust_stops():
    """动态调整所有持仓的止损 — 只收紧，不放松"""
    state = load_state()
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
        save_state(state)
    return state

def get_market_data():
    """获取所有品种的行情数据 — 用实时价 + 日线趋势"""
    data = {}
    for sym_key, cfg in SYMBOLS.items():
        df = fetch(cfg['code'], 60)
        if df is None or len(df) < 20: continue

        # 趋势用日线算
        ma20 = float(np.mean([float(df.iloc[k]['close']) for k in range(max(0, len(df)-20), len(df))]))
        ma60 = float(np.mean([float(df.iloc[k]['close']) for k in range(max(0, len(df)-60), len(df))]))
        atr_vals = [abs(float(df.iloc[k]['high']) - float(df.iloc[k]['low']))
                    for k in range(max(0, len(df)-20), len(df))]
        atr = np.mean(atr_vals) if atr_vals else 0

        # 日线上一根收盘
        prev_close = float(df.iloc[-2]['close']) if len(df) > 1 else 0

        # 用实时价
        rt = get_realtime_quote(sym_key)
        if rt:
            price = rt['price']
            chg = rt['changepercent']
            o = rt['open']
            h = rt['high']
            l = rt['low']
        else:
            # fallback: 日线收盘
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
    """计算权益 = 现金 + 浮动盈亏"""
    equity = state['cash']
    for k, p in state.get('positions', {}).items():
        multiplier = SYMBOLS.get(k, {}).get('multiplier', 10)
        cur = p['entry']  # 默认用入场价
        if market_data and k in market_data:
            cur = market_data[k]['price']
        if p['dir'] == 'LONG':
            pnl = (cur - p['entry']) * p['vol'] * multiplier
        else:
            pnl = (p['entry'] - cur) * p['vol'] * multiplier
        equity += pnl
    return equity

def compute_version_equities(market_data):
    """读取所有版本状态文件，计算各自权益"""
    versions = {}
    for label, state_file in [
        ('V25(基础)', 'paper_state.json'),
        ('V28(旧模型)', 'paper_state_v28.json'),
        ('V29(新模型)', 'paper_state_v29.json'),
    ]:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), state_file)
        if not os.path.exists(path):
            continue
        try:
            with open(path) as f:
                state = json.load(f)
        except:
            continue
        cash = state.get('cash', 300000)
        equity = cash
        for sym_key, pos_data in state.get('positions', {}).items():
            multiplier = SYMBOLS.get(sym_key, {}).get('multiplier', 10)
            cur = market_data.get(sym_key, {}).get('price', 0) if market_data else 0
            if isinstance(pos_data, list):
                for p in pos_data:
                    entry = p['entry']
                    vol = p['vol']
                    d = p['dir']
                    pnl = (cur - entry) * vol * multiplier if d == 'LONG' else (entry - cur) * vol * multiplier
                    equity += pnl
            else:
                entry = pos_data['entry']
                vol = pos_data['vol']
                d = pos_data['dir']
                pnl = (cur - entry) * vol * multiplier if d == 'LONG' else (entry - cur) * vol * multiplier
                equity += pnl
        versions[label] = {
            'cash': cash,
            'equity': equity,
            'pnl': equity - 300000,
            'pnl_pct': (equity - 300000) / 300000,
            'positions': len(state.get('positions', {})),
        }
    return versions


def generate_report(mode='morning'):
    """生成报告（文本 + 卡片），返回 (text, should_send_card)
    卡片只在实际有数据时发送
    """
    state = adjust_stops()
    now = datetime.now()
    today_str = now.strftime('%Y-%m-%d')
    wday = ['周一','周二','周三','周四','周五','周六','周日'][now.weekday()]
    positions = state.get('positions', {})
    market_data = get_market_data()
    equity = compute_equity(state, market_data)

    # ── 模型预测 ──
    model_preds = {}
    for sym_key in SYMBOLS:
        mp = get_model_prediction(sym_key)
        if mp:
            model_preds[sym_key] = mp

    if not market_data:
        return "⚠️ 无法获取行情数据", ("", "", "", [])

    # ── 文本输出（兼容）──
    mode_cn = '早报' if mode == 'morning' else '晚报'
    lines = ["═" * 36, f"  Prophet Futures {mode_cn}", f"  {today_str} {wday}", "═" * 36]

    # 持仓
    if positions:
        lines.append("\n📌 持仓")
        for sym_key, pos in positions.items():
            cfg = SYMBOLS.get(sym_key, {})
            m = market_data.get(sym_key)
            name = cfg.get('name', sym_key)
            d = '做多' if pos['dir'] == 'LONG' else '做空'
            cur = m['price'] if m else pos['entry']
            pnl_pts = cur - pos['entry'] if pos['dir'] == 'LONG' else pos['entry'] - cur
            pnl_pct = pnl_pts / pos['entry']
            emoji = '🟢' if pnl_pts > 0 else '🔴'
            lines.append(f"  {emoji} {name} {d} {pos['vol']}手 @{pos['entry']:.0f} → {cur:.0f}")
            lines.append(f"    浮{'盈' if pnl_pts>0 else '亏'} {abs(pnl_pts):.0f}点 ({pnl_pct:+.1%})")
    else:
        lines.append("\n📌 无持仓")

    # 行情
    lines.append("\n📊 行情")
    for sym_key, m in market_data.items():
        name = SYMBOLS[sym_key]['name']
        lines.append(f"  {m['trend'][:1]} {name} @{m['price']:.0f} ({m['chg']:+.1%}) | ATR {m['atr']:.0f}")

    # 模型预测
    if model_preds:
        ref_date = list(model_preds.values())[0]['last_date']
        lines.append(f"\n🧠 模型预测（基于 {ref_date} 日线）")
        for sym_key, mp in model_preds.items():
            name = SYMBOLS[sym_key]['name']
            dir_cn = '偏多' if mp['direction'] == 'LONG' else '偏空'
            emoji = '🟢' if mp['direction'] == 'LONG' else '🔴'
            # 检查持仓冲突
            pos = positions.get(sym_key)
            conflict = ''
            if pos:
                if pos['dir'] != mp['direction']:
                    conflict = ' ⚠️ 与你持仓方向冲突'
                else:
                    conflict = ' ✅ 持仓方向一致'
            lines.append(f"  {emoji} {name} {dir_cn} {mp['prob']:.1%} | 置信度 {mp['confidence']:.1%}{conflict}")

    # 操作
    actions_text = build_position_actions(positions, market_data, SYMBOLS, mode)
    lines.append(f"\n🎯 今日操作\n{actions_text}")

    # 权益
    lines.append(f"\n💰 权益 ¥{equity:,.0f} | 可用 ¥{state['cash']:,.0f}")
    lines.append(f"\n{'─'*36}")
    text_report = '\n'.join(lines)

    # ── 卡片构建 ──
    title = f"📊 Prophet {mode_cn}"
    subtitle = f"{today_str} {wday}"
    # 根据风险动态模板色
    has_urgent = '🚨' in actions_text if actions_text else False
    template = "red" if has_urgent else "blue"

    elements = []

    # ── 1. 持仓 ──
    if positions:
        pos_lines = ["**📌 持仓**\n"]
        for sym_key, pos in positions.items():
            cfg = SYMBOLS.get(sym_key, {})
            m = market_data.get(sym_key)
            name = cfg.get('name', sym_key)
            d = '做多' if pos['dir'] == 'LONG' else '做空'
            cur = m['price'] if m else pos['entry']
            pnl_pts = cur - pos['entry'] if pos['dir'] == 'LONG' else pos['entry'] - cur
            pnl_yuan = pnl_pts * pos['vol'] * cfg.get('multiplier', 10)
            pnl_atr = pnl_pts / m['atr'] if m and m['atr'] > 0 else 0
            emoji = '🟢' if pnl_pts > 0 else '🔴'
            pos_lines.append(
                f"{emoji} {name} {d} {pos['vol']}手 | "
                f"成本{pos['entry']:.0f} 现价{cur:.0f} | "
                f"+{pnl_pts:.0f}点({pnl_atr:.1f}ATR) ¥{pnl_yuan/10000:+.1f}万"
            )
        elements.append(md("\n".join(pos_lines)))
    else:
        elements.append(md("**📌 持仓**: 空仓"))
    elements.append(hr())

    # ── 2. 行情（对标扫描的紧凑格式）──
    mkt_lines = ["**📊 行情**\n"]
    for sym_key, m in market_data.items():
        name = SYMBOLS[sym_key]['name']
        chg = m['chg']
        emoji = '📈' if chg > 0.001 else ('📉' if chg < -0.001 else '➡️')
        mkt_lines.append(
            f"{emoji} {name} **{m['price']:.0f}** ({chg:+.2%}) | "
            f"O{m['open']:.0f} H{m['high']:.0f} L{m['low']:.0f} | "
            f"ATR{m['atr']:.0f} {m['trend']}"
        )
    elements.append(md("\n".join(mkt_lines)))
    elements.append(hr())

    # ── 3. 模型预测 ──
    if model_preds:
        ref_date = list(model_preds.values())[0]['last_date']
        pred_lines = [f"**🧠 模型预测**（{ref_date} 日线）\n"]
        for sym_key, mp in model_preds.items():
            name = SYMBOLS[sym_key]['name']
            dir_cn = '偏多' if mp['direction'] == 'LONG' else '偏空'
            emoji = '🟢' if mp['direction'] == 'LONG' else '🔴'
            pos = positions.get(sym_key)
            if pos:
                conflict = " ⚠️ 冲突" if pos['dir'] != mp['direction'] else " ✅ 一致"
            else:
                conflict = ""
            pred_lines.append(
                f"{emoji} {name} **{dir_cn}** {mp['prob']:.1%}{conflict}"
            )
        elements.append(md("\n".join(pred_lines)))
        elements.append(hr())

    # ── 4. 操作（精简版，对标扫描的直白风格）──
    if actions_text:
        elements.append(md(f"**🎯 操作**\n\n{actions_text}"))
        elements.append(hr())

    # ── 5. 权益 + 版本对比（合并）──
    floating_pnl = equity - state['cash']
    pnl_emoji = '🟢' if floating_pnl >= 0 else '🔴'
    ver_lines = [
        f"💰 **¥{equity:,.0f}** | 浮盈 {pnl_emoji} ¥{floating_pnl:,.0f} | 现金 ¥{state['cash']:,.0f}\n"
    ]
    versions = compute_version_equities(market_data)
    if len(versions) >= 2:
        ver_lines.append("**📊 版本**\n")
        best_equity = max(v['equity'] for v in versions.values())
        for label, ver in versions.items():
            crown = ' 👑' if ver['equity'] == best_equity else ''
            emoji = '🟢' if ver['pnl'] >= 0 else '🔴'
            ver_lines.append(
                f"{emoji} **{label}** ¥{ver['equity']:,.0f} ({ver['pnl']:+,.0f}){crown}"
            )
    elements.append(md("\n".join(ver_lines)))

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
