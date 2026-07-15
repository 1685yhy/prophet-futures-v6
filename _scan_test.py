#!/usr/bin/env python3
"""5分钟扫描 v4 — 实时行情 + 有事才推 + 标注数据日期"""
import xgboost  # workaround for glibc 2.39 ld.so bug — must be first
import sys, os, json, numpy as np, pandas as pd
from datetime import datetime
import pickle
from feishu_card import send_card, md, hr
from realtime_data import (
    get_realtime_quote, get_minute_history, get_daily_history,
    compute_atr_from_minutes, build_features, SYMBOL_MAP
)

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'paper_state.json')
PRED_HISTORY_FILE = os.path.join(MODEL_DIR, 'pred_history.json')

SYMBOLS = {
    'lh2609': {'stop_type': 'atr', 'stop_mult': 1.5, 'rr': 4, 'max_pos': 6,
               'be_atr': 1.0, 'reduce1_atr': 2.0, 'reduce2_atr': 4.0, 'multiplier': 16},
    'jm2609': {'stop_type': 'atr', 'stop_mult': 2.0, 'rr': 3.5, 'max_pos': 4,
               'be_atr': 2.0, 'reduce1_atr': 3.0, 'reduce2_atr': 5.0, 'multiplier': 60},
}

def load_state(path=None):
    p = path or STATE_FILE
    if os.path.exists(p):
        with open(p) as f: return json.load(f)
    return {'cash': 300000, 'positions': {}, 'trades': []}

def check_positions(sym_key, info, price, atr, cfg, pos_data, ticktime, ver_label):
    """检查单版本单个品种的持仓，返回告警文本或None"""
    if not pos_data:
        return None
    # V25 格式: {'entry':x, 'vol':x, ...} 单个
    # V28/V29 格式: [{'entry':x, ...}, ...] 列表
    positions_list = pos_data if isinstance(pos_data, list) else [pos_data]
    alerts = []
    for pos in positions_list:
        entry = pos['entry']
        vol = pos['vol']
        stop = pos.get('stop', pos.get('_trail', entry))
        tp = pos.get('take_profit', entry * 2)
        d = pos['dir']

        if d == 'LONG':
            pnl_pts = price - entry
            dist_stop = price - stop
            dist_tp = tp - price
        else:
            pnl_pts = entry - price
            dist_stop = stop - price
            dist_tp = price - tp

        pnl_atr = pnl_pts / atr if atr > 0 else 0
        dist_atr = dist_stop / atr if atr > 0 else 0
        emoji = '🟢' if pnl_pts > 0 else '🔴'
        dir_cn = '做多' if d == 'LONG' else '做空'

        if dist_atr < 0.5 and dist_stop > 0:
            alerts.append(
                f"🚨 [{ver_label}] **{info['name']} 止损逼近!** {ticktime}\n"
                f"{emoji} {dir_cn} {vol}手 | 实时价 **{price:.0f}** | 止损 **{stop:.0f}**\n"
                f"距止损仅 **{dist_stop:.0f}** 点 ({dist_atr:.1f} ATR)\n"
                f"→ 一旦触发立即出场，不要加仓"
            )
        elif dist_tp < atr and dist_tp > 0:
            alerts.append(
                f"🎯 [{ver_label}] **{info['name']} 止盈逼近!** {ticktime}\n"
                f"{emoji} {dir_cn} {vol}手 | 实时价 **{price:.0f}** | 止盈 **{tp:.0f}**\n"
                f"距止盈 **{dist_tp:.0f}** 点\n"
                f"→ 准备止盈出场"
            )
        elif pnl_atr >= cfg.get('reduce1_atr', 999) and vol >= cfg.get('max_pos', 99):
            cut = max(1, int(vol * 0.5))
            keep = vol - cut
            alerts.append(
                f"⚠️ [{ver_label}] **{info['name']} 减仓触发!** {ticktime}\n"
                f"{emoji} {dir_cn} {vol}手 | +{pnl_pts:.0f}点 ({pnl_atr:.1f} ATR)\n"
                f"实时价 **{price:.0f}** | 止盈 {tp:.0f}\n"
                f"→ 建议减 {cut}手 → 留 **{keep}手**"
            )
    return alerts

def is_trading_time():
    now = datetime.now()
    h, m = now.hour, now.minute
    t = h * 60 + m
    return (540 <= t < 690) or (810 <= t < 900)

def main():
    if not is_trading_time():
        return

    now = datetime.now()
    # 加载三个版本状态
    states = [
        ('V25', load_state()),
        ('V28', load_state(STATE_FILE.replace('.json', '_v28.json'))),
        ('V29', load_state(STATE_FILE.replace('.json', '_v29.json'))),
        ('V30', load_state(STATE_FILE.replace('.json', '_v30.json'))),
    ]
    alerts = []

    for sym_key, cfg in SYMBOLS.items():
        info = SYMBOL_MAP.get(sym_key)
        if not info: continue

        # ── 实时报价 ──
        rt = get_realtime_quote(sym_key)
        if not rt: continue
        price = rt['price']
        ticktime = rt['ticktime']

        # ── 分钟ATR ──
        min_df = get_minute_history(sym_key, 60)
        atr = compute_atr_from_minutes(min_df, 20) if min_df is not None else price * 0.002
        if atr is None or atr <= 0: atr = price * 0.002

        # ── 模型预测（用日线）──
        signal_text = None
        ref_date = ""
        # 优先加载校准模型
        for model_suffix in ['_xgb_calibrated.pkl', '_xgb.pkl']:
            mp = os.path.join(MODEL_DIR, f'{sym_key}{model_suffix}')
            if os.path.exists(mp):
                break
        if os.path.exists(mp):
            with open(mp, 'rb') as f: model = pickle.load(f)
            daily_df = get_daily_history(sym_key, 1200)
            if daily_df is not None and len(daily_df) >= 100:
                ref_date = str(daily_df.iloc[-1]['date'])
                feats = build_features(daily_df, len(daily_df)-1, 60)
                if feats is not None:
                    try:
                        prob = float(model.predict_proba(feats.reshape(1, -1))[0][1])
                        direction = 'LONG' if prob > 0.5 else 'SHORT'
                        confidence = prob if prob > 0.5 else (1 - prob)

                        # ── 前一日比对 ──
                        delta_str = ""
                        try:
                            if os.path.exists(PRED_HISTORY_FILE):
                                with open(PRED_HISTORY_FILE) as f:
                                    hist = json.load(f)
                                entries = hist.get(sym_key, [])
                                if entries and entries[-1].get('date') != ref_date:
                                    prev_p = entries[-1].get('prob', 0)
                                    d = prob - prev_p
                                    arrow = '↑' if d > 0 else '↓' if d < 0 else '→'
                                    delta_str = f"（{arrow}{abs(d):.1%}）"
                        except: pass

                        if confidence > 0.58:
                            stop_mult = cfg['stop_mult']
                            sd = atr * stop_mult
                            if direction == 'LONG':
                                sp = price - sd
                                tp = price + (price - sp) * cfg['rr']
                            else:
                                sp = price + sd
                                tp = price - (sp - price) * cfg['rr']

                            atr_pct = atr / price
                            if atr_pct < 0.01: lev = 3.0
                            elif atr_pct < 0.02: lev = 2.0
                            elif atr_pct < 0.03: lev = 1.5
                            elif atr_pct < 0.05: lev = 0.5
                            else: lev = 0
                            vol = max(1, int(lev * (cfg['max_pos'] // 2))) if lev > 0 else 0

                            if vol > 0:
                                emoji = '🟢' if direction == 'LONG' else '🔴'
                                dir_cn = '做多' if direction == 'LONG' else '做空'
                                mg = vol * price * cfg['multiplier'] * 0.15
                                signal_text = (
                                    f"{emoji} **{info['name']} {dir_cn}信号**\\n"
                                    f"🧠 模型{'偏多' if direction=='LONG' else '偏空'} {confidence:.1%}{delta_str}（基于 {ref_date} 日线）\\n"
                                    f"实时价 **{price:.0f}** | 止损 **{sp:.0f}** | 止盈 **{tp:.0f}**\\n"
                                    f"{vol}手 ¥{mg/10000:.1f}万 | RR=1:{cfg['rr']:.0f}"
                                )
                    except: pass

        # ── 持仓检查（三个版本）──
        for ver_label, ver_state in states:
            ver_positions = ver_state.get('positions', {})
            pos_data = ver_positions.get(sym_key)
            if pos_data:
                ver_alerts = check_positions(sym_key, info, price, atr, cfg, pos_data, ticktime, ver_label)
                if ver_alerts:
                    alerts.extend(ver_alerts)

        if signal_text: alerts.append(signal_text)

    # ── 没事件，静默 ──
    # ── 始终推送（不论有无提醒） ──
    if not alerts:
        alerts = ["✅ 持仓正常，无异常触发"]

    # ── 构建卡片 ──
    time_str = now.strftime('%H:%M')
    elements = []

    # 行情快照
    price_lines = ["**📊 实时行情**\n"]
    for sym_key in SYMBOLS:
        rt = get_realtime_quote(sym_key)
        if not rt: continue
        info = SYMBOL_MAP[sym_key]
        p = rt['price']
        chg = rt['changepercent']
        emoji = '📈' if chg > 0 else ('📉' if chg < 0 else '➡️')
        price_lines.append(
            f"{emoji} {info['name']} **{p:.0f}** ({chg:+.2%}) | "
            f"{rt['open']:.0f}→{rt['high']:.0f}→{rt['low']:.0f} | 量{rt['volume']}"
        )
    elements.append(md("\n".join(price_lines)))
    elements.append(hr())

    # 提醒事件
    elements.append(md("**🔔 提醒**\n\n" + "\n\n---\n\n".join(alerts)))

    # 发送
    alert_count = len(alerts)
    urgent = any('🚨' in a for a in alerts)
    title = f"{'🚨' if urgent else '🔔'} 行情提醒 · {time_str}"
    template = "red" if urgent else "blue"

    ok, result = send_card(title, elements, template)
    if ok:
        print(f"✅ {time_str} 已推送 {alert_count} 条提醒")
    else:
        print(f"❌ 发送失败: {result}")

if __name__ == '__main__':
    main()
