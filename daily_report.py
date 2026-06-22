#!/usr/bin/env python3
"""Prophet Futures — 早晚报告生成器 v25"""
import sys, os, json, numpy as np, pandas as pd
from datetime import datetime, timedelta
import akshare as ak

SYMBOLS = {
    'lh2609': {'code': 'LH0', 'name': 'LH 生猪', 'cost': 0.0006, 'multiplier': 16,
               'stop_type': 'atr', 'stop_mult': 1.5, 'rr': 4, 'max_pos': 6},
    'jm2609': {'code': 'JM0', 'name': 'JM 焦煤', 'cost': 0.0011, 'multiplier': 60,
               'stop_type': 'struct', 'struct_n': 20, 'rr': 3.5, 'max_pos': 4},
}
CAPITAL = 300000
STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'paper_state.json')

def fetch(sym, days=500):
    code = sym.upper() + '0'
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

def evening_report():
    """晚间报告 19:00"""
    state = load_state()
    now = datetime.now()
    today_str = now.strftime('%Y-%m-%d')
    
    lines = []
    lines.append("═" * 36)
    lines.append(f"  Prophet Futures 晚间报告")
    lines.append(f"  {today_str} {['周一','周二','周三','周四','周五','周六','周日'][now.weekday()]}")
    lines.append("═" * 36)
    
    # Cash
    lines.append(f"\n💰 账户资金: ¥{state['cash']:,.0f}")
    
    # Today's trades
    today_trades = [t for t in state.get('trades', []) if today_str in str(t.get('exit_time', ''))]
    if today_trades:
        lines.append(f"\n📊 今日成交 ({len(today_trades)}笔)")
        total_pnl = 0
        for t in today_trades:
            emoji = '🟢' if t['pnl_amount'] > 0 else '🔴'
            lines.append(f"  {emoji} {t['sym']} {t['dir']} {t['vol']}手"
                        f" @{t['entry']:.0f}→{t['exit']:.0f}"
                        f" PnL={t['pnl_amount']:+,.0f}")
            total_pnl += t['pnl_amount']
        lines.append(f"  合计: {total_pnl:+,.0f} 元")
    else:
        lines.append(f"\n📊 今日无平仓")
    
    # Current positions
    positions = state.get('positions', {})
    if positions:
        lines.append(f"\n📌 当前持仓 ({len(positions)}笔)")
        for sym_key, pos in positions.items():
            cfg = SYMBOLS.get(sym_key, {})
            name = cfg.get('name', sym_key)
            d = '做多' if pos['dir'] == 'LONG' else '做空'
            lines.append(f"  {d} {name}")
            lines.append(f"    开仓: {pos['entry']:.0f}  |  {pos['vol']}手")
            
            # Get current price
            code = cfg.get('code', sym_key[:2].upper()+'0')
            df = fetch(code, 10)
            if df is not None and len(df) > 0:
                cur = float(df.iloc[-1]['close'])
                if pos['dir'] == 'LONG':
                    pnl_pct = (cur - pos['entry']) / pos['entry']
                else:
                    pnl_pct = (pos['entry'] - cur) / pos['entry']
                emoji2 = '🟢' if pnl_pct > 0 else '🔴'
                margin = pos['vol'] * pos['entry'] * cfg.get('multiplier', 10) * 0.15
                pnl_amt = pnl_pct * margin
                lines.append(f"    现价: {cur:.0f} | 浮{emoji2} {pnl_amt:+,.0f} ({pnl_pct:+.1%})")
            
            lines.append(f"    止损: {pos['stop']:.0f} | 止盈: {pos['take_profit']:.0f}")
    else:
        lines.append(f"\n📌 当前无持仓")
    
    # Market analysis
    lines.append(f"\n📈 行情概览")
    for sym_key, cfg in SYMBOLS.items():
        df = fetch(cfg['code'], 120)
        if df is None or len(df) < 50: continue
        price = float(df.iloc[-1]['close'])
        ma20 = float(np.mean([float(df.iloc[k]['close']) for k in range(max(0, len(df)-20), len(df))]))
        ma60 = float(np.mean([float(df.iloc[k]['close']) for k in range(max(0, len(df)-60), len(df))]))
        
        # Trend
        if price > ma20 > ma60: trend = '上涨趋势 ✅'
        elif price < ma20 < ma60: trend = '下跌趋势 ⚠️'
        else: trend = '震荡整理'
        
        # Change
        prev_close = float(df.iloc[-2]['close']) if len(df) > 1 else price
        chg_pct = (price - prev_close) / prev_close

        atr_vals = [abs(float(df.iloc[k]['high']) - float(df.iloc[k]['low']))
                    for k in range(max(0, len(df)-20), len(df))]
        atr = np.mean(atr_vals) if atr_vals else 0
        
        lines.append(f"  {cfg['name']} @ {price:.0f} ({chg_pct:+.1%})")
        lines.append(f"    趋势: {trend} | 波幅: {atr:.0f}点")
        
        # Stop levels for tomorrow
        if cfg['stop_type'] == 'atr':
            stop_dist = atr * cfg['stop_mult']
            long_stop = price - stop_dist
            short_stop = price + stop_dist
            lines.append(f"    明日做多止损≈{long_stop:.0f} 做空止损≈{short_stop:.0f}")
        else:
            n = cfg['struct_n']
            recent_low = min(float(df.iloc[k]['low']) for k in range(max(0, len(df)-n), len(df)))
            recent_high = max(float(df.iloc[k]['high']) for k in range(max(0, len(df)-n), len(df)))
            lines.append(f"    明日做多止损≈{recent_low:.0f} 做空止损≈{recent_high:.0f}")
    
    lines.append(f"\n{'─'*36}")
    lines.append(f"  生成: {now.strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"  风险提示: 仅供学习参考")
    lines.append(f"{'─'*36}")
    
    return '\n'.join(lines)

def morning_report():
    """早间报告 08:50"""
    state = load_state()
    now = datetime.now()
    today_str = now.strftime('%Y-%m-%d')
    
    lines = []
    lines.append("═" * 36)
    lines.append(f"  Prophet Futures 早间报告")
    lines.append(f"  {today_str} {['周一','周二','周三','周四','周五','周六','周日'][now.weekday()]}")
    lines.append("═" * 36)
    
    # Overnight changes
    lines.append(f"\n🌙 隔夜变化")
    for sym_key, cfg in SYMBOLS.items():
        df = fetch(cfg['code'], 10)
        if df is None or len(df) < 2: continue
        
        prev_close = float(df.iloc[-2]['close'])
        cur = float(df.iloc[-1]['close'])
        chg = (cur - prev_close) / prev_close
        
        # Also get yesterday's close for comparison
        yesterday = cur
        df2 = fetch(cfg['code'], 3)
        if df2 is not None and len(df2) >= 2:
            yesterday = float(df2.iloc[-2]['close'])
            chg_vs_yesterday = (cur - yesterday) / yesterday
        else:
            chg_vs_yesterday = 0
            
        lines.append(f"  {cfg['name']}: {yesterday:.0f} → {cur:.0f} ({chg_vs_yesterday:+.1%})")
        
        # Position update
        if sym_key in state.get('positions', {}):
            pos = state['positions'][sym_key]
            if pos['dir'] == 'LONG':
                pnl_pct = (cur - pos['entry']) / pos['entry']
            else:
                pnl_pct = (pos['entry'] - cur) / pos['entry']
            margin = pos['vol'] * pos['entry'] * cfg['multiplier'] * 0.15
            pnl_amt = pnl_pct * margin
            lines.append(f"    持仓浮{('盈' if pnl_amt>0 else '亏')}: {pnl_amt:+,.0f}")
    
    # Current positions
    positions = state.get('positions', {})
    if positions:
        lines.append(f"\n📌 持仓状态")
        for sym_key, pos in positions.items():
            cfg = SYMBOLS.get(sym_key, {})
            name = cfg.get('name', sym_key)
            d = '做多' if pos['dir'] == 'LONG' else '做空'
            lines.append(f"  {d} {name} {pos['vol']}手 @ {pos['entry']:.0f}")
            lines.append(f"    止损: {pos['stop']:.0f} | 止盈: {pos['take_profit']:.0f}")
    
    # Today's plan
    lines.append(f"\n🎯 今日计划")
    lines.append(f"  模型: XGBoost 单模型")
    lines.append(f"  LH止损: ATR×1.5 (动态)")
    lines.append(f"  JM止损: 20日结构点")
    lines.append(f"  扫描频率: 每5分钟")
    lines.append(f"  交易时段: 9:00-10:15 | 10:30-11:30 | 13:30-15:00")
    
    # Cash
    equity = state['cash']
    for k, p in state.get('positions', {}).items():
        c2 = SYMBOLS.get(k, {}).get('multiplier', 10)
        equity += p['vol'] * p['entry'] * c2 * 0.15
    lines.append(f"\n💰 账户: ¥{state['cash']:,.0f} (含持仓权益约¥{equity:,.0f})")
    
    lines.append(f"\n{'─'*36}")
    lines.append(f"  生成: {now.strftime('%H:%M')}")
    lines.append(f"{'─'*36}")
    
    return '\n'.join(lines)

if __name__ == '__main__':
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else 'evening'
    if mode == 'morning':
        print(morning_report())
    else:
        print(evening_report())
