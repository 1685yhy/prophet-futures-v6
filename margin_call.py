#!/usr/bin/env python3
"""保证金强平模块 — 纸盘共享"""
import json

def check_margin_call(state_file, symbols):
    """检查是否需要强平，若需要则执行
    
    规则: 可用资金为负 → 标记 → 下次扫描仍为负 → 强平
    返回: (executed, message)
    """
    try:
        with open(state_file) as f:
            state = json.load(f)
    except:
        return False, "状态读取失败"
    
    positions = state.get('positions', {})
    if not positions:
        # 清除标记
        state.pop('_margin_call', None)
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2, default=str)
        return False, "无持仓"
    
    # 计算总权益和保证金
    total_equity = state['cash']
    total_margin = 0
    for sym_key, pos_list in positions.items():
        cfg = symbols.get(sym_key, {})
        mult = cfg.get('multiplier', 16)
        if isinstance(pos_list, dict):
            pos_list = [pos_list]
        for pos in pos_list:
            total_margin += pos['vol'] * pos['entry'] * mult * 0.15
    total_equity += total_margin  # margin is already deducted from cash, add back
    
    available = total_equity - total_margin
    had_call = state.get('_margin_call', False)
    
    if available >= 0:
        if had_call:
            state.pop('_margin_call', None)
            with open(state_file, 'w') as f:
                json.dump(state, f, indent=2, default=str)
            return False, f"保证金恢复(可用¥{available:,.0f})"
        return False, "正常"
    
    # 可用资金为负
    deficit = abs(available)
    
    if not had_call:
        # 首次发现，标记等下次
        state['_margin_call'] = True
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2, default=str)
        return False, f"⚠️ 保证金不足¥{deficit:,.0f}，标记待下次检查"
    
    # 第二次检查仍为负 → 强平
    # 按浮亏排序，先平最差的
    import numpy as np, pandas as pd, akshare as ak
    
    # 获取当前价格
    try:
        df = ak.futures_main_sina(symbol='LH0')
        df.columns = ['date','open','high','low','close','volume','oi','settle']
        for c in ['open','high','low','close']:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        df = df.dropna(subset=['close'])
        cur_price = float(df.iloc[-1]['close'])
    except:
        cur_price = None
    
    liquidated = []
    for sym_key in list(positions.keys()):
        cfg = symbols.get(sym_key, {})
        mult = cfg.get('multiplier', 16)
        cost = cfg.get('cost', 0.0006)
        pos_list = positions[sym_key]
        if isinstance(pos_list, dict):
            pos_list = [pos_list]
        
        surviving = []
        for pos in pos_list:
            if deficit <= 0:
                surviving.append(pos)
                continue
            
            d = pos['dir']; entry = pos['entry']; vol = pos['vol']
            margin = vol * entry * mult * 0.15
            price = cur_price if cur_price else entry
            
            # 平仓释放: 保证金 + 浮盈 - 手续费
            if d == 'LONG':
                pnl = (price - entry) * vol * mult
            else:
                pnl = (entry - price) * vol * mult
            commission = entry * vol * mult * cost * 2
            freed = margin + pnl - commission
            
            state['cash'] += freed
            deficit -= freed
            
            trade = {
                'sym': sym_key, 'dir': d, 'entry': entry, 'exit': price,
                'vol': vol, 'pnl': pnl - commission, 'type': 'LIQUIDATE',
                'time': pd.Timestamp.now().isoformat()
            }
            state.setdefault('trades', []).append(trade)
            liquidated.append(f'{d} {vol}手@{entry:.0f}→{price:.0f}')
        
        if surviving:
            positions[sym_key] = surviving
        else:
            del positions[sym_key]
    
    state.pop('_margin_call', None)
    with open(state_file, 'w') as f:
        json.dump(state, f, indent=2, default=str)
    
    liq_str = ' '.join(liquidated)
    cash_str = format(state['cash'], ',.0f')
    msg = f'🚨 强平: {liq_str}, 现金→¥{cash_str}'
    return True, msg
