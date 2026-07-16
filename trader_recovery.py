#!/usr/bin/env python3
"""纸盘崩溃恢复 — 启动时补执行已穿止损"""
import os, json, numpy as np, pandas as pd
from datetime import datetime
import akshare as ak

def run_recovery(state_file, symbols, label):
    """启动时检查持仓是否已穿止损，若穿则立即补平仓"""
    if not os.path.exists(state_file):
        return
    
    with open(state_file) as f:
        state = json.load(f)
    
    positions = state.get('positions', {})
    if not positions:
        return
    
    recovered = False
    
    for sym_key in list(positions.keys()):
        cfg = symbols.get(sym_key)
        if not cfg:
            continue
        
        # 获取最新日线
        try:
            df = ak.futures_main_sina(symbol=cfg['code'])
            df.columns = ['date','open','high','low','close','volume','oi','settle']
            for c in ['open','high','low','close']:
                df[c] = pd.to_numeric(df[c], errors='coerce')
            df = df.dropna(subset=['close'])
            cur_price = float(df.iloc[-1]['close'])
        except:
            continue
        
        pos_list = positions[sym_key]
        if not isinstance(pos_list, list):
            pos_list = [pos_list]
        
        surviving = []
        for pos in pos_list:
            d = pos['dir']; entry = pos['entry']; vol = pos['vol']
            
            # 获取止损价：V25用stop，动态版用_trail
            if 'stop' in pos:
                stop = pos['stop']
            elif '_trail' in pos:
                stop = pos['_trail']
            else:
                surviving.append(pos)
                continue
            
            # 检查是否穿止损
            breached = (d == 'LONG' and cur_price <= stop) or (d == 'SHORT' and cur_price >= stop)
            
            if breached:
                mult = cfg['multiplier']; cost = cfg['cost']
                margin = vol * entry * mult * 0.15
                pnl = margin * ((stop-entry)/entry/0.15 - cost*2) if d == 'LONG' else margin * ((entry-stop)/entry/0.15 - cost*2)
                state['cash'] += margin + pnl
                
                trade = {
                    'sym': sym_key, 'dir': d, 'entry': entry, 'exit': stop,
                    'vol': vol, 'pnl': pnl, 'type': 'RECOVERY',
                    'time': datetime.now().isoformat()
                }
                state.setdefault('trades', []).append(trade)
                
                print(f'  ⚠️ [{label}] 恢复止损 {sym_key} {d} {vol}手 @{entry:.0f}→{stop:.0f} PnL={pnl:+.0f}')
                recovered = True
            else:
                # 更新追踪止损到最新价格（只紧不松）
                if '_trail' in pos and d == 'LONG':
                    pos['_trail'] = max(pos['_trail'], cur_price - (entry - stop))
                elif '_trail' in pos and d == 'SHORT':
                    pos['_trail'] = min(pos['_trail'], cur_price + (stop - entry))
                surviving.append(pos)
        
        if surviving:
            positions[sym_key] = surviving
        else:
            del positions[sym_key]
    
    if recovered:
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2, default=str)
        cash_str = format(state['cash'], ',.0f')
        print(f'  [{label}] 恢复完成, 现金=¥{cash_str}')
