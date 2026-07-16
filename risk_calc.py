#!/usr/bin/env python3
"""共享风控计算 — 报告和纸盘都用这一套逻辑"""
import numpy as np, pandas as pd, akshare as ak

def get_latest_quote():
    """获取LH最新日线数据"""
    df = ak.futures_main_sina(symbol='LH0')
    df.columns = ['date','open','high','low','close','volume','oi','settle']
    for c in ['open','high','low','close','volume','oi']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=['close'])
    price = float(df.iloc[-1]['close'])
    av = [abs(float(df.iloc[i]['high'])-float(df.iloc[i]['low'])) 
          for i in range(max(0,len(df)-20),len(df))]
    atr = np.mean(av)
    return price, atr, df

def calc_stop(entry, direction, atr, atr_mult):
    """算止损价"""
    if direction == 'LONG':
        return entry - atr * atr_mult
    else:
        return entry + atr * atr_mult

def calc_tp(entry, stop, direction, rr):
    """算止盈价 = 入场 + 风险距离 × RR"""
    risk = abs(entry - stop)
    if direction == 'LONG':
        return entry + risk * rr
    else:
        return entry - risk * rr

def calc_trail(existing_trail, entry, direction, price, atr, trail_atr, be_atr, atr_stop):
    """算追踪止损 — 只紧不松"""
    if direction == 'LONG':
        fp = price - entry
        hard_stop = price - atr * atr_stop
        new_trail = existing_trail
        if fp > atr * trail_atr:
            new_trail = max(new_trail, price - atr * (atr_stop - 0.3))
        if fp > atr * be_atr:
            new_trail = max(new_trail, entry)
        return max(hard_stop, new_trail)
    else:
        fp = entry - price
        hard_stop = price + atr * atr_stop
        new_trail = existing_trail
        if fp > atr * trail_atr:
            new_trail = min(new_trail, price + atr * (atr_stop - 0.3))
        if fp > atr * be_atr:
            new_trail = min(new_trail, entry)
        return min(hard_stop, new_trail)
