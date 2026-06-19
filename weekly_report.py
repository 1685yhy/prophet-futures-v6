#!/usr/bin/env python3
"""周报 — 统计本周交易盈亏、胜率、最大回撤"""
import sys, json, os
from datetime import datetime, timedelta

LOG_FILE = "/tmp/prophet_trade_log.jsonl"

def this_week():
    """Return Monday of this week."""
    today = datetime.now()
    return today - timedelta(days=today.weekday())

def load_trades():
    if not os.path.exists(LOG_FILE): return []
    trades = []
    with open(LOG_FILE) as f:
        for line in f:
            try: trades.append(json.loads(line))
            except: pass
    return trades

def week_summary():
    trades = load_trades()
    monday = this_week()
    week_trades = [t for t in trades 
                   if "date" in t and datetime.strptime(t["date"][:10],"%Y-%m-%d") >= monday]
    
    if not week_trades:
        print("📊 本周无交易")
        return
    
    pnls = [t.get("pnl",0) for t in week_trades]
    n = len(pnls)
    wins = [p for p in pnls if p > 0]
    wr = len(wins)/n if n else 0
    tp = sum(pnls)
    
    print(f"📊 本周交易周报 ({monday.strftime('%m/%d')} ~ {datetime.now().strftime('%m/%d')})")
    print(f"  交易: {n}笔  胜率: {wr:.0%}  盈亏: {tp:+,.0f}元")
    print(f"  均盈: {np.mean(wins):+,.0f}" if wins else "  均盈: —")
    losses = [p for p in pnls if p <= 0]
    if losses: print(f"  均亏: {np.mean(losses):+,.0f}")
    print(f"  明细:")
    for t in week_trades:
        icon = "✅" if t.get("pnl",0) > 0 else "❌"
        print(f"    {icon} {t.get('date','')} {t.get('sym','')} {t.get('dir','')} {t.get('pnl',0):+,.0f}")

if __name__ == "__main__":
    import numpy as np
    week_summary()
