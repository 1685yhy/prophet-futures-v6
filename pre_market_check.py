#!/usr/bin/env python3
"""开盘前检查 — 验证昨日信号是否仍然有效"""

import sys; sys.path.insert(0, ".")
import numpy as np
from datetime import datetime

def fetch_quote(sym):
    import akshare as ak
    try:
        df = ak.futures_main_sina(sym.upper()+"0")
        if df is not None and len(df) > 0:
            return float(df.iloc[0]["close"]) if "close" in df.columns else None
        # Try spot price
        spot = ak.futures_zh_spot(symbol=sym.upper())
        if spot is not None and len(spot) > 0:
            return float(spot.iloc[0].get("last_price", 0)) or float(spot.iloc[0].get("trade", 0))
    except:
        pass
    return None

# 读取昨天的信号
signal_file = "/tmp/prophet_signal.txt"
try:
    with open(signal_file) as f:
        yesterday = f.read().strip()
except:
    yesterday = ""

if not yesterday:
    print("无昨日信号")
    sys.exit(0)

parts = yesterday.split(",")
sym, direction, entry_str = parts[0], parts[1], parts[2]
entry = float(entry_str)

# 获取当前价
current = fetch_quote(sym)

if current is None:
    print(f"⚠️ {sym}: 无法获取实时价格，手动判断")
    print(f"  昨日信号: {direction} @ {entry:.0f}")
    sys.exit(0)

gap = (current - entry) / entry

if abs(gap) > 0.015:
    print(f"⛔ {sym}: 跳空{abs(gap):.1%}！取消昨日{direction}信号")
    print(f"  昨日入场价: {entry:.0f}")
    print(f"  当前价: {current:.0f}")
    print(f"  建议: 观望，等市场稳定")
else:
    print(f"✅ {sym}: 价格正常 (偏离{gap:+.1%})，执行{direction}")
    print(f"  入场价: {entry:.0f} → 当前: {current:.0f}")

# 检查新闻（简单关键词）
import subprocess
try:
    result = subprocess.run(
        ["python", "-c", f"from tools.llm_utils import get_llm; llm=get_llm(); print(llm.invoke('今天{sym}期货有没有重大突发新闻或政策？只回答有或无，不要展开').content)"],
        capture_output=True, text=True, timeout=15,
        cwd="/home/a/prophet_futures/prophet_futures"
    )
    if "有" in result.stdout:
        print(f"\n⚠️ 检测到可能影响{sym}的新闻，建议人工确认")
except:
    pass
