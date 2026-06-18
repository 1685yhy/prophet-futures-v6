#!/usr/bin/env python3
"""AI最终确认 — 开盘时联网检查是否有影响信号的重大事件"""

import sys, json, os
from datetime import datetime

def ai_confirm(symbol, direction):
    """Ask DeepSeek if there's any reason to override the signal."""
    from openai import OpenAI
    
    client = OpenAI(
        api_key=os.environ.get("DEEPSEEK_API_KEY",""),
        base_url="https://api.deepseek.com"
    )
    
    prompt = f"""你是一个期货交易风控官。今天系统发出了{symbol}期货的{direction}信号。

请联网搜索今天关于{symbol}期货的最新消息（政策、供需、突发事件等）。
只回答"确认"或"取消"，并附一句简短理由（不超过20字）。"""

    try:
        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role":"user","content":prompt}],
            max_tokens=50, temperature=0
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"确认（AI不可用: {str(e)[:30]}）"

if __name__ == "__main__":
    # Read saved signal
    try:
        with open("/tmp/prophet_signal.txt") as f:
            sym, direction, entry = f.read().strip().split(",")
    except:
        print("无昨日信号")
        sys.exit(0)
    
    print(f"昨日信号: {sym} {direction} @ {entry}")
    
    # AI confirmation
    DEEPSEEK_API_KEY = os.popen("grep DEEPSEEK_API_KEY /home/a/.hermes/.env | cut -d= -f2").read().strip()
    os.environ["DEEPSEEK_API_KEY"] = DEEPSEEK_API_KEY
    
    result = ai_confirm(sym, direction)
    print(f"\nAI判断: {result}")
    
    if "取消" in result:
        print(f"⛔ 信号已取消: {result}")
    else:
        print(f"✅ 信号确认: 按计划执行 {direction}")
