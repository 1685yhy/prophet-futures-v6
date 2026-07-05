#!/usr/bin/env python3
"""Prophet Futures — 早间自检 08:45"""
import sys, os, json, subprocess
from datetime import datetime

CHECKS = []

def check(name, fn):
    try:
        ok, msg = fn()
        icon = "✅" if ok else "❌"
        CHECKS.append(f"{icon} {name}: {msg}")
        return ok
    except Exception as e:
        CHECKS.append(f"❌ {name}: {str(e)[:60]}")
        return False

def check_process(name):
    # Use bracket trick to avoid pgrep matching its own command line
    # pgrep -f '[p]aper_trader' won't match the grep itself
    pattern = f"[{name[0]}]{name[1:]}"
    r = subprocess.run(f"pgrep -f '{pattern}' | wc -l", shell=True, capture_output=True, text=True)
    ok = int(r.stdout.strip() or 0) > 0
    return ok, "运行中" if ok else "未启动"

def check_data():
    import akshare as ak, pandas as pd
    from datetime import datetime, timedelta
    today = datetime.now().strftime('%Y-%m-%d')
    yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    for sym in ['LH2609', 'JM2609']:
        df = ak.futures_zh_minute_sina(symbol=sym, period='1')
        df['dt'] = pd.to_datetime(df['datetime'])
        td = df[df['dt'].dt.strftime('%Y-%m-%d') == today]
        # Pre-market: if no data today, check yesterday (last trading day)
        if len(td) < 5:
            yd = df[df['dt'].dt.strftime('%Y-%m-%d') == yesterday]
            if len(yd) >= 5:
                continue  # yesterday had data, assume market not open yet
            return False, f"{sym}今日数据不足({len(td)}条),昨日({len(yd)}条)"
    return True, "数据正常"

def check_state():
    sf = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'paper_state.json')
    if not os.path.exists(sf):
        return False, "状态文件缺失"
    with open(sf) as f:
        s = json.load(f)
    pos = s.get('positions', {})
    if not pos:
        return True, "无持仓"
    lines = []
    for k, v in pos.items():
        lines.append(f"{k} {v['dir']} {v['vol']}手@{v['entry']}")
    return True, ", ".join(lines)

def check_models():
    md = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
    for sym in ['lh2609', 'jm2609']:
        if not os.path.exists(os.path.join(md, f'{sym}_xgb.pkl')):
            return False, f"{sym}模型缺失"
    return True, "模型完整"

def check_gateway():
    # Check if Hermes gateway is running (for Feishu)
    pattern = "[h]ermes gateway"
    r = subprocess.run(f"pgrep -f '{pattern}' | wc -l", shell=True, capture_output=True, text=True)
    if int(r.stdout.strip() or 0) == 0:
        return False, "飞书网关未启动"
    return True, "网关运行中"

if __name__ == '__main__':
    print(f"Prophet Futures 早间自检 — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 40)
    
    check("数据源", check_data)
    check("模型文件", check_models)
    check("状态文件", check_state)
    check("纸盘进程", lambda: check_process('paper_trader'))
    check("SimNow进程", lambda: check_process('simnow_live'))
    
    # Hermes gateway is optional — report_v4 uses direct Feishu API
    
    print("=" * 40)
    all_ok = all('✅' in c for c in CHECKS)
    for c in CHECKS:
        print(c)
    print(f"\n{'✅ 全部正常' if all_ok else '⚠️ 有问题需处理'}")
    sys.exit(0 if all_ok else 1)
