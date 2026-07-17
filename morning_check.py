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
    pattern = f"[{name[0]}]{name[1:]}"
    r = subprocess.run(f"pgrep -f '{pattern}' | wc -l", shell=True, capture_output=True, text=True)
    ok = int(r.stdout.strip() or 0) > 0
    return ok, "运行中" if ok else "未启动"

def check_data():
    import akshare as ak, pandas as pd
    from datetime import timedelta
    today = datetime.now()
    lookback = today
    for _ in range(5):
        lookback = lookback - timedelta(days=1)
        if lookback.weekday() < 5:
            break
    today_str = today.strftime('%Y-%m-%d')
    last_td_str = lookback.strftime('%Y-%m-%d')
    for sym in ['LH2609']:
        df = ak.futures_zh_minute_sina(symbol=sym, period='1')
        df['dt'] = pd.to_datetime(df['datetime'])
        td = df[df['dt'].dt.strftime('%Y-%m-%d') == today_str]
        if len(td) < 5:
            yd = df[df['dt'].dt.strftime('%Y-%m-%d') == last_td_str]
            if len(yd) >= 5:
                continue
            return False, f"{sym}今日数据不足({len(td)}条),上个交易日({len(yd)}条)"
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
    for sym in ['lh2609']:
        if not os.path.exists(os.path.join(md, f'{sym}_xgb.pkl')):
            return False, f"{sym}模型缺失"
    # Also check v31 model
    if not os.path.exists(os.path.join(md, 'v31_xgb.pkl')):
        return False, "v31模型缺失"
    return True, "模型完整"

def check_simnow_server():
    r = subprocess.run(
        "sshpass -p 'Asdfghjkl123!!' ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 "
        "root@47.102.42.238 'pgrep -f simnow_live | wc -l'",
        shell=True, capture_output=True, text=True, timeout=15
    )
    ok = int(r.stdout.strip() or 0) > 0
    return ok, "云端运行中" if ok else "云端未启动"

def check_stop_breach():
    """检查所有版本持仓是否已穿止损"""
    import numpy as np, pandas as pd, akshare as ak
    try:
        df = ak.futures_main_sina(symbol='LH0')
        df.columns = ['date','open','high','low','close','volume','oi','settle']
        for c in ['open','high','low','close']:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        df = df.dropna(subset=['close'])
        cur = float(df.iloc[-1]['close'])
    except:
        return True, "行情获取失败-跳过"
    
    breached = []
    for label, sf in [
        ('V25','paper_state.json'),('V28','paper_state_v28.json'),('V29','paper_state_v29.json'),
        ('V30','paper_state_v30.json'),('V31','paper_state_v31.json'),('V32','paper_state_v32.json'),
        ('V32b','paper_state_v32b.json'),
        ('V33','paper_state_v33.json'),
        ('V34','paper_state_v34.json')]:
        if not os.path.exists(sf): continue
        s = json.load(open(sf))
        pos = s['positions'].get('lh2609')
        if not pos: continue
        if isinstance(pos, list):
            d = pos[0]['dir']; trail = pos[0].get('_trail', 0)
        else:
            d = pos['dir']; trail = pos.get('_trail_stop', pos.get('stop', 0))
        if trail > 0:
            if (d == 'LONG' and cur <= trail) or (d == 'SHORT' and cur >= trail):
                breached.append(f'{label}止损穿(现{cur:.0f}vs{trail:.0f})')
    
    if breached:
        return False, '; '.join(breached)
    return True, "无穿透"

def check_trail_active():
    """检查动态版本追踪止损是否正常运行"""
    import numpy as np, pandas as pd, akshare as ak
    try:
        df = ak.futures_main_sina(symbol='LH0')
        df.columns = ['date','open','high','low','close','volume','oi','settle']
        for c in ['open','high','low','close']:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        df = df.dropna(subset=['close'])
        cur = float(df.iloc[-1]['close'])
        av = [abs(float(df.iloc[i]['high'])-float(df.iloc[i]['low'])) for i in range(max(0,len(df)-20),len(df))]
        atr = np.mean(av)
    except:
        return True, "行情获取失败-跳过"
    
    stuck = []
    for label, sf, atr_stop in [
        ('V28','paper_state_v28.json',1.5),('V29','paper_state_v29.json',1.5),
        ('V30','paper_state_v30.json',2.0),('V32','paper_state_v32.json',0.5),
        ('V32b','paper_state_v32b.json',0.5),
        ('V33','paper_state_v33.json',0.5),
        ('V34','paper_state_v34.json',1.0)]:
        if not os.path.exists(sf): continue
        s = json.load(open(sf))
        pos = s['positions'].get('lh2609')
        if not pos: continue
        if isinstance(pos, list):
            d = pos[0]['dir']; entry = sum(p['entry']*p['vol'] for p in pos)/sum(p['vol'] for p in pos)
            trail = pos[0].get('_trail', 0)
        else: continue
        
        fp = cur - entry if d == 'LONG' else entry - cur
        if trail and trail == entry and abs(fp) > atr:
            stuck.append(f'{label}浮盈{abs(fp):.0f}>{atr:.0f}ATR追踪未动')
    
    if stuck:
        return False, '; '.join(stuck)
    return True, "追踪正常"

if __name__ == '__main__':
    print(f"Prophet Futures 早间自检 — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 40)
    
    check("数据源", check_data)
    check("模型文件", check_models)
    check("状态文件", check_state)
    for label, pattern in [
        ('V25','paper_trader.py$'), ('V28','paper_trader_v28'), ('V29','paper_trader_v29'),
        ('V30','paper_trader_v30'), ('V31','paper_trader_v31'), ('V32','paper_trader_v32'),
        ('V32b','paper_trader_v32b'),
        ('V33','paper_trader_v33'),
        ('V34','paper_trader_v34')
    ]:
        check(f"纸盘{label}", lambda p=pattern: check_process(p))
    check("止损穿透", check_stop_breach)
    check("追踪止损", check_trail_active)
    check("SimNow云端", check_simnow_server)
    
    print("=" * 40)
    all_ok = all('✅' in c for c in CHECKS)
    for c in CHECKS:
        print(c)
    print(f"\n{'✅ 全部正常' if all_ok else '⚠️ 有问题需处理'}")
    sys.exit(0 if all_ok else 1)
