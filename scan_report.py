#!/usr/bin/env python3
"""5分钟扫描报告 — 系统驱动，直接飞书API"""
import json, sys, os, requests, numpy as np, pandas as pd, joblib, xgboost as xgb
import akshare as ak
from datetime import datetime

def _load_env():
    env_file = os.path.expanduser("~/.hermes/.env")
    if os.path.exists(env_file):
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    k, v = line.split('=', 1)
                    os.environ.setdefault(k.strip(), v.strip())
_load_env()
APP_ID = os.getenv('FEISHU_APP_ID', '')
APP_SECRET = os.getenv('FEISHU_APP_SECRET', '')
CHAT_ID = 'oc_e9bf3cb98e83f50ad4e71dff71f9dce8'

# Copy of paper_trader's build_features
def build_features(df, idx, window=60):
    if idx < window + 5: return None
    w = df.iloc[idx-window:idx+1]
    c = w['收盘价'].values; o = w['开盘价'].values
    h = w['最高价'].values; l = w['最低价'].values
    v = w['成交量'].values; oi = w['持仓量'].values
    f = []
    if idx >= 1: f.append((o[-1]-c[-2])/c[-2]); f.append(abs(f[-1]))
    else: f.extend([0, 0])
    for lag in [1,3,5,10,20]: f.append((c[-1]-c[-lag-1])/c[-lag-1] if len(c) > lag else 0)
    for p in [5,10,20,60]: ma = np.mean(c[-min(p,len(c)):]); f.append((c[-1]-ma)/ma)
    f.append(np.std(c[-20:])/np.mean(c[-20:])); f.append((h[-1]-l[-1])/c[-1])
    vma = np.mean(v[-20:]) if np.mean(v[-20:]) > 0 else 1; f.append(v[-1]/vma)
    f.append(oi[-1]/np.mean(oi[-20:]) if len(oi) >= 20 and np.mean(oi[-20:]) > 0 else 1)
    ema12 = c[-1]; ema26 = c[-1]
    for j in range(len(c)-2, -1, -1): ema12 = (2/13)*c[j] + (11/13)*ema12; ema26 = (2/27)*c[j] + (25/27)*ema26
    f.append((ema12-ema26)/c[-1])
    dd_ = np.diff(c[-15:]); g = dd_[dd_>0].sum() if len(dd_[dd_>0]) > 0 else 0
    lo = abs(dd_[dd_<0].sum()) if len(dd_[dd_<0]) > 0 else 1e-10
    f.append(100 - 100/(1+g/lo) if lo > 0 else 50)
    bb = np.std(c[-20:]); ma20 = np.mean(c[-20:]); f.append((c[-1]-ma20)/(2*bb+1e-10))
    f.append(c[-1]/1000.0)
    return np.array(f, dtype=np.float32)

def send_card(title, elements):
    token = requests.post('https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal',
        json={'app_id': APP_ID, 'app_secret': APP_SECRET}, timeout=10).json()['tenant_access_token']
    card = {
        'config': {'wide_screen_mode': True},
        'header': {'title': {'tag': 'plain_text', 'content': title}, 'template': 'blue'},
        'elements': elements,
    }
    r = requests.post(
        f'https://open.feishu.cn/open-apis/im/v1/messages?receive_id_type=chat_id',
        headers={'Authorization': f'Bearer {token}'},
        json={'receive_id': CHAT_ID, 'msg_type': 'interactive', 'content': json.dumps(card)},
        timeout=10
    )
    return r.json().get('code') == 0

today = datetime.now().strftime('%Y-%m-%d')
now = datetime.now().strftime('%H:%M')

# Load state
try:
    s = json.load(open('paper_state.json'))
except:
    s = {'cash': 300000, 'positions': {}}

V26 = {
    'lh2609': {'code': 'LH0', 'fut': 'LH2609', 'name': 'LH生猪', 'mult': 16,
               'hard_atr': 0.8, 'model_high': 0.65, 'trail_atr': 2},
    'jm2609': {'code': 'JM0', 'fut': 'JM2609', 'name': 'JM焦煤', 'mult': 60,
               'hard_atr': 1.8, 'model_high': 0.70, 'trail_atr': 3},
}

elements = []
total_pnl = 0

for sym_key, cfg in V26.items():
    try:
        df = ak.futures_main_sina(cfg['code'], start_date='2025-01-01', end_date=today)
        df = df.sort_values('日期').reset_index(drop=True)
        model = joblib.load(f'models/{sym_key}_xgb.pkl')
        feats = build_features(df, len(df)-1, 60)
        if feats is None: continue
        X = feats.reshape(1,-1).astype(np.float32)
        proba = model.predict_proba(X)[0]
        bear_pct = round(proba[0]*100)
        bull_pct = round(proba[1]*100)
        
        # Minute data
        mf = ak.futures_zh_minute_sina(symbol=cfg['fut'], period='5')
        mf['dt'] = pd.to_datetime(mf['datetime'])
        td = mf[mf['dt'].dt.strftime('%Y-%m-%d') == today]
        if len(td) == 0:
            elements.append({'tag': 'div', 'text': {'tag': 'lark_md', 'content': f'⚪ {cfg["name"]}: 无今日数据'}})
            continue
        cur = float(td.iloc[-1]['close'])
        opn = float(td.iloc[0]['open'])
        chg = round((cur - opn) / opn * 100, 1)
        
        # RSI
        c = [float(x) for x in df['收盘价']]
        dd_ = np.diff(np.array(c[-15:]))
        g = dd_[dd_>0].sum() if any(dd_>0) else 0
        lo = abs(dd_[dd_<0].sum()) if any(dd_<0) else 1e-10
        rsi = round(100 - 100/(1+g/lo) if lo > 0 else 50, 0)
        
        # Position
        pos = s['positions'].get(sym_key, {})
        has_pos = bool(pos)
        
        # Decision
        bearish = bear_pct > bull_pct
        model_conf = max(bear_pct, bull_pct)
        
        if has_pos:
            pos_dir = pos['dir']
            entry = pos['entry']
            pstop = pos.get('stop', 0)
            ptp = pos.get('take_profit', 0)
            vol = pos.get('vol', 0)
            
            if pos_dir == 'LONG':
                pnl = round((cur - entry) * cfg['mult'] * vol, 0)
                aligned = not bearish
                stop_dist = round(cur - pstop, 0)
            else:
                pnl = round((entry - cur) * cfg['mult'] * vol, 0)
                aligned = bearish
                stop_dist = round(pstop - cur, 0)
            
            total_pnl += pnl
            
            # v26 check
            model_exit_thresh = int(cfg['model_high'] * 100)
            if pos_dir == 'LONG':
                model_exit = bear_pct > model_exit_thresh
                gap = round(model_exit_thresh - bear_pct, 0)
            else:
                model_exit = bull_pct > model_exit_thresh
                gap = round(model_exit_thresh - bull_pct, 0)
            
            if model_exit:
                action = f'🔴 反转触发→平仓'
            elif aligned:
                action = f'✅ 持有'
            else:
                action = f'⚠️ 背离(距反转差{gap}%)'
            
            pnl_str = f'+{pnl/10000:.1f}万' if pnl >= 0 else f'{pnl/1000:.0f}千'
            
            line = (
                f'**{cfg["name"]}** {action} | {pnl_str}\n'
                f'{int(cur)}({chg:+.1f}%) | 模型{"看空" if bearish else "看多"}{model_conf}% | RSI{rsi}\n'
                f'止损{int(pstop)}(距{int(stop_dist)}pt) | 止盈{int(ptp)}'
            )
        else:
            # No position: show both directions
            atr_vals = [abs(float(h)-float(l)) for h,l in zip(df['最高价'].iloc[-20:], df['最低价'].iloc[-20:])]
            atr = np.mean(atr_vals)
            stop_dist = atr * 1.5 if 'lh' in sym_key else atr * 2.0
            rr = 4.0 if 'lh' in sym_key else 3.5
            base_lots = 6 if 'lh' in sym_key else 1
            margin = round(cur * cfg['mult'] * base_lots * 0.12, 0)
            
            long_stop = round(cur - stop_dist, 1)
            long_tp = round(cur + stop_dist * rr, 1)
            short_stop = round(cur + stop_dist, 1)
            short_tp = round(cur - stop_dist * rr, 1)
            
            sig = '看空' if bearish else '看多'
            arrow = '🔴' if bearish else '🟢'
            
            line = (
                f'**{cfg["name"]}** {arrow} {sig}{model_conf}% | {int(cur)}({chg:+.1f}%) RSI{rsi}\n'
                f'做多: {int(cur)}→{int(long_stop)}→{int(long_tp)} | 做空: {int(cur)}→{int(short_stop)}→{int(short_tp)}\n'
                f'{base_lots}手 ¥{margin/10000:.1f}万 RR=1:{rr}'
            )
        
        elements.append({'tag': 'div', 'text': {'tag': 'lark_md', 'content': line}})
        elements.append({'tag': 'hr'})
    except Exception as e:
        elements.append({'tag': 'div', 'text': {'tag': 'lark_md', 'content': f'⚪ {cfg["name"]}: 数据异常 {str(e)[:30]}'}})

# Remove trailing hr
if elements and elements[-1].get('tag') == 'hr':
    elements.pop()

if not elements:
    elements.append({'tag': 'div', 'text': {'tag': 'lark_md', 'content': '⚪ 当前无信号 | 数据获取异常'}})

p_str = f'+{total_pnl/10000:.1f}万' if total_pnl >= 0 else f'{total_pnl/1000:.0f}千'
elements.insert(0, {'tag': 'div', 'text': {'tag': 'lark_md', 'content': f'💰 ¥{s["cash"]:,.0f} | 浮盈 {p_str}'}})
elements.insert(1, {'tag': 'hr'})

elements.append({'tag': 'note', 'elements': [{'tag': 'plain_text', 'content': f'{today} {now} · 系统驱动 · v26'}]})

ok = send_card(f'Prophet 扫描 {now}', elements)
print(f'{"✅" if ok else "❌"} {now}')
