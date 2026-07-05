#!/usr/bin/env python3
"""
Prophet Futures — 智能报告/扫描卡片 v2
Feishu卡片 · column_set表格 · 一眼看懂
"""
import json, requests, numpy as np, pandas as pd, pickle, os, sys
from datetime import datetime, timedelta
import akshare as ak

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
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'paper_state.json')

SYMBOLS = {
    'lh2609': {
        'code': 'LH0', 'fut': 'LH2609', 'name': 'LH', 'cn': '生猪',
        'mp': 16, 'cost': 0.0006, 'mg': 0.15,
        'max_pos': 6, 'max_total': 12,
        'atr_stop': 1.5, 'rr': 4.0,
        'add_conf': 0.65, 'add_atr': 2.0, 'reduce_conf': 0.55, 'reverse_conf': 0.35,
        'trail_atr': 2.0, 'be_atr': 1.0, 'min_hold': 3,
    },
    'jm2609': {
        'code': 'JM0', 'fut': 'JM2609', 'name': 'JM', 'cn': '焦煤',
        'mp': 60, 'cost': 0.0011, 'mg': 0.15,
        'max_pos': 4, 'max_total': 8,
        'atr_stop': 2.0, 'rr': 3.5,
        'add_conf': 0.65, 'add_atr': 2.5, 'reduce_conf': 0.55, 'reverse_conf': 0.30,
        'trail_atr': 3.0, 'be_atr': 2.0, 'min_hold': 5,
    },
}

def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f: return json.load(f)
    return {'cash': 300000, 'positions': {}, 'trades': [], 'equity_history': []}

def bf(df, idx, w=60):
    if idx < w+5: return None
    s = df.iloc[idx-w:idx+1]
    c = s['close'].values.astype(float); o = s['open'].values.astype(float)
    h = s['high'].values.astype(float); l = s['low'].values.astype(float)
    v = s['volume'].values.astype(float); oi = s['oi'].values.astype(float)
    oc = float((o[-1]-c[-2])/c[-2]) if idx>=1 else 0.0
    f = [oc, abs(oc)]
    for lag in [1,3,5,10,20]: f.append(float((c[-1]-c[-lag-1])/c[-lag-1] if len(c)>lag else 0))
    for p in [5,10,20,60]: ma=np.mean(c[-min(p,len(c)):]);f.append(float((c[-1]-ma)/ma))
    f.append(float(np.std(c[-20:])/np.mean(c[-20:])))
    f.append(float((h[-1]-l[-1])/c[-1]))
    vm=np.mean(v[-20:])if np.mean(v[-20:])>0 else 1;f.append(float(v[-1]/vm))
    f.append(float(oi[-1]/np.mean(oi[-20:]))if len(oi)>=20 and np.mean(oi[-20:])>0 else 1)
    e12=c[-1];e26=c[-1]
    for j in range(len(c)-2,-1,-1):e12=(2/13)*c[j]+(11/13)*e12;e26=(2/27)*c[j]+(25/27)*e26
    f.append(float((e12-e26)/c[-1]))
    dd=np.diff(c[-15:]);g=float(dd[dd>0].sum())if len(dd[dd>0])>0 else 0
    lo=float(abs(dd[dd<0].sum()))if len(dd[dd<0])>0 else 1e-10
    f.append(float(100-100/(1+g/lo)if lo>0 else 50))
    bb=np.std(c[-20:]);m20=np.mean(c[-20:]);f.append(float((c[-1]-m20)/(2*bb+1e-10)))
    f.append(float(c[-1]/1000.0))
    return np.array(f,dtype=np.float32)

def fetch_daily(code):
    try:
        df = ak.futures_main_sina(symbol=code)
        df.columns = ['date','open','high','low','close','volume','oi','settle']
        for c in ['open','high','low','close','volume','oi']:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        return df.dropna(subset=['close']).reset_index(drop=True)
    except: return None

def fetch_minute(fut, today):
    try:
        df = ak.futures_zh_minute_sina(symbol=fut, period='5')
        df['dt'] = pd.to_datetime(df['datetime'])
        return df[df['dt'].dt.strftime('%Y-%m-%d') == today]
    except: return None

def get_token():
    r = requests.post('https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal',
        json={'app_id': APP_ID, 'app_secret': APP_SECRET}, timeout=10)
    return r.json()['tenant_access_token']

def send_card(title, elements, color='blue', pin=False):
    token = get_token()
    card = {
        'config': {'wide_screen_mode': True},
        'header': {'title': {'tag': 'plain_text', 'content': title}, 'template': color},
        'elements': elements,
    }
    r = requests.post(
        'https://open.feishu.cn/open-apis/im/v1/messages?receive_id_type=chat_id',
        headers={'Authorization': 'Bearer '+token, 'Content-Type': 'application/json'},
        json={'receive_id': CHAT_ID, 'msg_type': 'interactive', 'content': json.dumps(card, ensure_ascii=False)},
        timeout=10)
    result = r.json()
    if pin and result.get('code') == 0:
        mid = result.get('data', {}).get('message_id', '')
        if mid:
            requests.post('https://open.feishu.cn/open-apis/im/v1/messages/'+mid+'/pin',
                headers={'Authorization': 'Bearer '+token}, timeout=5)
    ok = result.get('code') == 0
    print('  send_card: %s code=%s' % ('OK' if ok else 'FAIL', result.get('code')))
    return ok

# ==== 卡片组件 ====
def md(text):
    return {'tag': 'div', 'text': {'tag': 'lark_md', 'content': text}}

def hr():
    return {'tag': 'hr'}

def note(text):
    return {'tag': 'note', 'elements': [{'tag': 'plain_text', 'content': text}]}

def row(*cells):
    """row(('标签','值','颜色?'), ...)"""
    cols = []
    for cell in cells:
        label, value = cell[0], str(cell[1])
        color = cell[2] if len(cell)>2 else ''
        v = "<font color='"+color+"'>"+value+"</font>" if color else value
        cols.append({
            'tag': 'column', 'width': 'weighted', 'weight': 1,
            'elements': [{'tag': 'div', 'text': {'tag': 'lark_md', 'content': '**'+label+'**\n'+v}}]
        })
    return {'tag': 'column_set', 'flex_mode': 'bisect', 'columns': cols}

# ==== V28 分析 ====
def v28_analyze(sym_key, cfg, df, pos, price, atr, prob):
    if not pos:
        stop_dist = atr * cfg['atr_stop']; entry = price
        long_stop = entry - stop_dist; long_tp = entry + stop_dist * cfg['rr']
        short_stop = entry + stop_dist; short_tp = entry - stop_dist * cfg['rr']
        atr_pct = atr/price
        if atr_pct < 0.01: lev = 3.0
        elif atr_pct < 0.02: lev = 2.0
        elif atr_pct < 0.03: lev = 1.5
        else: lev = 0.5
        lots = max(1, int(lev * (cfg['max_pos']//2))) if lev > 0 else 0
        mg = lots * price * cfg['mp'] * cfg['mg']
        signal = '看多' if prob>0.5 else '看空'; conf = prob if prob>0.5 else 1-prob
        return 'idle', 'ok', {
            'signal': signal, 'conf': conf, 'price': price, 'atr': atr,
            'long_entry': entry, 'long_stop': long_stop, 'long_tp': long_tp,
            'short_entry': entry, 'short_stop': short_stop, 'short_tp': short_tp,
            'lots': lots, 'margin': mg, 'rr': cfg['rr'],
        }

    d = pos['dir']; entry = pos['entry']; vol = pos['vol']
    cur_dir = 'LONG' if prob>0.5 else 'SHORT'
    conf = prob if prob>0.5 else 1-prob
    pnl_pts = price - entry if d=='LONG' else entry - price
    pnl_atr = pnl_pts / atr if atr>0 else 0
    margin = vol * entry * cfg['mp'] * cfg['mg']
    pnl_amt = pnl_pts * vol * cfg['mp'] * cfg['mg']
    pnl_pct = pnl_pts / entry
    
    stop_dist = atr * cfg['atr_stop']
    hard_stop = price - stop_dist if d=='LONG' else price + stop_dist
    trail = entry
    if pnl_atr > cfg['be_atr']: trail = entry
    if d=='LONG':
        if pnl_atr > cfg['trail_atr']: trail = max(trail, price - atr*(cfg['atr_stop']-0.3))
        eff_stop = max(hard_stop, trail); sp = price - eff_stop
    else:
        if pnl_atr > cfg['trail_atr']: trail = min(trail, price + atr*(cfg['atr_stop']-0.3))
        eff_stop = min(hard_stop, trail); sp = eff_stop - price
    
    should_reverse = (d=='LONG' and prob < cfg['reverse_conf']) or (d=='SHORT' and prob > 1-cfg['reverse_conf'])
    should_reduce = (d==cur_dir and conf < cfg['reduce_conf'])
    can_add = (d==cur_dir and conf > cfg['add_conf'] and pnl_atr > cfg['add_atr'])
    
    res = {'entry': entry, 'vol': vol, 'dir': d, 'pnl_pts': pnl_pts, 'pnl_atr': pnl_atr,
           'pnl_amt': pnl_amt, 'pnl_pct': pnl_pct, 'margin': margin,
           'eff_stop': eff_stop, 'stop_dist': sp, 'atr': atr,
           'signal': '看多' if prob>0.5 else '看空', 'conf': conf, 'prob': prob,
           'price': price, 'stop_atr': sp/atr if atr>0 else 0,
           'dir_cn': '做多' if d=='LONG' else '做空'}
    
    if should_reverse:
        res['action'] = '反手'; res['level'] = 'alert'
        rev_dir = '空' if d=='LONG' else '多'
        res['detail'] = '模型转%s(prob=%.2f) 平%d手' % (rev_dir, prob, vol)
    elif can_add:
        add = min(cfg['max_pos'], cfg['max_total']-vol)
        res['action'] = '加仓'; res['level'] = 'ok'
        res['detail'] = '同向高信(%d%%)+浮盈%.1fATR 加%d手' % (int(conf*100), pnl_atr, add)
    elif should_reduce and vol>1:
        cut = vol//2
        res['action'] = '减仓'; res['level'] = 'warn'
        res['detail'] = '信心降(%d%%) 减%d手留%d手' % (int(conf*100), cut, vol-cut)
    elif d!=cur_dir:
        gap = abs(cfg['reverse_conf'] - (prob if d=='LONG' else 1-prob))
        res['action'] = '背离'; res['level'] = 'warn'
        res['detail'] = '持仓%s vs 模型%s(%d%%) 距反手差%d%%' % (d, cur_dir, int(conf*100), int(gap*100))
    else:
        res['action'] = '持有'; res['level'] = 'ok'
        need_atr = cfg['add_atr'] - pnl_atr
        res['detail'] = '同向(%d%%)' % int(conf*100)
        if need_atr > 0:
            res['detail'] += ' 距加仓%.0f点(%.1fATR)' % (need_atr*atr, need_atr)
        if sp/atr < 0.5:
            res['level'] = 'warn'
            res['detail'] += ' 止损近(%.0f点)' % sp
    
    return 'pos', res['level'], res

# ==== 构建持仓卡 ====
def build_position_card(res, cfg, atr):
    """返回卡片元素列表"""
    lvl_icon = {'ok':'✅','warn':'⚠️','alert':'🔴'}[res['level']]
    sign = '+' if res['pnl_amt']>=0 else ''
    pnl_str = '%s%.1f万 (%.1f%%)' % (sign, res['pnl_amt']/10000, res['pnl_pct']*100)
    pnl_color = 'green' if res['pnl_amt']>=0 else 'red'
    
    elements = []
    # 标题
    title = '%s **%s** %s | %s%d手 | 浮<font color="%s">%s</font>' % (
        lvl_icon, cfg['cn'], res['action'], res['dir_cn'], res['vol'], pnl_color, pnl_str)
    elements.append(md(title))
    # 明细
    elements.append(row(
        ('入场', '%.0f' % res['entry']),
        ('现价', '%.0f' % res['price']),
        ('ATR', '%.0f' % atr),
    ))
    elements.append(row(
        ('止损', '%.0f' % res['eff_stop']),
        ('距离', '%.0f点(%.1fATR)' % (res['stop_dist'], res['stop_atr'])),
        ('模型', '%s%d%%' % (res['signal'], int(res['conf']*100))),
    ))
    elements.append(md(res['detail']))
    return elements

def build_idle_card(res, cfg):
    """空仓建议卡片"""
    elements = []
    title = '⚪ **%s** 空仓 | 现价%.0f | %s%d%%' % (
        cfg['cn'], res['price'], res['signal'], int(res['conf']*100))
    elements.append(md(title))
    elements.append(row(
        ('做多', '%.0f→%.0f→%.0f' % (res['long_entry'], res['long_stop'], res['long_tp'])),
        ('做空', '%.0f→%.0f→%.0f' % (res['short_entry'], res['short_stop'], res['short_tp'])),
    ))
    elements.append(row(
        ('手数', '%d手' % res['lots']),
        ('保证金', '¥%.1f万' % (res['margin']/10000)),
        ('RR', '1:%d' % res['rr']),
    ))
    return elements

# ==== 报告 ====
def morning_report():
    state = load_state()
    now = datetime.now(); today = now.strftime('%Y-%m-%d')
    wday = ['周一','周二','周三','周四','周五','周六','周日'][now.weekday()]
    elements = []; total_pnl = 0; has_alert = False
    
    for sk, cfg in SYMBOLS.items():
        df = fetch_daily(cfg['code'])
        if df is None or len(df)<20: continue
        price = float(df.iloc[-1]['close'])
        atr_vals = [abs(float(df.iloc[i]['high'])-float(df.iloc[i]['low'])) for i in range(max(0,len(df)-20), len(df))]
        atr = np.mean(atr_vals)
        feats = bf(df, len(df)-1, 60)
        if feats is None: continue
        mp = os.path.join(MODEL_DIR, sk+'_xgb.pkl')
        if not os.path.exists(mp): continue
        model = pickle.load(open(mp, 'rb'))
        prob = float(model.predict_proba(feats.reshape(1,-1))[0][1])
        pos = state['positions'].get(sk)
        status, level, res = v28_analyze(sk, cfg, df, pos, price, atr, prob)
        
        if status == 'pos':
            total_pnl += res['pnl_amt']
            if level != 'ok': has_alert = True
            elements.extend(build_position_card(res, cfg, atr))
        else:
            elements.extend(build_idle_card(res, cfg))
        elements.append(hr())
    
    if elements and elements[-1].get('tag')=='hr': elements.pop()
    
    equity = state['cash']
    for k,p in state['positions'].items():
        equity += p['vol'] * p['entry'] * SYMBOLS[k]['mp'] * 0.15
    pnl_str = '%+.1f万' % (total_pnl/10000)
    pc = 'green' if total_pnl>=0 else 'red'
    elements.insert(0, row(
        ('总权益', '¥%s' % format(int(equity), ',')),
        ('持仓浮盈', pnl_str, pc),
        ('可用', '¥%s' % format(int(state['cash']), ',')),
    ))
    elements.append(note('%s %s 08:50 | V28' % (today, wday)))
    send_card('Prophet 早报 | %s' % wday, elements, 'red' if has_alert else 'blue')

def scan_dual():
    """双版本扫描: V25 + V28 同框对比"""
    state_v25 = load_state()
    state28_path = STATE_FILE.replace('.json','_v28.json')
    state_v28 = json.load(open(state28_path)) if os.path.exists(state28_path) else {'positions':{},'cash':300000}
    
    now = datetime.now(); today = now.strftime('%Y-%m-%d')
    elements = []; total_pnl = 0; has_action = False
    
    elements.insert(0, md('**V25(原)** vs **V28(新)** 同框对比'))
    elements.append(hr())
    
    for sk, cfg in SYMBOLS.items():
        df = fetch_daily(cfg['code'])
        if df is None or len(df)<20: continue
        td = fetch_minute(cfg['fut'], today)
        price = float(td.iloc[-1]['close']) if td is not None and len(td)>0 else float(df.iloc[-1]['close'])
        atr_vals = [abs(float(df.iloc[i]['high'])-float(df.iloc[i]['low'])) for i in range(max(0,len(df)-20), len(df))]
        atr = np.mean(atr_vals)
        feats = bf(df, len(df)-1, 60)
        if feats is None: continue
        mp = os.path.join(MODEL_DIR, sk+'_xgb.pkl')
        if not os.path.exists(mp): continue
        model = pickle.load(open(mp, 'rb'))
        prob = float(model.predict_proba(feats.reshape(1,-1))[0][1])
        
        signal_icon = '🟢' if prob>0.5 else '🔴'
        signal_text = '看多' if prob>0.5 else '看空'
        conf = prob if prob>0.5 else 1-prob
        
        elements.append(md('%s **%s** %s%d%% | 现价%.0f ATR%.0f' % (
            signal_icon, cfg['cn'], signal_text, int(conf*100), price, atr)))
        
        # V25
        pos25 = state_v25['positions'].get(sk)
        if pos25:
            pnl_pts = price-pos25['entry'] if pos25['dir']=='LONG' else pos25['entry']-price
            pnl_amt = pnl_pts*pos25['vol']*cfg['mp']*cfg['mg']
            sign25 = '+' if pnl_amt>=0 else ''; pc25 = 'green' if pnl_amt>=0 else 'red'
            status25, level25, res25 = v28_analyze(sk, cfg, df, pos25, price, atr, prob)
            lvl_icon25 = {'ok':'✅','warn':'⚠️','alert':'🔴'}[level25]
            if level25 != 'ok': has_action = True
            v25_text = '%s V25: %s%d手 浮<font color="%s">%s%.1f万</font> | %s' % (
                lvl_icon25, ('多' if pos25['dir']=='LONG' else '空'), pos25['vol'],
                pc25, sign25, pnl_amt/10000, res25['action'])
        else:
            v25_text = '⚪ V25: 空仓'
        
        # V28
        pos28 = state_v28['positions'].get(sk, [])
        if pos28:
            total_vol = sum(p['vol'] for p in pos28)
            d28 = pos28[0]['dir']
            avg_entry = np.mean([p['entry'] for p in pos28])
            pnl_amt = (price-avg_entry)*total_vol*cfg['mp']*cfg['mg'] if d28=='LONG' else (avg_entry-price)*total_vol*cfg['mp']*cfg['mg']
            sign28 = '+' if pnl_amt>=0 else ''; pc28 = 'green' if pnl_amt>=0 else 'red'
            v28_text = 'V28: %s%d手(%d仓) @%.0f 浮<font color="%s">%s%.1f万</font>' % (
                ('多' if d28=='LONG' else '空'), total_vol, len(pos28), avg_entry, pc28, sign28, pnl_amt/10000)
        else:
            # V28 empty - show potential entry
            stop_dist = atr * cfg['atr_stop']
            if prob > 0.5:
                v28_text = 'V28: 空仓 | 信号做多 → %.0f止损%.0f' % (price, price-stop_dist)
            else:
                v28_text = 'V28: 空仓 | 信号做空 → %.0f止损%.0f' % (price, price+stop_dist)
        
        elements.append(row(
            ('V25', v25_text),
            ('V28', v28_text),
        ))
        
        # Action details if needed
        if pos25:
            _, lvl25, r25 = v28_analyze(sk, cfg, df, pos25, price, atr, prob)
            if lvl25 != 'ok':
                elements.append(md('⚠️ V25: '+r25['detail']))
        elements.append(hr())
    
    if elements and elements[-1].get('tag')=='hr': elements.pop()
    
    v25_eq = state_v25['cash']
    for k,p in state_v25['positions'].items():
        v25_eq += p['vol']*p['entry']*SYMBOLS[k]['mp']*0.15
    v28_eq = state_v28['cash']
    for k,v in state_v28.get('positions',{}).items():
        for p in v: v28_eq += p['vol']*p['entry']*SYMBOLS[k]['mp']*0.15
    
    elements.insert(1, row(
        ('V25权益', '¥%s' % format(int(v25_eq), ',')),
        ('V28权益', '¥%s' % format(int(v28_eq), ',')),
    ))
    elements.insert(2, hr())
    
    color = 'red' if has_action else 'blue'
    pin = has_action
    elements.append(note('%s %s | 有动作→pin置顶' % (today, now.strftime('%H:%M')) if has_action else '%s %s' % (today, now.strftime('%H:%M'))))
    send_card('扫描 %s [V25+V28]' % now.strftime('%H:%M'), elements, color, pin=pin)

def evening_report():
    state = load_state(); now = datetime.now(); today = now.strftime('%Y-%m-%d')
    elements = []; total_pnl = 0
    
    for sk, cfg in SYMBOLS.items():
        df = fetch_daily(cfg['code'])
        if df is None or len(df)<20: continue
        price = float(df.iloc[-1]['close'])
        prev = float(df.iloc[-2]['close']) if len(df)>1 else price
        atr_vals = [abs(float(df.iloc[i]['high'])-float(df.iloc[i]['low'])) for i in range(max(0,len(df)-20), len(df))]
        atr = np.mean(atr_vals)
        ma5 = np.mean([float(df.iloc[i]['close']) for i in range(max(0,len(df)-5), len(df))])
        ma20 = np.mean([float(df.iloc[i]['close']) for i in range(max(0,len(df)-20), len(df))])
        feats = bf(df, len(df)-1, 60)
        if feats is None: continue
        mp = os.path.join(MODEL_DIR, sk+'_xgb.pkl')
        if not os.path.exists(mp): continue
        model = pickle.load(open(mp, 'rb'))
        prob = float(model.predict_proba(feats.reshape(1,-1))[0][1])
        pos = state['positions'].get(sk)
        status, level, res = v28_analyze(sk, cfg, df, pos, price, atr, prob)
        
        trend = '📈' if price>ma5>ma20 else ('📉' if price<ma5<ma20 else '↔️')
        chg = (price-prev)/prev
        
        if status == 'pos':
            total_pnl += res['pnl_amt']
            sign = '+' if res['pnl_amt']>=0 else ''
            pnl_color = 'green' if res['pnl_amt']>=0 else 'red'
            pnl_str = '%s%.1f万' % (sign, res['pnl_amt']/10000)
            emoji = '🟢' if res['pnl_amt']>=0 else '🔴'
            
            elements.append(md('%s **%s** %s %.0f (%+.1f%%) | %s%d手 | 浮<font color="%s">%s</font>' % (
                emoji, cfg['cn'], trend, price, chg*100, res['dir_cn'], res['vol'], pnl_color, pnl_str)))
            elements.append(row(
                ('止损', '%.0f' % res['eff_stop']),
                ('距离', '%.0f点' % res['stop_dist']),
                ('模型', '%s%d%%' % (res['signal'], int(res['conf']*100))),
            ))
            elements.append(md(res['detail']))
        else:
            elements.append(md('⚪ **%s** %s %.0f (%+.1f%%) | %s%d%%' % (
                cfg['cn'], trend, price, chg*100, res['signal'], int(res['conf']*100))))
        elements.append(hr())
    
    if elements and elements[-1].get('tag')=='hr': elements.pop()
    equity = state['cash']
    for k,p in state['positions'].items():
        equity += p['vol'] * p['entry'] * SYMBOLS[k]['mp'] * 0.15
    pnl_str = '%+.1f万' % (total_pnl/10000)
    pc = 'green' if total_pnl>=0 else 'red'
    elements.insert(0, row(
        ('收盘权益', '¥%s' % format(int(equity), ',')),
        ('持仓浮盈', pnl_str, pc),
    ))
    elements.append(note('%s %s | V28' % (today, now.strftime('%H:%M'))))
    send_card('Prophet 晚报', elements)

if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv)>1 else 'scan'
    {'morning': morning_report, 'evening': evening_report, 'scan': scan_dual}[mode]()
