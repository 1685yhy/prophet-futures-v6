#!/usr/bin/env python3
"""
Prophet Futures — 报告系统 v3
飞书2.0卡片 | markdown表格 | V25/V28上下排列 | 建议说人话
早报08:50 / 午报11:30 / 晚报19:00 / 扫描每5分钟
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

# 品种配置
S = {
    'lh2609': {
        'code': 'LH0', 'fut': 'LH2609', 'cn': '生猪',
        'mp': 16, 'cost': 0.0006, 'mg': 0.15,
        'max_pos': 6, 'max_total': 12,
        'atr_stop': 1.5, 'rr': 4.0,
        'add_conf': 0.65, 'add_atr': 2.0, 'reduce_conf': 0.55, 'reverse_conf': 0.35,
        'trail_atr': 2.0, 'be_atr': 1.0, 'min_hold': 3,
    },
    'jm2609': {
        'code': 'JM0', 'fut': 'JM2609', 'cn': '焦煤',
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

# [特征函数省略,同前]
def bf(df, idx, w=60):
    if idx < w+5: return None
    s = df.iloc[idx-w:idx+1]; c=s['close'].values.astype(float); o=s['open'].values.astype(float)
    h=s['high'].values.astype(float); l=s['low'].values.astype(float)
    v=s['volume'].values.astype(float); oi=s['oi'].values.astype(float)
    oc=float((o[-1]-c[-2])/c[-2]) if idx>=1 else 0.0; f=[oc,abs(oc)]
    for lag in[1,3,5,10,20]:f.append(float((c[-1]-c[-lag-1])/c[-lag-1] if len(c)>lag else 0))
    for p in[5,10,20,60]:ma=np.mean(c[-min(p,len(c)):]);f.append(float((c[-1]-ma)/ma))
    f.append(float(np.std(c[-20:])/np.mean(c[-20:])));f.append(float((h[-1]-l[-1])/c[-1]))
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

def fd(code):
    try:
        df=ak.futures_main_sina(symbol=code)
        df.columns=['date','open','high','low','close','volume','oi','settle']
        for c in['open','high','low','close','volume','oi']:df[c]=pd.to_numeric(df[c],errors='coerce')
        return df.dropna(subset=['close']).reset_index(drop=True)
    except:return None

def fm(fut,today):
    try:
        df=ak.futures_zh_minute_sina(symbol=fut,period='5')
        df['dt']=pd.to_datetime(df['datetime'])
        return df[df['dt'].dt.strftime('%Y-%m-%d')==today]
    except:return None

def tk():
    r=requests.post('https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal',
        json={'app_id':APP_ID,'app_secret':APP_SECRET},timeout=10)
    return r.json()['tenant_access_token']

def send(title,elements,color='blue',pin=False):
    token=tk()
    card={'schema':'2.0','config':{'wide_screen_mode':True},
          'header':{'title':{'tag':'plain_text','content':title},'template':color},
          'body':{'elements':elements}}
    r=requests.post('https://open.feishu.cn/open-apis/im/v1/messages?receive_id_type=chat_id',
        headers={'Authorization':'Bearer '+token,'Content-Type':'application/json'},
        json={'receive_id':CHAT_ID,'msg_type':'interactive','content':json.dumps(card,ensure_ascii=False)},timeout=10)
    result=r.json();ok=result.get('code')==0
    if pin and ok:
        mid=result.get('data',{}).get('message_id','')
        if mid:requests.post('https://open.feishu.cn/open-apis/im/v1/messages/'+mid+'/pin',
            headers={'Authorization':'Bearer '+token},timeout=5)
    print('  send: %s code=%s pin=%s'%('OK'if ok else'FAIL',result.get('code'),pin))
    return ok

def mkd(text):return{'tag':'markdown','content':text}

def analyze(sk,cfg,df,pos,price,atr,prob):
    """返回(有持仓?, 级别ok/warn/alert, 建议文本)"""
    if not pos:
        sd=atr*cfg['atr_stop']
        signal='看多'if prob>0.5 else'看空';conf=int((prob if prob>0.5 else 1-prob)*100)
        lines=[]
        lines.append('**⚪ %s** 空仓 | 模型%s %d%% | 现价%.0f'%(cfg['cn'],signal,conf,price))
        lines.append('')
        lines.append('| 方向 | 入场 | 止损 | 止盈 |')
        lines.append('|------|------|------|------|')
        lines.append('| 🟢做多 | %.0f | %.0f | %.0f |'%(price,price-sd,price+sd*cfg['rr']))
        lines.append('| 🔴做空 | %.0f | %.0f | %.0f |'%(price,price+sd,price-sd*cfg['rr']))
        return False,'ok','\n'.join(lines)
    
    d=pos['dir'];entry=pos['entry'];vol=pos['vol']
    cd='LONG'if prob>0.5 else'SHORT'
    cf=prob if prob>0.5 else 1-prob
    pp=price-entry if d=='LONG'else entry-price
    pa=pp/atr if atr>0 else 0
    pnl_amt=pp*vol*cfg['mp']*cfg['mg']
    pnl_pct=pp/entry
    
    sd2=atr*cfg['atr_stop']
    hs=price-sd2 if d=='LONG'else price+sd2
    tr=entry
    if d=='LONG':
        if pa>cfg['be_atr']:tr=max(tr,entry)
        if pa>cfg['trail_atr']:tr=max(tr,price-atr*(cfg['atr_stop']-0.3))
        es=max(hs,tr);sp=price-es
    else:
        if pa>cfg['be_atr']:tr=min(tr,entry)
        if pa>cfg['trail_atr']:tr=min(tr,price+atr*(cfg['atr_stop']-0.3))
        es=min(hs,tr);sp=es-price
    
    signal='看多'if prob>0.5 else'看空';conf=int(cf*100)
    dc='做多'if d=='LONG'else'做空'
    
    # Build advice in plain Chinese
    lines=[]
    pnl_sign='+'if pnl_amt>=0 else''
    lines.append('**%s** %s%d手 @%.0f | 浮盈 %s%.1f万 (%+.1f%%) | 现价%.0f'%(
        cfg['cn'],dc,vol,entry,pnl_sign,pnl_amt/10000,pnl_pct*100,price))
    lines.append('')
    lines.append('| 项目 | 数值 |')
    lines.append('|------|------|')
    lines.append('| 止损价 | %.0f (距现价%.0f点=%.1f倍ATR) |'%(es,sp,sp/atr))
    lines.append('| 模型 | %s %d%% |'%(signal,conf))
    
    should_reverse=(d=='LONG'and prob<cfg['reverse_conf'])or(d=='SHORT'and prob>1-cfg['reverse_conf'])
    should_reduce=(d==cd and cf<cfg['reduce_conf'])
    can_add=(d==cd and cf>cfg['add_conf']and pa>cfg['add_atr'])
    
    if should_reverse:
        rev_dir='空'if d=='LONG'else'多'
        rev_price=price+sd2 if d=='LONG'else price-sd2
        lines.append('| ⚠️ 操作 | **立即反手做%s** |'%rev_dir)
        lines.append('| 原因 | 模型已确认反转(prob=%.2f) |'%prob)
        lines.append('| 反手入场 | %.0f 止损%.0f |'%(price,rev_price))
        return True,'alert','\n'.join(lines)
    elif can_add:
        add=min(cfg['max_pos'],cfg['max_total']-vol)
        lines.append('| 🟢 操作 | **加仓%d手** @%.0f |'%(add,price))
        lines.append('| 原因 | 同向高置信%+浮盈%.1f倍ATR |'%(conf,pa))
        return True,'ok','\n'.join(lines)
    elif should_reduce and vol>1:
        cut=vol//2
        lines.append('| 🟡 操作 | **减仓%d手**→留%d手 |'%(cut,vol-cut))
        lines.append('| 原因 | 模型信心降到%d%%，减仓锁利 |'%conf)
        return True,'warn','\n'.join(lines)
    elif d!=cd:
        gap_pct=int(abs(cfg['reverse_conf']-(prob if d=='LONG'else 1-prob))*100)
        rev_thresh=int(cfg['reverse_conf']*100)if d=='LONG'else int((1-cfg['reverse_conf'])*100)
        lines.append('| ⚠️ 注意 | 你%s但模型%s'%(dc,signal))
        lines.append('| 反手条件 | 模型看空超过%d%%就反手(当前%d%%，差%d%%) |'%(rev_thresh,conf,gap_pct))
        lines.append('| 现在 | 继续持有，等模型信号 |')
        return True,'warn','\n'.join(lines)
    else:
        if pa>=cfg['be_atr']:
            lines.append('| 保本 | ✅ 止损已移到入场价%.0f |'%entry)
        if pa>=cfg['trail_atr']:
            lines.append('| 移动止损 | ✅ 已启动(盈利>%.0fATR) |'%cfg['trail_atr'])
        need=cfg['add_atr']-pa
        if need>0:
            trigger_p=price+need*atr if d=='LONG'else price-need*atr
            lines.append('| 加仓条件 | 价到%.0f(还需%.0f点)+模型>%d%% |'%(trigger_p,need*atr,int(cfg['add_conf']*100)))
        else:
            lines.append('| 🟢 加仓条件 | 已满足！浮盈%.1fATR 模型%d%%，等信号确认 |'%(pa,conf))
        if sp/atr<0.5:
            lines.append('| ⚠️ | 止损很近！距现价仅%.0f点'%sp)
            return True,'warn','\n'.join(lines)
        return True,'ok','\n'.join(lines)

def load_v28():
    p=STATE_FILE.replace('.json','_v28.json')
    if os.path.exists(p):return json.load(open(p))
    return{'positions':{},'cash':300000}

def equity(st,ver):
    eq=st['cash']
    cfg_map=S if ver=='v25'else S
    for k,p in st.get('positions',{}).items():
        if ver=='v28':
            for pp in p:eq+=pp['vol']*pp['entry']*cfg_map[k]['mp']*0.15
        else:
            eq+=p['vol']*p['entry']*cfg_map[k]['mp']*0.15
    return eq

# ===== 通用报告构建 =====
def build_report(mode='scan'):
    """mode: scan/morning/midday/evening"""
    sv25=load_state();sv28=load_v28()
    now=datetime.now();today=now.strftime('%Y-%m-%d')
    wday=['周一','周二','周三','周四','周五','周六','周日'][now.weekday()]
    elements=[];has_action=False;pin=False
    
    # 头
    if mode=='morning':
        elements.append(mkd('**%s %s 08:50** | 📊 盘前分析'%(today,wday)))
    elif mode=='midday':
        elements.append(mkd('**%s %s 11:30** | 📊 午间总结'%(today,wday)))
    elif mode=='evening':
        elements.append(mkd('**%s %s 19:00** | 📊 收盘总结'%(today,wday)))
    else:
        elements.append(mkd('**%s %s** | 每5分钟自动扫描'%(today,now.strftime('%H:%M'))))
    
    elements.append(mkd(''))
    
    # ═══ V25 ═══
    elements.append(mkd('━━━ **V25 原版** ━━━'))
    for sk in['lh2609','jm2609']:
        cfg=S[sk]
        df=fd(cfg['code'])
        if df is None:continue
        td=fm(cfg['fut'],today)
        price=float(td.iloc[-1]['close'])if td is not None and len(td)>0 else float(df.iloc[-1]['close'])
        av=[abs(float(df.iloc[i]['high'])-float(df.iloc[i]['low']))for i in range(max(0,len(df)-20),len(df))]
        atr=np.mean(av)
        ma5=np.mean([float(df.iloc[i]['close'])for i in range(max(0,len(df)-5),len(df))])
        ma20=np.mean([float(df.iloc[i]['close'])for i in range(max(0,len(df)-20),len(df))])
        ft=bf(df,len(df)-1,60)
        if ft is None:continue
        mp=MODEL_DIR+'/'+sk+'_xgb.pkl'
        if not os.path.exists(mp):continue
        m=pickle.load(open(mp,'rb'))
        prob=float(m.predict_proba(ft.reshape(1,-1))[0][1])
        
        trend='📈'if price>ma5>ma20 else('📉'if price<ma5<ma20 else'↔️')
        if mode in('morning','midday','evening'):
            prev=float(df.iloc[-2]['close'])if len(df)>1 else price
            chg=(price-prev)/prev
            elements.append(mkd('**%s** %s %.0f (%+.1f%%) | MA5 %.0f MA20 %.0f | ATR %.0f'%(
                cfg['cn'],trend,price,chg*100,ma5,ma20,atr)))
        
        pos=sv25['positions'].get(sk)
        has,level,text=analyze(sk,cfg,df,pos,price,atr,prob)
        if level!='ok':has_action=True
        elements.append(mkd(text))
    
    # ═══ V28 ═══
    elements.append(mkd(''))
    elements.append(mkd('━━━ **V28 动态** ━━━'))
    for sk in['lh2609','jm2609']:
        cfg=S[sk]
        df=fd(cfg['code'])
        if df is None:continue
        td=fm(cfg['fut'],today)
        price=float(td.iloc[-1]['close'])if td is not None and len(td)>0 else float(df.iloc[-1]['close'])
        av=[abs(float(df.iloc[i]['high'])-float(df.iloc[i]['low']))for i in range(max(0,len(df)-20),len(df))]
        atr=np.mean(av)
        ft=bf(df,len(df)-1,60)
        if ft is None:continue
        mp=MODEL_DIR+'/'+sk+'_xgb.pkl'
        if not os.path.exists(mp):continue
        m=pickle.load(open(mp,'rb'))
        prob=float(m.predict_proba(ft.reshape(1,-1))[0][1])
        
        pos28=sv28['positions'].get(sk,[])
        signal='看多'if prob>0.5 else'看空';conf=int((prob if prob>0.5 else 1-prob)*100)
        si='🟢'if prob>0.5 else'🔴'
        
        if pos28:
            tv=sum(p['vol']for p in pos28);d28=pos28[0]['dir']
            ae=np.mean([p['entry']for p in pos28])
            pa=(price-ae)*tv*cfg['mp']*cfg['mg']if d28=='LONG'else(ae-price)*tv*cfg['mp']*cfg['mg']
            ps='+'if pa>=0 else''
            
            # V28 stop
            sd2=atr*cfg['atr_stop']
            tr=ae
            if d28=='LONG':
                if(price-ae)/atr>cfg['be_atr']:tr=max(tr,ae)
                es=max(price-sd2,tr);sp=price-es
            else:
                if(ae-price)/atr>cfg['be_atr']:tr=min(tr,ae)
                es=min(price+sd2,tr);sp=es-price
            
            # V28 advice
            cd='LONG'if prob>0.5 else'SHORT';cf=prob if prob>0.5 else 1-prob
            pa2=(price-ae)/atr if d28=='LONG'else(ae-price)/atr
            can_add=(d28==cd and cf>cfg['add_conf']and pa2>cfg['add_atr'])
            should_reduce=(d28==cd and cf<cfg['reduce_conf'])
            should_reverse=(d28=='LONG'and prob<cfg['reverse_conf'])or(d28=='SHORT'and prob>1-cfg['reverse_conf'])
            
            lines=[]
            lines.append('**V28 %s** %s%d手(%d子仓) @%.0f | 浮盈 %s%.1f万'%(
                cfg['cn'],('做多'if d28=='LONG'else'做空'),tv,len(pos28),ae,ps,pa/10000))
            lines.append('')
            lines.append('| 项目 | 数值 |')
            lines.append('|------|------|')
            lines.append('| 止损 | %.0f (距%.0f点) |'%(es,sp))
            lines.append('| 模型 | %s %s%d%% |'%(si,signal,conf))
            lines.append('| 子仓 | %s |'%'、'.join(['@%.0f(%d手)'%(p['entry'],p['vol'])for p in pos28]))
            
            if should_reverse:
                rev_dir='空'if d28=='LONG'else'多'
                lines.append('| 🔴 操作 | **反手做%s** @%.0f |'%(rev_dir,price))
                has_action=True
            elif can_add:
                add=min(cfg['max_pos'],cfg['max_total']-tv)
                lines.append('| 🟢 操作 | **加仓%d手** @%.0f |'%(add,price))
                has_action=True
            elif should_reduce and tv>1:
                cut=tv//2
                lines.append('| 🟡 操作 | **减仓%d手**→留%d手 |'%(cut,tv-cut))
                has_action=True
            elif d28!=cd:
                gap=int(abs(cfg['reverse_conf']-(prob if d28=='LONG'else 1-prob))*100)
                lines.append('| ⚠️ 注意 | 持仓方向与模型不一致，差%d%%触发反手 |'%gap)
            else:
                lines.append('| ✅ | 继续持有，模型同向 |')
            
            elements.append(mkd('\n'.join(lines)))
        else:
            sd=atr*cfg['atr_stop']
            if prob>0.5:
                advice='做多 %.0f→止损%.0f'%(price,price-sd)
            else:
                advice='做空 %.0f→止损%.0f'%(price,price+sd)
            elements.append(mkd('**V28 %s** 空仓 | %s %s%d%% | %s'%(cfg['cn'],si,signal,conf,advice)))
    
    # Footer
    v25_eq=equity(sv25,'v25');v28_eq=equity(sv28,'v28')
    elements.append(mkd(''))
    elements.append(mkd('V25 ¥%s | V28 ¥%s | %s'%(format(int(v25_eq),','),format(int(v28_eq),','),
        today if mode!='scan'else now.strftime('%H:%M'))))
    
    # Color + pin
    color='red'if has_action else'blue'
    if has_action:pin=True
    
    titles={'morning':'Prophet 早报 | '+wday,'midday':'Prophet 午报 | '+wday,
            'evening':'Prophet 晚报 | '+wday,'scan':'扫描 %s'%now.strftime('%H:%M')}
    send(titles.get(mode,'Prophet'),elements,color,pin)

# ===== 入口 =====
if __name__=='__main__':
    mode=sys.argv[1]if len(sys.argv)>1 else'scan'
    build_report(mode)
