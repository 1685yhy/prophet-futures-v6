#!/usr/bin/env python3
"""
Prophet Futures — 报告系统 v5
实时价 | 明细表始终显示 | 成交记录+原因 | 中文操作名
"""
import json, requests, numpy as np, pandas as pd, pickle, os, sys
from datetime import datetime, timedelta
import akshare as ak
from realtime_data import get_realtime_quote

def _load_env():
    env_file=os.path.expanduser("~/.hermes/.env")
    if os.path.exists(env_file):
        with open(env_file) as f:
            for line in f:
                line=line.strip()
                if line and not line.startswith('#') and '=' in line:
                    k,v=line.split('=',1)
                    os.environ.setdefault(k.strip(),v.strip())
_load_env()
APP_ID=os.getenv('FEISHU_APP_ID','');APP_SECRET=os.getenv('FEISHU_APP_SECRET','')
CHAT_ID='oc_e9bf3cb98e83f50ad4e71dff71f9dce8'
MD=os.path.join(os.path.dirname(os.path.abspath(__file__)),'models')
SF=os.path.join(os.path.dirname(os.path.abspath(__file__)),'paper_state.json')

S={'lh2609':{'code':'LH0','fut':'LH2609','cn':'生猪','mp':16,'cost':0.0006,'mg':0.15,
    'max_pos':6,'max_total':12,'atr_stop':1.5,'rr':4.0,
    'add_conf':0.65,'add_atr':2.0,'reduce_conf':0.55,'reverse_conf':0.35,'trail_atr':2.0,'be_atr':1.0,'min_hold':3},
  'jm2609':{'code':'JM0','fut':'JM2609','cn':'焦煤','mp':60,'cost':0.0011,'mg':0.15,
    'max_pos':4,'max_total':8,'atr_stop':2.0,'rr':3.5,
    'add_conf':0.65,'add_atr':2.5,'reduce_conf':0.55,'reverse_conf':0.30,'trail_atr':3.0,'be_atr':2.0,'min_hold':5}}

def ls():
    if os.path.exists(SF):
        with open(SF)as f:return json.load(f)
    return{'cash':300000,'positions':{},'trades':[],'equity_history':[]}

def bf(df,idx,w=60):
    if idx<w+5:return None
    s=df.iloc[idx-w:idx+1];c=s['close'].values.astype(float);o=s['open'].values.astype(float)
    h=s['high'].values.astype(float);l=s['low'].values.astype(float)
    v=s['volume'].values.astype(float);oi=s['oi'].values.astype(float)
    oc=float((o[-1]-c[-2])/c[-2])if idx>=1 else 0.0;f=[oc,abs(oc)]
    for lag in[1,3,5,10,20]:f.append(float((c[-1]-c[-lag-1])/c[-lag-1]if len(c)>lag else 0))
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

def gp(sk):
    """获取实时价（盘中）或最新日线收盘（盘后）"""
    rt=get_realtime_quote(sk)
    if rt:return rt['price']
    df=fd(S[sk]['code'])
    if df is not None and len(df)>0:return float(df.iloc[-1]['close'])
    return None

def tp_cn(t):
    """交易类型中文"""
    m={'STOP':'止损','REDUCE':'减仓','REVERSE':'反手','OPEN':'开仓','ADD':'加仓'}
    return m.get(t,t)

def tp_reason(t,cfg):
    """交易原因说明"""
    r={'STOP':'价格触及止损位，按规则强制平仓',
       'REDUCE':'模型信心降至%d%%以下，主动减仓锁利'%int(cfg['reduce_conf']*100),
       'REVERSE':'模型信号反转，平仓准备反向操作',
       'OPEN':'模型新信号开仓',
       'ADD':'浮盈达标+模型高信心，顺势加仓'}
    return r.get(t,'')

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
    return ok

def md(t):return{'tag':'markdown','content':t}
def hr():return{'tag':'hr'}

def lv28():
    p=SF.replace('.json','_v28.json')
    if os.path.exists(p):return json.load(open(p))
    return{'positions':{},'cash':300000}

def lv29():
    p=SF.replace('.json','_v29.json')
    if os.path.exists(p):return json.load(open(p))
    return{'positions':{},'cash':300000}

def lv30():
    p=SF.replace('.json','_v30.json')
    if os.path.exists(p):return json.load(open(p))
    return{'positions':{},'cash':300000}

def lv31():
    p=SF.replace('.json','_v31.json')
    if os.path.exists(p):return json.load(open(p))
    return{'positions':{},'cash':300000}

def lv32():
    p=SF.replace('.json','_v32.json')
    if os.path.exists(p):return json.load(open(p))
    return{'positions':{},'cash':300000}

def lv32b():
    p=SF.replace('.json','_v32b.json')
    if os.path.exists(p):return json.load(open(p))
    return{'positions':{},'cash':300000}

def eq(st,prices=None):
    """权益 = 现金 + 持仓保证金 + 浮动盈亏(逐日盯市)
    prices: {sym_key: 当前价格} 不传则只算保证金(兼容旧调用)"""
    e=st['cash']
    for k,p in st.get('positions',{}).items():
        if k not in S:continue
        mp=S[k]['mp'];mg=S[k]['mg']
        if isinstance(p,list):
            for pp in p:
                entry=pp['entry'];vol=pp['vol'];d=pp['dir']
                e+=vol*entry*mp*mg  # 锁定的保证金
                if prices and k in prices:
                    cur=prices[k]
                    e+=(cur-entry)*vol*mp if d=='LONG' else(entry-cur)*vol*mp
        else:
            entry=p['entry'];vol=p['vol'];d=p['dir']
            e+=vol*entry*mp*mg
            if prices and k in prices:
                cur=prices[k]
                e+=(cur-entry)*vol*mp if d=='LONG' else(entry-cur)*vol*mp
    return e

def analyze(sk,cfg,df,pos,price,atr,prob):
    """返回 (持仓?, 级别, 摘要行, 详情行列表)"""
    if not pos:
        sd=atr*cfg['atr_stop'];signal='看多'if prob>0.5 else'看空'
        conf=int((prob if prob>0.5 else 1-prob)*100);si='🟢'if prob>0.5 else'🔴'
        lines=[]
        lines.append('| 方向 | 入场→止损→止盈 | RR |')
        lines.append('|------|------|------|')
        lines.append('| %s做多 | %.0f→%.0f→%.0f | 1:%d |'%(si,price,price-sd,price+sd*cfg['rr'],cfg['rr']))
        lines.append('| %s做空 | %.0f→%.0f→%.0f | 1:%d |'%(si,price,price+sd,price-sd*cfg['rr'],cfg['rr']))
        return False,'ok','⚪ %s 空仓 | 模型%s %d%% | 现价%.0f'%(cfg['cn'],signal,conf,price),lines
    
    d=pos['dir'];entry=pos['entry'];vol=pos['vol']
    cd='LONG'if prob>0.5 else'SHORT';cf=prob if prob>0.5 else 1-prob
    pp=price-entry if d=='LONG'else entry-price
    pa=pp/atr if atr>0 else 0
    pa_amt=pp*vol*cfg['mp']  # 浮盈 = 点数×乘数×手数
    pa_pct=pp/entry
    dc='做多'if d=='LONG'else'做空';signal='看多'if prob>0.5 else'看空'
    conf=int(cf*100);si='🟢'if prob>0.5 else'🔴'
    ps='+'if pa_amt>=0 else'';pc='🟢'if pa_amt>=0 else'🔴'
    
    sd2=atr*cfg['atr_stop']
    hs=price-sd2 if d=='LONG'else price+sd2;tr=entry
    if d=='LONG':
        if pa>cfg['be_atr']:tr=max(tr,entry)
        if pa>cfg['trail_atr']:tr=max(tr,price-atr*(cfg['atr_stop']-0.3))
        es=max(hs,tr);sp=price-es;spa=sp/atr
    else:
        if pa>cfg['be_atr']:tr=min(tr,entry)
        if pa>cfg['trail_atr']:tr=min(tr,price+atr*(cfg['atr_stop']-0.3))
        es=min(hs,tr);sp=es-price;spa=sp/atr
    
    should_reverse=(d=='LONG'and prob<cfg['reverse_conf'])or(d=='SHORT'and prob>1-cfg['reverse_conf'])
    should_reduce=(d==cd and cf<cfg['reduce_conf'])
    can_add=(d==cd and cf>cfg['add_conf']and pa>cfg['add_atr'])
    
    # 摘要行: 品种 + 方向 + 浮盈 + 一句话建议
    if should_reverse:
        rev_dir='空'if d=='LONG'else'多'
        summary='🔴 **%s** %s%d手 | 浮盈%s%.1f万 | **立即反手做%s @%.0f**'%(
            cfg['cn'],dc,vol,ps,pa_amt/10000,rev_dir,price)
        level='alert'
    elif can_add:
        add=min(cfg['max_pos'],cfg['max_total']-vol)
        summary='🟢 **%s** %s%d手 | 浮盈%s%.1f万 | **加仓%d手 @%.0f**'%(
            cfg['cn'],dc,vol,ps,pa_amt/10000,add,price)
        level='ok'
    elif should_reduce and vol>1:
        cut=vol//2
        summary='🟡 **%s** %s%d手 | 浮盈%s%.1f万 | **减仓%d手→留%d手**'%(
            cfg['cn'],dc,vol,ps,pa_amt/10000,cut,vol-cut)
        level='warn'
    elif d!=cd:
        gap=int(abs(cfg['reverse_conf']-(prob if d=='LONG'else 1-prob))*100)
        rev_th=int(cfg['reverse_conf']*100)if d=='LONG'else int((1-cfg['reverse_conf'])*100)
        summary='⚠️ **%s** %s%d手 | 浮盈%s%.1f万 | 持仓%s 模型偏%s(距反手差%d%%)'%(
            cfg['cn'],dc,vol,ps,pa_amt/10000,dc,signal,gap)
        level='warn'
    elif spa<0.5:
        summary='⚠️ **%s** %s%d手 | 浮盈%s%.1f万 | 止损很近(距%.0f点)'%(
            cfg['cn'],dc,vol,ps,pa_amt/10000,sp)
        level='warn'
    else:
        need=cfg['add_atr']-pa
        if need>0:
            tp=price+need*atr if d=='LONG'else price-need*atr
            summary='✅ **%s** %s%d手 | 浮盈%s%.1f万 | 持有 加仓等%.0f'%(
                cfg['cn'],dc,vol,ps,pa_amt/10000,tp)
        else:
            summary='✅ **%s** %s%d手 | 浮盈%s%.1f万 | 持有 加仓条件已满足'%(
                cfg['cn'],dc,vol,ps,pa_amt/10000)
        level='ok'
    
    # 详情行
    lines=[]
    lines.append('| 项目 | 数值 |')
    lines.append('|------|------|')
    lines.append('| 入场 | %.0f |'%entry)
    lines.append('| 现价 | %.0f (ATR %.0f) |'%(price,atr))
    lines.append('| 浮盈 | %s%.1f万 (%+.1f%%) |'%(ps,pa_amt/10000,pa_pct*100))
    lines.append('| 止损 | %.0f → 跌破就平仓 |'%es)
    sys_tp = pos.get('take_profit', 0)
    if sys_tp:
        lines.append('| 止盈 | %.0f (固定) |'%sys_tp)
    lines.append('| 模型 | %s%s %d%% |'%(si,signal,conf))
    
    if should_reverse:
        lines.append('| 🔴 反手 | 确认反转，平仓后做%s @%.0f |'%('空'if d=='LONG'else'多',price))
    elif can_add:
        add=min(cfg['max_pos'],cfg['max_total']-vol)
        lines.append('| 🟢 加仓 | 浮盈%.1fATR+模型%d%% → 加%d手 |'%(pa,conf,add))
    elif should_reduce:
        lines.append('| 🟡 减仓 | 模型信心降到%d%% → 减仓锁利 |'%conf)
    elif d!=cd:
        rev_th=int(cfg['reverse_conf']*100)if d=='LONG'else int((1-cfg['reverse_conf'])*100)
        gap=int(abs(cfg['reverse_conf']-(prob if d=='LONG'else 1-prob))*100)
        lines.append('| ⚠️ 方向冲突 | 持仓%s 模型偏%s | 需模型<%.0f%%才反手(距反手差%d%%) |'%(
            dc,signal,cfg['reverse_conf']*100,gap))
    else:
        need=cfg['add_atr']-pa
        if need>0:
            tp=price+need*atr if d=='LONG'else price-need*atr
            lines.append('| 加仓触发 | 价到%.0f(还需%.0f点)+模型>%d%% |'%(tp,need*atr,int(cfg['add_conf']*100)))
        if pa>=cfg['be_atr']:lines.append('| 保本 | ✅ 止损已移到入场价 |')
        if pa>=cfg['trail_atr']:lines.append('| 移动止损 | ✅ 已启动 |')
    
    return True,level,summary,lines

def analyze_v28(sk,cfg,df,pos28,price,atr,prob):
    """V28持仓分析 — 读系统真实状态，不自己算"""
    if not pos28:return False,'ok','',''
    
    tv=sum(p['vol']for p in pos28);d28=pos28[0]['dir']
    ae=np.mean([p['entry']for p in pos28]);dc28='做多'if d28=='LONG'else'做空'
    pa_amt=(price-ae)*tv*cfg['mp']if d28=='LONG'else(ae-price)*tv*cfg['mp']  # 浮盈=点数×乘数×手数
    ps28='+'if pa_amt>=0 else'';pa_pct=(price-ae)/ae if d28=='LONG'else(ae-price)/ae
    pa_atr=(price-ae)/atr if d28=='LONG'else(ae-price)/atr
    
    cd='LONG'if prob>0.5 else'SHORT';cf=prob if prob>0.5 else 1-prob
    signal='看多'if prob>0.5 else'看空';conf=int(cf*100);si='🟢'if prob>0.5 else'🔴'
    
    # 用系统真实的_trail,不用自己算
    sys_trail=pos28[0].get('_trail',None)
    if sys_trail is not None:
        es=sys_trail
        sp=es-price if d28=='SHORT'else price-es
        spa=sp/atr if atr>0 else 0
    else:
        # Fallback: 自己算
        sd2=atr*cfg['atr_stop']
        es=price-sd2 if d28=='LONG'else price+sd2
        sp=price-es if d28=='LONG'else es-price
        spa=sp/atr if atr>0 else 0
    
    # 从交易记录推算持仓时长(用第一条子仓的entry_time)
    first_entry=pos28[0].get('_entry_time','')
    
    should_reverse=(d28=='LONG'and prob<cfg['reverse_conf'])or(d28=='SHORT'and prob>1-cfg['reverse_conf'])
    should_reduce=(d28==cd and cf<cfg['reduce_conf'])
    can_add=(d28==cd and cf>cfg['add_conf']and pa_atr>cfg['add_atr'])
    
    # 摘要
    if should_reverse:
        rev_dir='空'if d28=='LONG'else'多'
        summary='🔴 **%s** %s%d手(%d仓) | 浮盈%s%.1f万 | **模型建议:反手做%s @%.0f**'%(
            cfg['cn'],dc28,tv,len(pos28),ps28,pa_amt/10000,rev_dir,price)
        level='alert'
    elif can_add:
        add=min(cfg['max_pos'],cfg['max_total']-tv)
        summary='🟢 **%s** %s%d手(%d仓) | 浮盈%s%.1f万 | **模型建议:加仓%d手 @%.0f**'%(
            cfg['cn'],dc28,tv,len(pos28),ps28,pa_amt/10000,add,price)
        level='ok'
    elif should_reduce and tv>1:
        cut=tv//2
        summary='🟡 **%s** %s%d手(%d仓) | 浮盈%s%.1f万 | **模型建议:减仓%d→%d手**'%(
            cfg['cn'],dc28,tv,len(pos28),ps28,pa_amt/10000,cut,tv-cut)
        level='warn'
    elif d28!=cd:
        gap=int(abs(cfg['reverse_conf']-(prob if d28=='LONG'else 1-prob))*100)
        rev_th=int(cfg['reverse_conf']*100)if d28=='LONG'else int((1-cfg['reverse_conf'])*100)
        summary='⚠️ **%s** %s%d手(%d仓) | 浮盈%s%.1f万 | 持仓%s 模型偏%s(距反手差%d%%)'%(
            cfg['cn'],dc28,tv,len(pos28),ps28,pa_amt/10000,dc28,signal,gap)
        level='warn'
    elif spa<0.5:
        summary='⚠️ **%s** %s%d手(%d仓) | 浮盈%s%.1f万 | 止损很近(距%.0f点)'%(
            cfg['cn'],dc28,tv,len(pos28),ps28,pa_amt/10000,sp)
        level='warn'
    else:
        need=cfg['add_atr']-pa_atr
        if need>0:
            tp=price+need*atr if d28=='LONG'else price-need*atr
            summary='✅ **%s** %s%d手(%d仓) | 浮盈%s%.1f万 | 持有 加仓等%.0f'%(
                cfg['cn'],dc28,tv,len(pos28),ps28,pa_amt/10000,tp)
        else:
            summary='✅ **%s** %s%d手(%d仓) | 浮盈%s%.1f万 | 持有 加仓条件满足'%(
                cfg['cn'],dc28,tv,len(pos28),ps28,pa_amt/10000)
        level='ok'
    
    # 详情表
    lines=[]
    lines.append('| 项目 | 数值(系统) |')
    lines.append('|------|------|')
    lines.append('| 均价 | %.0f (%d子仓) |'%(ae,len(pos28)))
    lines.append('| 现价 | %.0f (ATR %.0f) |'%(price,atr))
    lines.append('| 浮盈 | %s%.1f万 (%+.1f%%) |'%(ps28,pa_amt/10000,pa_pct*100))
    if sys_trail is not None:
        lines.append('| 系统止损 | %.0f (距%.0f点) |'%(es,sp))
        lines.append('| 动态出场 | %.0f (追踪止盈) |'%sys_trail)
    else:
        lines.append('| 止损(算) | %.0f |'%es)
    lines.append('| 模型 | %s%s %d%% |'%(si,signal,conf))
    lines.append('| 子仓 | %s |'%'、'.join(['@%.0f(%d手)'%(p['entry'],p['vol'])for p in pos28]))
    
    if should_reverse:
        lines.append('| 🔴 模型建议 | 确认反转，平仓后做%s @%.0f |'%('空'if d28=='LONG'else'多',price))
    elif can_add:
        add=min(cfg['max_pos'],cfg['max_total']-tv)
        lines.append('| 🟢 模型建议 | 浮盈%.1fATR+模型%d%% → 加%d手 |'%(pa_atr,conf,add))
    elif should_reduce:
        lines.append('| 🟡 模型建议 | 信心降到%d%% → 减仓锁利 |'%conf)
    elif d28!=cd:
        rev_th=int(cfg['reverse_conf']*100)if d28=='LONG'else int((1-cfg['reverse_conf'])*100)
        gap=int(abs(cfg['reverse_conf']-(prob if d28=='LONG'else 1-prob))*100)
        lines.append('| ⚠️ 方向冲突 | 持仓%s 模型偏%s | 需模型概率<%.0f%%反手(距反手差%d%%) |'%(
            dc28,signal,cfg['reverse_conf']*100,gap))
    else:
        need=cfg['add_atr']-pa_atr
        if need>0:
            tp=price+need*atr if d28=='LONG'else price-need*atr
            lines.append('| 加仓触发 | 价到%.0f(还需%.0f点)+模型>%d%% |'%(tp,need*atr,int(cfg['add_conf']*100)))
        if pa_atr>=cfg['be_atr']:lines.append('| 保本 | ✅ 止损已移到均价 |')
        if pa_atr>=cfg['trail_atr']:lines.append('| 移动止损 | ✅ 已启动 |')
    
    return True,level,summary,lines

# ===== 扫描 =====
def scan():
    sv=ls();s28=lv28();now=datetime.now();today=now.strftime('%Y-%m-%d')
    ele=[];act=[];warn=[]
    
    # 预拉数据(一次)
    data={}
    for sk in['lh2609']:
        cfg=S[sk];df=fd(cfg['code'])
        if df is None:continue
        td=fm(cfg['fut'],today)
        if td is not None and len(td)>0:price=float(td.iloc[-1]['close'])
        else:price=float(df.iloc[-1]['close'])
        # 用实时价覆盖（盘中优先）
        rt=get_realtime_quote(sk)
        if rt:price=rt['price']
        av=[abs(float(df.iloc[i]['high'])-float(df.iloc[i]['low']))for i in range(max(0,len(df)-20),len(df))]
        atr=np.mean(av);ft=bf(df,len(df)-1,60)
        if ft is None:continue
        mp=MD+'/'+sk+'_xgb.pkl'
        if not os.path.exists(mp):continue
        m=pickle.load(open(mp,'rb'));prob=float(m.predict_proba(ft.reshape(1,-1))[0][1])
        # V29 用新模型
        mp_new=MD+'/'+sk+'_xgb_new.pkl'
        prob_new=prob  # 默认fallback
        if os.path.exists(mp_new):
            m_new=pickle.load(open(mp_new,'rb'))
            prob_new=float(m_new.predict_proba(ft.reshape(1,-1))[0][1])
        # V30 用校准模型
        mp_cal=MD+'/'+sk+'_xgb_calibrated.pkl'
        prob_cal=prob
        if os.path.exists(mp_cal):
            m_cal=pickle.load(open(mp_cal,'rb'))
            prob_cal=float(m_cal.predict_proba(ft.reshape(1,-1))[0][1])
        data[sk]={'cfg':cfg,'df':df,'price':price,'atr':atr,'prob':prob,'prob_new':prob_new,'prob_cal':prob_cal}
    
    # ═══ V25 大块 ═══
    ele.append(md('━━━ **V25 原版** ━━━'))
    for sk in['lh2609']:
        if sk not in data:continue
        d=data[sk];cfg=d['cfg'];price=d['price'];atr=d['atr'];prob=d['prob']
        pos=sv['positions'].get(sk)
        has,level,summary,lines=analyze(sk,cfg,d['df'],pos,price,atr,prob)
        if level=='alert':act.append('🔴 V25 %s: %s'%(cfg['cn'],summary.replace('**','')))
        elif level=='warn':warn.append('⚠️ V25 %s: %s'%(cfg['cn'],summary.replace('**','')))
        ele.append(md(summary))
        ele.append(md('\n'.join(lines)))  # 始终显示明细表
    
    # V25 成交
    t25=sv.get('trades',[]);t25_today=[t for t in t25 if today in str(t.get('time',''))]
    if t25_today:
        ele.append(md('**今日成交**'))
        for t in t25_today[-5:]:
            cfg=S.get(t.get('sym',''),{})
            cn=tp_cn(t.get('type','?'));reason=tp_reason(t.get('type','?'),cfg)
            ele.append(md('%s %s %d手 %+.0f → %s'%(t.get('time','')[-8:-3],cn,t.get('vol',0),t.get('pnl',0),reason)))
    
    # ═══ V28 大块 ═══
    ele.append(md(''))
    ele.append(md('━━━ **V28 动态** ━━━'))
    for sk in['lh2609']:
        if sk not in data:continue
        d=data[sk];cfg=d['cfg'];price=d['price'];atr=d['atr'];prob=d['prob']
        pos28=s28['positions'].get(sk,[])
        if pos28:
            has28,level28,sum28,lines28=analyze_v28(sk,cfg,d['df'],pos28,price,atr,prob)
            if level28=='alert':act.append('🔴 V28 %s: %s'%(cfg['cn'],sum28.replace('**','')))
            elif level28=='warn':warn.append('⚠️ V28 %s: %s'%(cfg['cn'],sum28.replace('**','')))
            ele.append(md(sum28))
            ele.append(md('\n'.join(lines28)))  # 始终显示明细表
        else:
            s_txt='看多'if prob>0.5 else'看空';conf=int((prob if prob>0.5 else 1-prob)*100)
            ele.append(md('⚪ **%s** 空仓 | 模型%s %d%% | 现价%.0f'%(cfg['cn'],s_txt,conf,price)))
    
    # V28 成交
    t28=s28.get('trades',[]);t28_today=[t for t in t28 if today in str(t.get('time',''))]
    if t28_today:
        ele.append(md('**今日成交**'))
        for t in t28_today[-5:]:
            cfg=S.get(t.get('sym',''),{})
            cn=tp_cn(t.get('type','?'));reason=tp_reason(t.get('type','?'),cfg)
            ele.append(md('%s %s %d手 %+.0f → %s'%(t.get('time','')[-8:-3],cn,t.get('vol',0),t.get('pnl',0),reason)))

    # ── 发送第一张卡 V25+V28 ──
    if act:
        banner=['**⚠️ 需要操作**']+act+['']; [ele.insert(0,md(b)) for b in banner]
    elif warn:
        banner=['**⚡ 关注**']+warn+['']; [ele.insert(0,md(b)) for b in banner]
    
    prices={sk:d['price']for sk,d in data.items()}
    v25_eq=eq(sv,prices);v28_eq=eq(s28,prices)
    ele.append(md(''))
    ele.append(md('V25 ¥%s | V28 ¥%s | %s'%(format(int(v25_eq),','),format(int(v28_eq),','),now.strftime('%H:%M'))))
    send('扫描① %s'%now.strftime('%H:%M'),ele,'red'if act else('yellow'if warn else'blue'),bool(act))

    # ── 第二张卡 V29+V30 ──
    ele2=[];act2=[];warn2=[]
    s29=lv29();s30=lv30()
    for ver_lbl,s_ver,prob_key in[('V29 新模型',s29,'prob_new'),('V30 校准版',s30,'prob_cal')]:
        ele2.append(md(''))
        ele2.append(md('━━━ **%s** ━━━'%ver_lbl))
        for sk in['lh2609']:
            if sk not in data:continue
            d=data[sk];cfg=d['cfg'];price=d['price'];atr=d['atr'];prob=d[prob_key]
            pos=s_ver['positions'].get(sk,[])
            if pos:
                has,level,summary,lines=analyze_v28(sk,cfg,d['df'],pos,price,atr,prob)
                if level=='alert':act2.append('🔴 %s %s: %s'%(ver_lbl[:3],cfg['cn'],summary.replace('**','')))
                elif level=='warn':warn2.append('⚠️ %s %s: %s'%(ver_lbl[:3],cfg['cn'],summary.replace('**','')))
                ele2.append(md(summary))
                ele2.append(md('\n'.join(lines)))
            else:
                s_txt='看多'if prob>0.5 else'看空';conf=int((prob if prob>0.5 else 1-prob)*100)
                ele2.append(md('⚪ **%s** 空仓 | 模型%s %d%% | 现价%.0f'%(cfg['cn'],s_txt,conf,price)))
        ts=s_ver.get('trades',[]);ts_t=[t for t in ts if today in str(t.get('time',''))]
        if ts_t:
            ele2.append(md('**%s 成交**'%ver_lbl[:3]))
            for t in ts_t[-5:]:
                cfg=S.get(t.get('sym',''),{})
                cn=tp_cn(t.get('type','?'));reason=tp_reason(t.get('type','?'),cfg)
                ele2.append(md('%s %s %d手 %+.0f → %s'%(t.get('time','')[-8:-3],cn,t.get('vol',0),t.get('pnl',0),reason)))
    
    if act2: banner2=['**⚠️ 需要操作**']+act2+['']; [ele2.insert(0,md(b)) for b in banner2]
    elif warn2: banner2=['**⚡ 关注**']+warn2+['']; [ele2.insert(0,md(b)) for b in banner2]
    
    v29_eq=eq(s29,prices);v30_eq=eq(s30,prices)
    ele2.append(md(''))
    ele2.append(md('V29 ¥%s | V30 ¥%s | %s'%(format(int(v29_eq),','),format(int(v30_eq),','),now.strftime('%H:%M'))))
    send('扫描② %s'%now.strftime('%H:%M'),ele2,'red'if act2 else('yellow'if warn2 else'blue'),bool(act2))

    # ── 第三张卡 V31+V32+V32b ──
    ele3=[];act3=[];warn3=[]
    s31=lv31();s32=lv32();s32b=lv32b()
    # V31/V32/V32b model probs
    mp_v32=MD+'/'+'v31_xgb.pkl'
    prob32=None
    if os.path.exists(mp_v32) and data.get('lh2609'):
        m32=pickle.load(open(mp_v32,'rb'))
        d0=data['lh2609']; df0=d0.get('df')
        if df0 is not None and len(df0)>70:
            ft=bf(df0,len(df0)-1,60)
            if ft is not None:
                try: prob32=float(m32.predict_proba(ft.reshape(1,-1))[0][1])
                except: prob32=None
    
    for ver_lbl,s_ver,model_info in [
        ('V31 基线',s31,'_xgb.pkl 旧模型'),
        ('V32 优化',s32,'v31_xgb.pkl 回测最优'),
        ('V32b 保守',s32b,'v31_xgb.pkl 半仓不反手')]:
        ele3.append(md(''))
        ele3.append(md('━━━ **%s** ━━━'%ver_lbl))
        ele3.append(md('模型: %s'%model_info))
        for sk in['lh2609']:
            if sk not in data:continue
            d=data[sk];cfg=d['cfg'];price=d['price'];atr=d['atr']
            pos=s_ver['positions'].get(sk,[])
            prob=prob32 if prob32 is not None else d['prob']
            if pos:
                if isinstance(pos, list):
                    has,level,summary,lines=analyze_v28(sk,cfg,d['df'],pos,price,atr,prob)
                else:
                    has,level,summary,lines=analyze(sk,cfg,d['df'],pos,price,atr,prob)
                if level=='alert':act3.append('🔴 %s %s: %s'%(ver_lbl[:4],cfg['cn'],summary.replace('**','')))
                elif level=='warn':warn3.append('⚠️ %s %s: %s'%(ver_lbl[:4],cfg['cn'],summary.replace('**','')))
                ele3.append(md(summary))
                ele3.append(md('\n'.join(lines)))
            else:
                s_txt='做多'if prob>0.5 else'做空';conf=int((prob if prob>0.5 else 1-prob)*100)
                ele3.append(md('⚪ **%s** 空仓 | 模型%s %d%% | 现价%.0f 待信号入场'%(cfg['cn'],s_txt,conf,price)))
        ts=s_ver.get('trades',[]);ts_t=[t for t in ts if today in str(t.get('time',''))]
        if ts_t:
            ele3.append(md('**今日成交**'))
            for t in ts_t[-3:]:
                cfg=S.get(t.get('sym',''),{})
                cn=tp_cn(t.get('type','?'));reason=tp_reason(t.get('type','?'),cfg)
                ele3.append(md('%s %s %d手 %+.0f → %s'%(t.get('time','')[-8:-3],cn,t.get('vol',0),t.get('pnl',0),reason)))
    
    if act3: banner3=['**⚠️ 需要操作**']+act3+['']; [ele3.insert(0,md(b)) for b in banner3]
    elif warn3: banner3=['**⚡ 关注**']+warn3+['']; [ele3.insert(0,md(b)) for b in banner3]
    
    v31_eq=eq(s31,prices);v32_eq=eq(s32,prices);v32b_eq=eq(s32b,prices)
    ele3.append(md(''))
    ele3.append(md('V31 ¥%s | V32 ¥%s | V32b ¥%s | %s'%(
        format(int(v31_eq),','),format(int(v32_eq),','),format(int(v32b_eq),','),now.strftime('%H:%M'))))
    send('扫描③ %s'%now.strftime('%H:%M'),ele3,'red'if act3 else('yellow'if warn3 else'blue'),bool(act3))

# ===== 早报 =====
def morning():
    sv=ls();s28=lv28();s29=lv29();s30=lv30()
    s31=lv31();s32=lv32();s32b=lv32b()
    now=datetime.now();today=now.strftime('%Y-%m-%d')
    wday=['周一','周二','周三','周四','周五','周六','周日'][now.weekday()]
    ele=[md('**%s %s 盘前** | 行情+模型预测'%(today,wday)),hr()]
    
    # 预拉行情
    m_prices={}
    for sk in['lh2609']:
        rt=get_realtime_quote(sk)
        if rt:m_prices[sk]=rt['price']
        else:
            df=fd(S[sk]['code'])
            if df is not None:m_prices[sk]=float(df.iloc[-1]['close'])
    
    # 六个版本 + 各自模型
    ver_config=[
        ('V25 原版',sv,False,'_xgb.pkl'),
        ('V28 动态',s28,True,'_xgb.pkl'),
        ('V29 新模型',s29,True,'_xgb_new.pkl'),
        ('V30 校准版',s30,True,'_xgb_calibrated.pkl'),
        ('V31 基线',s31,False,'_xgb.pkl'),
        ('V32 优化',s32,True,'v31_xgb.pkl'),
        ('V32b 保守',s32b,True,'v31_xgb.pkl'),
    ]
    
    for ver_name,ver_st,is_v28,msuffix in ver_config:
        ele.append(md('━━━ **%s** ━━━'%ver_name))
        for sk in['lh2609']:
            cfg=S[sk];df=fd(cfg['code'])
            if df is None:continue
            price=float(df.iloc[-1]['close']);prev=float(df.iloc[-2]['close'])if len(df)>1 else price
            chg=(price-prev)/prev
            av=[abs(float(df.iloc[i]['high'])-float(df.iloc[i]['low']))for i in range(max(0,len(df)-20),len(df))]
            atr=np.mean(av);ma5=np.mean([float(df.iloc[i]['close'])for i in range(max(0,len(df)-5),len(df))])
            ma20=np.mean([float(df.iloc[i]['close'])for i in range(max(0,len(df)-20),len(df))])
            ft=bf(df,len(df)-1,60)
            if ft is None:continue
            mp=MD+'/'+sk+msuffix
            if not os.path.exists(mp):continue
            m=pickle.load(open(mp,'rb'));prob=float(m.predict_proba(ft.reshape(1,-1))[0][1])
            trend='📈'if price>ma5>ma20 else('📉'if price<ma5<ma20 else'↔️')
            signal='看多'if prob>0.5 else'看空';conf=int((prob if prob>0.5 else 1-prob)*100)
            
            # 模型预测行
            ele.append(md('**%s** %s %.0f(%+.1f%%) | MA5 %.0f MA20 %.0f | 模型%s %d%%'%(
                cfg['cn'],trend,price,chg*100,ma5,ma20,signal,conf)))
            
            # 持仓/明细
            pos=ver_st['positions'].get(sk,[] if is_v28 else None)
            if is_v28:
                if pos:
                    has28,level28,sum28,lines28=analyze_v28(sk,cfg,df,pos,price,atr,prob)
                    ele.append(md(sum28.replace('**%s** '%cfg['cn'],'')))
                    ele.append(md('\n'.join(lines28)))
                else:
                    ele.append(md('⚪ 空仓'))
            else:
                if pos:
                    has,level,summary,lines=analyze(sk,cfg,df,pos,price,atr,prob)
                    ele.append(md(summary.replace('**%s** '%cfg['cn'],'')))
                    ele.append(md('\n'.join(lines)))
                else:
                    ele.append(md('⚪ 空仓'))
    
    v25_eq=eq(sv,m_prices);v28_eq=eq(s28,m_prices)
    v29_eq=eq(s29,m_prices);v30_eq=eq(s30,m_prices)
    ele.append(md('**账户** V25 ¥%s | V28 ¥%s | V29 ¥%s | V30 ¥%s'%(
        format(int(v25_eq),','),format(int(v28_eq),','),
        format(int(v29_eq),','),format(int(v30_eq),','))))
    ok1=send('早报① | '+wday,ele,'blue')
    print('早报①:','✅'if ok1 else'❌')
    
    # 卡2: V31/V32/V32b (超单卡限制,拆开)
    if ver_config[4:]:
        ele2=[md('**%s %s 盘前②**'%(today,wday)),hr()]
        for ver_name,ver_st,is_v28,msuffix in ver_config[4:]:
            ele2.append(md('━━━ **%s** ━━━'%ver_name))
            for sk in['lh2609']:
                cfg=S[sk];df=fd(cfg['code'])
                if df is None:continue
                price=float(df.iloc[-1]['close']);prev=float(df.iloc[-2]['close'])if len(df)>1 else price
                chg=(price-prev)/prev
                av=[abs(float(df.iloc[i]['high'])-float(df.iloc[i]['low']))for i in range(max(0,len(df)-20),len(df))]
                atr=np.mean(av);ma5=np.mean([float(df.iloc[i]['close'])for i in range(max(0,len(df)-5),len(df))])
                ma20=np.mean([float(df.iloc[i]['close'])for i in range(max(0,len(df)-20),len(df))])
                ft=bf(df,len(df)-1,60)
                if ft is None:continue
                mp=MD+'/'+sk+msuffix
                if not os.path.exists(mp):continue
                m=pickle.load(open(mp,'rb'));prob=float(m.predict_proba(ft.reshape(1,-1))[0][1])
                trend='📈'if price>ma5>ma20 else('📉'if price<ma5<ma20 else'↔️')
                signal='看多'if prob>0.5 else'看空';conf=int((prob if prob>0.5 else 1-prob)*100)
                ele2.append(md('**%s** %s %.0f(%+.1f%%) | MA5 %.0f MA20 %.0f | 模型%s %d%%'%(
                    cfg['cn'],trend,price,chg*100,ma5,ma20,signal,conf)))
                pos=ver_st['positions'].get(sk,[] if is_v28 else None)
                if is_v28:
                    if pos:
                        has28,level28,sum28,lines28=analyze_v28(sk,cfg,df,pos,price,atr,prob)
                        ele2.append(md(sum28.replace('**%s** '%cfg['cn'],'')))
                        ele2.append(md('\\n'.join(lines28)))
                    else:ele2.append(md('⚪ 空仓'))
                else:
                    if pos:
                        has,level,summary,lines=analyze(sk,cfg,df,pos,price,atr,prob)
                        ele2.append(md(summary.replace('**%s** '%cfg['cn'],'')))
                        ele2.append(md('\\n'.join(lines)))
                    else:ele2.append(md('⚪ 空仓'))
        ok2=send('早报② | '+wday,ele2,'blue')
        print('早报②:','✅'if ok2 else'❌')

# ===== 晚报 =====
def evening():
    sv=ls();s28=lv28();s29=lv29();s30=lv30()
    s31=lv31();s32=lv32();s32b=lv32b()
    now=datetime.now();today=now.strftime('%Y-%m-%d')
    ele=[md('**%s 收盘**'%today),hr()]
    
    e_prices={}
    for sk in['lh2609']:
        rt=get_realtime_quote(sk)
        if rt:e_prices[sk]=rt['price']
        else:
            df=fd(S[sk]['code'])
            if df is not None:e_prices[sk]=float(df.iloc[-1]['close'])
    
    ver_config=[
        ('V25 原版',sv,False,'_xgb.pkl'),
        ('V28 动态',s28,True,'_xgb.pkl'),
        ('V29 新模型',s29,True,'_xgb_new.pkl'),
        ('V30 校准版',s30,True,'_xgb_calibrated.pkl'),
    ]
    
    for ver_name,ver_st,is_v28,msuffix in ver_config:
        ele.append(md('━━━ **%s** ━━━'%ver_name))
        for sk in['lh2609']:
            cfg=S[sk];df=fd(cfg['code'])
            if df is None:continue
            price=float(df.iloc[-1]['close']);prev=float(df.iloc[-2]['close'])if len(df)>1 else price
            chg=(price-prev)/prev
            av=[abs(float(df.iloc[i]['high'])-float(df.iloc[i]['low']))for i in range(max(0,len(df)-20),len(df))]
            atr=np.mean(av);ma5=np.mean([float(df.iloc[i]['close'])for i in range(max(0,len(df)-5),len(df))])
            ma20=np.mean([float(df.iloc[i]['close'])for i in range(max(0,len(df)-20),len(df))])
            ft=bf(df,len(df)-1,60)
            if ft is None:continue
            mp=MD+'/'+sk+msuffix
            if not os.path.exists(mp):continue
            m=pickle.load(open(mp,'rb'));prob=float(m.predict_proba(ft.reshape(1,-1))[0][1])
            trend='📈'if price>ma5>ma20 else('📉'if price<ma5<ma20 else'↔️')
            signal='看多'if prob>0.5 else'看空';conf=int((prob if prob>0.5 else 1-prob)*100)
            
            ele.append(md('**%s** %s %.0f(%+.1f%%) | MA5 %.0f MA20 %.0f | 模型%s %d%%'%(
                cfg['cn'],trend,price,chg*100,ma5,ma20,signal,conf)))
            
            pos=ver_st['positions'].get(sk,[] if is_v28 else None)
            if is_v28:
                if pos:
                    has28,level28,sum28,lines28=analyze_v28(sk,cfg,df,pos,price,atr,prob)
                    ele.append(md(sum28.replace('**%s** '%cfg['cn'],'')))
                    ele.append(md('\n'.join(lines28)))
                else:
                    ele.append(md('⚪ 空仓'))
            else:
                if pos:
                    has,level,summary,lines=analyze(sk,cfg,df,pos,price,atr,prob)
                    ele.append(md(summary.replace('**%s** '%cfg['cn'],'')))
                    ele.append(md('\n'.join(lines)))
                else:
                    ele.append(md('⚪ 空仓'))
    
    v25_eq=eq(sv,e_prices);v28_eq=eq(s28,e_prices)
    v29_eq=eq(s29,e_prices);v30_eq=eq(s30,e_prices)
    ele.append(md('**账户** V25 ¥%s | V28 ¥%s | V29 ¥%s | V30 ¥%s'%(
        format(int(v25_eq),','),format(int(v28_eq),','),
        format(int(v29_eq),','),format(int(v30_eq),','))))
    send('晚报 | %s'%today,ele,'blue')

def midday():
    morning()

# ===== 周报 =====
def _eq_trend(eq_week):
    """从权益历史生成文字趋势描述"""
    if len(eq_week)<3:return '数据不足'
    vals=[e['equity']for e in eq_week]
    start=vals[0];end=vals[-1];peak=max(vals);trough=min(vals)
    # 找拐点: 上涨/下跌段
    segs=[];cur_dir=None;cur_start=0
    for i in range(1,len(vals)):
        d='up'if vals[i]>vals[i-1]*1.001 else('down'if vals[i]<vals[i-1]*0.999 else'flat')
        if d!=cur_dir:
            if cur_dir and cur_dir!='flat':
                segs.append((cur_dir,i-cur_start))
            cur_dir=d;cur_start=i
    if cur_dir and cur_dir!='flat':segs.append((cur_dir,len(vals)-cur_start))
    if not segs:return '权益基本持平'
    days=['一','二','三','四','五']
    desc=[];pos=0
    for d,count in segs:
        end_pos=pos+count
        ds=days[min(pos,len(days)-1)] if pos<len(days)else''
        de=days[min(end_pos-1,len(days)-1)]if end_pos-1<len(days)else''
        if ds==de:label='周%s'%ds
        else:label='周%s→%s'%(ds,de)
        if d=='up':desc.append('%s📈'%label)
        else:desc.append('%s📉'%label)
        pos=end_pos
    chg=(end-start)/start*100
    return '%s | 波动%.1f%%'%(','.join(desc),chg)

def _weekly_comment(ver_name,net,pct,week_trades,eq_week,model_driven,rule_driven,win_rate):
    """生成版本要点点评"""
    parts=[]
    if net>0:parts.append('✅盈利')
    elif net<0:parts.append('❌亏损')
    else:parts.append('➖持平')
    if week_trades:
        stops=sum(1 for t in week_trades if tp_cn(t.get('type','')or'')=='止损')
        if stops>=2:parts.append('止损%d次偏多'%stops)
        if win_rate>=60:parts.append('胜率%.0f%%不错'%win_rate)
        elif win_rate<40:parts.append('胜率%.0f%%偏低'%win_rate)
        if model_driven>rule_driven:parts.append('模型主导')
        else:parts.append('规则触发多')
    else:parts.append('本周无交易')
    return ' | '.join(parts)

def _eval_models(week_start,week_end):
    """评估各模型本周方向准确率：预测涨跌 vs 实际次日涨跌"""
    # 模型列表: (标签, 模型文件后缀, 使用的版本)
    model_list=[
        ('V25/V28','_xgb.pkl'),
        ('V29','_xgb_new.pkl'),
        ('V30','_xgb_calibrated.pkl'),
    ]
    results=[]
    for sk in['lh2609']:
        cfg=S[sk];df=fd(cfg['code'])
        if df is None or len(df)<80:continue
        # 筛选本周的日线bar
        df['dt']=pd.to_datetime(df['date'])
        week_df=df[(df['dt']>=week_start)&(df['dt']<=week_end)].reset_index(drop=True)
        if len(week_df)<3:continue
        
        for mlbl,msuffix in model_list:
            mp=MD+'/'+sk+msuffix
            if not os.path.exists(mp):continue
            m=pickle.load(open(mp,'rb'))
            
            correct=0;total=0;bull_returns=[];bear_returns=[];bullish_count=0
            for i in range(len(week_df)-1):
                # 在整个df中定位这个bar的索引
                full_idx=df[df['dt']==week_df.iloc[i]['dt']].index
                if len(full_idx)==0:continue
                idx=full_idx[0]
                if idx<65:continue
                ft=bf(df,idx,60)
                if ft is None:continue
                try:
                    prob=float(m.predict_proba(ft.reshape(1,-1))[0][1])
                except:continue
                # 实际次日涨跌
                next_ret=float((df.iloc[idx+1]['close']-df.iloc[idx]['close'])/df.iloc[idx]['close'])
                pred_up=prob>0.5
                actual_up=next_ret>0
                if pred_up==actual_up:correct+=1
                total+=1
                if pred_up:
                    bull_returns.append(next_ret);bullish_count+=1
                else:
                    bear_returns.append(next_ret)
            
            if total==0:continue
            bull_mean=np.mean(bull_returns)if bull_returns else 0
            bear_mean=np.mean(bear_returns)if bear_returns else 0
            results.append({
                'model':mlbl,'symbol':cfg['cn'],
                'accuracy':correct/total*100,
                'bull_mean':bull_mean,'bear_mean':bear_mean,
                'bullish_pct':bullish_count/total*100,
                'total':total,
            })
    return results

def weekly_report():
    """生成四版本周报：卡1总览 + 卡2明细"""
    sv=ls();s28=lv28();s29=lv29();s30=lv30()
    s31=lv31();s32=lv32();s32b=lv32b()
    now=datetime.now()
    today=now.date()
    days_since_monday=today.weekday()
    last_monday=today-timedelta(days=days_since_monday+7)
    last_friday=last_monday+timedelta(days=4)
    week_label='%s~%s'%(last_monday.strftime('%m/%d'),last_friday.strftime('%m/%d'))
    week_start=last_monday.strftime('%Y-%m-%d')
    week_end=last_friday.strftime('%Y-%m-%d')
    
    versions=[
        ('V25 原版',sv),
        ('V28 动态',s28),
        ('V29 新模型',s29),
        ('V30 校准版',s30),
    ]
    
    # 预拉行情
    prices={}
    for sk in['lh2609']:
        rt=get_realtime_quote(sk)
        if rt:prices[sk]=rt['price']
        else:
            df=fd(S[sk]['code'])
            if df is not None:prices[sk]=float(df.iloc[-1]['close'])
    
    # ── 收集所有版本数据 ──
    all_data=[]
    for ver_name,st in versions:
        eq_hist=st.get('equity_history',[])
        trades_all=st.get('trades',[])
        week_trades=[t for t in trades_all if _in_week(t.get('time','')or t.get('exit_time','')or'',week_start,week_end)]
        
        # 权益
        eq_week=[e for e in eq_hist if week_start<=e['time'][:10]<=week_end]
        if eq_week:
            eq_start=eq_week[0]['equity'];eq_end=eq_week[-1]['equity']
            eq_high=max(e['equity']for e in eq_week)
            eq_low=min(e['equity']for e in eq_week)
        else:
            start_equity=300000
            for t in trades_all:
                t_time=t.get('time','')or t.get('exit_time','')
                if t_time and t_time[:10]<week_start:
                    pnl=t.get('pnl',0)or t.get('pnl_amount',0)
                    start_equity+=pnl
            end_equity=eq(st,prices)
            eq_start=start_equity;eq_end=end_equity
            eq_high=max(start_equity,end_equity);eq_low=min(start_equity,end_equity)
        
        net=eq_end-eq_start;pct=net/eq_start*100 if eq_start else 0
        
        # 胜率
        wins=sum(1 for t in week_trades if(t.get('pnl',0)or t.get('pnl_amount',0))>0)
        win_rate=wins/len(week_trades)*100 if week_trades else 0
        
        # 驱动统计
        model_driven=0;rule_driven=0
        for t in week_trades:
            tt=tp_cn(t.get('type','')or('TP'if t.get('exit_type')=='TP'else'?'))
            if tt in('止损','TP','止盈'):rule_driven+=1
            else:model_driven+=1
        
        # 趋势描述
        trend_desc=_eq_trend(eq_week)if eq_week else'无历史数据'
        
        # 要点
        comment=_weekly_comment(ver_name,net,pct,week_trades,eq_week,model_driven,rule_driven,win_rate)
        
        all_data.append({
            'ver':ver_name,'st':st,'eq_start':eq_start,'eq_end':eq_end,
            'eq_high':eq_high,'eq_low':eq_low,'net':net,'pct':pct,
            'week_trades':week_trades,'eq_week':eq_week,
            'model':model_driven,'rule':rule_driven,
            'win_rate':win_rate,'wins':wins,
            'trend':trend_desc,'comment':comment,
        })
    
    # ═══════════════ 卡1: 总览 ═══════════════
    ele1=[md('**📊 周报 %s**'%week_label),md('')]
    
    # 四版本对比总表
    rows=['| 版本 | 周初 | 周末 | 净盈亏 | 收益率 | 胜率 | 要点 |']
    rows.append('|------|------|------|------|------|------|------|')
    for d in all_data:
        net_sign='+'if d['net']>=0 else''
        rows.append('| **%s** | ¥%s | ¥%s | %s¥%s | %+.1f%% | %d/%d(%.0f%%) | %s |'%(
            d['ver'],format(int(d['eq_start']/10000),',')+'万',format(int(d['eq_end']/10000),',')+'万',
            net_sign,format(int(d['net']),','),d['pct'],
            d['wins'],len(d['week_trades']),d['win_rate'],
            d['comment']))
    ele1.append(md('\n'.join(rows)))
    ele1.append(md(''))
    
    # 排名
    ranked=sorted(all_data,key=lambda x:x['pct'],reverse=True)
    ranks=[]
    for i,d in enumerate(ranked):
        icon='🥇'if i==0 else('🥈'if i==1 else('🥉'if i==2 else'  %d.'%(i+1)))
        ranks.append('%s%s %+.1f%%'%(icon,d['ver'][:3],d['pct']))
    ele1.append(md('**排名** %s'%' | '.join(ranks)))
    ele1.append(md(''))
    
    # 各版本趋势
    ele1.append(md('**权益趋势**'))
    for d in all_data:
        arrow='📈'if d['net']>0 else('📉'if d['net']<0 else'➖')
        ele1.append(md('%s **%s** ¥%s→¥%s %+.1f%% | %s'%(
            arrow,d['ver'],
            format(int(d['eq_start']),','),format(int(d['eq_end']),','),
            d['pct'],d['trend'])))
    ele1.append(md(''))
    
    # 模型准确率
    ele1.append(md('**模型准确率（本周日线）**'))
    ele1.append(md('> 计算方法：每日收盘后用当日特征预测次日涨跌方向，对比次日实际涨跌。准确率=方向正确的天数/总天数。看多/看空均收益=按信号方向模拟持仓的次日平均收益率。'))
    acc_data=_eval_models(week_start,week_end)
    if acc_data:
        acc_rows=['| 模型 | 品种 | 方向准确率 | 看多均收益 | 看空均收益 | 信号偏好 |']
        acc_rows.append('|------|------|------|------|------|------|')
        for a in acc_data:
            bias='偏多'if a['bullish_pct']>60 else('偏空'if a['bullish_pct']<40 else'均衡')
            acc_rows.append('| %s | %s | %.0f%% | %+.2f%% | %+.2f%% | %s(%.0f%%) |'%(
                a['model'],a['symbol'],a['accuracy'],
                a['bull_mean']*100,a['bear_mean']*100,
                bias,a['bullish_pct']))
        ele1.append(md('\n'.join(acc_rows)))
        # 一句话解读
        best_acc=max(acc_data,key=lambda x:x['accuracy'])
        worst_acc=min(acc_data,key=lambda x:x['accuracy'])
        ele1.append(md('最佳: %s-%s %.0f%% | 最差: %s-%s %.0f%%'%(
            best_acc['model'],best_acc['symbol'],best_acc['accuracy'],
            worst_acc['model'],worst_acc['symbol'],worst_acc['accuracy'])))
    else:
        ele1.append(md('数据不足，无法评估'))
    ele1.append(md(''))
    
    # 当前持仓快照
    ele1.append(md('**周末持仓**'))
    pos_rows=['| 版本 | 品种 | 方向 | 手数 | 均价 | 浮盈 |']
    pos_rows.append('|------|------|------|------|------|------|')
    for d in all_data:
        pos=d['st'].get('positions',{})
        if not pos:
            pos_rows.append('| %s | — | 空仓 | — | — | — |'%d['ver'])
        else:
            first=True
            for sk,pl in pos.items():
                cfg=S.get(sk,{})
                if isinstance(pl,list):
                    tv=sum(p['vol']for p in pl);dr=pl[0]['dir']
                    ae=sum(p['entry']*p['vol']for p in pl)/tv
                    cur=prices.get(sk,ae)
                    pp=(cur-ae)*tv*cfg['mp']if dr=='LONG'else(ae-cur)*tv*cfg['mp']
                    ps='+'if pp>=0 else''
                    dir_cn='多'if dr=='LONG'else'空'
                    lbl=d['ver']if first else''
                    pos_rows.append('| %s | %s | %s | %d | %.0f | %s%.1f万 |'%(
                        lbl,cfg['cn'],dir_cn,tv,ae,ps,pp/10000))
                else:
                    cur=prices.get(sk,pl['entry'])
                    pp=(cur-pl['entry'])*pl['vol']*cfg['mp']if pl['dir']=='LONG'else(pl['entry']-cur)*pl['vol']*cfg['mp']
                    ps='+'if pp>=0 else''
                    dir_cn='多'if pl['dir']=='LONG'else'空'
                    lbl=d['ver']if first else''
                    pos_rows.append('| %s | %s | %s | %d | %.0f | %s%.1f万 |'%(
                        lbl,cfg['cn'],dir_cn,pl['vol'],pl['entry'],ps,pp/10000))
                first=False
    ele1.append(md('\n'.join(pos_rows)))
    ele1.append(md(''))
    ele1.append(md('生成 %s | 明细见下一张卡'%now.strftime('%m/%d %H:%M')))
    
    send('📊 周报总览 | %s'%week_label,ele1,'blue')
    
    # ═══════════════ 卡2: 明细 ═══════════════
    ele2=[md('**📋 周报明细 | %s**'%week_label),md('')]
    
    for d in all_data:
        week_trades=d['week_trades']
        ele2.append(md('━━━ **%s** ━━━'%d['ver']))
        
        # 统计行
        ele2.append(md('权益 %+.1f%% | 交易%d笔 | 胜率%.0f%%(%d/%d) | 🧠%d ⚙️%d | %s'%(
            d['pct'],len(week_trades),d['win_rate'],d['wins'],len(week_trades),
            d['model'],d['rule'],d['trend'])))
        
        if not week_trades:
            ele2.append(md('本周无交易'))
        else:
            rows=['| 时间 | 品种 | 操作 | 盈亏 | 原因 |']
            rows.append('|------|------|------|------|------|')
            for t in week_trades:
                t_time=(t.get('time','')or t.get('exit_time','')or'')[:16].replace('T',' ')
                if len(t_time)>11:t_time=t_time[5:]  # MM-DD HH:MM
                t_sym=t.get('sym','?');cfg=S.get(t_sym,{})
                cn=cfg.get('cn',t_sym)
                t_type=tp_cn(t.get('type','')or('TP'if t.get('exit_type')=='TP'else'?'))
                t_pnl=t.get('pnl',0)or t.get('pnl_amount',0)
                pnl_str='%+.0f'%t_pnl
                # 原因
                reason=tp_reason(t.get('type','')or('TP'if t.get('exit_type')=='TP'else'?'),cfg)
                if len(reason)>20:reason=reason[:18]+'…'
                rows.append('| %s | %s | %s | %s | %s |'%(t_time,cn,t_type,pnl_str,reason))
            ele2.append(md('\n'.join(rows)))
        
        # 模型对齐
        total_t=len(week_trades)
        if total_t>0:
            ele2.append(md('**模型对齐** 🧠模型%d笔(%.0f%%) ⚙️规则%d笔(%.0f%%)'%(
                d['model'],d['model']/total_t*100,d['rule'],d['rule']/total_t*100)))
        ele2.append(md(''))
    
    ele2.append(md('生成 %s'%now.strftime('%m/%d %H:%M')))
    send('📋 周报明细 | %s'%week_label,ele2,'blue')

def _in_week(ts,ws,we):
    if not ts:return False
    d=ts[:10]
    try:return ws<=d<=we
    except:return False

if __name__=='__main__':
    mode=sys.argv[1]if len(sys.argv)>1 else'scan'
    {'morning':morning,'midday':midday,'evening':evening,'scan':scan,'weekly':weekly_report}[mode]()
