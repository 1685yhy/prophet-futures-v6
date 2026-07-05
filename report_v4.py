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
    for sk in['lh2609','jm2609']:
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
        data[sk]={'cfg':cfg,'df':df,'price':price,'atr':atr,'prob':prob,'prob_new':prob_new}
    
    # ═══ V25 大块 ═══
    ele.append(md('━━━ **V25 原版** ━━━'))
    for sk in['lh2609','jm2609']:
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
    for sk in['lh2609','jm2609']:
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

    # ═══ V29 大块（新模型）═══
    s29=lv29()
    ele.append(md(''))
    ele.append(md('━━━ **V29 新模型** ━━━'))
    for sk in['lh2609','jm2609']:
        if sk not in data:continue
        d=data[sk];cfg=d['cfg'];price=d['price'];atr=d['atr'];prob=d['prob_new']
        pos29=s29['positions'].get(sk,[])
        if pos29:
            has29,level29,sum29,lines29=analyze_v28(sk,cfg,d['df'],pos29,price,atr,prob)
            if level29=='alert':act.append('🔴 V29 %s: %s'%(cfg['cn'],sum29.replace('**','')))
            elif level29=='warn':warn.append('⚠️ V29 %s: %s'%(cfg['cn'],sum29.replace('**','')))
            ele.append(md(sum29))
            ele.append(md('\n'.join(lines29)))
        else:
            s_txt='看多'if prob>0.5 else'看空';conf=int((prob if prob>0.5 else 1-prob)*100)
            ele.append(md('⚪ **%s** 空仓 | 模型%s %d%% | 现价%.0f'%(cfg['cn'],s_txt,conf,price)))
    # V29 成交
    t29=s29.get('trades',[]);t29_today=[t for t in t29 if today in str(t.get('time',''))]
    if t29_today:
        ele.append(md('**今日成交**'))
        for t in t29_today[-5:]:
            cfg=S.get(t.get('sym',''),{})
            cn=tp_cn(t.get('type','?'));reason=tp_reason(t.get('type','?'),cfg)
            ele.append(md('%s %s %d手 %+.0f → %s'%(t.get('time','')[-8:-3],cn,t.get('vol',0),t.get('pnl',0),reason)))

    # Action banner at top
    if act:
        banner=['**⚠️ 需要操作**']+act+['']
        for b in banner:ele.insert(0,md(b))
    elif warn:
        banner=['**⚡ 关注**']+warn+['']
        for b in banner:ele.insert(0,md(b))
    
    prices={sk:d['price']for sk,d in data.items()}
    v25_eq=eq(sv,prices);v28_eq=eq(s28,prices);v29_eq=eq(s29,prices)
    ele.append(md(''))
    ele.append(md('V25 ¥%s | V28 ¥%s | V29 ¥%s | %s'%(format(int(v25_eq),','),format(int(v28_eq),','),format(int(v29_eq),','),now.strftime('%H:%M'))))
    
    color='red'if act else('yellow'if warn else'blue')
    pin=bool(act)
    send('扫描 %s'%now.strftime('%H:%M'),ele,color,pin)

# ===== 早报 =====
def morning():
    sv=ls();s28=lv28();now=datetime.now();today=now.strftime('%Y-%m-%d')
    wday=['周一','周二','周三','周四','周五','周六','周日'][now.weekday()]
    ele=[md('**%s %s 盘前** | MA趋势+模型信号'%(today,wday)),hr()]
    
    # 预拉行情(用于eq浮盈计算)
    m_prices={}
    for sk in['lh2609','jm2609']:
        df=fd(S[sk]['code'])
        if df is not None:m_prices[sk]=float(df.iloc[-1]['close'])
    
    for ver_name,ver_st,is_v28 in[('V25 原版',sv,False),('V28 动态',s28,True)]:
        ele.append(md('━━━ **%s** ━━━'%ver_name))
        for sk in['lh2609','jm2609']:
            cfg=S[sk];df=fd(cfg['code'])
            if df is None:continue
            price=float(df.iloc[-1]['close']);prev=float(df.iloc[-2]['close'])if len(df)>1 else price
            chg=(price-prev)/prev
            av=[abs(float(df.iloc[i]['high'])-float(df.iloc[i]['low']))for i in range(max(0,len(df)-20),len(df))]
            atr=np.mean(av);ma5=np.mean([float(df.iloc[i]['close'])for i in range(max(0,len(df)-5),len(df))])
            ma20=np.mean([float(df.iloc[i]['close'])for i in range(max(0,len(df)-20),len(df))])
            ft=bf(df,len(df)-1,60)
            if ft is None:continue
            mp=MD+'/'+sk+'_xgb.pkl'
            if not os.path.exists(mp):continue
            m=pickle.load(open(mp,'rb'));prob=float(m.predict_proba(ft.reshape(1,-1))[0][1])
            trend='📈'if price>ma5>ma20 else('📉'if price<ma5<ma20 else'↔️')
            
            if is_v28:
                pos28=s28['positions'].get(sk,[])
                if pos28:
                    has28,level28,sum28,lines28=analyze_v28(sk,cfg,df,pos28,price,atr,prob)
                    ele.append(md('**%s** %s %.0f(%+.1f%%) | MA5 %.0f MA20 %.0f'%(
                        cfg['cn'],trend,price,chg*100,ma5,ma20)))
                    ele.append(md(sum28.replace('**%s** '%cfg['cn'],'')))
                    ele.append(md('\n'.join(lines28)))
                else:
                    s_txt='看多'if prob>0.5 else'看空';conf=int((prob if prob>0.5 else 1-prob)*100)
                    ele.append(md('**%s** %s %.0f(%+.1f%%) | 空仓 | 模型%s %d%%'%(
                        cfg['cn'],trend,price,chg*100,s_txt,conf)))
            else:
                pos=ver_st['positions'].get(sk)
                has,level,summary,lines=analyze(sk,cfg,df,pos,price,atr,prob)
                ele.append(md('**%s** %s %.0f(%+.1f%%) | MA5 %.0f MA20 %.0f'%(
                    cfg['cn'],trend,price,chg*100,ma5,ma20)))
                ele.append(md(summary.replace('**%s** '%cfg['cn'],'')))
                ele.append(md('\n'.join(lines)))
    
    v25_eq=eq(sv,m_prices);v28_eq=eq(s28,m_prices)
    ele.append(md('**账户** V25 ¥%s | V28 ¥%s'%(format(int(v25_eq),','),format(int(v28_eq),','))))
    send('早报 | '+wday,ele,'blue')

# ===== 晚报 =====
def evening():
    sv=ls();s28=lv28();now=datetime.now();today=now.strftime('%Y-%m-%d')
    ele=[md('**%s 收盘**'%today),hr()]
    
    # 预拉行情(用于eq浮盈计算)
    e_prices={}
    for sk in['lh2609','jm2609']:
        df=fd(S[sk]['code'])
        if df is not None:e_prices[sk]=float(df.iloc[-1]['close'])
    
    for ver_name,ver_st,is_v28 in[('V25 原版',sv,False),('V28 动态',s28,True)]:
        ele.append(md('━━━ **%s** ━━━'%ver_name))
        for sk in['lh2609','jm2609']:
            cfg=S[sk];df=fd(cfg['code'])
            if df is None:continue
            price=float(df.iloc[-1]['close']);prev=float(df.iloc[-2]['close'])if len(df)>1 else price
            chg=(price-prev)/prev
            av=[abs(float(df.iloc[i]['high'])-float(df.iloc[i]['low']))for i in range(max(0,len(df)-20),len(df))]
            atr=np.mean(av);ma5=np.mean([float(df.iloc[i]['close'])for i in range(max(0,len(df)-5),len(df))])
            ma20=np.mean([float(df.iloc[i]['close'])for i in range(max(0,len(df)-20),len(df))])
            ft=bf(df,len(df)-1,60)
            if ft is None:continue
            mp=MD+'/'+sk+'_xgb.pkl'
            if not os.path.exists(mp):continue
            m=pickle.load(open(mp,'rb'));prob=float(m.predict_proba(ft.reshape(1,-1))[0][1])
            trend='📈'if price>ma5>ma20 else('📉'if price<ma5<ma20 else'↔️')
            signal='看多'if prob>0.5 else'看空';conf=int((prob if prob>0.5 else 1-prob)*100)
            
            if is_v28:
                pos28=s28['positions'].get(sk,[])
                if pos28:
                    has28,level28,sum28,lines28=analyze_v28(sk,cfg,df,pos28,price,atr,prob)
                    ele.append(md('**%s** %s %.0f(%+.1f%%) | 模型%s %d%%'%(
                        cfg['cn'],trend,price,chg*100,signal,conf)))
                    ele.append(md(sum28.replace('**%s** '%cfg['cn'],'')))
                    ele.append(md('\n'.join(lines28)))
                else:
                    ele.append(md('**%s** %s %.0f(%+.1f%%) | 空仓 | 模型%s %d%%'%(
                        cfg['cn'],trend,price,chg*100,signal,conf)))
            else:
                pos=ver_st['positions'].get(sk)
                has,level,summary,lines=analyze(sk,cfg,df,pos,price,atr,prob)
                ele.append(md('**%s** %s %.0f(%+.1f%%) | MA5 %.0f MA20 %.0f | 模型%s %d%%'%(
                    cfg['cn'],trend,price,chg*100,ma5,ma20,signal,conf)))
                ele.append(md(summary.replace('**%s** '%cfg['cn'],'')))
                ele.append(md('\n'.join(lines)))
    
    v25_eq=eq(sv,e_prices);v28_eq=eq(s28,e_prices)
    ele.append(md('**账户** V25 ¥%s | V28 ¥%s'%(format(int(v25_eq),','),format(int(v28_eq),','))))
    send('晚报 | %s'%today,ele,'blue')

def midday():
    morning()

if __name__=='__main__':
    mode=sys.argv[1]if len(sys.argv)>1 else'scan'
    {'morning':morning,'midday':midday,'evening':evening,'scan':scan}[mode]()
