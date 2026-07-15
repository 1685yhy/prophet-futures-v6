#!/usr/bin/env python3
"""V29 vs V30 — Walk-Forward 回测
  V29: 方向预测 + 固定仓位
  V30: 方向预测 + 波动率动态调仓（高波半仓/低波双倍）
"""
import sys, os, numpy as np, xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from datetime import datetime, timedelta
import akshare as ak
import warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from realtime_data import SYMBOL_MAP, build_features

CASH = 300000; MIN_T = 200; RETRAIN = 30; WIN = 60

def daily(code, n=800):
    e=datetime.now();s=e-timedelta(days=n+50)
    try:
        df=ak.futures_main_sina(symbol=code,start_date=s.strftime("%Y%m%d"),end_date=e.strftime("%Y%m%d"))
        df.columns=["date","open","high","low","close","volume","oi","settle"]
        for c in["open","high","low","close","volume","oi"]:df[c]=df[c].apply(float)
        return df.dropna(subset=["close"]).reset_index(drop=True)
    except:return None

def metrics(eq,tr):
    eq=np.array(eq);ret=(eq[-1]-CASH)/CASH
    me=np.maximum.accumulate(eq);dd=abs((eq-me)/(me+1)).min()
    rs=np.diff(eq)/(eq[:-1]+1);rs=rs[rs!=0]
    sh=np.mean(rs)/np.std(rs)*np.sqrt(252)if len(rs)>1 and np.std(rs)>0 else 0
    n=len(tr);wt=[t for t in tr if t["pnl"]>0];lt=[t for t in tr if t["pnl"]<=0]
    wr=len(wt)/n if n>0 else 0
    aw=np.mean([t["pnl"]for t in wt])if wt else 0
    al=np.mean([t["pnl"]for t in lt])if lt else 0
    return{"total_return":ret,"max_dd":dd,"sharpe":sh,"n_trades":n,"win_rate":wr,"avg_win":aw,"avg_loss":al}

def simulate_v29(df):
    cash=CASH;eq=[cash];pos=0;entry=0;sp=0;trades=[];lr=0;m=None
    for t in range(MIN_T,len(df)-1):
        if t-lr>=RETRAIN:
            X,y=[],[]
            for i in range(WIN,t-1):
                f=build_features(df,i,WIN)
                if f is None:continue
                X.append(f);y.append(1 if df.iloc[i+1]["close"]>df.iloc[i]["close"] else 0)
            if len(X)>=100:
                m=xgb.XGBClassifier(n_estimators=100,max_depth=4,learning_rate=0.05,eval_metric="logloss",verbosity=0,random_state=42)
                m.fit(np.array(X,dtype=np.float32),np.array(y));lr=t
        if m is None:eq.append(eq[-1]);continue
        f=build_features(df,t,WIN)
        if f is None:eq.append(eq[-1]);continue
        p=float(m.predict_proba(f.reshape(1,-1))[0][1])
        pr=df.iloc[t]["close"]
        av=[abs(df.iloc[i]["high"]-df.iloc[i]["low"])for i in range(max(0,t-20),t+1)]
        at=np.mean(av)if av else pr*0.005
        if pos!=0:
            if pos==1:
                trl=pr-at*1.5
                if trl>sp:sp=trl
                hit=pr<=sp
            else:
                trl=pr+at*1.5
                if trl<sp:sp=trl
                hit=pr>=sp
            rev=(pos==1 and p<0.4)or(pos==-1 and p>0.6)
            if hit or rev:
                pnl=(pr-entry)*pos;trades.append({"pnl":pnl});cash+=pnl;pos=0;eq.append(cash);continue
            eq.append(cash+(pr-entry)*pos);continue
        if p>0.55:pos=1;entry=pr;sp=pr-at*1.5;eq.append(cash)
        elif p<0.45:pos=-1;entry=pr;sp=pr+at*1.5;eq.append(cash)
        else:eq.append(cash)
    if pos!=0:trades.append({"pnl":(float(df.iloc[-1]["close"])-entry)*pos})
    return metrics(eq,trades)

def simulate_v30(df):
    cash=CASH;eq=[cash];pos=0;entry=0;sp=0;trades=[];lr=0;m=None;vm=None
    for t in range(MIN_T,len(df)-1):
        if t-lr>=RETRAIN:
            X,y=[],[]
            for i in range(WIN,t-1):
                f=build_features(df,i,WIN)
                if f is None:continue
                X.append(f);y.append(1 if df.iloc[i+1]["close"]>df.iloc[i]["close"] else 0)
            if len(X)>=100:
                m=xgb.XGBClassifier(n_estimators=100,max_depth=4,learning_rate=0.05,eval_metric="logloss",verbosity=0,random_state=42)
                m.fit(np.array(X,dtype=np.float32),np.array(y))
            # 波动率模型
            Xv,yv=[],[]
            for i in range(WIN,t-1):
                f=build_features(df,i,WIN)
                if f is None:continue
                rs=max(0,i-20)
                rng=[abs(df.iloc[j]["high"]-df.iloc[j]["low"])for j in range(rs,i+1)]
                ar=np.mean(rng)if rng else 0
                if ar<=0:continue
                hi=1 if abs(df.iloc[i+1]["high"]-df.iloc[i+1]["low"])>ar*1.5 else 0
                Xv.append(f);yv.append(hi)
            if len(Xv)>=80:
                pr_pos=sum(yv)/len(yv)
                vm=xgb.XGBClassifier(n_estimators=200,max_depth=3,learning_rate=0.05,subsample=0.8,colsample_bytree=0.8,reg_alpha=0.1,reg_lambda=1.0,scale_pos_weight=(1-pr_pos)/pr_pos if pr_pos>0 else 1,eval_metric="logloss",verbosity=0,random_state=42)
                vm.fit(np.array(Xv,dtype=np.float32),np.array(yv))
            lr=t
        if m is None:eq.append(eq[-1]);continue
        f=build_features(df,t,WIN)
        if f is None:eq.append(eq[-1]);continue
        p=float(m.predict_proba(f.reshape(1,-1))[0][1])
        pr=df.iloc[t]["close"]
        av=[abs(df.iloc[i]["high"]-df.iloc[i]["low"])for i in range(max(0,t-20),t+1)]
        at=np.mean(av)if av else pr*0.005
        # 波动率仓位系数
        vp=0.5
        if vm is not None:
            vf=build_features(df,t,WIN)
            if vf is not None:vp=float(vm.predict_proba(vf.reshape(1,-1))[0][1])
        mult=0.5 if vp>0.6 else(2.0 if vp<0.35 else 1.0)
        if pos!=0:
            if pos>0:
                trl=pr-at*1.5
                if trl>sp:sp=trl
                hit=pr<=sp
            else:
                trl=pr+at*1.5
                if trl<sp:sp=trl
                hit=pr>=sp
            rev=(pos>0 and p<0.4)or(pos<0 and p>0.6)
            if vp>0.6 and abs(pos)>1:
                pnl=(pr-entry)*(1 if pos>0 else -1);trades.append({"pnl":pnl});cash+=pnl
                pos=pos//2
                if pos==0:eq.append(cash);continue
            if hit or rev:
                pnl=(pr-entry)*pos;trades.append({"pnl":pnl});cash+=pnl;pos=0;eq.append(cash);continue
            eq.append(cash+(pr-entry)*pos);continue
        if p>0.55:
            lots=max(1,int(mult));pos=lots;entry=pr;sp=pr-at*1.5;eq.append(cash)
        elif p<0.45:
            lots=max(1,int(mult));pos=-lots;entry=pr;sp=pr+at*1.5;eq.append(cash)
        else:eq.append(cash)
    if pos!=0:trades.append({"pnl":(float(df.iloc[-1]["close"])-entry)*pos})
    return metrics(eq,trades)

def main():
    print("V29(方向固定仓) vs V30(方向+波动调仓) Walk-Forward")
    for sk in["lh2609","jm2609"]:
        nm=SYMBOL_MAP[sk]["name"];cd=SYMBOL_MAP[sk].get("daily_code","LH0"if"lh"in sk else"JM0")
        df=daily(cd,800)
        if df is None:print(f"{nm}: 数据缺失");continue
        print(f"\n{'='*50}\n  {nm}")
        r1=simulate_v29(df);r2=simulate_v30(df)
        print(f"  V29: {r1['total_return']:+.2%} DD{r1['max_dd']:.1%} Sh{r1['sharpe']:.2f} {r1['n_trades']}笔 {r1['win_rate']:.0%}")
        print(f"  V30: {r2['total_return']:+.2%} DD{r2['max_dd']:.1%} Sh{r2['sharpe']:.2f} {r2['n_trades']}笔 {r2['win_rate']:.0%}")
        d=r2["total_return"]-r1["total_return"]
        print(f"  差: {d:+.2%} {'✅'if d>0 else'➖'if abs(d)<0.001 else'❌'}")

if __name__=="__main__":main()
