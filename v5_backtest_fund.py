#!/usr/bin/env python3
"""
Prophet Futures v5 — Comprehensive Backtest System
===================================================
V31: Simple baseline (entry-only, fixed stop/tp)
V32: V28 dynamic strategy (add/reduce/reverse/trail)
V33: Ensemble (XGB+LGBM+CB voting)
V34: Prediction validity window test
V35: Stop-loss/take-profit grid optimization

CRITICAL: Single continuous capital run (¥300,000), NEVER reset.
Positions carry through retraining points.
"""
import sys, os, json, time, pickle
import numpy as np
import pandas as pd
from datetime import datetime

# ── Paths ──
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
RESULTS_PATH = os.path.join(BASE_DIR, 'v5_results.json')
os.makedirs(MODEL_DIR, exist_ok=True)

# ── Contract Specs ──
SPECS = {
    'LH': {'code':'LH0','sym':'lh2609','name':'LH 生猪','multiplier':16,'cost':0.0006,'margin_rate':0.15,
           'max_pos':6,'max_total':12,'atr_stop':1.5,'rr':4.0,'min_hold':3,
           'add_conf':0.65,'add_atr':2.0,'reduce_conf':0.55,'reverse_conf':0.35,
           'trail_atr':2.0,'be_atr':1.0,'xgb_params':{'n_est':200,'depth':5,'lr':0.05}},
    'JM': {'code':'JM0','sym':'jm2609','name':'JM 焦煤','multiplier':60,'cost':0.0011,'margin_rate':0.15,
           'max_pos':4,'max_total':8,'atr_stop':2.0,'rr':3.5,'min_hold':5,
           'add_conf':0.65,'add_atr':2.5,'reduce_conf':0.55,'reverse_conf':0.30,
           'trail_atr':3.0,'be_atr':2.0,'xgb_params':{'n_est':100,'depth':4,'lr':0.03}},
}

START_CAPITAL = 300_000.0
FEAT_NAMES = ['开盘缺口','|缺口|','1日涨跌','3日涨跌','5日涨跌','10日涨跌','20日涨跌',
              'MA5偏离','MA10偏离','MA20偏离','MA60偏离','波动率','日内振幅','量比',
              '持仓比','MACD','RSI','布林带','价格/1000']

# ═══════════════════════════════════════════════════════════════
# PHASE 1: Data Fetch
# ═══════════════════════════════════════════════════════════════

def fetch_data():
    """Fetch LH and JM daily data via akshare"""
    import akshare as ak
    data = {}
    for name, spec in [('LH',SPECS['LH']), ('JM',SPECS['JM'])]:
        print(f"  📡 获取 {spec['name']} ({spec['code']})...")
        try:
            df = ak.futures_main_sina(symbol=spec['code'])
            # Handle Chinese column names from newer akshare
            cn_map = {'日期':'date','开盘价':'open','最高价':'high','最低价':'low',
                      '收盘价':'close','成交量':'volume','持仓量':'oi','动态结算价':'settle'}
            if df.columns[0] in cn_map:
                df = df.rename(columns=cn_map)
            else:
                df.columns = ['date','open','high','low','close','volume','oi','settle']
            for c in ['open','high','low','close','volume','oi']:
                df[c] = pd.to_numeric(df[c], errors='coerce')
            df = df.dropna(subset=['close']).reset_index(drop=True)
            df['date'] = pd.to_datetime(df['date'])
            print(f"    ✅ {len(df)}行  {df.iloc[0]['date'].strftime('%Y-%m-%d')} → {df.iloc[-1]['date'].strftime('%Y-%m-%d')}")
            data[name] = df
        except Exception as e:
            print(f"    ❌ {e}")
    return data

# ═══════════════════════════════════════════════════════════════
# PHASE 2: Feature Engineering
# ═══════════════════════════════════════════════════════════════

def build_features(df, idx, window=60, sym='LH'):
    """Build 19-dim feature vector matching realtime_data.py exactly"""
    if idx < window + 5:
        return None
    w = df.iloc[idx - window:idx + 1]
    c = w['close'].values.astype(float)
    o = w['open'].values.astype(float)
    h = w['high'].values.astype(float)
    l_low = w['low'].values.astype(float)
    v = w['volume'].values.astype(float)
    oi_v = w['oi'].values.astype(float)
    f = []
    # 1-2: open gap
    if idx >= 1:
        gap = float((o[-1] - c[-2]) / c[-2])
        f.append(gap)
        f.append(abs(gap))
    else:
        f.extend([0.0, 0.0])
    # 3-7: returns
    for lag in [1, 3, 5, 10, 20]:
        f.append(float((c[-1] - c[-lag - 1]) / c[-lag - 1] if len(c) > lag else 0))
    # 8-11: MA deviations
    for p in [5, 10, 20, 60]:
        ma = np.mean(c[-min(p, len(c)):])
        f.append(float((c[-1] - ma) / ma))
    # 12: volatility
    f.append(float(np.std(c[-20:]) / np.mean(c[-20:])))
    # 13: intraday range
    f.append(float((h[-1] - l_low[-1]) / c[-1]))
    # 14: volume ratio
    vma = np.mean(v[-20:]) if np.mean(v[-20:]) > 0 else 1.0
    f.append(float(v[-1] / vma))
    # 15: OI ratio
    oi_mean = np.mean(oi_v[-20:])
    f.append(float(oi_v[-1] / oi_mean) if len(oi_v) >= 20 and oi_mean > 0 else 1.0)
    # 16: MACD
    ema12 = c[-1]; ema26 = c[-1]
    for j in range(len(c)-2, -1, -1):
        ema12 = (2/13)*c[j] + (11/13)*ema12
        ema26 = (2/27)*c[j] + (25/27)*ema26
    f.append(float((ema12 - ema26) / c[-1]))
    # 17: RSI
    dd = np.diff(c[-15:])
    g = float(dd[dd > 0].sum()) if len(dd[dd > 0]) > 0 else 0
    lo = float(abs(dd[dd < 0].sum())) if len(dd[dd < 0]) > 0 else 1e-10
    f.append(float(100 - 100/(1 + g/lo) if lo > 0 else 50))
    # 18: Bollinger
    bb_std = np.std(c[-20:]); m20 = np.mean(c[-20:])
    f.append(float((c[-1] - m20) / (2*bb_std + 1e-10)))
    # 19: price level
    f.append(float(c[-1] / 1000.0))
    # === 基本面特征(精选3维:现货指数+猪粮比+周指) — 仅LH ===
    if sym != 'LH':
        return np.array(f, dtype=np.float32)
    try:
        import akshare, requests
        from datetime import datetime, timedelta
        cur_date = pd.to_datetime(df.iloc[idx]['date'])
        cur_close = float(df.iloc[idx]['close'])
        if not hasattr(build_features, '_fd'):
            idx_df = akshare.index_hog_spot_price()
            idx_df['date'] = pd.to_datetime(idx_df['日期'])
            rt = akshare.futures_hog_supply(symbol='猪粮比价')
            rt['date'] = pd.to_datetime(rt['date'])
            r = requests.post('https://xt.yangzhu.vip/data/getmapdata',
                params={'ptype':'6','areno':'-1'}, timeout=10,
                headers={'User-Agent':'Mozilla/5.0'})
            wr = r.json()['data']; wd_d=[]; wv_d=[]
            for row in wr:
                yr=int(row[0][:4]); wk=int(row[0].split('第')[1].split('周')[0])
                wd_d.append(datetime(yr,1,1)+timedelta(weeks=wk-1,days=3))
                wv_d.append(float(row[1]))
            wkdf = pd.DataFrame({'date':wd_d,'value':wv_d})
            build_features._fd = (idx_df, rt, wkdf)
        idx_df, rt, wkdf = build_features._fd
        def gv_d(data, col, cd, db=1):
            s = data[data['date'] <= cd]
            if len(s) < db: return None
            return float(s[col].iloc[-db])
        # 3维全部相对化: 期现价差 + 猪粮比20日Z + 周指4周变化率
        ia = gv_d(idx_df, '成交均价', cur_date)
        f.append((ia*1000 - cur_close)/cur_close if ia and cur_close>0 else 0.0)
        # 猪粮比 20期Z-score(水平漂移免疫)
        rt_hist = rt[rt['date'] <= cur_date]['value'].values[-20:]
        if len(rt_hist) >= 5:
            m_, s_ = float(np.mean(rt_hist)), float(np.std(rt_hist))
            f.append((float(rt_hist[-1]) - m_) / s_ if s_ > 1e-6 else 0.0)
        else:
            f.append(0.0)
        # 周指 4周变化率
        wk_hist = wkdf[wkdf['date'] <= cur_date]['value'].values[-5:]
        if len(wk_hist) >= 2:
            f.append((float(wk_hist[-1]) - float(wk_hist[0])) / max(abs(float(wk_hist[0])), 0.01))
        else:
            f.append(0.0)
    except:
        f.extend([0.0]*3)
    return np.array(f, dtype=np.float32)

def compute_atr(df, idx, period=20):
    """Compute ATR at given index"""
    if idx < period:
        return None
    trs = []
    for i in range(idx - period + 1, idx + 1):
        h = float(df.iloc[i]['high']); l = float(df.iloc[i]['low'])
        trs.append(abs(h - l))
    return np.mean(trs)

# ═══════════════════════════════════════════════════════════════
# PHASE 3: Model Training
# ═══════════════════════════════════════════════════════════════

def train_xgb(X, y, params):
    """Train XGBoost classifier"""
    import xgboost as xgb
    model = xgb.XGBClassifier(
        n_estimators=params['n_est'], max_depth=params['depth'],
        learning_rate=params['lr'], subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=1, verbosity=0)
    model.fit(X, y)
    return model

def train_lgb(X, y):
    """Train LightGBM classifier"""
    import lightgbm as lgb
    model = lgb.LGBMClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
        verbose=-1, force_col_wise=True)
    model.fit(X, y)
    return model

def train_cb(X, y):
    """Train CatBoost classifier"""
    import catboost as cb
    model = cb.CatBoostClassifier(
        iterations=100, depth=4, learning_rate=0.05,
        random_seed=42, verbose=False, allow_writing_files=False)
    model.fit(X, y)
    return model

# ═══════════════════════════════════════════════════════════════
# PHASE 4: Backtest Engines
# ═══════════════════════════════════════════════════════════════

class ContinuousBacktest:
    """Single continuous capital run for LH+JM combined backtest."""
    
    def __init__(self, lh_df, jm_df, initial_capital=START_CAPITAL):
        self.lh_df = lh_df
        self.jm_df = jm_df
        self.initial_capital = initial_capital
        
        # Align dates
        all_dates = sorted(set(
            pd.to_datetime(self.lh_df['date']).dt.strftime('%Y-%m-%d').values
        ) | set(
            pd.to_datetime(self.jm_df['date']).dt.strftime('%Y-%m-%d').values
        ))
        self.all_dates = all_dates
        
        self.lh_date_to_idx = {}
        for i, row in self.lh_df.iterrows():
            d = pd.to_datetime(row['date']).strftime('%Y-%m-%d')
            self.lh_date_to_idx[d] = i
        
        self.jm_date_to_idx = {}
        for i, row in self.jm_df.iterrows():
            d = pd.to_datetime(row['date']).strftime('%Y-%m-%d')
            self.jm_date_to_idx[d] = i
        
        # Precompute features for all indices
        print("  🔄 预计算特征...")
        self._precompute_features()
        
        # Current state
        self.cash = initial_capital
        self.positions = []  # list of position dicts
        self.trades = []     # list of trade dicts
        self.equity_curve = [(0, initial_capital)]  # (date_idx, equity)
    
    def _precompute_features(self):
        """Precompute features and labels for both instruments"""
        self.feats = {'LH': {}, 'JM': {}}
        self.labels = {'LH': {}, 'JM': {}}
        
        for name, df, df_dates in [('LH', self.lh_df, self.lh_date_to_idx),
                                    ('JM', self.jm_df, self.jm_date_to_idx)]:
            for date_str, idx in df_dates.items():
                feats = build_features(df, idx, 60, sym=name)
                if feats is not None and idx + 1 < len(df):
                    self.feats[name][idx] = feats
                    self.labels[name][idx] = 1 if float(df.iloc[idx+1]['close']) > float(df.iloc[idx]['close']) else 0
    
    def get_available_feat_indices(self, name):
        """Get sorted list of indices that have features"""
        return sorted(self.feats[name].keys())
    
    def train_model_at(self, name, train_indices):
        """Train XGBoost model on given indices"""
        df = self.lh_df if name == 'LH' else self.jm_df
        params = SPECS[name]['xgb_params']
        
        X_list, y_list = [], []
        for idx in train_indices:
            if idx in self.feats[name] and idx in self.labels[name]:
                X_list.append(self.feats[name][idx])
                y_list.append(self.labels[name][idx])
        
        if len(X_list) < 50:
            return None
        
        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list)
        return train_xgb(X, y, params)
    
    def train_ensemble_at(self, name, train_indices):
        """Train XGB+LGB+CB ensemble"""
        df = self.lh_df if name == 'LH' else self.jm_df
        params = SPECS[name]['xgb_params']
        
        X_list, y_list = [], []
        for idx in train_indices:
            if idx in self.feats[name] and idx in self.labels[name]:
                X_list.append(self.feats[name][idx])
                y_list.append(self.labels[name][idx])
        
        if len(X_list) < 50:
            return None
        
        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list)
        
        models = {
            'xgb': train_xgb(X, y, params),
            'lgb': train_lgb(X, y),
            'cb': train_cb(X, y),
        }
        return models
    
    def predict(self, models, name, idx):
        """Get prediction probability and direction"""
        if idx not in self.feats[name]:
            return None
        
        feats = self.feats[name][idx].reshape(1, -1)
        
        if isinstance(models, dict):
            # Ensemble: all three must agree
            votes = []
            for k, m in models.items():
                try:
                    p = float(m.predict_proba(feats)[0][1])
                    votes.append(p)
                except:
                    votes.append(0.5)
            prob = np.mean(votes)
            # All agree check
            directions = [1 if v > 0.5 else 0 for v in votes]
            all_agree = len(set(directions)) == 1
            direction = 'LONG' if prob > 0.5 else 'SHORT'
            confidence = prob if prob > 0.5 else (1 - prob)
            return {'prob': prob, 'direction': direction, 'confidence': confidence, 'all_agree': all_agree}
        else:
            # Single model
            try:
                prob = float(models.predict_proba(feats)[0][1])
            except:
                return None
            direction = 'LONG' if prob > 0.5 else 'SHORT'
            confidence = prob if prob > 0.5 else (1 - prob)
            return {'prob': prob, 'direction': direction, 'confidence': confidence}
    
    def _position_margin(self, pos):
        """Calculate margin for a position"""
        return pos['vol'] * pos['entry'] * SPECS[pos['name']]['multiplier'] * SPECS[pos['name']]['margin_rate']
    
    def _total_margin(self):
        """Total margin of all active positions"""
        return sum(self._position_margin(p) for p in self.positions)
    
    def _available_cash(self):
        """Available cash for new positions"""
        return self.cash - self._total_margin()
    
    def _close_position(self, pos, exit_price, exit_type, bar_idx):
        """Close a position and realize PnL"""
        name = pos['name']
        spec = SPECS[name]
        mult = spec['multiplier']
        cost = spec['cost']
        
        if pos['direction'] == 'LONG':
            pnl = (exit_price - pos['entry']) * pos['vol'] * mult
        else:
            pnl = (pos['entry'] - exit_price) * pos['vol'] * mult
        
        commission = pos['entry'] * pos['vol'] * mult * cost * 2
        
        self.cash += pnl - commission
        
        trade = {
            'name': name, 'direction': pos['direction'],
            'entry': pos['entry'], 'exit': exit_price,
            'vol': pos['vol'], 'pnl_abs': pnl - commission,
            'pnl_pct': (pnl - commission) / (pos['entry'] * pos['vol'] * mult),
            'bars': bar_idx - pos['entry_bar'],
            'type': exit_type,
            'date': bar_idx,
        }
        self.trades.append(trade)
        self.positions.remove(pos)
    
    def _process_v31_positions(self, bar_idx):
        """V31: Simple fixed stop/tp — check stops and targets"""
        surviving = []
        for pos in list(self.positions):
            name = pos['name']
            df = self.lh_df if name == 'LH' else self.jm_df
            spec = SPECS[name]
            
            if bar_idx >= len(df):
                surviving.append(pos)
                continue
            
            row = df.iloc[bar_idx]
            high = float(row['high']); low = float(row['low'])
            
            if pos['direction'] == 'LONG':
                if low <= pos['stop']:
                    self._close_position(pos, pos['stop'], 'STOP', bar_idx)
                elif high >= pos['tp']:
                    self._close_position(pos, pos['tp'], 'TP', bar_idx)
                else:
                    surviving.append(pos)
            else:
                if high >= pos['stop']:
                    self._close_position(pos, pos['stop'], 'STOP', bar_idx)
                elif low <= pos['tp']:
                    self._close_position(pos, pos['tp'], 'TP', bar_idx)
                else:
                    surviving.append(pos)
        
        self.positions = surviving
    
    def _process_v32_positions(self, bar_idx, predictions):
        """V32: Dynamic add/reduce/reverse/trail"""
        surviving = []
        for pos in list(self.positions):
            name = pos['name']
            df = self.lh_df if name == 'LH' else self.jm_df
            spec = SPECS[name]
            
            if bar_idx >= len(df):
                surviving.append(pos)
                continue
            
            row = df.iloc[bar_idx]
            price = float(row['close']); high = float(row['high']); low = float(row['low'])
            atr = compute_atr(df, bar_idx, 20)
            if atr is None:
                surviving.append(pos)
                continue
            
            bars_held = bar_idx - pos['entry_bar']
            
            # Current PnL
            if pos['direction'] == 'LONG':
                pnl_pct = (price - pos['entry']) / pos['entry']
                pnl_atr = pnl_pct * pos['entry'] / atr if atr > 0 else 0
                hard_stop = price - atr * spec['atr_stop']
                
                # Trail stop updates
                trail = pos.get('trail', pos['stop'])
                if pnl_atr > spec['trail_atr']:
                    trail = max(trail, price - atr * (spec['atr_stop'] - 0.3))
                if pnl_atr > spec['be_atr']:
                    trail = max(trail, pos['entry'])
                effective_stop = max(hard_stop, trail)
                
                # Get current prediction for this instrument
                pred = predictions.get(name, {})
                cur_dir = pred.get('direction', 'LONG') if pred else 'LONG'
                cur_conf = pred.get('confidence', 0.5) if pred else 0.5
                cur_prob = pred.get('prob', 0.5) if pred else 0.5
                
                should_reduce = (cur_dir == 'LONG' and cur_conf < spec['reduce_conf'] 
                                and bars_held >= spec['min_hold'])
                should_reverse = (cur_prob < spec['reverse_conf'] 
                                 and bars_held >= spec['min_hold'])
                
                # Check stop
                if low <= effective_stop:
                    self._close_position(pos, effective_stop, 'TRAIL_STOP', bar_idx)
                elif should_reverse:
                    self._close_position(pos, price, 'REVERSE', bar_idx)
                elif should_reduce and pos['vol'] > 1:
                    # Close half
                    rv = pos['vol'] // 2
                    mult = spec['multiplier']; cost = spec['cost']
                    pnl = (price - pos['entry']) * rv * mult
                    commission = pos['entry'] * rv * mult * cost * 2
                    self.cash += pnl - commission
                    trade = {
                        'name': name, 'direction': pos['direction'],
                        'entry': pos['entry'], 'exit': price,
                        'vol': rv, 'pnl_abs': pnl - commission,
                        'pnl_pct': (pnl - commission) / (pos['entry'] * rv * mult),
                        'bars': bars_held, 'type': 'REDUCE', 'date': bar_idx,
                    }
                    self.trades.append(trade)
                    pos['vol'] -= rv
                    pos['trail'] = trail
                    surviving.append(pos)
                else:
                    pos['trail'] = trail
                    surviving.append(pos)
            else:
                # SHORT
                pnl_pct = (pos['entry'] - price) / pos['entry']
                pnl_atr = pnl_pct * pos['entry'] / atr if atr > 0 else 0
                hard_stop = price + atr * spec['atr_stop']
                
                trail = pos.get('trail', pos['stop'])
                if pnl_atr > spec['trail_atr']:
                    trail = min(trail, price + atr * (spec['atr_stop'] - 0.3))
                if pnl_atr > spec['be_atr']:
                    trail = min(trail, pos['entry'])
                effective_stop = min(hard_stop, trail)
                
                pred = predictions.get(name, {})
                cur_dir = pred.get('direction', 'SHORT') if pred else 'SHORT'
                cur_conf = pred.get('confidence', 0.5) if pred else 0.5
                cur_prob = pred.get('prob', 0.5) if pred else 0.5
                
                should_reduce = (cur_dir == 'SHORT' and cur_conf < spec['reduce_conf'] 
                                and bars_held >= spec['min_hold'])
                should_reverse = (cur_prob > (1 - spec['reverse_conf']) 
                                 and bars_held >= spec['min_hold'])
                
                if high >= effective_stop:
                    self._close_position(pos, effective_stop, 'TRAIL_STOP', bar_idx)
                elif should_reverse:
                    self._close_position(pos, price, 'REVERSE', bar_idx)
                elif should_reduce and pos['vol'] > 1:
                    rv = pos['vol'] // 2
                    mult = spec['multiplier']; cost = spec['cost']
                    pnl = (pos['entry'] - price) * rv * mult
                    commission = pos['entry'] * rv * mult * cost * 2
                    self.cash += pnl - commission
                    trade = {
                        'name': name, 'direction': pos['direction'],
                        'entry': pos['entry'], 'exit': price,
                        'vol': rv, 'pnl_abs': pnl - commission,
                        'pnl_pct': (pnl - commission) / (pos['entry'] * rv * mult),
                        'bars': bars_held, 'type': 'REDUCE', 'date': bar_idx,
                    }
                    self.trades.append(trade)
                    pos['vol'] -= rv
                    pos['trail'] = trail
                    surviving.append(pos)
                else:
                    pos['trail'] = trail
                    surviving.append(pos)
        
        self.positions = surviving
    
    def _try_entry_v31(self, name, bar_idx, predictions, atr_stop_mult=None, rr_mult=None):
        """V31: Simple entry"""
        spec = SPECS[name]
        df = self.lh_df if name == 'LH' else self.jm_df
        atr_s = atr_stop_mult if atr_stop_mult is not None else spec['atr_stop']
        rr = rr_mult if rr_mult is not None else spec['rr']
        
        if bar_idx >= len(df):
            return
        
        row = df.iloc[bar_idx]
        price = float(row['close']); high = float(row['high']); low = float(row['low'])
        atr = compute_atr(df, bar_idx, 20)
        if atr is None or price <= 0:
            return
        
        pred = predictions.get(name, {})
        if not pred:
            return
        
        prob = pred['prob']; direction = pred['direction']; conf = pred['confidence']
        
        # Entry conditions
        _ec = spec.get('entry_conf', 0.55)
        if not (conf > _ec and ((direction == 'LONG' and prob > _ec) or (direction == 'SHORT' and prob < 1 - _ec))):
            return
        
        # Position sizing
        atr_pct = atr / price
        if atr_pct < 0.01: lev = 3.0
        elif atr_pct < 0.02: lev = 2.0
        elif atr_pct < 0.03: lev = 1.5
        else: lev = 0.5
        
        ps = max(1, int(lev * (spec['max_pos'] // 2))) if lev > 0 else 0
        if ps <= 0:
            return
        
        # Check margin
        mult = spec['multiplier']
        margin = ps * price * mult * spec['margin_rate']
        if self._available_cash() < margin:
            return
        
        # Max position check per instrument
        inst_positions = [p for p in self.positions if p['name'] == name]
        inst_lots = sum(p['vol'] for p in inst_positions)
        if inst_lots + ps > spec['max_total']:
            ps = spec['max_total'] - inst_lots
            if ps <= 0:
                return
            margin = ps * price * mult * spec['margin_rate']
            if self._available_cash() < margin:
                return
        
        sd = atr * atr_s
        if direction == 'LONG':
            stop = price - sd
            tp = price + sd * rr
            if low > stop:
                # Check if no conflicting positions
                if not inst_positions or inst_positions[0]['direction'] == 'LONG':
                    self.positions.append({
                        'name': name, 'direction': 'LONG',
                        'entry': price, 'stop': stop, 'tp': tp,
                        'vol': ps, 'entry_bar': bar_idx,
                    })
        else:
            stop = price + sd
            tp = price - sd * rr
            if high < stop:
                if not inst_positions or inst_positions[0]['direction'] == 'SHORT':
                    self.positions.append({
                        'name': name, 'direction': 'SHORT',
                        'entry': price, 'stop': stop, 'tp': tp,
                        'vol': ps, 'entry_bar': bar_idx,
                    })
    
    def _try_entry_v32(self, name, bar_idx, predictions):
        """V32: Dynamic entry with add capability"""
        spec = SPECS[name]
        df = self.lh_df if name == 'LH' else self.jm_df
        atr_s = spec['atr_stop']
        
        if bar_idx >= len(df):
            return
        
        row = df.iloc[bar_idx]
        price = float(row['close']); high = float(row['high']); low = float(row['low'])
        atr = compute_atr(df, bar_idx, 20)
        if atr is None or price <= 0:
            return
        
        pred = predictions.get(name, {})
        if not pred:
            return
        
        prob = pred['prob']; direction = pred['direction']; conf = pred['confidence']
        
        # Position sizing
        atr_pct = atr / price
        if atr_pct < 0.01: lev = 3.0
        elif atr_pct < 0.02: lev = 2.0
        elif atr_pct < 0.03: lev = 1.5
        else: lev = 0.5
        
        ps = max(1, int(lev * (spec['max_pos'] // 2))) if lev > 0 else 0
        if ps <= 0:
            return
        
        mult = spec['multiplier']
        margin = ps * price * mult * spec['margin_rate']
        
        inst_positions = [p for p in self.positions if p['name'] == name]
        inst_lots = sum(p['vol'] for p in inst_positions)
        
        sd = atr * atr_s
        
        if not inst_positions:
            # No existing position — fresh entry
            _ec = spec.get('entry_conf', 0.55)
            if not (conf > _ec and ((direction == 'LONG' and prob > _ec) or (direction == 'SHORT' and prob < 1 - _ec))):
                return
            if self._available_cash() < margin:
                return
            if ps > spec['max_total']:
                ps = spec['max_total']
            
            if direction == 'LONG':
                stop = price - sd
                if low > stop:
                    self.positions.append({
                        'name': name, 'direction': 'LONG',
                        'entry': price, 'stop': stop,
                        'vol': ps, 'entry_bar': bar_idx, 'trail': stop,
                    })
            else:
                stop = price + sd
                if high < stop:
                    self.positions.append({
                        'name': name, 'direction': 'SHORT',
                        'entry': price, 'stop': stop,
                        'vol': ps, 'entry_bar': bar_idx, 'trail': stop,
                    })
        else:
            # Has existing position — check add
            existing_dir = inst_positions[0]['direction']
            if inst_lots + ps > spec['max_total']:
                ps = spec['max_total'] - inst_lots
                if ps <= 0:
                    return
                margin = ps * price * mult * spec['margin_rate']
            
            if direction == existing_dir and conf > spec['add_conf']:
                # Calculate average entry profit in ATR
                avg_entry = np.mean([p['entry'] for p in inst_positions])
                if existing_dir == 'LONG':
                    pa = (price - avg_entry) / atr
                else:
                    pa = (avg_entry - price) / atr
                
                if pa > spec['add_atr']:
                    if self._available_cash() < margin:
                        return
                    if existing_dir == 'LONG':
                        stop = price - sd
                        if low > stop:
                            self.positions.append({
                                'name': name, 'direction': 'LONG',
                                'entry': price, 'stop': stop,
                                'vol': ps, 'entry_bar': bar_idx, 'trail': stop,
                            })
                    else:
                        stop = price + sd
                        if high < stop:
                            self.positions.append({
                                'name': name, 'direction': 'SHORT',
                                'entry': price, 'stop': stop,
                                'vol': ps, 'entry_bar': bar_idx, 'trail': stop,
                            })
    
    def _calc_unrealized_pnl(self, bar_idx):
        """Calculate total unrealized PnL of all positions at current bar"""
        total = 0.0
        for pos in self.positions:
            name = pos['name']
            df = self.lh_df if name == 'LH' else self.jm_df
            if bar_idx >= len(df):
                continue
            price = float(df.iloc[bar_idx]['close'])
            mult = SPECS[name]['multiplier']
            if pos['direction'] == 'LONG':
                total += (price - pos['entry']) * pos['vol'] * mult
            else:
                total += (pos['entry'] - price) * pos['vol'] * mult
        return total

    def _force_liquidate(self, bar_idx, reason='MARGIN'):
        """次日14:00强平：可用资金仍为负则按市价平最差持仓至够保证金"""
        df_date = self.lh_df
        if bar_idx >= len(df_date):
            return
        price_map = {}
        for name in ['LH', 'JM']:
            df = self.lh_df if name == 'LH' else self.jm_df
            if bar_idx < len(df):
                price_map[name] = float(df.iloc[bar_idx]['close'])
        
        if not self.positions:
            return
        
        # 计算当前总权益 = cash + 未实现盈亏
        equity = self.cash + self._calc_unrealized_pnl(bar_idx)
        margin_locked = self._total_margin()
        available = equity - margin_locked
        
        if available >= 0:
            return  # 次日回升了，不需要强平
        
        deficit = abs(available)
        
        # 按浮亏从大到小排序（先平亏最多的）
        sorted_pos = sorted(self.positions, key=lambda p: (
            (price_map.get(p['name'], p['entry']) - p['entry']) * p['vol'] * SPECS[p['name']]['multiplier']
            if p['direction'] == 'LONG' else
            (p['entry'] - price_map.get(p['name'], p['entry'])) * p['vol'] * SPECS[p['name']]['multiplier']
        ))
        
        freed = 0.0
        liquidated = []
        for pos in sorted_pos:
            if freed >= deficit:
                break
            price = price_map.get(pos['name'], pos['entry'])
            mult = SPECS[pos['name']]['multiplier']
            margin = self._position_margin(pos)
            # 平仓释放: 保证金 + (浮盈 - 手续费)
            if pos['direction'] == 'LONG':
                float_pnl = (price - pos['entry']) * pos['vol'] * mult
            else:
                float_pnl = (pos['entry'] - price) * pos['vol'] * mult
            cost = pos['entry'] * pos['vol'] * mult * SPECS[pos['name']]['cost'] * 2
            freed += margin + float_pnl - cost
            liquidated.append(pos)
        
        for pos in liquidated:
            self._close_position(pos, price_map.get(pos['name'], pos['entry']), reason, bar_idx)

    def _check_margin_daily(self, bar_idx):
        """每日收盘检查：可用资金为负则标记，次日再判断是否强平"""
        equity = self.cash + self._calc_unrealized_pnl(bar_idx)
        available = equity - self._total_margin()
        
        if available < 0 and hasattr(self, '_pending_margin_call'):
            # 昨天已标记，今天仍为负 → 强平
            self._force_liquidate(bar_idx, 'LIQUIDATE')
            self._pending_margin_call = False
        elif available < 0:
            # 今天首次为负 → 标记，等明天14:00再判断
            self._pending_margin_call = True
        else:
            self._pending_margin_call = False

    def close_all_at_end(self, final_bar_idx):
        """Close all positions at end of backtest"""
        for pos in list(self.positions):
            name = pos['name']
            df = self.lh_df if name == 'LH' else self.jm_df
            if final_bar_idx < len(df):
                price = float(df.iloc[final_bar_idx]['close'])
            else:
                price = pos['entry']
            self._close_position(pos, price, 'EOD', final_bar_idx)
    
    def get_stats(self):
        """Compute summary statistics from trades"""
        trades = self.trades
        if not trades:
            return {
                'total_return': 0, 'mdd': 0, 'win_rate': 0, 'n_trades': 0,
                'profit_factor': 0, 'final_equity': self.cash,
                'avg_win': 0, 'avg_loss': 0, 'sharpe': 0,
            }
        
        wins = [t for t in trades if t['pnl_abs'] > 0]
        losses = [t for t in trades if t['pnl_abs'] <= 0]
        
        n = len(trades)
        wr = len(wins) / n if n > 0 else 0
        
        # Profit factor (absolute PnL)
        gw = sum(t['pnl_abs'] for t in wins)
        gl = abs(sum(t['pnl_abs'] for t in losses)) if losses else 0
        pf = gw / gl if gl > 0 else 99
        
        avg_win = np.mean([t['pnl_abs'] for t in wins]) if wins else 0
        avg_loss = np.mean([t['pnl_abs'] for t in losses]) if losses else 0
        
        total_return = (self.cash - self.initial_capital) / self.initial_capital
        
        # MDD from equity curve
        eq_values = [self.initial_capital]
        running = self.initial_capital
        for t in sorted(trades, key=lambda x: x['date']):
            running += t['pnl_abs']
            eq_values.append(running)
        
        peak = self.initial_capital
        mdd = 0.0
        for eq in eq_values:
            peak = max(peak, eq)
            dd = (eq - peak) / peak
            mdd = min(mdd, dd)
        
        # Sharpe ratio (approximate from trade returns)
        pnls = [t['pnl_abs'] for t in trades]
        if len(pnls) > 1:
            sharpe = np.mean(pnls) / (np.std(pnls) + 1e-10) * np.sqrt(252 / max(np.mean([t['bars'] for t in trades]), 1))
        else:
            sharpe = 0
        
        # Trade type breakdown
        types = {}
        for t in trades:
            types[t['type']] = types.get(t['type'], 0) + 1
        
        return {
            'total_return': round(total_return * 100, 2),
            'mdd': round(mdd * 100, 2),
            'win_rate': round(wr * 100, 1),
            'n_trades': n,
            'profit_factor': round(pf, 2),
            'final_equity': round(self.cash, 2),
            'avg_win': round(avg_win, 2),
            'avg_loss': round(avg_loss, 2),
            'sharpe': round(sharpe, 2),
            'trade_types': types,
        }


def run_combined_backtest(lh_df, jm_df, strategy='V31', retrain_freq=30, train_window=500,
                          atr_stop_lh=None, rr_lh=None, atr_stop_jm=None, rr_jm=None):
    """Run a single continuous combined backtest with specified strategy.
    
    Args:
        train_window: number of bars to use for training (default 500 ~2 years)
        retrain_freq: retrain every N trading bars
    """
    
    bt = ContinuousBacktest(lh_df, jm_df)
    
    # Get all feature indices (sorted)
    lh_indices = bt.get_available_feat_indices('LH')
    jm_indices = bt.get_available_feat_indices('JM')
    
    # Train data: all indices up to the first test point
    # For continuous backtest, we process sequentially through dates
    # Models are retrained every retrain_freq trading bars
    
    # Create a mapping from bar counter to date index
    # We process in date order, incrementing a bar counter when either instrument has data
    
    all_dates = bt.all_dates
    min_feat_bar = 70
    
    # Track models
    lh_model = None; jm_model = None
    lh_ensemble = None; jm_ensemble = None
    
    bar_count = 0
    last_retrain_bar = -999
    lh_train_start = 0
    jm_train_start = 0
    
    for date_idx, date_str in enumerate(all_dates):
        # Check if we can process this date
        has_lh = date_str in bt.lh_date_to_idx
        has_jm = date_str in bt.jm_date_to_idx
        
        lh_idx = bt.lh_date_to_idx.get(date_str, -1)
        jm_idx = bt.jm_date_to_idx.get(date_str, -1)
        
        # Skip early bars without features
        if has_lh and lh_idx < min_feat_bar:
            continue
        if has_jm and jm_idx < min_feat_bar:
            continue
        
        bar_count += 1
        
        # ── Retrain check ──
        if bar_count - last_retrain_bar >= retrain_freq:
            last_retrain_bar = bar_count
            
            if strategy == 'V33':
                # Ensemble training
                lh_train_indices = [i for i in lh_indices if i < lh_idx and lh_idx - i <= train_window]
                jm_train_indices = [i for i in jm_indices if i < jm_idx and jm_idx - i <= train_window]
                if len(lh_train_indices) >= 50:
                    lh_ensemble = bt.train_ensemble_at('LH', lh_train_indices)
                if len(jm_train_indices) >= 50:
                    jm_ensemble = bt.train_ensemble_at('JM', jm_train_indices)
            else:
                # Single XGBoost
                lh_train_indices = [i for i in lh_indices if i < lh_idx and lh_idx - i <= train_window]
                jm_train_indices = [i for i in jm_indices if i < jm_idx and jm_idx - i <= train_window]
                if len(lh_train_indices) >= 50:
                    lh_model = bt.train_model_at('LH', lh_train_indices)
                if len(jm_train_indices) >= 50:
                    jm_model = bt.train_model_at('JM', jm_train_indices)
        
        # ── Generate predictions for this bar ──
        predictions = {}
        if has_lh and lh_idx in bt.feats['LH']:
            if strategy == 'V33':
                if lh_ensemble:
                    pred = bt.predict(lh_ensemble, 'LH', lh_idx)
                    if pred and pred.get('all_agree', False):
                        predictions['LH'] = pred
            else:
                if lh_model:
                    pred = bt.predict(lh_model, 'LH', lh_idx)
                    if pred:
                        predictions['LH'] = pred
        
        if has_jm and jm_idx in bt.feats['JM']:
            if strategy == 'V33':
                if jm_ensemble:
                    pred = bt.predict(jm_ensemble, 'JM', jm_idx)
                    if pred and pred.get('all_agree', False):
                        predictions['JM'] = pred
            else:
                if jm_model:
                    pred = bt.predict(jm_model, 'JM', jm_idx)
                    if pred:
                        predictions['JM'] = pred
        
        # ── Process positions ──
        if strategy == 'V31' or strategy == 'V35':
            # Check existing positions for stop/tp
            bt._process_v31_positions(lh_idx if has_lh else bar_count)
            
            # Try new entries
            if has_lh and 'LH' in predictions:
                bt._try_entry_v31('LH', lh_idx, predictions, atr_stop_lh, rr_lh)
            if has_jm and 'JM' in predictions:
                bt._try_entry_v31('JM', jm_idx, predictions, atr_stop_jm, rr_jm)
        
        elif strategy in ('V32', 'V33', 'V34'):
            # Dynamic strategy
            bt._process_v32_positions(lh_idx if has_lh else bar_count, predictions)
            
            if has_lh and 'LH' in predictions:
                bt._try_entry_v32('LH', lh_idx, predictions)
            if has_jm and 'JM' in predictions:
                bt._try_entry_v32('JM', jm_idx, predictions)
        
        # ── 每日保证金检查（可用资金为负则标记，次日14:00判断强平）──
        bt._check_margin_daily(lh_idx if has_lh else jm_idx if has_jm else bar_count)
    
    # Close remaining positions
    last_idx = max(
        bt.lh_date_to_idx.get(all_dates[-1], len(bt.lh_df) - 1) if all_dates else 0,
        bt.jm_date_to_idx.get(all_dates[-1], len(bt.jm_df) - 1) if all_dates else 0
    )
    bt.close_all_at_end(last_idx)
    
    return bt.get_stats(), bt


# ═══════════════════════════════════════════════════════════════
# PHASE 5: Output
# ═══════════════════════════════════════════════════════════════

def print_header(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")

def print_table(headers, rows, aligns=None):
    """Print a formatted table"""
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(cell)))
    
    # Header
    sep = '  '
    header_line = sep.join(f'{h:<{col_widths[i]}}' for i, h in enumerate(headers))
    print(f"  {header_line}")
    # Separator
    sep_line = sep.join('-' * col_widths[i] for i in range(len(headers)))
    print(f"  {sep_line}")
    # Rows
    for row in rows:
        row_line = sep.join(f'{str(c):<{col_widths[i]}}' for i, c in enumerate(row))
        print(f"  {row_line}")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    start_time = time.time()
    print_header(f"Prophet Futures v5 综合回测系统")
    print(f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  起始资金: ¥{START_CAPITAL:,.0f}")
    print(f"  模式: 连续资金（不重置）")
    
    # ── Phase 1: Fetch Data ──
    print_header("Phase 1: 数据获取")
    data = fetch_data()
    
    if 'LH' not in data or 'JM' not in data:
        print("\n❌ 数据获取失败，无法继续")
        return
    
    lh_df = data['LH']
    jm_df = data['JM']
    
    # ── Phase 2: Features (done inside ContinuousBacktest) ──
    print_header("Phase 2: 特征工程预计算")
    print(f"  特征维度: {len(FEAT_NAMES)}")
    print(f"  特征名称: {', '.join(FEAT_NAMES[:6])}...")
    
    # ── Phase 3: Models trained during backtest ──
    
    # ── Phase 4: Five Backtests ──
    all_results = {}
    
    # ─── V31: Simple Baseline ───
    print_header("V31 简单基线 | XGBoost 新模型 | 固定ATR止损止盈 | 30天重训 | 无动态管理")
    t0 = time.time()
    v31_stats, v31_bt = run_combined_backtest(lh_df, jm_df, strategy='V31', retrain_freq=30)
    all_results['V31'] = v31_stats
    print(f"  ✅ V31完成 ({time.time()-t0:.0f}s)")
    print(f"  总收益: {v31_stats['total_return']:+.2f}%")
    print(f"  最大回撤: {v31_stats['mdd']:.2f}%")
    print(f"  胜率: {v31_stats['win_rate']:.1f}%")
    print(f"  交易次数: {v31_stats['n_trades']}")
    print(f"  盈利因子: {v31_stats['profit_factor']:.2f}")
    print(f"  最终权益: ¥{v31_stats['final_equity']:,.2f}")
    if v31_stats['n_trades'] > 0:
        liq = v31_stats['trade_types'].get('LIQUIDATE', 0)
        if liq: print(f"  ⚠️ 强平: {liq}次")
    
    # ─── V32: Dynamic Strategy ───
    print_header("V32 动态策略 | XGBoost 新模型 | 加仓/减仓/反手/追踪止损 | 30天重训")
    t0 = time.time()
    v32_stats, v32_bt = run_combined_backtest(lh_df, jm_df, strategy='V32', retrain_freq=30)
    all_results['V32'] = v32_stats
    print(f"  ✅ V32完成 ({time.time()-t0:.0f}s)")
    print(f"  总收益: {v32_stats['total_return']:+.2f}%")
    print(f"  最大回撤: {v32_stats['mdd']:.2f}%")
    print(f"  胜率: {v32_stats['win_rate']:.1f}%")
    print(f"  交易次数: {v32_stats['n_trades']}")
    print(f"  盈利因子: {v32_stats['profit_factor']:.2f}")
    print(f"  最终权益: ¥{v32_stats['final_equity']:,.2f}")
    if v32_stats['n_trades'] > 0:
        liq = v32_stats['trade_types'].get('LIQUIDATE', 0)
        if liq: print(f"  ⚠️ 强平: {liq}次")
    
    # ─── V33: Ensemble ───
    print_header("V33 集成投票 | XGBoost+LGBM+CatBoost 三模型一致才交易 | 30天重训")
    t0 = time.time()
    v33_stats, v33_bt = run_combined_backtest(lh_df, jm_df, strategy='V33', retrain_freq=30)
    all_results['V33'] = v33_stats
    print(f"  ✅ V33完成 ({time.time()-t0:.0f}s)")
    print(f"  总收益: {v33_stats['total_return']:+.2f}%")
    print(f"  最大回撤: {v33_stats['mdd']:.2f}%")
    print(f"  胜率: {v33_stats['win_rate']:.1f}%")
    print(f"  交易次数: {v33_stats['n_trades']}")
    print(f"  盈利因子: {v33_stats['profit_factor']:.2f}")
    print(f"  最终权益: ¥{v33_stats['final_equity']:,.2f}")
    if v33_stats['n_trades'] > 0:
        liq = v33_stats['trade_types'].get('LIQUIDATE', 0)
        if liq: print(f"  ⚠️ 强平: {liq}次")
    
    # ─── V34: Prediction Validity Window ───
    print_header("V34 预测有效期 | 同V32策略 | 变更重训频率 5/10/20/30/60天对比")
    v34_results = {}
    for freq in [5, 10, 20, 30, 60]:
        t0 = time.time()
        stats, _ = run_combined_backtest(lh_df, jm_df, strategy='V32', retrain_freq=freq)
        v34_results[freq] = stats
        liq_info = f" | ⚠️强平{stats['trade_types'].get('LIQUIDATE',0)}次" if stats['n_trades']>0 and stats['trade_types'].get('LIQUIDATE',0) else ""
        print(f"  重训频率 {freq}天: 收益{stats['total_return']:+.1f}% | MDD{stats['mdd']:.1f}% | "
              f"胜率{stats['win_rate']:.1f}% | 交易{stats['n_trades']}笔{liq_info} ({time.time()-t0:.0f}s)")
    all_results['V34'] = v34_results
    
    # ─── V35: Stop-Loss/Take-Profit Grid ───
    print_header("V35 止盈止损网格 | 同V31策略 | 仅LH | 6×5=30组合扫描最优参数")
    
    atr_stops = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    rrs = [2.0, 3.0, 4.0, 5.0, 6.0]
    v35_grid = []
    
    total_combos = len(atr_stops) * len(rrs)
    done = 0
    for atr_s in atr_stops:
        for rr in rrs:
            t0 = time.time()
            stats, _ = run_combined_backtest(
                lh_df, jm_df, strategy='V31', retrain_freq=30,
                atr_stop_lh=atr_s, rr_lh=rr,
                atr_stop_jm=SPECS['JM']['atr_stop'], rr_jm=SPECS['JM']['rr']
            )
            score = stats['total_return'] / 100 * (1 + stats['mdd'] / 100)  # return*(1-MDD) normalized
            v35_grid.append({
                'atr_stop': atr_s, 'rr': rr,
                'return': stats['total_return'],
                'mdd': stats['mdd'],
                'win_rate': stats['win_rate'],
                'n_trades': stats['n_trades'],
                'profit_factor': stats['profit_factor'],
                'score': round(score, 4),
                'time': round(time.time() - t0, 1),
            })
            done += 1
            print(f"  [{done}/{total_combos}] atr_stop={atr_s} rr={rr} → "
                  f"收益{stats['total_return']:+.1f}% MDD{stats['mdd']:.1f}% "
                  f"评分{score:.4f} ({v35_grid[-1]['time']}s)")
    
    # Sort by score
    v35_grid.sort(key=lambda x: x['score'], reverse=True)
    all_results['V35'] = v35_grid[:10]  # top 10
    
    # ────────────────────────────────────────────────
    # Phase 5: Output
    # ────────────────────────────────────────────────
    
    print_header("📊 综合对比报告")
    
    # V31/V32/V33 comparison
    ver_labels = {
        'V31': 'V31 简单基线 | 固定止损止盈 | 无动态管理',
        'V32': 'V32 动态策略 | 加仓减仓反手追踪 | 新模型30天重训',
        'V33': 'V33 集成投票 | 三模型一致才交易 | 30天重训',
    }
    print("\n  ▸ V31/V32/V33 策略对比（含保证金强平风控）")
    headers = ['版本', '策略说明', '总收益%', '最大回撤%', '胜率%', '交易', '强平', '盈利因子', '最终¥']
    rows = []
    for ver in ['V31', 'V32', 'V33']:
        s = all_results[ver]
        rows.append([
            ver,
            ver_labels[ver],
            f"{s['total_return']:+.2f}",
            f"{s['mdd']:.2f}",
            f"{s['win_rate']:.1f}",
            str(s['n_trades']),
            str(s['trade_types'].get('LIQUIDATE', 0)),
            f"{s['profit_factor']:.2f}",
            f"{s['final_equity']:,.0f}",
        ])
    print_table(headers, rows)
    
    # Trade type breakdown
    print("\n  ▸ 交易类型分布")
    for ver in ['V31', 'V32', 'V33']:
        types = all_results[ver].get('trade_types', {})
        types_str = ', '.join(f'{k}:{v}' for k, v in sorted(types.items()))
        print(f"  {ver}: {types_str}")
    
    # V34 validity table
    print("\n  ▸ V34 预测有效期测试")
    headers = ['重训频率(天)', '总收益%', '最大回撤%', '胜率%', '交易次数', '盈利因子']
    rows = []
    for freq in [5, 10, 20, 30, 60]:
        s = v34_results[freq]
        rows.append([
            str(freq),
            f"{s['total_return']:+.2f}",
            f"{s['mdd']:.2f}",
            f"{s['win_rate']:.1f}",
            str(s['n_trades']),
            f"{s['profit_factor']:.2f}",
        ])
    print_table(headers, rows)
    
    # V35 optimization table
    print("\n  ▸ V35 止损/止盈网格优化 Top-5 (评分 = 收益×(1-MDD))")
    headers = ['排名', 'ATR止损', 'RR止盈', '总收益%', '最大回撤%', '胜率%', '交易', '盈利因子', '评分']
    rows = []
    for i, g in enumerate(v35_grid[:5]):
        rows.append([
            str(i+1),
            str(g['atr_stop']),
            str(g['rr']),
            f"{g['return']:+.2f}",
            f"{g['mdd']:.2f}",
            f"{g['win_rate']:.1f}",
            str(g['n_trades']),
            f"{g['profit_factor']:.2f}",
            f"{g['score']:.4f}",
        ])
    print_table(headers, rows)
    
    # ── Save models ──
    print_header("💾 保存模型和结果")
    
    # Train final models on full data and save
    for name, df, spec in [('LH', lh_df, SPECS['LH']), ('JM', jm_df, SPECS['JM'])]:
        X_list, y_list = [], []
        for idx in range(70, len(df) - 1):
            feats = build_features(df, idx, 60)
            if feats is not None:
                X_list.append(feats)
                y_list.append(1 if float(df.iloc[idx+1]['close']) > float(df.iloc[idx]['close']) else 0)
        
        if len(X_list) >= 100:
            X = np.array(X_list, dtype=np.float32)
            y = np.array(y_list)
            
            # XGBoost
            model = train_xgb(X, y, spec['xgb_params'])
            path = os.path.join(MODEL_DIR, 'v31_xgb.pkl' if name == 'LH' else 'v31_jm_xgb.pkl')
            with open(path, 'wb') as f:
                pickle.dump(model, f)
            print(f"  ✅ 已保存: {path}")
            
            # LGBM
            model_lgb = train_lgb(X, y)
            path_lgb = os.path.join(MODEL_DIR, 'v31_lgb.pkl' if name == 'LH' else 'v31_jm_lgb.pkl')
            with open(path_lgb, 'wb') as f:
                pickle.dump(model_lgb, f)
            
            # CatBoost
            model_cb = train_cb(X, y)
            path_cb = os.path.join(MODEL_DIR, 'v31_cb.pkl' if name == 'LH' else 'v31_jm_cb.pkl')
            with open(path_cb, 'wb') as f:
                pickle.dump(model_cb, f)
    
    # ── Save JSON results ──
    # Convert numpy types for JSON
    def convert_for_json(obj):
        if isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(v) for v in obj]
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        else:
            return obj
    
    json_results = convert_for_json(all_results)
    json_results['metadata'] = {
        'timestamp': datetime.now().isoformat(),
        'initial_capital': START_CAPITAL,
        'lh_rows': len(lh_df),
        'jm_rows': len(jm_df),
        'lh_date_range': f"{pd.to_datetime(lh_df.iloc[0]['date']).strftime('%Y-%m-%d')} → {pd.to_datetime(lh_df.iloc[-1]['date']).strftime('%Y-%m-%d')}",
        'jm_date_range': f"{pd.to_datetime(jm_df.iloc[0]['date']).strftime('%Y-%m-%d')} → {pd.to_datetime(jm_df.iloc[-1]['date']).strftime('%Y-%m-%d')}",
        'total_time_s': round(time.time() - start_time, 1),
    }
    
    with open(RESULTS_PATH, 'w', encoding='utf-8') as f:
        json.dump(json_results, f, indent=2, ensure_ascii=False)
    print(f"  ✅ 结果已保存: {RESULTS_PATH}")
    
    total_time = time.time() - start_time
    print_header(f"✅ 全部完成！总耗时: {total_time:.0f}s ({total_time/60:.1f}分钟)")
    
    return all_results

if __name__ == '__main__':
    main()
