#!/usr/bin/env python3
"""V35深度学习实验 — LSTM/GRU/Transformer vs XGB基线
输入: 60日窗口×6序列特征 | 输出: 次日涨跌二分类
WF: 300训/30测/15步 (与V32-V34网格同协议, 可直接对比)
决策规则: DL赢XGB基线才上纸盘, 输了如实报告
"""
import sys, json, time
import numpy as np, pandas as pd
sys.path.insert(0, '.')

import torch
import torch.nn as nn
import xgboost as xgb
import akshare as ak

torch.manual_seed(42); np.random.seed(42)
DEV = 'cpu'
SEQ = 60          # 序列长度
NF = 6            # 序列特征数
EPOCHS = 60
PATIENCE = 8

# ===== 数据 =====
df = ak.futures_main_sina(symbol='LH0')
df.columns = ['date','open','high','low','close','volume','oi','settle']
for c in ['open','high','low','close','volume','oi']:
    df[c] = pd.to_numeric(df[c], errors='coerce')
df = df.dropna(subset=['close']).reset_index(drop=True)
c_arr = df['close'].values.astype(float)
o_arr = df['open'].values.astype(float)
h_arr = df['high'].values.astype(float)
l_arr = df['low'].values.astype(float)
v_arr = df['volume'].values.astype(float)
oi_arr = df['oi'].values.astype(float)
N = len(df)
print(f'数据: {N}条 {df.iloc[0]["date"]} → {df.iloc[-1]["date"]}')

# 序列特征(每日6维, 全部相对量防漂移)
feat = np.zeros((N, NF), dtype=np.float32)
for i in range(1, N):
    feat[i, 0] = (c_arr[i] - c_arr[i-1]) / c_arr[i-1]                  # 日收益
    feat[i, 1] = (h_arr[i] - l_arr[i]) / c_arr[i]                      # 振幅
    feat[i, 2] = (o_arr[i] - c_arr[i-1]) / c_arr[i-1]                  # 缺口
    vm = v_arr[max(0,i-20):i].mean() or 1
    feat[i, 3] = min(v_arr[i] / vm, 5.0)                               # 量比(截尾)
    om = oi_arr[max(0,i-20):i].mean() or 1
    feat[i, 4] = min(oi_arr[i] / om, 3.0)                              # 持仓比(截尾)
    feat[i, 5] = (c_arr[i] - c_arr[max(0,i-5)]) / c_arr[max(0,i-5)]    # 5日动量

labels = np.zeros(N, dtype=np.int64)
labels[:-1] = (c_arr[1:] > c_arr[:-1]).astype(np.int64)

def make_seq(idx):
    """idx日的输入 = [idx-SEQ+1, idx]的特征矩阵"""
    if idx < SEQ: return None
    return feat[idx-SEQ+1:idx+1]

# ===== 模型 =====
class RNNClf(nn.Module):
    def __init__(self, kind='lstm', hidden=32, layers=1, drop=0.3):
        super().__init__()
        rnn_cls = nn.LSTM if kind == 'lstm' else nn.GRU
        self.rnn = rnn_cls(NF, hidden, layers, batch_first=True,
                           dropout=drop if layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Dropout(drop), nn.Linear(hidden, 2))
    def forward(self, x):
        out, _ = self.rnn(x)
        return self.head(out[:, -1])

class TransClf(nn.Module):
    def __init__(self, d=32, heads=4, layers=2, drop=0.3):
        super().__init__()
        self.proj = nn.Linear(NF, d)
        self.pos = nn.Parameter(torch.randn(1, SEQ, d) * 0.02)
        enc = nn.TransformerEncoderLayer(d, heads, d*2, drop, batch_first=True)
        self.enc = nn.TransformerEncoder(enc, layers)
        self.head = nn.Sequential(nn.Dropout(drop), nn.Linear(d, 2))
    def forward(self, x):
        z = self.proj(x) + self.pos
        z = self.enc(z)
        return self.head(z.mean(1))

def train_dl(model, Xtr, ytr, Xva, yva):
    model = model.to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    lf = nn.CrossEntropyLoss()
    Xtr_t = torch.tensor(Xtr, device=DEV); ytr_t = torch.tensor(ytr, device=DEV)
    Xva_t = torch.tensor(Xva, device=DEV); yva_t = torch.tensor(yva, device=DEV)
    best_acc, best_state, bad = 0.0, None, 0
    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(len(Xtr_t))
        for b in range(0, len(perm), 64):
            idx = perm[b:b+64]
            opt.zero_grad()
            loss = lf(model(Xtr_t[idx]), ytr_t[idx])
            loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            acc = (model(Xva_t).argmax(1) == yva_t).float().mean().item()
        if acc > best_acc:
            best_acc, bad = acc, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE: break
    if best_state: model.load_state_dict(best_state)
    return model

# ===== Walk-Forward =====
n_tr, n_te, step = 300, 30, 15
first = SEQ + 10
folds = list(range(first + n_tr, N - n_te - 1, step))
print(f'WF折数: {len(folds)}')

def wf_eval(name, factory):
    t0 = time.time()
    ok = tot = 0
    for ts in folds:
        Xtr = np.stack([make_seq(i) for i in range(ts - n_tr, ts)])
        ytr = labels[ts - n_tr:ts]
        Xte = np.stack([make_seq(i) for i in range(ts, ts + n_te)])
        yte = labels[ts:ts + n_te]
        va = int(len(Xtr) * 0.85)  # 时序尾部15%做早停验证
        model = train_dl(factory(), Xtr[:va], ytr[:va], Xtr[va:], ytr[va:])
        model.eval()
        with torch.no_grad():
            pred = model(torch.tensor(Xte, device=DEV)).argmax(1).numpy()
        ok += int((pred == yte).sum()); tot += len(yte)
    acc = ok / tot
    print(f'  {name}: {acc:.1%} ({ok}/{tot}) {time.time()-t0:.0f}s', flush=True)
    return acc

def wf_xgb():
    t0 = time.time(); ok = tot = 0
    for ts in folds:
        Xtr = np.stack([make_seq(i).flatten() for i in range(ts - n_tr, ts)])
        ytr = labels[ts - n_tr:ts]
        Xte = np.stack([make_seq(i).flatten() for i in range(ts, ts + n_te)])
        yte = labels[ts:ts + n_te]
        m = xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05,
                              n_jobs=4, verbosity=0, random_state=42)
        m.fit(Xtr, ytr)
        ok += int((m.predict(Xte) == yte).sum()); tot += len(yte)
    acc = ok / tot
    print(f'  XGB基线(同序列展平): {acc:.1%} ({ok}/{tot}) {time.time()-t0:.0f}s', flush=True)
    return acc

print('\n=== WF方向准确率对比(同折同数据) ===')
results = {}
results['XGB'] = wf_xgb()
results['LSTM'] = wf_eval('LSTM(h32)', lambda: RNNClf('lstm'))
results['GRU'] = wf_eval('GRU(h32)', lambda: RNNClf('gru'))
results['Transformer'] = wf_eval('Transformer(d32x2)', lambda: TransClf())

json.dump(results, open('v35_dl_results.json', 'w'), indent=2)
best = max(results, key=results.get)
print(f'\n最优: {best} {results[best]:.1%}')
print(f'XGB基线: {results["XGB"]:.1%}')
print('结论: ' + ('DL赢,值得接策略回测' if best != 'XGB' and results[best] > results['XGB'] + 0.01
                 else 'DL未显著超越XGB(差距<1pp),维持GBDT'))
