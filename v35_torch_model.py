#!/usr/bin/env python3
"""TorchSeqClassifier — sklearn兼容LSTM包装(fit/predict_proba)
输入X: (n, 360)展平序列 → 内部reshape(n,60,6)。引擎无感知替换XGB。
"""
import numpy as np
import torch
import torch.nn as nn

SEQ, NF = 60, 6

class _LSTM(nn.Module):
    def __init__(self, hidden=32, drop=0.3):
        super().__init__()
        self.rnn = nn.LSTM(NF, hidden, 1, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(drop), nn.Linear(hidden, 2))
    def forward(self, x):
        out, _ = self.rnn(x)
        return self.head(out[:, -1])

class TorchSeqClassifier:
    """与train_v35_dl.py同架构/同训练协议(早停+AdamW)"""
    def __init__(self, epochs=60, patience=8, seed=42):
        self.epochs, self.patience, self.seed = epochs, patience, seed
        self.model = None

    def fit(self, X, y):
        torch.manual_seed(self.seed); np.random.seed(self.seed)
        X = np.asarray(X, dtype=np.float32).reshape(-1, SEQ, NF)
        y = np.asarray(y, dtype=np.int64)
        va = int(len(X) * 0.85)  # 时序尾部15%早停验证
        Xtr, ytr = torch.tensor(X[:va]), torch.tensor(y[:va])
        Xva, yva = torch.tensor(X[va:]), torch.tensor(y[va:])
        m = _LSTM()
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
        lf = nn.CrossEntropyLoss()
        best_acc, best_state, bad = 0.0, None, 0
        for _ in range(self.epochs):
            m.train()
            perm = torch.randperm(len(Xtr))
            for b in range(0, len(perm), 64):
                idx = perm[b:b+64]
                opt.zero_grad()
                lf(m(Xtr[idx]), ytr[idx]).backward()
                opt.step()
            m.eval()
            with torch.no_grad():
                acc = (m(Xva).argmax(1) == yva).float().mean().item() if len(Xva) else 0.5
            if acc > best_acc:
                best_acc, bad = acc, 0
                best_state = {k: v.clone() for k, v in m.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience: break
        if best_state: m.load_state_dict(best_state)
        m.eval(); self.model = m
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float32).reshape(-1, SEQ, NF)
        with torch.no_grad():
            logits = self.model(torch.tensor(X))
            return torch.softmax(logits, 1).numpy()

    def predict(self, X):
        return self.predict_proba(X).argmax(1)
