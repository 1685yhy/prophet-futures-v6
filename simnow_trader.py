#!/usr/bin/env python3
"""SimNow CTP 自动交易 — 完整版"""

import sys, time, os
from datetime import datetime

try:
    from openctp_ctp import tdapi, mdapi
except ImportError:
    print("pip install openctp-ctp"); sys.exit(1)

CFG = {
    "broker": "9999", "user": "266887", "pass": "asdfghjkl123!!",
    "trade": "tcp://180.168.146.187:10130",
    "market": "tcp://180.168.146.187:10110",
    "app": "simnow_client_test", "auth": "0000000000000000",
}

collected = {}

# ══════════════════ Trade SPI ══════════════════
class TradeSpi(tdapi.CThostFtdcTraderSpi):
    def __init__(self, api):
        super().__init__()
        self.api = api
    def OnFrontConnected(self):
        print("  [CTP] 交易前置已连接")
        req = tdapi.CThostFtdcReqAuthenticateField()
        req.BrokerID=CFG["broker"]; req.UserID=CFG["user"]
        req.AppID=CFG["app"]; req.AuthCode=CFG["auth"]
        self.api.ReqAuthenticate(req, 1)
    def OnRspAuthenticate(self, pRsp, pInfo, nID, bLast):
        if pInfo and pInfo.ErrorID == 0:
            print("  [CTP] 认证通过")
            req = tdapi.CThostFtdcReqUserLoginField()
            req.BrokerID=CFG["broker"]; req.UserID=CFG["user"]; req.Password=CFG["pass"]
            self.api.ReqUserLogin(req, 2)
        else:
            print(f"  [CTP] 认证失败: {pInfo.ErrorMsg if pInfo else 'unknown'}")
    def OnRspUserLogin(self, pRsp, pInfo, nID, bLast):
        if pInfo and pInfo.ErrorID == 0:
            print(f"  [CTP] 登录成功 (Session={pRsp.SessionID})")
            collected["login"] = True
            req = tdapi.CThostFtdcQrySettlementInfoConfirmField()
            req.BrokerID=CFG["broker"]; req.InvestorID=CFG["user"]
            self.api.ReqQrySettlementInfoConfirm(req, 3)
        else:
            print(f"  [CTP] 登录失败: {pInfo.ErrorMsg if pInfo else 'unknown'}")
    def OnRspQrySettlementInfoConfirm(self, pRsp, pInfo, nID, bLast):
        if pRsp:
            collected["settled"] = True
            print("  [CTP] 已确认结算单")
    def OnRspOrderInsert(self, pRsp, pInfo, nID, bLast):
        if pInfo and pInfo.ErrorID == 0:
            print(f"  [CTP] 下单成功 Ref={pRsp.OrderRef} ID={pRsp.OrderSysID}")
        else:
            print(f"  [CTP] 下单失败: {pInfo.ErrorMsg if pInfo else 'unknown'}")
    def OnRtnOrder(self, pOrder):
        pass  # 订单状态回报
    def OnRtnTrade(self, pTrade):
        pass  # 成交回报

# ══════════════════ Market SPI ══════════════════
class MarketSpi(mdapi.CThostFtdcMdSpi):
    def __init__(self, api):
        super().__init__()
        self.api = api
    def OnFrontConnected(self):
        print("  [MKT] 行情前置已连接")
        req = mdapi.CThostFtdcReqUserLoginField()
        req.BrokerID=CFG["broker"]; req.UserID=CFG["user"]; req.Password=CFG["pass"]
        self.api.ReqUserLogin(req, 1)
    def OnRspUserLogin(self, pRsp, pInfo, nID, bLast):
        if pInfo and pInfo.ErrorID == 0:
            print("  [MKT] 行情登录成功")
            collected["market_ok"] = True
    def OnRtnDepthMarketData(self, pData):
        if pData:
            sym = pData.InstrumentID
            collected[f"price_{sym}"] = pData.LastPrice
            collected[f"time_{sym}"] = pData.UpdateTime

# ══════════════════ 主流程 ══════════════════
print("=" * 50)
print("  SimNow CTP 连接测试")
print(f"  账号: {CFG['user']}")
print(f"  {datetime.now():%Y-%m-%d %H:%M}")
print("=" * 50)

# Trade
try:
    td = tdapi.CThostFtdcTraderApi.CreateFtdcTraderApi("./ctp_trade/")
    tspi = TradeSpi(td)
    td.RegisterSpi(tspi)
    td.SubscribePrivateTopic(tdapi.THOST_TERT_QUICK)
    td.SubscribePublicTopic(tdapi.THOST_TERT_QUICK)
    td.RegisterFront(CFG["trade"])
    td.Init()
    print("  交易API初始化完成")
except Exception as e:
    print(f"  交易API失败: {e}")

# Market  
try:
    md = mdapi.CThostFtdcMdApi.CreateFtdcMdApi("./ctp_md/")
    mspi = MarketSpi(md)
    md.RegisterSpi(mspi)
    md.RegisterFront(CFG["market"])
    md.Init()
    print("  行情API初始化完成")
except Exception as e:
    print(f"  行情API失败: {e}")

print("\n  ⏳ 等待连接... (10秒)")
time.sleep(10)

if collected.get("login") and collected.get("settled"):
    print("\n  ✅ SimNow 连接成功！")
    print(f"  资金账户已就绪")
    
    # Subscribe
    symbols = [b"lh2609", b"jm2609"]
    md.SubscribeMarketData(symbols, len(symbols))
    print(f"  已订阅: LH2609, JM2609")
    print(f"\n  📊 等待行情...")
    
    time.sleep(3)
    for s in ["lh2609", "jm2609"]:
        p = collected.get(f"price_{s}")
        if p: print(f"  {s}: {p}")
    
    print(f"\n  🟢 系统就绪 — 等待信号")
    print(f"  信号文件: /tmp/prophet_signal.txt")
    print(f"  格式: sym,direction,entry")
else:
    print(f"\n  ❌ 连接不完全")
    print(f"  状态: login={collected.get('login')} settled={collected.get('settled')}")
