#!/bin/bash
# Prophet Futures — Start All Paper Traders
# NOTE: libcuda stub already in xgboost.libs — no LD_PRELOAD needed
set -e
cd /home/a/prophet_futures/prophet_futures
source .venv/bin/activate

pkill -f "paper_trader.py " 2>/dev/null || true
pkill -f "paper_trader_v28.py" 2>/dev/null || true
pkill -f "paper_trader_v29.py" 2>/dev/null || true
pkill -f "paper_trader_v30.py" 2>/dev/null || true
pkill -f "paper_trader_v32.py" 2>/dev/null || true
pkill -f "paper_trader_v32b.py" 2>/dev/null || true
pkill -f "paper_trader_v33.py" 2>/dev/null || true
pkill -f "paper_trader_v34.py" 2>/dev/null || true
pkill -f "paper_trader_v35.py" 2>/dev/null || true
sleep 2

nohup python3 -u paper_trader_v31.py > /home/a/prophet_futures/logs/paper_v31.log 2>&1 &
echo "V31  PID $!"
nohup python3 -u paper_trader.py > /home/a/prophet_futures/logs/paper_v25.log 2>&1 &
echo "V25  PID $!"
nohup python3 -u paper_trader_v28.py > /home/a/prophet_futures/logs/paper_v28.log 2>&1 &
echo "V28  PID $!"
nohup python3 -u paper_trader_v29.py --mode paper_trading --symbol lh > /home/a/prophet_futures/logs/paper_v29.log 2>&1 &
echo "V29  PID $!"
nohup python3 -u paper_trader_v30.py --mode paper_trading --symbol lh > /home/a/prophet_futures/logs/paper_v30.log 2>&1 &
echo "V30  PID $!"
nohup python3 -u paper_trader_v32.py > /home/a/prophet_futures/logs/paper_v32.log 2>&1 &
echo "V32  PID $!"
nohup python3 -u paper_trader_v32b.py > /home/a/prophet_futures/logs/paper_v32b.log 2>&1 &
echo "V32b PID $!"
nohup python3 -u paper_trader_v33.py > /home/a/prophet_futures/logs/paper_v33.log 2>&1 &
echo "V33  PID $!"
nohup python3 -u paper_trader_v34.py > /home/a/prophet_futures/logs/paper_v34.log 2>&1 &
echo "V34  PID $!"
nohup python3 -u paper_trader_v35.py > /home/a/prophet_futures/logs/paper_v35.log 2>&1 &
echo "V35  PID $!"
echo "done"
