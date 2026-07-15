#!/bin/bash
cd /home/a/prophet_futures/prophet_futures
export LD_PRELOAD=/home/a/prophet_futures/prophet_futures/libcuda_stub.so
exec .venv/bin/python -u paper_trader.py
