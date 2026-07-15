#!/usr/bin/env python3
"""Wrapper to import xgboost before other heavy libraries to avoid glibc 2.39 bug."""
import xgboost  # Must be FIRST to avoid ld.so assertion failure
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import runpy
runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scan_5min.py'), run_name='__main__')
