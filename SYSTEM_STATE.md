# Prophet Futures v2.1 — System State Snapshot
# Date: 2026-06-18

## Strategy Parameters (Recommended)
{
  "strategy": "稳健方向 (Conservative-Regime)",
  "min_conditions": 7,
  "stop_atr_mult": 1.5,
  "target_atr_mult": 3.0,
  "require_regime_filter": true,
  "require_volume_confirm": false,
  "use_dynamic_sizing": false,
  "use_pyramiding": false,
  "use_time_stop": true,
  "time_stop_days": 8
}

## Performance (3-year LH backtest)
- Trades: 48 (1.1/month)
- Win Rate: 54%
- Total PnL: +373,338 CNY (+37.3%)
- Annualized: ~12.4%
- Max Drawdown: 8.4%
- Sharpe Ratio: 5.642
- Profit Factor: ~2.0

## Agent Status (8/8 AI-driven)
✅ Technician   ✅ Fund Analyst   ✅ Macro Analyst
✅ Scenario     ✅ Trap Detector   ✅ Memory Retriever
✅ Vision Tech  ✅ Causal Reasoner

## Infrastructure
- LLM: DeepSeek v4-pro
- Gateway: Feishu (飞书) connected
- Cron: Daily LH analysis @ 15:00 Mon-Fri → Feishu DM
- GitHub: https://github.com/1685yhy/prophet-futures (needs token with repo scope)

## Key Files Modified
- agents/*.py: All @tool decorator refactor
- tools/causal_graph.py: Full LH supply chain graph
- tools/llm_utils.py: JSON extraction + schema tolerance
- tools/market_data.py: Volume column fix
- graph/workflow.py: Enhanced output + symbol override
- config.yaml: DeepSeek provider config
- advanced_strategy.py: New strategy engine
- comprehensive_backtest.py: 3-year backtest + WF
- fine_tune.py: Sweet spot search
- daily_run.sh: Daily analysis script

## Rollback
git log --oneline
git checkout <hash>
