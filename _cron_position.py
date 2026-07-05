import json, os, sys

sf = 'paper_state.json'
if os.path.exists(sf):
    with open(sf) as f:
        s = json.load(f)
    pos = s.get('positions', {})
    if pos:
        print('📌 当前持仓')
        for k, v in pos.items():
            print(f"  {k}: {v['dir']} {v['vol']}手 @{v['entry']:.0f} | 止损{v['stop']:.0f} 止盈{v['take_profit']:.0f}")
    else:
        print('📌 无持仓')
    print(f"💰 现金 ¥{s['cash']:,.0f}")
else:
    print('📌 paper_state.json 不存在')
