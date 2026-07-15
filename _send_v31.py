#!/usr/bin/env python3
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from report_v4 import send, md

ele = [
    md('**V31 新版本设计方案**'),
    md(''),
    md('**当前痛点**'),
]

rows = [
    '| 问题 | 表现 |',
    '|------|------|',
    '| LH方向难预测 | 四个模型最高才50%,抛硬币水平 |',
    '| 模型偏见严重 | V29永远喊多 V30永远喊空,方向对了赚错了亏 |',
    '| 信号质量低 | 不区分高/低置信度,>50%就开仓 |',
    '| 逆势开仓 | 下跌趋势中V29还在喊多,被反复打脸 |',
]
ele.append(md('\n'.join(rows)))

ele.append(md(''))
ele.append(md('**V31方案: 三过滤器+动态退出**'))

rows2 = [
    '| 层级 | 过滤器 | 规则 | 作用 |',
    '|------|------|------|------|',
    '| 1 | 模型共识 | V29和V30方向一致才开仓 | 过滤偏见,不一致=观望 |',
    '| 2 | 趋势对齐 | 做多需价格>MA20,做空需价格<MA20 | 只顺大势,不逆势 |',
    '| 3 | 高置信度 | prob>60%或<40%才交易 | 只在模型确定时出手 |',
    '| 退出 | V28动态 | 移动止损+模型退出+保本 | 已证明优于V25固定止损 |',
]
ele.append(md('\n'.join(rows2)))

ele.append(md(''))
ele.append(md('**预期效果**'))

rows3 = [
    '| 指标 | 当前(V25-V30) | V31预期 |',
    '|------|------|------|',
    '| 交易频率 | 几乎每天开仓 | 3-4天一次 |',
    '| LH准确率 | 43-50% | 55-65%(过滤低质量信号) |',
    '| 信号偏见 | V29偏多100% V30偏空0% | 均衡(共识后才交易) |',
    '| 逆势交易 | 经常 | 被MA20趋势过滤掉 |',
    '| 最大回撤 | 12-15% | <10%(少交易=少犯错) |',
]
ele.append(md('\n'.join(rows3)))

ele.append(md(''))
ele.append(md('**实现**'))
ele.append(md('1. 新建 paper_trader_v31.py,基于V28框架'))
ele.append(md('2. 加载两个模型(_xgb_new + _xgb_calibrated)'))
ele.append(md('3. 每个tick: 两模型都预测→方向一致→检查MA20→检查置信度→开仓'))
ele.append(md('4. 退出沿用V28的trail+reverse+reduce逻辑'))
ele.append(md('5. 初始跑纸盘,攒20天数据后评估'))

ele.append(md(''))
ele.append(md('**要不要现在创建V31?**'))

send('V31设计方案', ele, 'blue')
print('OK')
