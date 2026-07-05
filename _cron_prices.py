import akshare as ak
import pandas as pd
from datetime import datetime

now = datetime.now()
today_str = now.strftime('%Y-%m-%d')

for sym in ['lh2609', 'jm2609']:
    try:
        df = ak.futures_zh_minute_sina(symbol=sym.upper(), period='5')
        if df is None or len(df) == 0:
            print(f'{sym}: 获取失败（无数据）')
            continue

        df['dt'] = pd.to_datetime(df['datetime'])
        today = df[df['dt'].dt.strftime('%Y-%m-%d') == today_str]

        if len(today) == 0:
            # Try latest row anyway
            latest = df.iloc[-1]
            print(f'{sym}: 暂无今日数据 | 最新收盘{latest["close"]:.0f} (时间{latest["datetime"]})')
            continue

        latest = today.iloc[-1]
        high = today['high'].max()
        low = today['low'].min()
        open_price = today.iloc[0]['open']
        chg = (latest['close'] - open_price) / open_price if open_price else 0

        print(f'{sym}: 现价{latest["close"]:.0f} | 日高{high:.0f} 日低{low:.0f} | 涨跌{chg:+.1%} | 时间{latest["datetime"]}')
    except Exception as e:
        print(f'{sym}: 获取失败 ({e})')
